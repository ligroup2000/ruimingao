# -*- coding: utf-8 -*-
"""论文算例4.1；rigid body rotation
直接运行: python rigid_body_rk_pinn_two_newton.py
依赖: pip install numpy scipy matplotlib torch
默认T=50,h=0.1,float64。输出CSV/PNG/JSON/PT，支持--device cuda。
两阶段：RK网络训练 -> 可微牛顿能量层+松弛RK残差+连续ODE残差。
两个校正参数同时Newton求解G1、G2守恒；实际dt=(1+gamma1+gamma2)*h。
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import ellipj
from scipy.optimize import root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALPHA=1+1/math.sqrt(1.51)
BETA=1-0.51/math.sqrt(1.51)
# 原本的RK方法burcher表
A_NP=np.array([[1/9,(-1-math.sqrt(6))/18,(-1+math.sqrt(6))/18],
 [1/9,(88+7*math.sqrt(6))/360,(88-43*math.sqrt(6))/360],
 [1/9,(88+43*math.sqrt(6))/360,(88-7*math.sqrt(6))/360]])
B_NP=np.array([1/9,(16+math.sqrt(6))/36,(16-math.sqrt(6))/36])
C_NP=np.array([0.,(6-math.sqrt(6))/10,(6+math.sqrt(6))/10])
Q_NP=np.array([1.,BETA,ALPHA])
# 新增的嵌入方法：sum b_tilde=1,b_tilde@c=1/2；只要求二阶。
B_EMBED_NP=np.array([0.,(C_NP[2]-.5)/(C_NP[2]-C_NP[1]),
                       (.5-C_NP[1])/(C_NP[2]-C_NP[1])])
QS_NP=np.stack((np.ones(3),Q_NP))
Y_INITIAL=np.array([0.,1.,1.])


# the source term of the function
def rhs(y):
    if isinstance(y,torch.Tensor):
        return torch.stack(((ALPHA-BETA)*y[...,1]*y[...,2],
          (1-ALPHA)*y[...,2]*y[...,0],(BETA-1)*y[...,0]*y[...,1]),dim=-1)
    return np.stack(((ALPHA-BETA)*y[...,1]*y[...,2],
          (1-ALPHA)*y[...,2]*y[...,0],(BETA-1)*y[...,0]*y[...,1]),axis=-1)


def exact(t):
    # scipy的第二个参数为m=k^2；此处m=0.51，与论文解的微分方程一致。
    sn,cn,dn,_=ellipj(np.asarray(t),0.51)
    return np.stack((math.sqrt(1.51)*sn,cn,dn),axis=-1)

# conservation [G1,G2]
def invariants(y):
    return np.stack((np.sum(y*y,axis=-1),np.sum(Q_NP*y*y,axis=-1)),axis=-1)


class NewtonConvergenceError(RuntimeError):
    pass


class NewtonTwoGamma(torch.autograd.Function):
    """F=(G1-target1,G2-target2)，2x2阻尼Newton；两参数从(0,0)开始。
    论文式(11)：u=y+(1+g1)*d1+g2*d2，实际dt=(1+g1+g2)*h。
    小残差采用二次型增量表达式，避免每次相减两个大能量。
    """
    @staticmethod
    def forward(ctx,y,d1,d2,qs,target):
        D=torch.stack((d1,d2),dim=1)
        star=y+d1
        # F(0)用展开形式；固定全局初始守恒量。
        delta=(qs*y.square()).sum(1)-target
        F0=delta+2*(qs*y)@d1+qs@d1.square()
        gamma=d1.new_zeros(2)
        def evaluate(g):
            correction=D@g
            u=star+correction
            F=F0+2*(qs*star)@correction+qs@correction.square()
            J=2*(qs*u)@D #jacobi矩阵
            return F,J,u
        for _ in range(30):
            F,J,u=evaluate(gamma)
            if float(F.abs().max())<=2e-15:
                break
            cond=float(torch.linalg.cond(J))
            # 计算条件数，判断jacobi矩阵是否为病态矩阵
            if not math.isfinite(cond) or cond>1e12:
                raise NewtonConvergenceError(f'双参数Newton雅可比病态: cond={cond:.3e}')
            # 列缩放改善线性求解的量纲，返回的是原参数的Newton方向。
            scale=torch.linalg.vector_norm(J,dim=0)
            direction=torch.linalg.solve(J/scale, F)/scale
            damping=1.
            for _ in range(30):
                # newton迭代
                candidate=gamma-damping*direction
                theta=1+candidate.sum()
                # 判断gamma是否与0相距过大
                if float(candidate.abs().max())<=.25 and .5<=float(theta)<=1.5:
                    Fc,_,_=evaluate(candidate)
                    if float(Fc.abs().max())<float(F.abs().max()):
                        gamma=candidate
                        break
                damping*=.5
            else:
                raise NewtonConvergenceError('双参数Newton线搜索失败，本步不接受。')
        F,J,u=evaluate(gamma)
        if float(F.abs().max())>1e-13:
            raise NewtonConvergenceError('双参数Newton未收敛，本步不接受。')
        if float(torch.linalg.cond(J))>1e12:
            raise NewtonConvergenceError('隐函数雅可比退化，无法可靠反向传播。')
        ctx.save_for_backward(u,qs,gamma,J)
        return gamma

    @staticmethod
    def backward(ctx,grad_gamma):
        u,qs,gamma,J=ctx.saved_tensors
        # J*d_gamma/d_theta=-F_theta；反向仅解一次2x2转置系统。
        adjoint=torch.linalg.solve(J.T,grad_gamma)
        common=-(2*qs*u).T@adjoint
        return common,(1+gamma[0])*common,gamma[1]*common,None,adjoint


class PINN(nn.Module):
    """只输入局部时间s的紧凑多项式网络，无空间变量。
    RK阶段是代数阶段，C[0]=0但A[0,:]非零，不把第一阶段强制为y_n。
    连续时间分支单独表示物理轨迹，避免把代数阶段误当真实时刻解。
    其y(0)=y_n、dy/dt(0)=f(y_n)、y(1)=y_gamma均为硬约束。
    """
    def __init__(self,y,h,A,B,C):
        super().__init__()
        self.y,self.h,self.A,self.B,self.C=y,h,A,B,C
        self.register_buffer("B_embed",y.new_tensor(B_EMBED_NP))
        self.register_buffer('nodes',torch.cat((C,C.new_tensor([1.]))))
        self.rk=nn.Linear(4,3,bias=False).to(device=y.device,dtype=y.dtype)
        self.curve=nn.Linear(6,3,bias=False).to(device=y.device,dtype=y.dtype)
        with torch.no_grad():
            predicted=rhs(y+self.nodes[:,None]*h*rhs(y))
            self.rk.weight.copy_(predicted.T*self.nodes[None,:])
            self.curve.weight.zero_()

    def stage_increment(self,s):
        features=[]
        for i in range(4):
            p=torch.ones_like(s)
            for j in range(4):
                if i!=j:
                    p=p*(s-self.nodes[j])/(self.nodes[i]-self.nodes[j])
            features.append(p)
        return self.h*self.rk(torch.stack(features,dim=-1))

    def rk_fields(self):
        Y=self.y+self.stage_increment(self.C)
        f=rhs(Y)
        d=self.stage_increment(self.C.new_tensor([1.]))[0]
        stage=(Y-self.y)/self.h-self.A@f
        endpoint=d/self.h-self.B@f
        return d,f,stage,endpoint

    def physical_solution(self,s,increment,theta):
        dt=theta*self.h
        features=torch.stack([s**i for i in range(6)],dim=-1)
        correction=self.curve(features)
        return (self.y+dt*s[:,None]*rhs(self.y)+s[:,None]**2*
          (increment-dt*rhs(self.y)+(1-s[:,None])*correction))

    def losses(self,phase,q,target):
        d,f,stage,endpoint=self.rk_fields()
        rk=stage.square().mean()+endpoint.square().mean()
        if phase==1:
            return rk,rk,rk.new_tensor(0.),d,d.new_tensor(1.)
        d2=self.h*(self.B_embed@f)
        gamma=NewtonTwoGamma.apply(self.y,d,d2,q,target)
        theta=1+gamma.sum()
        increment=(1+gamma[0])*d+gamma[1]*d2
        relaxed=(1+gamma[0])*endpoint
        s=torch.linspace(0.,1.,12,device=d.device,dtype=d.dtype,requires_grad=True)
        u=self.physical_solution(s,increment,theta)
        derivative=torch.stack([torch.autograd.grad(u[:,k].sum(),s,
            create_graph=True,retain_graph=True)[0] for k in range(3)],dim=1)/(theta*self.h)

        ode = (derivative-rhs(u)).square().mean()
        # 高权重控制RK训练误差，连续ODE分支通过gamma反传微调网络。
        total = 1e6 * (rk + relaxed.square().mean()) + ode
        return total,rk,ode,increment,gamma


def optimize(model,phase,q,target,max_iter):
    if phase==2:
        # 先拟合连续轨迹分支，避免初始ODE梯度破坏已求准的RK阶段。
        for p in model.rk.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            d,f,_,_=model.rk_fields()
            d2=model.h*(model.B_embed@f)
            g=NewtonTwoGamma.apply(model.y,d,d2,q,target)
            increment=(1+g[0])*d+g[1]*d2
            dt=(1+g.sum())*model.h
            hermite=2*increment-dt*(rhs(model.y)+rhs(model.y+increment))
            model.curve.weight[:,0].copy_(hermite)
        # 优化函数
        curve_optimizer=torch.optim.LBFGS(model.curve.parameters(),lr=1.,
            max_iter=100,tolerance_grad=1e-13,tolerance_change=1e-24,
            history_size=30,line_search_fn='strong_wolfe')

        # LBFGS需要运行closure
        def curve_closure():
            curve_optimizer.zero_grad()
            ode=model.losses(2,q,target)[2]
            ode.backward()
            return ode
        curve_optimizer.step(curve_closure)
        for p in model.rk.parameters():
            p.requires_grad_(True)
    parameters=list(model.rk.parameters()) if phase==1 else list(model.parameters())
    optimizer=torch.optim.LBFGS(parameters,lr=1.,max_iter=max_iter,
        max_eval=max_iter*2,tolerance_grad=1e-13,tolerance_change=1e-22,
        history_size=30,line_search_fn='strong_wolfe')
    reference=[p.detach().clone() for p in parameters]
    best_loss=float('inf');best_state=None
    def closure():
        nonlocal best_loss,best_state
        optimizer.zero_grad()
        try:
            loss=model.losses(phase,q,target)[0]
        except NewtonConvergenceError:
            if phase!=2:
                raise
            # 无守恒根的试探参数不作为有效解；返回大惩罚促使线搜索回退。
            penalty=sum(1e6*(p-r).square().sum() for p,r in zip(parameters,reference))+1.
            penalty.backward()
            return penalty
        if not torch.isfinite(loss):
            raise RuntimeError('loss非有限值。')
        if float(loss.detach())<best_loss:
            best_loss=float(loss.detach());best_state=copy.deepcopy(model.state_dict())
        loss.backward()
        return loss
    optimizer.step(closure)
    if best_state is None:
        raise RuntimeError('优化未找到有效的双松弛模型。')
    model.load_state_dict(best_state)
    return model.losses(phase,q,target)


# 主程序
def calculate_pinn(args):
    device=torch.device(args.device if args.device!='auto' else
                        ('cuda' if torch.cuda.is_available() else 'cpu'))
    tensor=lambda a:torch.tensor(a,device=device,dtype=torch.float64)
    A,B,C,qs=map(tensor,(A_NP,B_NP,C_NP,QS_NP))
    y=tensor(Y_INITIAL); target=(qs*y.square()).sum(1).detach()
    # t,u1,u2,u3,G1,G2,g1,g2,theta,h,rk_rms,ode_mse,error,condJ
    rows=[[0.,*Y_INITIAL,*invariants(Y_INITIAL),0.,0.,1.,0.,0.,0.,0.,0.]]
    snapshots=[]; t=0.; start=time.perf_counter()
    for n in range(math.ceil(args.T/args.h)+100):
        remaining=args.T-t
        if remaining<=1e-12:
            break
        h=min(args.h,remaining)
        for end_iteration in range(12):
            model=PINN(y,h,A,B,C)
            _,rk0,_,_,_=optimize(model,1,qs,target,args.rk_iters)
            if math.sqrt(float(rk0.detach()))>args.rk_tol:
                raise RuntimeError(f'阶段1 RK残差不合格: {float(rk0):.3e}')
            _,rk,ode,increment,g=optimize(model,2,qs,target,args.relax_iters)
            theta=1+float(g.detach().sum()); dt=theta*h
            if h<args.h*.999 or remaining<args.h*1.5:
                if abs(dt-remaining)<1e-12 or (dt<remaining and h==args.h):
                    break
                h=h*remaining/dt
            else:
                break
        else:
            raise RuntimeError('最后一步实际时间匹配未收敛。')
        rms=math.sqrt(float(rk.detach()))
        if rms>args.rk_tol:
            raise RuntimeError(f'阶段2 RK残差{rms:.3e}超出{args.rk_tol:.3e}')
        ynew=(y+increment).detach()
        value=ynew.cpu().numpy(); inv=invariants(value)
        if np.max(np.abs(inv-target.cpu().numpy()))>1e-13:
            raise RuntimeError('接受前双守恒量复核失败。')
        if dt>remaining+1e-12:
            raise RuntimeError('终点越界。')
        d,f,_,_=model.rk_fields(); d2=h*(model.B_embed@f)
        J=2*(qs*ynew)@torch.stack((d,d2),dim=1)
        cond=float(torch.linalg.cond(J).detach())
        t+=dt; error=np.linalg.norm(value-exact(t))
        gv=g.detach().cpu().tolist()
        rows.append([t,*value,*inv,*gv,theta,h,rms,float(ode.detach()),error,cond])
        snapshots.append({'t_start':t-dt,'h':h,'gamma':gv,'theta':theta,
            'y_start':y.cpu(),'state_dict':model.state_dict()})
        y=ynew
        if (n+1)%10==0 or args.T-t<1e-12:
            print(f'ACCEPT {n+1:3d} t={t:.12f} G1={inv[0]:.15f} '
              f'G2={inv[1]:.15f} g1={gv[0]:+.3e} g2={gv[1]:+.3e} '
              f'RK={rms:.2e} ODE={math.sqrt(float(ode.detach())):.2e} '
              f'error={error:.2e} condJ={cond:.2e}',flush=True)
    else:
        raise RuntimeError('超过最大步数。')
    torch.save({'steps':snapshots,'A':A_NP,'B':B_NP,'B_embed':B_EMBED_NP,
                'C':C_NP,'alpha':ALPHA,'beta':BETA,'args':vars(args)},args.out/'models.pt')
    return np.asarray(rows),time.perf_counter()-start,str(device)


def calculate_baseline(T,h):
    """相同Butcher系数；独立高精度代数RK求解，无网络、无松弛，用作核验。"""
    y=Y_INITIAL.copy(); t=0.; rows=[[t,*y,*invariants(y),0.]]
    while T-t>1e-12:
        step=min(h,T-t)
        def residual(z):
            Y=z.reshape(3,3)
            return (Y-y-step*A_NP@rhs(Y)).ravel()
        sol=root(residual,(y+C_NP[:,None]*step*rhs(y)).ravel(),tol=1e-11)
        if np.max(np.abs(residual(sol.x)))>1e-12:
            raise RuntimeError('基准RK阶段求解失败。')
        y=y+step*B_NP@rhs(sol.x.reshape(3,3)); t+=step
        rows.append([t,*y,*invariants(y),np.linalg.norm(y-exact(t))])
    return np.asarray(rows)


def calculate_unrelaxed_pinn(args):
    """独立推进的RK-PINN：仅训练阶段1，不作守恒或时间松弛。"""
    device=torch.device(args.device if args.device!='auto' else
                        ('cuda' if torch.cuda.is_available() else 'cpu'))
    tensor=lambda a:torch.tensor(a,device=device,dtype=torch.float64)
    A,B,C,qs=map(tensor,(A_NP,B_NP,C_NP,QS_NP))
    y=tensor(Y_INITIAL); target=(qs*y.square()).sum(1).detach()
    # time,u1,u2,u3,G1,G2,exact_error_l2,rk_rms
    rows=[[0.,*Y_INITIAL,*invariants(Y_INITIAL),0.,0.]]
    t=0.; start=time.perf_counter(); n=0
    while args.T-t>1e-12:
        h=min(args.h,args.T-t)
        model=PINN(y,h,A,B,C)
        _,rk,_,increment,_=optimize(model,1,qs,target,args.rk_iters)
        rms=math.sqrt(float(rk.detach()))
        if rms>args.rk_tol:
            raise RuntimeError(f'无松弛RK-PINN残差{rms:.3e}超出{args.rk_tol:.3e}')
        y=(y+increment).detach(); t+=h; n+=1
        value=y.cpu().numpy(); inv=invariants(value)
        error=np.linalg.norm(value-exact(t))
        rows.append([t,*value,*inv,error,rms])
        if n%25==0 or args.T-t<=1e-12:
            print(f'UNRELAXED {n:3d} t={t:.12f} RK={rms:.2e} error={error:.2e}',flush=True)
    return np.asarray(rows),time.perf_counter()-start


def summarize_unrelaxed(unrelaxed,elapsed):
    initial=invariants(Y_INITIAL)
    errors=np.abs(unrelaxed[:,4:6]-initial)
    return {'method':'RK-PINN phase 1 only, independent fixed-time-step rollout',
        'steps':len(unrelaxed)-1,'final_time':float(unrelaxed[-1,0]),
        'final_state':unrelaxed[-1,1:4].tolist(),
        'elapsed_seconds':elapsed,
        'max_invariant_errors':errors.max(axis=0).tolist(),
        'max_relative_invariant_errors':(errors/np.abs(initial)).max(axis=0).tolist(),
        'max_solution_error':float(unrelaxed[:,6].max()),
        'final_solution_error':float(unrelaxed[-1,6]),
        'max_rk_rms':float(unrelaxed[:,7].max())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--T',type=float,default=50.)
    parser.add_argument('--h',type=float,default=.1)
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--rk-iters',type=int,default=100)
    parser.add_argument('--relax-iters',type=int,default=100)
    parser.add_argument('--rk-tol',type=float,default=1e-7)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'rigid_body_two_T10')
    args=parser.parse_args()
    if args.h<=0 or args.T<=0:
        parser.error('T和h必须为正。')
    torch.set_num_threads(args.threads); torch.manual_seed(42)
    args.out.mkdir(parents=True,exist_ok=True)
    print(f'alpha={ALPHA:.15f} beta={BETA:.15f}; T={args.T} h={args.h}',flush=True)
    relaxed,elapsed,device=calculate_pinn(args)
    baseline=calculate_baseline(args.T,args.h)
    unrelaxed,unrelaxed_elapsed=calculate_unrelaxed_pinn(args)
    np.savetxt(args.out/'relaxed_history.csv',relaxed,delimiter=',',comments='',
        header='actual_time,u1,u2,u3,G1,G2,gamma1,gamma2,time_factor,nominal_h,rk_rms,ode_mse,exact_error_l2,newton_jacobian_condition')
    np.savetxt(args.out/'baseline_history.csv',baseline,delimiter=',',comments='',
        header='time,u1,u2,u3,G1,G2,exact_error_l2')
    np.savetxt(args.out/'unrelaxed_pinn_history.csv',unrelaxed,delimiter=',',comments='',
        header='time,u1,u2,u3,G1,G2,exact_error_l2,rk_rms')
    initial=invariants(Y_INITIAL)
    summary={'T':args.T,'nominal_h':args.h,'alpha':ALPHA,'beta':BETA,
      'initial_invariants':initial.tolist(),'device':device,'elapsed_seconds':elapsed,
      'method':'original RK + second-order embedded weights + time-only PINN + differentiable 2x2 Newton G1/G2 relaxation', 'embedded_weights':B_EMBED_NP.tolist(), 'embedded_order':2,
      'relaxed':{'steps':len(relaxed)-1,'final_time':relaxed[-1,0],
        'final_state':relaxed[-1,1:4].tolist(),
        'max_invariant_errors':np.max(np.abs(relaxed[:,4:6]-initial),axis=0).tolist(),
        'max_solution_error':float(np.max(relaxed[:,12])),
        'final_solution_error':float(relaxed[-1,12]),
        'max_rk_rms':float(np.max(relaxed[:,10])),
        'max_ode_rms':float(np.sqrt(np.max(relaxed[:,11]))),
        'gamma_ranges':[[float(relaxed[1:,j].min()),float(relaxed[1:,j].max())] for j in (6,7)], 'time_factor_range':[float(relaxed[1:,8].min()),float(relaxed[1:,8].max())], 'max_jacobian_condition':float(relaxed[1:,13].max())},
      'baseline':{'steps':len(baseline)-1,
        'max_invariant_errors':np.max(np.abs(baseline[:,4:6]-initial),axis=0).tolist(),
        'final_solution_error':float(baseline[-1,-1])},
       'unrelaxed_pinn':summarize_unrelaxed(unrelaxed,unrelaxed_elapsed),
       'exact_at_T':exact(args.T).tolist()}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plot_conservation(relaxed,baseline,unrelaxed,args.T,args.out)
    print(json.dumps(summary,indent=2),flush=True)


def plot_conservation(relaxed,baseline,unrelaxed,T,out):
    initial=invariants(Y_INITIAL)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    tt=np.linspace(0,T,1000)
    fig, axes = plt.subplots(1, 1, layout='constrained')
    for k in range(3):
        color=f'C{k}'
        axes.plot(tt,exact(tt)[:,k],color=color,label=f'exact u{k+1}')
        axes.plot(relaxed[:,0],relaxed[:,1+k],'.',color=color,ms=3,label=f'relaxed u{k+1}')
        # axes[0,0].plot(unrelaxed[:,0],unrelaxed[:,1+k],'--',color=color,lw=1,label=f'unrelaxed u{k+1}')
    axes.set_title('The simulation of rigid body'); axes.legend(ncol=3,fontsize=8)
    fig.savefig(out / 'prediction.png', dpi=180);
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')
    relaxed_relative=np.abs(relaxed[:,4:6]-initial)/np.abs(initial)
    unrelaxed_relative=np.abs(unrelaxed[:,4:6]-initial)/np.abs(initial)
    for k in range(2):
        axes[0].loglog(relaxed[1:,0],np.maximum(relaxed_relative[1:,k],1e-17),label=f'relaxation method')
        axes[0].loglog(unrelaxed[1:,0],np.maximum(unrelaxed_relative[1:,k],1e-17),':',label=f'unrelaxation method')
    axes[0].set_title('Relative invariant errors (log-log, floor 1e-17)'); axes[0,1].legend()
    axes[0].set_ylabel(r'$|G_1(t)-G_1(0)|/|G_1(0)|$')

    for k in range(2):
        axes[1].semilogy(relaxed[1:,0],np.maximum(relaxed_relative[1:,k],1e-17),label=f'relaxation method')
        axes[1].semilogy(unrelaxed[1:,0],np.maximum(unrelaxed_relative[1:,k],1e-17),':',label=f'unrelaxation method')
    axes[1].set_title('Relative invariant errors (floor 1e-17)'); axes[1,0].legend()
    axes[1].set_ylabel(r'$|G_2(t)-G_2(0)|/|G_2(0)|$')
    for ax in axes.ravel():
        ax.set_xlabel('t'); ax.grid(alpha=.25)
    fig.savefig(out/'conservation.png',dpi=180); plt.close(fig)

if __name__=='__main__':
    main()
