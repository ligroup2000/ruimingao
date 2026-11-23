# -*- coding: utf-8 -*-
"""3维周期 Schrödinger--Klein--Gordon 耦合算例4.3.2。

i*u_t + kappa1/2*Delta(u) + xi*u*v = 0，
v_tt - kappa2*Delta(v) + mu^2*v - xi*|u|^2 = 0。
默认区域 [-20,20)^2，不重复右端点；state=(Re(u),Im(u),v,w)，w=v_t。

"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


COEFFICIENTS = dict(kappa1=-.4, kappa2=.1, gamma=.2, mu=.1, nu=.01)
STATE_COMPONENTS = ['real_u', 'imag_u', 'v', 'v_t']
SNAPSHOT_TIMES = (0., .75, 2., 3.25)


def configure(threads, seed, device):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch.device('cuda' if device == 'auto' and torch.cuda.is_available()
                        else 'cpu' if device == 'auto' else device)


class PeriodicGrid:
    def __init__(self, n, length, device):
        if n < 4 or n % 2 or not math.isfinite(length) or length <= 0:
            raise ValueError('n must be even and >=4; length must be finite and positive')
        self.n, self.length = n, length
        self.dx = length/n
        self.cell_volume = self.dx**3
        self.x = -length/2+self.dx*torch.arange(n, device=device)
        k = 2*math.pi*torch.fft.fftfreq(n, d=self.dx, device=device)
        self.k_squared = (k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2)
        k = k.clone()
        k[n//2] = 0  # Real first derivatives have zero Nyquist multiplier.
        self.ks = (k[:,None,None], k[None,:,None], k[None,None,:])

    def physical_coordinates(self):
        return torch.meshgrid(self.x, self.x, self.x, indexing='ij')

    def coordinate_arrays(self):
        return {name:self.x.cpu().numpy() for name in ('x1','x2','x3')}

    @staticmethod
    def fft(f):
        return torch.fft.fftn(f, dim=(-3,-2,-1))

    @staticmethod
    def ifft(f):
        return torch.fft.ifftn(f, dim=(-3,-2,-1)).real

    def derivative(self, f, axis):
        return self.ifft(1j*self.ks[axis]*self.fft(f))

    def laplacian(self, f):
        return self.ifft(-self.k_squared*self.fft(f))

    def rhs(self, state):
        real,imag,v,w = state.unbind(dim=-4)
        c = COEFFICIENTS
        return torch.stack([
            -.5*c['kappa1']*self.laplacian(imag)-c['gamma']*v*imag,
            .5*c['kappa1']*self.laplacian(real)+c['gamma']*v*real,
            w, c['kappa2']*self.laplacian(v)-c['mu']**2*v+c['gamma']*(real**2+imag**2)], dim=-4)

    def mass(self, state):
        real,imag,_,_ = state.unbind(dim=-4)
        return self.cell_volume*(real**2+imag**2).sum(dim=(-3,-2,-1))

    def energy(self, state):
        real,imag,v,w = state.unbind(dim=-4)
        c = COEFFICIENTS
        # -q*Delta(q) includes the Nyquist contribution consistently with rhs.
        density = -.5*c['kappa1']*(real*self.laplacian(real)+imag*self.laplacian(imag))
        density = density+.5*(w**2-c['kappa2']*v*self.laplacian(v)+c['mu']**2*v**2)
        density = density-c['gamma']*(real**2+imag**2)*v
        return self.cell_volume*density.sum(dim=(-3,-2,-1))

    def invariants(self, state):
        return torch.stack([self.mass(state), self.energy(state)],dim=-1)

    def invariant_gradients(self, state):
        real,imag,v,w = state.unbind(dim=-4)
        rhs = self.rhs(state)
        gm = torch.stack([2*real,2*imag,torch.zeros_like(v),torch.zeros_like(w)],dim=-4)
        ge = torch.stack([-2*rhs.select(-4,1),2*rhs.select(-4,0),-rhs.select(-4,3),w],dim=-4)
        return self.cell_volume*torch.stack([gm,ge])


def initial_state(grid):
    x1,x2,x3 = grid.physical_coordinates()
    amplitude = torch.zeros_like(x1)
    v = torch.exp(-x1**2-x2**2-(x3-2)**2)
    for k in range(2):
        amplitude += torch.exp(-(x1+(-1)**k*2)**2-x2**2-x3**2)
        v += torch.exp(-(x1+(-1)**k*math.sqrt(3))**2-x2**2-(x3+1)**2)
    phase = COEFFICIENTS['nu']*(x1+x2+x3)
    w = torch.exp(-x1**2-x2**2-x3**2)
    return torch.stack([amplitude*torch.cos(phase),amplitude*torch.sin(phase),v,w])


class SpectralConv3d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(4,width,width,modes,modes,modes,2)/(width*math.sqrt(2)))

    def forward(self, z):
        zh = torch.fft.rfftn(z, dim=(-3,-2,-1))
        m = min(self.modes,z.shape[-3]//2,z.shape[-2]//2)
        mz = min(self.modes,zh.shape[-1])
        out = torch.zeros_like(zh)
        weights = torch.view_as_complex(self.weight.contiguous())
        bands = (slice(0,m),slice(-m,None))
        for i,sx in enumerate(bands):
            for j,sy in enumerate(bands):
                out[...,sx,sy,:mz] = torch.einsum('bixyz,ioxyz->boxyz',
                    zh[...,sx,sy,:mz],weights[2*i+j,...,:m,:m,:mz])
        return torch.fft.irfftn(out,s=z.shape[-3:],dim=(-3,-2,-1))


class FourierCNN(nn.Module):
    def __init__(self, channels, width, modes):
        super().__init__()
        self.lift = nn.Conv3d(channels,width,1)
        self.spectral = nn.ModuleList([SpectralConv3d(width,modes) for _ in range(2)])
        self.local = nn.ModuleList([nn.Conv3d(width,width,1) for _ in range(2)])
        self.project = nn.Conv3d(width,4,1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, features):
        z = self.lift(features)
        for spectral,local in zip(self.spectral,self.local):
            z = torch.tanh(spectral(z)+local(z))
        return self.project(z)


def lobatto_iiia_tableau(state):
    """四阶段六阶 Lobatto IIIA 主表；六阶指未松弛的主 RK 法。

    来源：https://juliagni.github.io/RungeKutta.jl/stable/lobatto/
    b2 为独立的一阶辅助权重，用于现有双参数 MRRK，不是六阶权重。
    """
    r = math.sqrt(5.)
    c = state.new_tensor([0., .5-r/10, .5+r/10, 1.])
    a = state.new_tensor([
        [0., 0., 0., 0.],
        [(11+r)/120, (25-r)/120, (25-13*r)/120, (r-1)/120],
        [(11-r)/120, (25+13*r)/120, (25+r)/120, -(1+r)/120],
        [1/12, 5/12, 5/12, 1/12],
    ])
    b1 = state.new_tensor([1/12, 5/12, 5/12, 1/12])
    b2 = state.new_tensor([1/4, 1/4, 1/4, 1/4])
    return a, b1, b2, c


def gauss_legendre_tableau(state):
    """三阶段、六阶 Gauss-Legendre 配置法与一套嵌入权重。

    c 是 [0,1] 上的三个 Gauss 点。
    A_ij = integral_0^{c_i} L_j(s) ds，b1_j = integral_0^1 L_j(s) ds，
    其中 L_j 为这三个节点上的 Lagrange 基函数。
    保留当前 b2=(1/6,1/6,2/3)，sum(b2)=1，仅保证一阶。
    配置主表来源：https://juliagni.github.io/RungeKutta.jl/stable/gauss/
    """
    r = math.sqrt(15.)
    c = state.new_tensor([.5-r/10, .5, .5+r/10])
    a = state.new_tensor([
        [5/36,       2/9-r/15, 5/36-r/30],
        [5/36+r/24,  2/9,      5/36-r/24],
        [5/36+r/30,  2/9+r/15, 5/36],
    ])
    b1 = state.new_tensor([5/18, 4/9, 5/18])
    b2 = state.new_tensor([1/6, 1/6, 2/3])
    return a, b1, b2, c



class RKStep(nn.Module):
    def __init__(self, grid, y_n, h, width=8, modes=4, method='lobatto'):
        super().__init__()
        self.grid,self.h = grid,h
        self.register_buffer('y_n',y_n.detach().clone())
        self.method = method
        tableau = {'gauss': gauss_legendre_tableau,
                   'lobatto': lobatto_iiia_tableau}[method]
        for name,value in zip(('a','b1','b2','c'),tableau(y_n)):
            self.register_buffer(name,value)
        self.stage_count = len(self.c)
        self.network = FourierCNN(5,width,min(modes,grid.n//2))

    def predict_slopes(self):
        features = torch.stack([torch.cat([self.y_n,c*torch.ones_like(self.y_n[:1])]) for c in self.c])
        return self.network(features)

    def stage_states(self, K):
        return self.y_n[None]+self.h*torch.einsum('ij,jcxyz->icxyz',self.a,K)

    def residual(self, K):
        return K-self.grid.rhs(self.stage_states(K))

    def rk_loss(self):
        return self.residual(self.predict_slopes()).square().mean()

    def rk_directions(self,K):
        return [torch.einsum('i,icxyz->cxyz',b,K) for b in (self.b1,self.b2)]


def optimize(network, objective, adam, lbfgs, lr):
    if adam:
        opt = torch.optim.Adam(network.parameters(),lr=lr)
        for _ in range(adam):
            opt.zero_grad()
            loss = objective()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite FNO loss')
            loss.backward()
            opt.step()
    if lbfgs:
        opt = torch.optim.LBFGS(network.parameters(),max_iter=lbfgs,line_search_fn='strong_wolfe',
                                tolerance_grad=1e-10,tolerance_change=1e-14)
        def closure():
            opt.zero_grad()
            loss = objective()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite FNO loss')
            loss.backward()
            return loss
        opt.step(closure)


@torch.no_grad()
def polish_rk_stages(step,K,iterations,tol):
    """FNO-initialized Picard iteration, implicit linear Schrodinger/wave solves."""
    g,h,A = step.grid,step.h,step.a
    A2 = A@A
    eye = torch.eye(step.stage_count,device=K.device,dtype=K.dtype)
    stages = step.stage_states(K)
    for it in range(iterations+1):
        rms = float(step.residual(K).square().mean().sqrt())
        if rms <= tol or iterations == 0:
            return K,rms,it
        if it == iterations:
            break
        c = COEFFICIENTS
        def mix(M,f):
            return torch.einsum('ij,jxyz->ixyz',M.to(f.dtype),f)
        def solve(matrix,rhs):
            return torch.linalg.solve(matrix.to(rhs.dtype),rhs.movedim(0,-1)[...,None]).squeeze(-1).movedim(-1,0)
        # u_t = -i*kappa1*k^2*u/2 + i*gamma*u*v.
        # Keep the inverse FFT complex here: its real/imaginary parts are fields.
        u = torch.complex(stages[:,0],stages[:,1])
        u0 = torch.complex(step.y_n[0],step.y_n[1])
        nh_u = g.fft(1j*c['gamma']*u*stages[:,2])
        matrix_u = eye+(.5j*h*c['kappa1']*g.k_squared)[...,None,None]*A
        uh = solve(matrix_u,g.fft(u0)[None]+h*mix(A,nh_u))
        new_u = torch.fft.ifftn(uh,dim=(-3,-2,-1))
        omega2 = c['kappa2']*g.k_squared+c['mu']**2
        nh = g.fft(c['gamma']*(stages[:,0]**2+stages[:,1]**2))
        rhs_v = g.fft(step.y_n[2])[None]+h*step.c[:,None,None,None]*g.fft(step.y_n[3])[None]+h*h*mix(A2,nh)
        vh = solve(eye+h*h*omega2[...,None,None]*A2,rhs_v)
        wh = g.fft(step.y_n[3])[None]+h*mix(A,-omega2*vh+nh)
        stages = torch.stack([new_u.real,new_u.imag,g.ifft(vh),g.ifft(wh)],dim=1)
        K = g.rhs(stages)
    raise RuntimeError(f'Stage polish failed: RMS={rms:.3e}')


@torch.no_grad()
def multiple_relaxation(step,K,targets,cfg):
    d1,d2 = step.rk_directions(K)
    directions = torch.stack([step.h*d1,step.h*(d2-d1)])
    p = step.y_n.new_zeros(2)  # total gamma, gamma2
    tolerance = p.new_tensor([cfg.mass_tol,cfg.energy_tol])
    def evaluate(p):
        change = (1+p[0])*directions[0]+p[1]*directions[1]
        y = step.y_n+change
        return y,step.grid.invariants(y)-targets
    for it in range(cfg.newton_iters):
        y,F = evaluate(p)
        if bool(torch.isfinite(F).all()) and bool((F.abs()<=tolerance).all()):
            return y,float(step.h*(1+p[0])),dict(gamma1=float(p[0]-p[1]),gamma2=float(p[1]),newton_iterations=it)
        grads = step.grid.invariant_gradients(y)
        J = torch.einsum('icxyz,jcxyz->ij',grads,directions)/tolerance[:,None]
        scale = torch.linalg.vector_norm(J,dim=0).clamp_min(1e-30)
        delta = (torch.linalg.pinv(J/scale,rtol=1e-12)@(-F/tolerance))/scale
        merit = float(torch.linalg.vector_norm(F/tolerance))
        for k in range(31):
            trial = p+(.5**k)*delta
            gamma1,gamma2 = trial[0]-trial[1],trial[1]
            if trial[0] <= -1 or max(abs(float(gamma1)),abs(float(gamma2)))>cfg.gamma_bound:
                continue
            _,Ft = evaluate(trial)
            if float(torch.linalg.vector_norm(Ft/tolerance))<merit:
                p = trial
                break
        else:
            raise RuntimeError('MRRK Newton line search failed')
    raise RuntimeError('MRRK Newton iteration limit')


class HermiteExtension(nn.Module):
    """在实际松弛时间区间内插值，端点斜率取半离散方程右端。"""

    def __init__(self, grid, state_start, state_end, duration):
        super().__init__()
        self.grid = grid
        self.duration = duration
        for name, value in (('state_start', state_start), ('state_end', state_end),
                            ('slope_start', grid.rhs(state_start)),
                            ('slope_end', grid.rhs(state_end))):
            self.register_buffer(name, value)

    def fields(self, s):
        """s 为归一化时间；返回状态和物理时间导数。"""
        s = s.reshape(-1, 1, 1, 1, 1)
        H = self.duration
        # 使用端点差值，避免将两个接近的状态分别除以 H。
        delta = self.state_end - self.state_start
        f0, f1 = self.slope_start, self.slope_end
        blend = s.square()*(3.-2.*s)
        state = ((1.-blend)*self.state_start + blend*self.state_end
                 + H*s*(1.-s)*((1.-s)*f0-s*f1))
        slope = (6.*s*(1.-s)*(delta/H)
                 + (3.*s.square()-4.*s+1.)*f0 + (3.*s.square()-2.*s)*f1)
        return state, slope

    def interior_residual(self, interior_s):
        state, slope = self.fields(interior_s)
        return slope - self.grid.rhs(state)

    def endpoint_residual(self, change=None):
        """用一阶后向差分近似右端点时间导数。"""
        if change is None:
            change = self.state_end-self.state_start
        return change/self.duration - self.slope_end

    def residual_info(self, interior_s):
        residual = self.interior_residual(interior_s)
        time_mse = residual.square().mean(dim=(-4, -3, -2, -1))
        mse = time_mse.mean()
        endpoint_residual = self.endpoint_residual()
        return dict(pde_mse=float(mse),
                    pde_rms=float(mse.sqrt()),
                    pde_max_time_rms=float(time_mse.max().sqrt()),
                    pde_interior_points=interior_s.numel(),
                    endpoint_backward_pde_rms=float(endpoint_residual.square().mean().sqrt()))


    def residual(self, s):
        return self.interior_residual(s)


def differentiable_relaxation(step, K, targets, config):
    """Newton forward solve with implicit differentiation of both invariants."""
    _, duration, info = multiple_relaxation(step, K.detach(), targets, config)
    parameters = step.y_n.new_tensor([duration/step.h, info['gamma2']])
    d1, d2 = step.rk_directions(K)
    D1, difference = step.h*d1, step.h*(d2-d1)
    change = parameters[0]*D1 + parameters[1]*difference
    F = step.grid.invariants(step.y_n+change)-targets
    with torch.no_grad():
        tolerance = step.y_n.new_tensor([config.mass_tol, config.energy_tol])
        gradients = step.grid.invariant_gradients(step.y_n+change)
        J = torch.einsum('icxyz,jcxyz->ij', gradients,
                         torch.stack([D1, difference]))/tolerance[:, None]
        scale = torch.linalg.vector_norm(J, dim=0).clamp_min(torch.finfo(K.dtype).tiny)
        J = J/scale
        if not bool(torch.isfinite(J).all()):
            raise RuntimeError('nonfinite_endpoint_training_jacobian')
        singular_values = torch.linalg.svdvals(J)
        if float(singular_values[-1]) <= 1e-12*float(singular_values[0]):
            raise RuntimeError('singular_endpoint_training_jacobian')
    parameters = parameters-torch.linalg.solve(J, (F-F.detach())/tolerance)/scale
    change = parameters[0]*D1+parameters[1]*difference
    return step.y_n+change, step.h*parameters[0], change


class StageCorrector(nn.Module):
    """输入精修后的全部阶段、未松弛端点和配点标签，输出阶段修正。"""

    def __init__(self, polished_stages, polished_endpoint, nodes, width, modes):
        super().__init__()
        self.register_buffer('polished_stages', polished_stages.detach())
        self.register_buffer('polished_endpoint', polished_endpoint.detach())
        self.register_buffer('nodes', nodes)
        channels = (len(nodes)+1)*polished_stages.shape[1]+1
        self.network = FourierCNN(channels=channels, width=width, modes=modes)

    def forward(self):
        s, _, nx, ny, nz = self.polished_stages.shape
        context = torch.cat([self.polished_stages.flatten(0, 1), self.polished_endpoint], dim=0)
        context = context.unsqueeze(0).expand(s, -1, -1, -1, -1)
        labels = self.nodes.reshape(s, 1, 1, 1, 1).expand(-1, 1, nx, ny, nz)
        correction = self.network(torch.cat([context, labels], dim=1))
        # 对 c_i=0 的配点，阶段值就是左端点。
        return self.polished_stages + (labels != 0.)*correction


def correct_stages(step, polished_stages, polished_endpoint, baseline, targets, config):
    """用端点和 Hermite 步内 PDE loss 训练第二个 FNO。"""
    y_base, H_base, _ = baseline
    interior_s = torch.linspace(0., 1., config.corrector_points+2,
                                device=step.y_n.device, dtype=step.y_n.dtype)[1:-1]

    def residual_mses(y_end, duration, change=None):
        curve = HermiteExtension(step.grid, step.y_n, y_end, duration)
        return torch.stack([curve.endpoint_residual(change).square().mean(),
                            curve.interior_residual(interior_s).square().mean()])

    base_mses = residual_mses(y_base, H_base)
    # 按 Picard+Newton 基线的残差平方归一化，防止一阶端点差分项主导训练。
    scales = base_mses.detach().clamp_min(1e-24)
    weights = step.y_n.new_tensor([1., config.corrector_interior_weight])
    best_loss = float((weights*(base_mses/scales)).sum())
    endpoint_rms, interior_rms = base_mses.sqrt().tolist()
    info = dict(corrector_updates=0, corrector_selected_epoch=None,
                corrector_pde_before_rms=endpoint_rms, corrector_pde_after_rms=endpoint_rms,
                corrector_interior_before_rms=interior_rms, corrector_interior_after_rms=interior_rms,
                corrector_loss_before=best_loss, corrector_loss_after=best_loss,
                corrector_interior_points=config.corrector_points)
    best_stages, best_result = polished_stages, baseline
    if not config.corrector_adam:
        return best_stages, best_result, info

    model = StageCorrector(polished_stages, polished_endpoint, step.c,
                           config.width, min(config.modes, step.grid.n//2))
    model = model.to(step.y_n)
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.corrector_lr)

    def evaluate():
        stages = model()
        K = step.grid.rhs(stages)
        y_end, H, change = differentiable_relaxation(step, K, targets, config)
        mses = residual_mses(y_end, H, change)
        loss = (weights*(mses/scales)).sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite_corrector_loss')
        return stages, K, loss, mses

    try:
        _, _, loss, _ = evaluate()
    except (RuntimeError, torch.linalg.LinAlgError) as error:
        info['corrector_stop_reason'] = str(error)
        return best_stages, best_result, info

    # Adam 给出更新方向。若该更新过大导致松弛失败，沿同一方向缩步。
    info['corrector_backtracks'] = 0
    for epoch in range(1, config.corrector_adam + 1):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 100.)
        if not bool(torch.isfinite(grad_norm)):
            info['corrector_stop_reason'] = 'nonfinite_corrector_gradient'
            break
        before = [value.detach().clone() for value in parameters]
        optimizer.step()
        direction = [value.detach()-old for value, old in zip(parameters, before)]
        for backtrack in range(13):
            fraction = 2.**(-backtrack)
            if backtrack:
                info['corrector_backtracks'] += 1
                with torch.no_grad():
                    for value, old, delta in zip(parameters, before, direction):
                        value.copy_(old + fraction*delta)
            try:
                stages, K, candidate_loss, mses = evaluate()
                score = float(candidate_loss.detach())
                if score >= best_loss:
                    reason = 'corrector_pde_loss_did_not_decrease'
                    continue
                candidate = multiple_relaxation(step, K.detach(), targets, config)
            except (RuntimeError, torch.linalg.LinAlgError) as error:
                reason = str(error)
                continue
            best_stages, best_result, best_loss = stages.detach(), candidate, score
            loss = candidate_loss
            optimizer.param_groups[0]['lr'] *= fraction
            endpoint_rms, interior_rms = mses.detach().sqrt().tolist()
            info.update(corrector_selected_epoch=epoch, corrector_pde_after_rms=endpoint_rms,
                        corrector_interior_after_rms=interior_rms, corrector_loss_after=best_loss,
                        corrector_updates=epoch, corrector_final_lr=optimizer.param_groups[0]['lr'])
            break
        else:
            info['corrector_stop_reason'] = reason
            break
    return best_stages, best_result, info

def save_plots(output,x,snapshots,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax.plot([0.]+[r['t'] for r in history],[0.]+[r['relative_energy_error'] for r in history])
    ax.set(xlabel='t',ylabel='|E(t)-E(0)|/|E(0)|')
    fig.savefig(output/'energy_relative_error.png',dpi=160)
    plt.close(fig)
    fig,ax = plt.subplots()
    ax.plot([0.]+[r['t'] for r in history],[0.]+[r['mass_error'] for r in history])
    ax.set(xlabel='t',ylabel='M(t) - M(0)',title='Mass conservation')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax.grid(True,alpha=.3)
    fig.savefig(output/'mass_error.png',dpi=160)
    plt.close(fig)
    mid = len(x)//2
    for component,name in ((0,'abs_u'),(2,'v')):
        fig,axes = plt.subplots(2,2,figsize=(10,8),layout='constrained')
        for ax,t in zip(axes.flat,SNAPSHOT_TIMES):
            if t not in snapshots:
                ax.set_axis_off()
                continue
            state = snapshots[t]
            field = np.hypot(state[0],state[1]) if component==0 else state[2]
            im = ax.pcolormesh(x,x,field[:,mid,:].T,shading='auto')
            ax.set(xlabel='x1',ylabel='x3',title=f'{name}, x2={x[mid]:g}, t={t:g}')
            fig.colorbar(im,ax=ax)
        fig.savefig(output/f'time_snapshots_{name}.png',dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=20,help='Even periodic grid size per axis; differs from paper Hermite degree N=40')
    parser.add_argument('--L',type=float,default=10.)
    parser.add_argument('--length',type=float,default=None,help='Cube side length; overrides 2*L')
    parser.add_argument('--dt',type=float,default=.1)
    parser.add_argument('--t-end',type=float,default=50.)
    parser.add_argument('--steps',type=int,default=0,help='Minimum accepted steps; t-end also controls stopping')
    parser.add_argument('--width',type=int,default=8)
    parser.add_argument('--modes',type=int,default=4)
    parser.add_argument('--adam',type=int,default=400)
    parser.add_argument('--lbfgs',type=int,default=100)
    parser.add_argument('--lr',type=float,default=.003)
    parser.add_argument('--rk-tol',type=float,default=1e-2)
    parser.add_argument('--stage-polish-iters',type=int,default=30)
    parser.add_argument('--stage-polish-tol',type=float,default=1e-10)
    parser.add_argument('--energy-tol',type=float,default=1e-12)
    parser.add_argument('--mass-tol',type=float,default=1e-12)
    parser.add_argument('--newton-iters',type=int,default=20)
    parser.add_argument('--gamma-bound',type=float,default=.5)
    parser.add_argument('--corrector-adam',type=int,default=10)
    parser.add_argument('--corrector-lr',type=float,default=1e-4)
    parser.add_argument('--corrector-points',type=int,default=9)
    parser.add_argument('--corrector-interior-weight',type=float,default=1.)
    parser.add_argument('--pde-points',type=int,default=9)
    parser.add_argument('--pde-tol',type=float,default=1e-2)
    parser.add_argument('--max-refinements',type=int,default=3)
    parser.add_argument('--max-step-halvings',type=int,default=2)
    parser.add_argument('--min-dt',type=float,default=None)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--output',default='kgs3d_guo43_fno_mrrk_output')
    parser.add_argument('--method',choices=['both','gauss','lobatto'],default='both',
                        help='Gauss: 3 stages, order 6; Lobatto IIIA: 4 stages, order 6; both: run separately')
    cfg = parser.parse_args(argv)
    length = cfg.length if cfg.length is not None else 2*cfg.L
    positive = [length,cfg.dt,cfg.width,cfg.modes,cfg.lr,cfg.rk_tol,cfg.energy_tol,
                cfg.mass_tol,cfg.newton_iters,cfg.gamma_bound,cfg.corrector_lr,
                cfg.corrector_points,cfg.pde_tol,cfg.stage_polish_tol,cfg.threads]
    if any(not math.isfinite(v) or v<=0 for v in positive):
        parser.error('Lengths, step size, dimensions, learning rates and tolerances must be finite and positive')
    if min(cfg.adam,cfg.lbfgs,cfg.corrector_adam,cfg.stage_polish_iters,
           cfg.max_refinements,cfg.max_step_halvings,cfg.steps)<0 or not math.isfinite(cfg.t_end) or cfg.t_end<0 or cfg.pde_points<3:
        parser.error('Invalid iteration counts, end time or PDE points')
    if not math.isfinite(cfg.corrector_interior_weight) or cfg.corrector_interior_weight < 0:
        parser.error('corrector-interior-weight must be finite and nonnegative')
    cfg.min_dt = cfg.dt/64 if cfg.min_dt is None else cfg.min_dt
    if not 0<cfg.min_dt<=cfg.dt:
        parser.error('Require 0 < min-dt <= dt')
    methods = ('gauss','lobatto') if cfg.method == 'both' else (cfg.method,)
    summaries = {}
    for method in methods:
        run_cfg = argparse.Namespace(**vars(cfg))
        run_cfg.method = method
        if cfg.method == 'both':
            base = Path(cfg.output)
            run_cfg.output = str(base.with_name(base.name+'_'+method))
        summaries[method] = run_method(run_cfg)
    return summaries if cfg.method == 'both' else summaries[cfg.method]


def run_method(cfg):
    length = cfg.length if cfg.length is not None else 2*cfg.L
    method_name = {'gauss':'Gauss', 'lobatto':'Lobatto IIIA'}[cfg.method]
    device = configure(cfg.threads,cfg.seed,cfg.device)
    grid = PeriodicGrid(cfg.n,length,device)
    y = initial_state(grid)
    targets = grid.invariants(y).detach()
    output = Path(cfg.output)
    output.mkdir(parents=True,exist_ok=True)
    stepdir = output/'steps'
    stepdir.mkdir(exist_ok=True)
    tableau = {'gauss':gauss_legendre_tableau, 'lobatto':lobatto_iiia_tableau}[cfg.method]
    a,b1,b2,c = tableau(y)
    (output/'butcher_tableau.json').write_text(json.dumps(dict(
        method=method_name, stage_count=len(c), main_order=6, embedded_order=1,
        a=a.cpu().tolist(), b1=b1.cpu().tolist(), b2=b2.cpu().tolist(),
        c=c.cpu().tolist()),indent=2),encoding='utf-8')
    history = []
    snapshots = {0.:y.cpu().numpy()}
    t,h,previous = 0.,cfg.dt,None
    def save_history():
        (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    save_history()
    while len(history)<cfg.steps or t<cfg.t_end:
        step = RKStep(grid,y,h,cfg.width,cfg.modes,method=cfg.method).to(device)
        if previous is not None:
            step.network.load_state_dict(previous)
        accepted = False
        for halving in range(cfg.max_step_halvings+1):
            for refinement in range(cfg.max_refinements+1):
                try:
                    optimize(step.network,step.rk_loss,cfg.adam,cfg.lbfgs,cfg.lr)
                    with torch.no_grad():
                        K = step.predict_slopes()
                        fno_rms = float(step.residual(K).square().mean().sqrt())
                        K,rk_rms,polish_iterations = polish_rk_stages(step,K,cfg.stage_polish_iters,cfg.stage_polish_tol)
                        if not math.isfinite(rk_rms) or rk_rms>cfg.rk_tol:
                            raise RuntimeError(f'RK residual {rk_rms}')
                        baseline = multiple_relaxation(step,K,targets,cfg)
                        polished_stages = step.stage_states(K).detach()
                        d1, _ = step.rk_directions(K)
                        polished_endpoint = (step.y_n+step.h*d1).detach()
                    stages, result, corrector_info = correct_stages(
                        step, polished_stages, polished_endpoint, baseline, targets, cfg)
                    yn,duration,info = result
                    info.update(corrector_info)
                    with torch.no_grad():
                        if corrector_info['corrector_selected_epoch'] is not None:
                            K = grid.rhs(stages)
                        info['corrector_rk_rms'] = float(step.residual(K).square().mean().sqrt())
                        info['corrector_stage_defect_rms'] = float(
                            (stages-step.stage_states(K)).square().mean().sqrt())
                    curve = HermiteExtension(grid,y,yn,duration).to(device)
                    with torch.no_grad():
                        validation = torch.linspace(0,1,2*cfg.pde_points-1,device=device)
                        residual = curve.residual(validation)
                        pde_rms = float(residual.square().mean().sqrt())
                        pde_max = float(residual.square().mean(dim=(-4,-3,-2,-1)).sqrt().max())
                    if not math.isfinite(pde_max) or pde_max>cfg.pde_tol:
                        raise RuntimeError(f'Continuous PDE residual {pde_max}')
                    accepted = True
                    break
                except RuntimeError as error:
                    with (output/'retry_history.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(t=t,h=step.h,halving=halving,refinement=refinement,error=str(error)))+'\n')
                    if refinement==cfg.max_refinements and (halving==cfg.max_step_halvings or step.h/2<cfg.min_dt):
                        raise
            if accepted:
                break
            step.h /= 2
        with torch.no_grad():
            for requested in SNAPSHOT_TIMES[1:]:
                if requested not in snapshots and t<=requested<=t+duration:
                    value,_ = curve.fields(y.new_tensor([(requested-t)/duration]))
                    snapshots[requested] = value[0].cpu().numpy()
            dense,_ = curve.fields(validation)
        index = len(history)+1
        torch.save(dict(state_dict=curve.state_dict(),duration=duration,t_start=t,t_end=t+duration,
                        n=grid.n,length=length,width=cfg.width,modes=cfg.modes,interpolation='cubic Hermite',
                        coefficients=COEFFICIENTS,state_components=STATE_COMPONENTS,method=cfg.method),
                   stepdir/f'continuous_{index:08d}.pt')
        np.savez_compressed(stepdir/f'step_{index:08d}.npz',state_start=y.cpu().numpy(),
                            state=yn.cpu().numpy(),K=K.cpu().numpy(),Y=stages.cpu().numpy(),Y_picard=polished_stages.cpu().numpy(),
                            state_picard=polished_endpoint.cpu().numpy(),
                            Y_rk_reconstructed=step.stage_states(K).detach().cpu().numpy(),
                            dense_t=t+duration*validation.cpu().numpy(),dense_states=dense.cpu().numpy(),
                            t=t+duration,h=step.h,method=cfg.method,
                            a=step.a.cpu().numpy(),b1=step.b1.cpu().numpy(),
                            b2=step.b2.cpu().numpy(),c=step.c.cpu().numpy(),**info)
        y = yn.detach()
        t += duration
        h = step.h
        previous = {k:v.detach().clone() for k,v in step.network.state_dict().items()}
        energy_error = float(grid.energy(y)-targets[1])
        mass = float(grid.mass(y))
        mass_error = mass-float(targets[0])
        row = dict(step=index,t=t,h=h,physical_time_step=duration,rk_rms=rk_rms,
                   fno_rk_rms=fno_rms,stage_polish_iterations=polish_iterations,
                   energy=float(grid.energy(y)),energy_error=energy_error,
                   relative_energy_error=abs(energy_error)/max(abs(float(targets[1])),torch.finfo(y.dtype).eps),
                   mass=mass,mass_error=mass_error,
                   relative_mass_error=abs(mass_error)/max(abs(float(targets[0])),torch.finfo(y.dtype).eps),
                   pde_rms=pde_rms,pde_max_time_rms=pde_max,**info)
        history.append(row)
        save_history()
        print(f'{method_name} step={index}, t={t:.6f}, RK={rk_rms:.3e}, dE={energy_error:+.3e}, '
              f'M={mass:.12e}, dM={mass_error:+.3e}, '
              f'PDE={pde_max:.3e}',flush=True)
    summary = dict(config=vars(cfg),coefficients=COEFFICIENTS,state_components=STATE_COMPONENTS,
                   domain='[-length/2,length/2)^3',boundary='periodic in x1,x2,x3',
                   spatial_axis_order=['x1','x2','x3'],
                   invariant_order=['mass','energy'],
                   paper='Guo et al. 2023, DOI 10.1137/22M1484109, section 4.3, equation (60)',
                   paper_boundary='decay at infinity on R^3; here approximated with periodic truncation',
                   mass_definition='integral(|u|^2)',
                   energy_definition='integral((kappa1*|grad u|^2+kappa2*|grad v|^2+v_t^2+mu^2*v^2)/2-gamma*|u|^2*v)',
                   equations=['i*u_t+kappa1/2*Delta(u)+gamma*u*v=0','v_tt-kappa2*Delta(v)+mu^2*v-gamma*|u|^2=0'],
                   final_time=t,steps=len(history),main_rk_order=6,embedded_rk_order=1,stage_count=len(c),
                   method=f'3D FNO {method_name} + two-parameter MRRK Newton + FNO stage correction + cubic Hermite dense output',
                   max_energy_error=max([abs(r['energy_error']) for r in history],default=0.),
                   initial_mass=float(targets[0]),final_mass=float(grid.mass(y)),
                   initial_energy=float(targets[1]),final_energy=float(grid.energy(y)),
                   max_mass_error=max([abs(r['mass_error']) for r in history],default=0.))
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    np.savez_compressed(output/'snapshots.npz',**grid.coordinate_arrays(),t=list(snapshots),
                        states=np.asarray(list(snapshots.values())),state_components=STATE_COMPONENTS)
    save_plots(output,grid.x.cpu().numpy(),snapshots,history)
    return summary


if __name__ == '__main__':
    main()
