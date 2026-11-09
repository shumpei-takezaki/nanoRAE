"""Riemannian Jacobi flow (RJF, "Learning on the Manifold", arXiv 2602.10099, Alg. 1-2) on per-token unit spheres.

Tensors are [B, C, H, W]; each token (dim 1) lives on its own sphere. t = 0 is data, t = 1 is noise, as in the linear path.
The paper leaves open whether the sphere is per token; we use per token because the encoder's LayerNorm acts per token.
"""

import torch
import torch.nn.functional as F


def plan(t, noise, data):
    """Slerp x_t and its velocity u_t between unit data and noise tokens; Omega is the geodesic angle per token."""
    x, e = F.normalize(data, dim=1), F.normalize(noise, dim=1)
    omega = (x * e).sum(1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6).arccos()
    t = t.view(-1, 1, 1, 1)
    xt = (((1 - t) * omega).sin() * x + (t * omega).sin() * e) / omega.sin()
    ut = omega / omega.sin() * ((t * omega).cos() * e - ((1 - t) * omega).cos() * x)
    return xt, ut, omega


def tangent(v, x):
    return v - (v * x).sum(1, keepdim=True) * x


def loss(model, t, noise, data, model_kwargs, jacobi_weight=True):
    """Jacobi-weighted tangent MSE, summed over token dims (Alg. 1) and averaged over tokens: [B].
    jacobi_weight=False drops the sinc^2 weight (RFM in RJF Table 3)."""
    xt, ut, omega = plan(t, noise, data)
    v = tangent(model(xt, t, **model_kwargs).float(), xt)
    a = (1 - t.view(-1, 1, 1, 1)) * omega
    w = torch.sinc(a / torch.pi) ** 2 if jacobi_weight else 1.0  # torch.sinc(z) = sin(pi z) / (pi z)
    return (w * (v - ut).pow(2).sum(1, keepdim=True)).flatten(1).mean(1)


def sample(init, model, ts, radius, **model_kwargs):
    """Euler steps along geodesics (exponential map, Alg. 2) from noise at ts[0] to data; the result is scaled to `radius`."""
    x = F.normalize(init.float(), dim=1)
    for cur, nxt in zip(ts[:-1], ts[1:]):
        t = torch.full((x.size(0),), cur.item(), device=x.device)
        step = tangent(model(x, t, **model_kwargs).float(), x) * (nxt - cur)  # time runs backwards, so the step is v * dt
        theta = step.norm(dim=1, keepdim=True).clamp_min(1e-12)
        x = F.normalize(theta.cos() * x + theta.sin() * step / theta, dim=1)
    return [x * radius]
