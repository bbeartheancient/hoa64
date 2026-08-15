"""Muon/Dion3-style optimizer: orthogonalized momentum via Newton–Schulz.

Theory
------
Muon (Keller Jordan et al., 2024, "Muon: An optimizer for hidden layers")
updates a weight matrix W with the **orthogonal polar factor** of its momentum
matrix: for momentum M with SVD M = U Σ Vᵀ the update is O = U Vᵀ — the
nearest semi-orthogonal matrix to M, which equalizes the update across all
singular directions instead of letting the top singular vectors dominate
(as plain SGD/AdamW do). The orthogonalization is computed with the quintic
Newton–Schulz iteration on the spectral-norm-normalized input

    X₀      = M / ‖M‖_F                    (σ_max(X₀) ≤ 1 since σ_max ≤ ‖·‖_F)
    X_{k+1} = a X_k + b (X_k X_kᵀ) X_k + c (X_k X_kᵀ)² X_k
    (a, b, c) = (3.4445, −4.7750, 2.0315)

whose coefficients are the **cursed quintic** (Keller / Bernstein): they
maximize the slope at 0 so 5 steps lift tiny singular values as fast as
possible. The polynomial is *not* convergent — p(1) = 0.701 ≠ 1 — and
the output is US′Vᵀ with S′ roughly in [0.5, 1.5], not exact U Vᵀ. That
is the published Muon design; Polar Express coefficients would be used
if a tighter polar factor were needed.

Dion3 revisions (arXiv 2608.11612) — implemented here
-----------------------------------------------------
Dion3 keeps Muon's update rule but attacks its cost — the NS iteration is the
optimizer's dominant FLOPs:

1. **Gram NS.** Writing the iteration as X_{k+1} = Q_k X_k with
   Q_k = aI + bA_k + cA_k² and A_k = X_k X_kᵀ, the Gram matrix satisfies the
   *closed* recursion A_{k+1} = Q_k A_k Q_k (Q_k, A_k symmetric). So for an
   m×n matrix with m ≤ n one can iterate entirely on the m×m Gram, accumulate
   the left multiplier P = Q_{s−1}···Q₀, and reconstruct O = P X₀ once at the
   end: cost O(steps·m³ + m²n) instead of O(steps·m²n) — a win whenever
   n ≫ m (wide layers). Orthogonalizing G this way exploits that NS on GGᵀ
   acts on Σ² with the same left vectors U of G = UΣVᵀ.
2. **Row-subsampled orthogonalization.** Only a random fraction
   ``row_frac`` of momentum rows is orthogonalized per step (scaled by
   1/row_frac to keep the expected update magnitude); the remaining rows take
   a plain momentum step. Dion3 reports matching Muon loss at up to 6× step
   speed.

The parameter update uses Keller Jordan's fan scaling
``p ← p − lr · O · max(1, fan_out/fan_in)**0.5`` with the matrix flattened to
(fan_out, −1). Parameters with ndim < 2 (biases, norms, embeddings-as-vectors)
get a plain **AdamW** fallback inside the same optimizer at ``adamw_lr`` —
the standard Muon companion rule. Choices kept simple on purpose: heavy-ball
momentum M ← μM + g (no Nesterov look-ahead), Frobenius normalization as the
spectral bound, weight decay applied (decoupled) only on the AdamW fallback
params, and float32 NS (no bf16 rounding tricks).

PyTorch is an optional dependency and is imported lazily: the module imports
torch-free; ``Muon(...)`` is a factory that builds (and caches) the real
``torch.optim.Optimizer`` subclass on first call.
"""
from __future__ import annotations

import math

#: Quintic Newton–Schulz coefficients (Muon standard).
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(G, steps: int = 5, gram: bool = True, eps: float = 1e-7):
    """Orthogonalize ``G`` (2-D tensor) → approximate U Vᵀ of its SVD.

    Quintic NS iteration with ``_NS_COEFFS`` on the Frobenius-normalized input
    (σ_max ≤ 1 guaranteed). Tall inputs are transposed so the working matrix
    is always wide (rows ≤ cols). With ``gram=True`` on a wide matrix the
    iteration runs on the small rows×rows Gram matrix and reconstructs
    O = P X₀ once at the end (see the module docstring); with ``gram=False``
    it runs the plain full-matrix iteration. Same math, different FLOPs.
    The cursed quintic does **not** drive every σ to 1 — see the module
    docstring.
    """
    import torch

    a, b, c = _NS_COEFFS
    X = G.to(torch.float64) if G.dtype == torch.float64 else G.to(torch.float32)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + eps)
    m = X.size(0)
    eye = torch.eye(m, dtype=X.dtype, device=X.device)
    if gram:
        # Iterate A_{k+1} = Q_k A_k Q_k on the m×m Gram, P ← Q_k P; O = P X₀.
        A = X @ X.mT
        P = eye.clone()
        for _ in range(steps):
            Q = a * eye + b * A + c * (A @ A)
            A = Q @ A @ Q
            P = Q @ P
        O = P @ X
    else:
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        O = X
    return O.mT if transposed else O


_MUON_CLS = None


def _muon_class():
    """Build (and cache) the real ``torch.optim.Optimizer`` subclass lazily."""
    global _MUON_CLS
    if _MUON_CLS is not None:
        return _MUON_CLS
    import torch

    class _Muon(torch.optim.Optimizer):
        """Muon/Dion3 optimizer — see the module docstring.

        ndim ≥ 2 params: momentum buffer M ← μM + g, then M (flattened to
        (fan_out, −1)) is orthogonalized by ``newton_schulz`` and applied with
        the fan scaling ``max(1, fan_out/fan_in)**0.5``. With
        ``row_frac < 1`` only a fresh random subset of
        ``max(1, ceil(row_frac · rows))`` momentum rows is orthogonalized per
        step (scaled by 1/row_frac); the other rows take the plain momentum
        step (Dion3 row subsampling). ndim < 2 params: decoupled-wd AdamW at
        ``adamw_lr``.
        """

        def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5,
                     row_frac=1.0, adamw_lr=3e-4, weight_decay=0.0):
            if not 0.0 < row_frac <= 1.0:
                raise ValueError(f"row_frac must be in (0, 1], got {row_frac}")
            defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                            row_frac=row_frac, adamw_lr=adamw_lr,
                            weight_decay=weight_decay)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr, mu = group["lr"], group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if p.ndim >= 2:
                        if "momentum" not in state:
                            state["momentum"] = torch.zeros_like(g)
                        M = state["momentum"]
                        M.mul_(mu).add_(g)
                        M2 = M.view(M.size(0), -1)
                        rows = M2.size(0)
                        rf = group["row_frac"]
                        if rf >= 1.0:
                            upd = newton_schulz(M2, steps=group["ns_steps"])
                        else:
                            k = max(1, math.ceil(rf * rows))
                            idx = torch.randperm(rows, device=M2.device)[:k]
                            upd = M2.clone()
                            upd[idx] = newton_schulz(M2[idx],
                                                     steps=group["ns_steps"]) / rf
                        scale = max(1.0, rows / M2.size(1)) ** 0.5
                        p.add_(upd.view_as(p).to(p.dtype), alpha=-lr * scale)
                    else:
                        # AdamW fallback (decoupled weight decay).
                        if "exp_avg" not in state:
                            state["exp_avg"] = torch.zeros_like(g)
                            state["exp_avg_sq"] = torch.zeros_like(g)
                            state["step"] = 0
                        state["step"] += 1
                        t = state["step"]
                        m, v = state["exp_avg"], state["exp_avg_sq"]
                        b1, b2, eps = 0.9, 0.999, 1e-8
                        m.mul_(b1).add_(g, alpha=1 - b1)
                        v.mul_(b2).addcmul_(g, g, value=1 - b2)
                        m_hat = m / (1 - b1 ** t)
                        v_hat = v / (1 - b2 ** t)
                        wd = group["weight_decay"]
                        if wd:
                            p.mul_(1 - group["adamw_lr"] * wd)
                        p.addcdiv_(m_hat, v_hat.sqrt().add_(eps),
                                   value=-group["adamw_lr"])
            return loss

    _MUON_CLS = _Muon
    return _MUON_CLS


def Muon(params, lr: float = 0.02, momentum: float = 0.95, ns_steps: int = 5,
         row_frac: float = 1.0, adamw_lr: float = 3e-4, weight_decay: float = 0.0):
    """Build a Muon optimizer over ``params`` → ``torch.optim.Optimizer``.

    Factory (not a class) so the module imports without torch; the returned
    object is a genuine ``torch.optim.Optimizer`` subclass instance. All
    params may be handed over together — ndim ≥ 2 params get the
    orthogonalized-momentum update, the rest the AdamW fallback.
    """
    return _muon_class()(params, lr=lr, momentum=momentum, ns_steps=ns_steps,
                         row_frac=row_frac, adamw_lr=adamw_lr,
                         weight_decay=weight_decay)


if __name__ == "__main__":
    import time

    import numpy as np
    import torch

    t0 = time.time()
    torch.manual_seed(0)

    # (a) cursed quintic on a well-conditioned 16×16: inflates the spectrum
    # toward ~[0.5, 1.5], does NOT land on σ=1 (p(1)=0.701 by design).
    U, _ = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64))
    s0 = torch.linspace(0.20, 1.0, 16, dtype=torch.float64)
    G = U @ torch.diag(s0) @ V.T
    O = newton_schulz(G, steps=5)
    sv = torch.linalg.svdvals(O)
    cond_g = (s0.max() / s0.min()).item()
    cond_o = (sv.max() / sv.min()).item()
    print(f"(a) NS well-cond 16×16: σ ∈ [{sv.min():.4f}, {sv.max():.4f}], "
          f"cond {cond_g:.1f} → {cond_o:.2f}")
    assert 0.40 < sv.min().item() < sv.max().item() < 1.50, (sv.min().item(), sv.max().item())
    assert cond_o < cond_g, (cond_o, cond_g)

    # (b) Gram NS ≈ plain NS on a wide 32×128.
    W = torch.randn(32, 128, dtype=torch.float64)
    d = (newton_schulz(W, gram=True) - newton_schulz(W, gram=False)).norm().item()
    print(f"(b) gram vs plain NS 32×128: frob diff {d:.2e}")
    assert d < 1e-3, d

    # (c) teacher-linear regression, 2-layer MLP (2D + 1D params), 300 steps.
    def fit(opt_fn, steps=300):
        torch.manual_seed(1)
        X = torch.randn(512, 16)
        Wt = torch.randn(16, 1)
        y = X @ Wt + 0.01 * torch.randn(512, 1)
        net = torch.nn.Sequential(
            torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        opt = opt_fn(net)
        lossf = torch.nn.MSELoss()
        first = None
        for i in range(steps):
            opt.zero_grad()
            loss = lossf(net(X), y)
            loss.backward()
            opt.step()
            if first is None:
                first = loss.item()
        return first, loss.item()

    muon_first, muon_last = fit(lambda net: Muon(net.parameters(), lr=0.02,
                                                 adamw_lr=3e-3, weight_decay=0.0))
    adamw_first, adamw_last = fit(lambda net: torch.optim.AdamW(net.parameters(),
                                                                lr=3e-3))
    print(f"(c) teacher fit 300 steps: muon {muon_first:.4f} → {muon_last:.6f}, "
          f"adamw {adamw_first:.4f} → {adamw_last:.6f}")
    assert muon_last < adamw_last * 1.5, (muon_last, adamw_last)

    # (d) row_frac=0.5 still descends.
    rf_first, rf_last = fit(lambda net: Muon(net.parameters(), lr=0.02,
                                             row_frac=0.5, adamw_lr=3e-3))
    print(f"(d) row_frac=0.5: {rf_first:.4f} → {rf_last:.6f}")
    assert rf_last < rf_first, (rf_first, rf_last)

    # (e) one Muon step on a tiny DiT noise classifier.
    from . import dit_noise, noise_data

    model = dit_noise.build_model(n_classes=3, dim=32, depth=1, heads=2)
    mel = torch.from_numpy(
        noise_data.mel_spectrogram(
            np.random.default_rng(0).standard_normal(12987).astype(np.float32),
            noise_data.FS)[None, None])
    mel = torch.from_numpy(dit_noise._fit_mel(mel[0, 0].numpy(), 64)[None, None])
    opt = Muon(model.parameters())
    loss = torch.nn.CrossEntropyLoss()(model(mel), torch.tensor([1]))
    loss.backward()
    opt.step()
    print(f"(e) tiny DiT muon step OK (loss {loss.item():.4f})")

    print(f"muon self-check OK ({time.time() - t0:.1f} s)")
