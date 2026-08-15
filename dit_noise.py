"""DiT-style transformer classifier for NOISEX-92 noise type recognition.

Architecture (honest description)
---------------------------------
This is the *backbone* of the Diffusion Transformer (Peebles & Xie, "Scalable
Diffusion Models with Transformers", ICCV 2023) repurposed as a plain
discriminative classifier — **there is no diffusion process, no noise
schedule, no timestep**. What is borrowed from DiT is the block structure:

- patch embedding by a strided ``patch × patch`` Conv2d over the
  ``(n_mels, T)`` log-mel spectrogram (in place of DiT's latent patches),
- learned positional embeddings + a prepended classification token,
- ``depth`` transformer blocks using DiT's **adaLN-Zero** modulation: instead
  of the usual affine LayerNorm, each block regresses six vectors
  (shift/scale/gate for the attention sub-block, shift/scale/gate for the MLP
  sub-block) from a conditioning embedding, with the regression head
  zero-initialized so every block starts as the identity map. Since there is
  no diffusion timestep to condition on, the conditioning embedding is simply
  a **learned constant vector** — the modulation then acts as per-block
  learned featurewise rescaling/gating, which is the useful inductive bias
  that survives the removal of the diffusion context.
- final LayerNorm, cls token → linear head → per-class logits.

The default config (dim 192, depth 6, heads 6, patch 8 on a 64×64 mel patch
grid) is ≈2.0 M parameters: the MLPs use expansion ratio 2 and each adaLN
regression is bottlenecked (dim → dim//8 → 6·dim, last layer zero-init),
which keeps the per-block modulation tables cheap. Grow ``dim``/``depth``
for capacity.

Front-end & resampling
----------------------
Inputs are log-mel spectrograms from :mod:`hoa64.noise_data` with its fixed
dB normalization (``DB_MEAN``/``DB_STD``) — training and inference share that
exact transform. The time axis of every window is linearly interpolated to
``t_len`` columns before patching. ``classify`` resamples inputs whose sample
rate differs from the NOISEX-92 rate (19.98 kHz) with plain ``np.interp``
linear resampling — adequate for classification (the mel front-end smears
sub-bin detail anyway), not for critical listening.

PyTorch is an optional dependency and is imported lazily inside every
function, per project convention.
"""
from __future__ import annotations

import pathlib

import numpy as np

from . import noise_data

_FS = noise_data.FS
_DEFAULT_PATH = pathlib.Path(__file__).resolve().parent / "dit_noise.pt"  # gitignored
_MODEL_CACHE: dict = {}


def _device(device=None):
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fit_mel(mel: np.ndarray, t_len: int) -> np.ndarray:
    """Resize/crop a (n_mels, T) log-mel to (n_mels, t_len) by linear interp."""
    n_mels, t = mel.shape
    if t == t_len:
        return np.ascontiguousarray(mel, dtype=np.float32)
    old = np.linspace(0.0, 1.0, t)
    new = np.linspace(0.0, 1.0, t_len)
    return np.stack([np.interp(new, old, row) for row in mel]).astype(np.float32)


def build_model(n_classes: int = 15, n_mels: int = 64, dim: int = 192, depth: int = 6,
                heads: int = 6, patch: int = 8, t_len: int = 64):
    """DiT-backbone noise classifier → ``nn.Module`` (see module docstring).

    Conv2d patch embed on the (n_mels, t_len) log-mel, learned positional
    embeddings, prepended cls token, ``depth`` adaLN-Zero DiT blocks
    conditioned on a learned constant embedding, final LayerNorm, cls →
    linear head → ``n_classes`` logits. ``model.config`` / ``model.classes``
    carry the metadata needed to rebuild and interpret it.
    """
    import torch
    from torch import nn

    grid = (n_mels // patch) * (t_len // patch)

    class DiTBlock(nn.Module):
        """Transformer block with DiT adaLN-Zero modulation.

        mod(c) → (shift_att, scale_att, gate_att, shift_mlp, scale_mlp,
        gate_mlp); the modulation head is a bottleneck MLP whose last layer
        is zero-initialized so the block is the identity at init (DiT's
        adaLN-Zero). Note this slows the first ~hundred optimizer steps —
        few-epoch training needs enough steps per epoch for the gates to
        open.
        """

        def __init__(self, dim: int, heads: int):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
            self.mlp = nn.Sequential(
                nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
            self.mod = nn.Sequential(
                nn.Linear(dim, dim // 8), nn.SiLU(), nn.Linear(dim // 8, 6 * dim))
            nn.init.zeros_(self.mod[2].weight)
            nn.init.zeros_(self.mod[2].bias)

        def forward(self, x, c):
            s_a, g_a_shift, gate_a, s_m, g_m_shift, gate_m = self.mod(c).chunk(6, dim=-1)
            # chunk order: (shift1, scale1, gate1, shift2, scale2, gate2)
            h = self.norm1(x) * (1.0 + g_a_shift[:, None, :]) + s_a[:, None, :]
            a, _ = self.attn(h, h, h, need_weights=False)
            x = x + gate_a[:, None, :] * a
            h = self.norm2(x) * (1.0 + g_m_shift[:, None, :]) + s_m[:, None, :]
            x = x + gate_m[:, None, :] * self.mlp(h)
            return x

    class DiTNoiseClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid, dim))
            self.cond = nn.Parameter(torch.zeros(dim))  # learned constant (no diffusion t)
            self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
            self.norm_f = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, n_classes)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.cond, std=0.02)

        def forward(self, mel):
            # mel: (B, 1, n_mels, t_len)
            b = mel.shape[0]
            tok = self.patch_embed(mel).flatten(2).transpose(1, 2)      # (B, grid, dim)
            tok = torch.cat([self.cls_token.expand(b, -1, -1), tok], dim=1)
            tok = tok + self.pos_embed
            c = self.cond.unsqueeze(0).expand(b, -1)                    # (B, dim)
            for blk in self.blocks:
                tok = blk(tok, c)
            return self.head(self.norm_f(tok[:, 0]))                    # (B, n_classes)

    model = DiTNoiseClassifier()
    model.config = dict(n_classes=n_classes, n_mels=n_mels, dim=dim, depth=depth,
                        heads=heads, patch=patch, t_len=t_len)
    model.classes = None
    return model


def _train_loop(model, x_tr, y_tr, x_va, y_va, epochs, batch_size, lr, device,
                classes, callback=None, stop_flag=None, optimizer: str = "adamw"
                ) -> list[dict]:
    """Cross-entropy epochs over in-memory float32 mel windows.

    ``optimizer`` is ``"adamw"`` (default, used by the tiny self-check)
    or ``"muon"`` (Dion3/Muon polar-factor update on ndim≥2 weights,
    AdamW fallback on biases/norms — see ``muon.Muon``).
    """
    import torch

    model.to(device)
    if optimizer == "muon":
        from .muon import Muon
        opt = Muon(model.parameters(), lr=lr, adamw_lr=3e-4, weight_decay=1e-2)
    elif optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}; expected muon|adamw")
    lossf = torch.nn.CrossEntropyLoss()
    n = len(x_tr)
    rng = np.random.default_rng(0)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        tot_loss = tot_correct = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.from_numpy(x_tr[idx][:, None]).to(device)
            yb = torch.from_numpy(y_tr[idx]).to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += int((out.argmax(1) == yb).sum())
        loss_avg = tot_loss / n
        acc = tot_correct / n
        model.eval()
        with torch.no_grad():
            if len(x_va):
                val_acc = float((model(torch.from_numpy(x_va[:, None]).to(device))
                                 .argmax(1).cpu().numpy() == y_va).mean())
            else:
                val_acc = acc
        rec = {"epoch": epoch, "loss": loss_avg, "acc": acc, "val_acc": val_acc,
               "classes": list(classes)}
        history.append(rec)
        if callback is not None:
            callback(dict(rec))
        if stop_flag is not None and stop_flag.is_set():
            break
    return history


def train_model(cache_dir=None, epochs: int = 8, batch_size: int = 64,
                lr: float | None = None, win_s: float = 0.65, hop_s: float = 0.32,
                device=None, max_windows_per_class: int | None = None,
                callback=None, stop_flag=None, out_path=None,
                optimizer: str = "muon") -> dict:
    """Train the DiT noise classifier on NOISEX-92 + synthetic RF classes.

    Every class in ``noise_data.NOISE_CLASSES`` is loaded via
    ``noise_data.load_noise``: the 15 SPIB recordings are downloaded
    lazily, the four RF labels (``ble``/``wifi``/``zigbee``/``lora``)
    are synthesized locally (no network). Classes whose load fails are
    skipped and reported in the returned dict. Windows of ``win_s``
    (hop ``hop_s``) are converted to fixed-normalized log-mel, resized
    to the model's ``t_len``, split 90/10 train/val, and trained with
    Muon (default) or AdamW + cross-entropy. ``lr`` defaults to 0.02
    for Muon and 3e-4 for AdamW. ``callback({"epoch", "loss", "acc",
    "val_acc", "classes"})`` fires per epoch; ``stop_flag.is_set()``
    is honored between epochs. Saves ``{"state_dict", "classes",
    "config", "optimizer"}`` to ``out_path`` (default: ``dit_noise.pt``
    next to this file — gitignored). Returns an info dict with the
    final val accuracy and per-class window counts.
    """
    import torch  # noqa: F401  (lazy optional dep; presence check)

    if optimizer not in ("muon", "adamw"):
        raise ValueError(f"unknown optimizer {optimizer!r}; expected muon|adamw")
    if lr is None:
        lr = 0.02 if optimizer == "muon" else 3e-4

    dev = _device(device)
    feats, labels, counts, skipped = [], [], {}, []
    for ci, name in enumerate(noise_data.NOISE_CLASSES):
        try:
            x, fs = noise_data.load_noise(name, cache_dir)
        except Exception as exc:  # download or parse failure → skip, report
            skipped.append(f"{name}: {exc}")
            continue
        wins = noise_data.frame_windows(x, fs, win_s, hop_s)
        if max_windows_per_class is not None and len(wins) > max_windows_per_class:
            keep = np.linspace(0, len(wins) - 1, max_windows_per_class).astype(int)
            wins = [wins[i] for i in keep]
        mels = np.stack([noise_data.mel_spectrogram(x[w], fs) for w in wins])
        feats.append(mels)
        labels.append(np.full(len(wins), ci, dtype=np.int64))
        counts[name] = len(wins)
    if not feats:
        raise RuntimeError(f"no NOISEX-92 data available; skipped: {skipped}")

    classes = [n for n in noise_data.NOISE_CLASSES if n in counts]
    remap = {noise_data.NOISE_CLASSES.index(n): i for i, n in enumerate(classes)}
    x_all = np.concatenate(feats)
    y_all = np.vectorize(remap.get)(np.concatenate(labels))

    model = build_model(n_classes=len(classes))
    t_len = model.config["t_len"]
    x_all = np.stack([_fit_mel(m, t_len) for m in x_all]).astype(np.float32)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x_all))
    n_tr = int(0.9 * len(perm))
    tr, va = perm[:n_tr], perm[n_tr:]

    model.classes = classes
    history = _train_loop(model, x_all[tr], y_all[tr], x_all[va], y_all[va],
                          epochs, batch_size, lr, dev, classes, callback,
                          stop_flag, optimizer=optimizer)

    path = pathlib.Path(out_path) if out_path is not None else _DEFAULT_PATH
    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "config": model.config, "optimizer": optimizer}, str(path))
    _MODEL_CACHE.clear()
    return {"val_acc": history[-1]["val_acc"], "epochs_ran": len(history),
            "classes": classes, "counts": counts, "skipped": skipped,
            "optimizer": optimizer, "out_path": str(path)}


def load_model(path=None, device=None):
    """Load a saved ``dit_noise.pt`` (cached by (path, device))."""
    import torch

    p = pathlib.Path(path) if path is not None else _DEFAULT_PATH
    dev = _device(device)
    key = (str(p), str(dev))
    if key not in _MODEL_CACHE:
        ckpt = torch.load(str(p), map_location=dev, weights_only=False)
        model = build_model(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.classes = list(ckpt["classes"])
        model.to(dev).eval()
        _MODEL_CACHE[key] = (model, dev)
    return _MODEL_CACHE[key]


def classify(x: np.ndarray, fs: float, model=None, device=None,
             classes: list[str] | None = None) -> dict:
    """Classify one waveform → ``{"probs": {name: p}, "top": name, "mel": ...}``.

    ``x`` is mono (multichannel is averaged). If ``fs`` ≠ 19.98 kHz the input
    is resampled with ``np.interp`` linear resampling (documented above).
    The first ``win_s`` = 0.65 s window is analyzed (short inputs are
    zero-padded); the returned ``mel`` is the exact ``(n_mels, t_len)``
    network input after fixed dB normalization and time-resize.
    """
    import torch

    if model is None:
        model, dev = load_model(device=device)
    else:
        dev = _device(device)
        model.to(dev).eval()
    names = classes or getattr(model, "classes", None) or noise_data.NOISE_CLASSES
    t_len = model.config["t_len"]

    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=tuple(range(1, x.ndim)))
    if fs != _FS:
        t_old = np.linspace(0.0, 1.0, len(x))
        t_new = np.linspace(0.0, 1.0, int(round(len(x) * _FS / fs)))
        x = np.interp(t_new, t_old, x).astype(np.float32)
    win = int(round(0.65 * _FS))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    mel = _fit_mel(noise_data.mel_spectrogram(x[:win], _FS), t_len)

    with torch.no_grad():
        logits = model(torch.from_numpy(mel[None, None]).to(dev))[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    out = {names[i]: float(probs[i]) for i in range(len(names))}
    return {"probs": out, "top": names[int(probs.argmax())], "mel": mel}


if __name__ == "__main__":
    # Network-free self-check: tiny DiT on synthetic white / pink / 1 kHz tone.
    import time

    import torch

    t0 = time.time()
    dev = _device()
    rng = np.random.default_rng(1)
    fs = _FS
    win = int(0.65 * fs)
    # adaLN-Zero gates open slowly → the 3 epochs need enough steps each.
    per_class = 600

    def synth(kind):
        if kind == "white":
            v = rng.standard_normal(win)
        elif kind == "pink":
            v = noise_data._pink(win, rng)
        else:  # 1 kHz tone burst with random phase/gate + light noise
            t = np.arange(win) / fs
            gate = np.hanning(win) ** rng.uniform(0.5, 2.0)
            v = np.sin(2 * np.pi * 1000.0 * t + rng.uniform(0, 2 * np.pi)) * gate
            v += 0.05 * rng.standard_normal(win)
        return (v / np.max(np.abs(v))).astype(np.float32)

    names = ["white", "pink", "tone"]
    xs, ys = [], []
    for ci, kind in enumerate(names):
        for _ in range(per_class):
            xs.append(noise_data.mel_spectrogram(synth(kind), fs))
            ys.append(ci)
    xs = np.stack([_fit_mel(m, 64) for m in xs]).astype(np.float32)
    ys = np.asarray(ys, dtype=np.int64)

    model = build_model(n_classes=3, dim=64, depth=2, heads=4)
    model.classes = names
    n_par = sum(p.numel() for p in model.parameters())
    hist = _train_loop(model, xs, ys, xs[:0], ys[:0], epochs=3, batch_size=32,
                       lr=3e-3, device=dev, classes=names)
    acc = hist[-1]["acc"]
    print(f"tiny DiT ({n_par} params) on {dev}: 3 epochs, final train acc {acc:.3f}")
    assert acc > 0.8, acc

    fresh = rng.standard_normal(win).astype(np.float32)
    fresh /= np.max(np.abs(fresh))
    res = classify(fresh, fs, model=model, device=dev, classes=names)
    print(f"classify(fresh white) → top={res['top']!r}, "
          f"probs={ {k: round(v, 3) for k, v in res['probs'].items()} }, "
          f"mel shape {res['mel'].shape}")
    assert res["top"] == "white", res

    n_default = sum(p.numel() for p in build_model().parameters())
    print(f"default-config param count: {n_default / 1e6:.2f} M")
    print(f"dit_noise self-check OK ({time.time() - t0:.1f} s)")
