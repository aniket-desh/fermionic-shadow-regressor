"""B9 2A (E2.2) — Fourier-feature MLP baseline at matched ~12M-param budget.

Generic coordinate network: inputs (HF orbital-energy features eps(R), t) -> random Fourier
features -> MLP -> 120 channels. SAME data/split as the FSR (split_r_indices, seed 42), SAME
held-out metric. Isolates whether the FSR's EXPLICIT harmonic structure beats a generic neural
surrogate with identical inputs. Code pending Aniket review for dev.
"""
import os, sys, json, argparse
import numpy as np, h5py, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fermionic_pipeline.data.exact_conditional_dataset import split_r_indices

class FourierMLP(nn.Module):
    def __init__(self, in_dim, K, n_ff=256, ff_scale=20.0, hidden=2304, layers=3):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_ff) * ff_scale)
        h = [nn.Linear(2 * n_ff, hidden), nn.GELU()]
        for _ in range(layers - 1): h += [nn.Linear(hidden, hidden), nn.GELU()]
        h += [nn.Linear(hidden, K)]
        self.net = nn.Sequential(*h)
    def forward(self, x):
        p = x @ self.B
        return self.net(torch.cat([torch.cos(p), torch.sin(p)], -1))

def per_geom_pearson(De, Dm):
    ps = []
    for i in range(De.shape[1]):
        a, b = De[:, i], Dm[:, i]
        if a.std() < 1e-12 or b.std() < 1e-12: continue
        ps.append(np.corrcoef(a, b)[0, 1])
    return float(np.nanmean(ps)) if ps else np.nan

def run(mol, h5path, dev="cuda", steps=40000, bs=4096, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    with h5py.File(h5path) as f:
        R = f["R_values"][:].astype(np.float32); t = f["times"][:].astype(np.float32)
        D = f["expectations"][:].astype(np.float32)             # (nR,nT,K)
        eps = f["hf_orbital_energies"][:].astype(np.float32)    # (nR,n_orb)
    a, b = split_r_indices(len(R), 0.2, seed); train_idx, test_idx = (a, b) if len(a) >= len(b) else (b, a)
    K = D.shape[2]; nT = len(t)
    em, es = eps[train_idx].mean(0), eps[train_idx].std(0) + 1e-6
    tm, ts = t.mean(), t.std() + 1e-6
    epn = (eps - em) / es; tn = (t - tm) / ts
    # flatten training (R,t) -> input (eps(R)..., t), target D
    Xtr = np.concatenate([np.repeat(epn[train_idx], nT, 0),
                          np.tile(tn, len(train_idx))[:, None]], 1).astype(np.float32)
    Ytr = D[train_idx].reshape(-1, K)
    dev = torch.device(dev)
    Xtr = torch.from_numpy(Xtr).to(dev); Ytr = torch.from_numpy(Ytr).to(dev)
    model = FourierMLP(Xtr.shape[1], K).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    N = len(Xtr)
    for s in range(steps):
        idx = torch.randint(0, N, (bs,), device=dev)
        opt.zero_grad(); loss = ((model(Xtr[idx]) - Ytr[idx]) ** 2).mean(); loss.backward(); opt.step(); sched.step()
        if s % 8000 == 0: print(f"  {mol} step {s} mse {loss.item():.4e}", flush=True)
    model.eval()
    held = []
    with torch.no_grad():
        for g in test_idx:
            Xq = np.concatenate([np.repeat(epn[g][None], nT, 0), tn[:, None]], 1).astype(np.float32)
            Dh = model(torch.from_numpy(Xq).to(dev)).cpu().numpy()
            held.append(dict(R=float(R[g]), pearson=per_geom_pearson(D[g], Dh)))
    pm = float(np.nanmean([h["pearson"] for h in held]))
    print(f"{mol}: Fourier-MLP ({npar/1e6:.1f}M) held-out pearson_mean={pm:.4f}", flush=True)
    return dict(mol=mol, params=int(npar), heldout_pearson_mean=pm, per_geom=held)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mols", nargs="*"); ap.add_argument("--steps", type=int, default=40000)
    args = ap.parse_args()
    R = "results/fermionic_pipeline/regression"; OUT = f"{R}/revision/2A_fourmlp"; os.makedirs(OUT, exist_ok=True)
    for m in (args.mols or ["h4", "lih", "beh2", "n2"]):
        o = run(m, f"{R}/{m}_regress_v1/regression_targets.h5", steps=args.steps)
        json.dump(o, open(f"{OUT}/baseline_fourmlp_{m}.json", "w"), indent=2)
    print("[ok] ->", OUT)
