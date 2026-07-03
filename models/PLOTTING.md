# Regenerating the paper figures

Everything you need to re-make any figure in the FSR paper: one command per plot, where
the code lives, and where to change **fonts and centering**. Written to be followed by a
human or an agent — every command is copy-paste and **run from `models/`**.

---

## TL;DR (3 steps)

```bash
cd models
export HF_TOKEN=hf_xxx                 # your token, read access to the dataset repo
python -m scripts.fetch_plot_data      # pulls the ~16 MB plot cache from HF into results/
python -m scripts.plot_baseline_comparison   # ... or any command in the table below
```

Every script writes its PDF into `results/paper_figures/`. To put a figure into the
manuscript, copy it into the Overleaf repo: `cp results/paper_figures/<fig>.pdf ../paper_sync/imgs/`.

---

## Where to change fonts and centering

There is **one** shared style module — edit it and every figure changes consistently:

- **`fermionic_pipeline/eval/nature_style.py`** — `apply_nature_style()` sets all Matplotlib
  `rcParams`. The knobs you want:
  - **Fonts:** `FONT_BASE / FONT_LABEL / FONT_TICK / FONT_LEGEND / FONT_TITLE` (pt, near the top);
    `font.family`, `font.serif`, `font.sans-serif`. Output fonts are embedded as editable
    TrueType (`pdf.fonttype = 42`), so they stay editable in Illustrator/Inkscape.
  - **Column widths (figure sizing):** `SINGLE_COL` (3.50 in) and `DOUBLE_COL` (7.20 in); helper
    `grid_figsize(ncols, nrows, aspect)`.
- **Per-figure centering** lives in each script: the `figsize=(...)`, `fig.tight_layout(...)` /
  `subplots_adjust(...)`, colorbar padding, and any `fig.text(...)` footer. Grep a script for
  `figsize`, `tight_layout`, `subplots_adjust`.
- A few figures also nudge horizontal centering **in LaTeX** (not Python): in
  `../paper_sync/includes/include-body.tex`, look for `\hspace*{...}\includegraphics` (e.g. the
  coherence heatmap). Centering there is a manuscript-side tweak, not a script re-run.

All 9 plotting modules call `apply_nature_style(...)`, so a change in `nature_style.py` is global.

---

## Getting the data

Two things live on HuggingFace, private repo **`aniketdesh/molecular-shadows-datasets`**
(you need read access; set `HF_TOKEN`):

1. **`plot_cache/`** — the derived per-figure inputs (baseline stats, 5-seed eval JSONs,
   freq/smoothness `.npz`, dipole `.pkl`, overlay JSONs, coherence grids; ~16 MB). Its layout
   mirrors `results/` exactly. `python -m scripts.fetch_plot_data` downloads it and drops it into
   `models/results/`. **This is all you need for the 8 fast figures below.**
2. **Raw datasets + checkpoints** — `{mol}/regression_targets.h5`, `{mol}/extrap_regression_targets.h5`,
   `{mol}/regressor.pt` for `mol ∈ {h4, lih, beh2, n2}` (+ `h2o`, `*_v2` variants). Needed **only**
   for the two model-gated figures (`time_series`, `regression_summary`) and for authoritative
   coherence-map regeneration. Get them with `python -m scripts.fetch_plot_data --with-datasets`
   (prints the local snapshot path to feed to `--data_path` / `--checkpoint`).

---

## The 8 fast figures — model-free, one command each

No GPU, no checkpoint. After `fetch_plot_data`, run from `models/`:

| Output PDF (paper) | Command | Reads |
|---|---|---|
| `preflight_diagnostic.pdf` (pre-flight/**screen** diagnostic) | `python -m scripts.plot_preflight_diagnostic` | *nothing* — exact 8-qubit diagonalization on the fly |
| `baseline_comparison.pdf` (classical baselines) | `python -m scripts.plot_baseline_comparison` | `revision/1A/stats/`, `1C_linharm/`, `2A_fourmlp/`, `2B_gpkrr/` JSONs |
| `performance_aggregate.pdf` (multi-seed suite) | `python -m scripts.plot_performance_aggregate` | `revision/1A/{mol}_s{42,1729,7,13,101}/eval/regressor_eval.json` |
| `freq_recovery.pdf` **and** `smoothness.pdf` (diagnostics) | `python -m scripts.plot_diagnostics` | `revision/3B_freqrec/*.npz`, `3C_smoothness/*.npz` |
| `finite_shot.pdf` (LiH shot/library sweep) | `python -m scripts.plot_finite_shot` | `revision/3A_fs/`, `2D_nq/` eval JSONs |
| `dipole_resource.pdf` (dipole reconstruction) | `python -m scripts.plot_dipole_resource` | `*/bo*/dipole_bo_plotdata_{h4,beh2}.pkl` |
| `cross_molecule_overlay.pdf` (appendix overlay) | `python -m scripts.plot_overlay_summary` | `paper_figures/overlay_jsons/*.json` |
| `coherence_average.pdf` (appendix averaged map) | `python -m scripts.plot_coherence_average` | the four `*/plots_extrap/coherence_grid.npz` |

`plot_diagnostics` emits **two** figures. `plot_overlay_summary` also emits `overlay_mse_appendix.pdf`.

---

## Coherence heatmaps — restyle GPU-free from cache

The coherence maps (H4 `coherence_heatmap.pdf`, LiH `lih_coherence_heatmap.pdf`, and the four
appendix `coherence_extrap_{mol}.pdf`) are normally produced by running the model. But the
`corr(R,t)` grid is cached (`coherence_grid.npz`, in `plot_cache/`), so you can **redraw them
without a GPU** to iterate on fonts/centering:

```bash
# H4 main-text map
python -m scripts.replot_coherence_from_grid \
  --grid results/fermionic_pipeline/regression/h4_regress_v1_orb_s42_model/plots_extrap/coherence_grid.npz \
  --out  results/paper_figures/coherence_heatmap.pdf
# LiH main-text map (swap h4 -> lih); appendix panels -> --out coherence_extrap_<mol>.pdf
```

Fonts here come from `nature_style` (shared); the per-panel centering (figsize, colorbar,
`tight_layout`) lives in `scripts/replot_coherence_from_grid.py`, mirroring the model-path
drawer in `fermionic_pipeline/eval/plot_regression.py::plot_coherence_heatmap`.

> **Caveat:** the cached grids are the `v1_orb` extended-grid runs (H4 in-box r̄≈0.94). The
> paper's *main-text* H4 map (Fig. `coherence_heatmap.pdf`) was the oracle-ceiling version
> (r̄=0.89). Use the cache for **style iteration**; for the final, numerically-authoritative
> figure regenerate via the model path below with the exact settings.

---

## The 2 model-gated figures (need `.h5` + `.pt` + GPU)

`time_series.pdf` and `regression_summary.pdf` (H4 appendix) run the network forward, so they
need a checkpoint + dataset + a GPU (CPU works but is slow). Fetch the raw data, then:

```bash
python -m scripts.fetch_plot_data --with-datasets      # note the printed snapshot path $SNAP
python3 -m fermionic_pipeline.eval.plot_regression \
  --data_path $SNAP/h4/regression_targets.h5 \
  --checkpoint $SNAP/h4/regressor.pt \
  --save_dir results/paper_figures/_h4_eval --device cuda
```

This also (re)writes `coherence_heatmap.pdf` + `coherence_grid.npz`. The extrapolation-grid
maps use the sibling module with the extended `.h5`:

```bash
python3 -m fermionic_pipeline.eval.extrapolation_heatmap \
  --data_path $SNAP/h4/extrap_regression_targets.h5 --checkpoint $SNAP/h4/regressor.pt \
  --save_dir results/paper_figures/_h4_extrap --train_r_range 0.5 3.0 --train_t_range 0 300 --device cuda
```

(For LiH/BeH2/N2 the coherence maps use `--omega_op_source train-interp --train_data_path
$SNAP/<mol>/regression_targets.h5`; see `RUNPOD_FIG_LABELS.md` for the per-molecule train ranges.)

---

## Not Python scripts

- **`fsr_architecture.pdf`** (architecture schematic) — hand-authored TikZ at
  `results/paper_figures/fsr_architecture.tex`; rebuild with `pdflatex fsr_architecture.tex`.
- **`basf-spectrum.png`** (Fig. 1 workflow) — a hand-made diagram, not generated by code here.

---

## Agent quick-reference

- Run everything from `models/`. Fast figures: `python -m scripts.<name>` (see table). Data:
  `python -m scripts.fetch_plot_data` (needs `HF_TOKEN`). Coherence restyle:
  `python -m scripts.replot_coherence_from_grid --grid <coherence_grid.npz> --out <pdf>`.
- Global style: `fermionic_pipeline/eval/nature_style.py` (`apply_nature_style`, `FONT_*`,
  `SINGLE_COL`/`DOUBLE_COL`). Per-figure centering: the `figsize`/`tight_layout` in each script.
- Outputs land in `results/paper_figures/`; the manuscript includes them from `../paper_sync/imgs/`.
