# RunPod Briefing — re-render figure PDFs with clean axis labels (peer-review R2)

**No science change, no retraining, no new datasets.** A peer-review BLOCKING item is that
several committed figure PDFs show raw LaTeX/Python in their labels — they were authored for
`usetex=True` but render with mathtext, so the commands leaked through:
`R (\A A)` (should be Å), `r=̄` (broken `\bar r`), `np.int32(13)` (a numpy scalar in a title),
and `model vs.\ exact`. The label code is now fixed on `dev`; the PDFs just need re-rendering
where the datasets + checkpoints live (none are off-pod).

## Step 0 — pull the label fix (REQUIRED)
```bash
cd <repo> && git pull origin dev    # must include commit 5f34e2a
```
`5f34e2a` patches `models/fermionic_pipeline/eval/plot_regression.py` only:
`r"$R$ (\AA)"`→`"$R$ (Å)"`, `model vs.\ exact`→`model vs. exact`, `$\bar r$`→`$\bar{r}$`,
unwrap numpy scalars in the time-series obs title, **and curate `plot_time_series` to a 3×3 grid**
(was 7×4: compressed/near-eq/stretched geometries × loud→quiet channels, suptitle dropped).
Every figure below routes through this file, so a re-render is all that's needed.

## The 7 paper PDFs to refresh (paper `imgs/` name → what it is)
| `imgs/` file | role | generator |
|---|---|---|
| `coherence_heatmap.pdf` | H4 in/out-of-box map (main §4.1) | `extrapolation_heatmap` (H4 extrap grid) |
| `lih_coherence_heatmap.pdf` | LiH in-box map (main §4.2) | LiH in-box `coherence_heatmap.pdf`, renamed |
| `coherence_extrap_{h4,lih,beh2,n2}.pdf` | per-molecule extended maps (appendix E) | `extrapolation_heatmap` ×4, renamed |
| `regression_summary.pdf` | H4 in-box accuracy (appendix D) | `plot_regression` main |
| `time_series.pdf` | H4 representative traces, **now 3×3** (appendix D) | `plot_regression` main |

These are the SAME grids/checkpoints you generated originally — only the labels changed. Use the
checkpoints already on-pod; if a dataset was cleaned, regen just it (no retrain).

## Step 1 — coherence maps (extrap + the per-molecule appendix panels)
Same per-molecule loop as RUNPOD_LIH_EXTRAP Step 2 (eval only, no retrain):
```bash
RES=results/fermionic_pipeline/regression
for m in h4 lih beh2 n2; do
  python -m fermionic_pipeline.eval.extrapolation_heatmap \
    --data_path $RES/${m}_regress_v1_extrap/regression_targets.h5 \
    --checkpoint $RES/${m}_regress_v1_orb_s42_model/regressor.pt \
    --save_dir   $RES/${m}_regress_v1_orb_s42_model/plots_extrap \
    --train_r_range <m train R> --train_t_range 0 <m train t_max> \
    --omega_op_source train-interp \
    --train_data_path $RES/${m}_regress_v1/regression_targets.h5 --device cuda
done
```
(Train ranges: h4 R[0.5,3.0] t300; lih R[1.0,3.2] t1200; n2 R[0.9,2.3] t500; beh2 R[1.0,3.0] t500.
H4's checkpoint path is the v18-orb one if not under the `_v1` name — use whatever you used before.)
Then map the outputs to the paper names: H4's extrap `coherence_heatmap.pdf` → `coherence_heatmap.pdf`;
each molecule's extrap map → `coherence_extrap_<m>.pdf`; LiH's **in-box** `coherence_heatmap.pdf`
(from `plots/`, not `plots_extrap/`) → `lih_coherence_heatmap.pdf`.

## Step 2 — H4 in-box summary + curated traces
```bash
python -m fermionic_pipeline.eval.plot_regression \
  --data_path $RES/h4_regress_v1/regression_targets.h5 \
  --checkpoint $RES/h4_regress_v18_v18_orb_s42_model/regressor.pt \
  --save_dir   $RES/h4_regress_v18_v18_orb_s42_model/plots --device cuda
```
Produces `regression_summary.pdf` and the new 3×3 `time_series.pdf`.
- **Eyeball `time_series.pdf`:** confirm 3×3, real titles ("obs 13", not `np.int32(13)`), and that the
  loud→quiet channel span actually shows an easy→hard contrast. If the quietest channel is just
  flat noise, nudge the selection in `plot_time_series` (the `order`/`pick` lines).
- **Eyeball `regression_summary.pdf`:** the MSE (semilogy) panel — if the y-ticks still look
  cluttered, add a `LogLocator(numticks=...)`/clean formatter on `ax2`.

## Step 3 — verify NO artifacts remain
Open all 7 PDFs (or `pdftotext | grep`): there must be **no** `\AA`, `\bar`, `vs.\`, or `np.int32`
text, and `Å` must render as `Å`.

## Step 4 — push the refreshed PDFs back (imgs only)
Push the 7 PDFs to the **`runpod-results`** branch (`git add -f`), as in prior briefings — **do NOT
touch `paper_sync` / any `.tex`** (Aniket is editing the manuscript on Overleaf). I'll copy the
refreshed PDFs into `paper_sync/imgs/` and push to Overleaf off-pod. In your final message, list
the pushed paths and flag any figure whose numbers/appearance drifted from the originals.
