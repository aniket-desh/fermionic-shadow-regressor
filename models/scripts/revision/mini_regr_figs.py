"""Re-render only the 3 paper PDFs from plot_regression (summary, time_series,
in-box coherence_heatmap), skipping the slow plot_spectra + plot_chan_pipeline
per-geometry Ljung-Box loops. Identical figures, just the ones the paper uses."""
import sys, os, torch
import fermionic_pipeline.eval.plot_regression as P

data_path, checkpoint, save_dir = sys.argv[1], sys.argv[2], sys.argv[3]
P.apply_nature_style()
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
handle = P.RegressionDatasetHandle(data_path)
model, payload = P.load_checkpoint_model(checkpoint, device=device)
test_r_indices = payload.get("test_r_indices", list(range(len(handle.R_values))))
print(f"[info] {len(test_r_indices)} test geometries (mini driver)")
P.plot_summary(handle, model, test_r_indices, device, save_dir)
print("[done] regression_summary.pdf")
P.plot_time_series(handle, model, test_r_indices, device, save_dir)
print("[done] time_series.pdf")
P.plot_coherence_heatmap(handle, model, test_r_indices, device, save_dir)
print("[done] coherence_heatmap.pdf")
print("MINI_REGR_DONE")
