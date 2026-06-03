# Running the Heisenberg pipeline on Trillium

## 1. First-time SSH setup (one-time)

Run this **on your Mac**:

```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519

# Upload public key at https://ccdb.alliancecan.ca → My Account → SSH Keys
# Then test:
ssh -i ~/.ssh/id_ed25519 aniketrd@trillium-gpu.scinet.utoronto.ca  
```

## 2. Deploy code to Trillium (one-time, then repeat when code changes)

Run this **on your Mac**:

```bash
cd ~/Documents/university/research/vector/generative-quantum-states
bash slurm/deploy.sh
```

This rsyncs the repo to `$SCRATCH/generative-quantum-states/` on Trillium.
`$SCRATCH` is the only filesystem writable from compute nodes.

## 3. First-time cluster setup (one-time)

Run this **on Trillium** (in your SSH tab), after step 2 has synced the code:

```bash
cd $SCRATCH/generative-quantum-states
bash slurm/setup_env.sh
```

## 4. Submit jobs

Run this **on your Mac**:

```bash
cd ~/Documents/university/research/vector/generative-quantum-states
bash slurm/deploy.sh --run
```

Or **on Trillium** directly:

```bash
cd $SCRATCH/generative-quantum-states
bash slurm/run_all.sh 4 4
```

## 5. Pipeline (5 SLURM jobs, chained with dependencies)

| Step | Script | What it does | Resources |
|------|--------|-------------|-----------|
| 1 | `01_generate_data.sh` | Coupling graphs + exact diag + shadow sampling (4x4, 80 train / 20 test) | CPU only, 2h |
| 2 | `02_train_transformer.sh` | GCN encoder + transformer training | 1 GPU + 4 CPU, 6h |
| 3 | `03_sample_transformer.sh` | Generate 20k samples from trained model | 1 GPU + 4 CPU, 2h |
| 4 | `04_evaluate.sh` | Correlation matrices + entanglement entropy via classical shadows | CPU only, 1h |
| 5 | `05_plot.sh` | Correlation heatmaps + RMSE (paper Fig. 3 style) | CPU only, 15min |

Steps 1-2 chain automatically. After training finishes, you need to find the results dir and submit steps 3-5 manually (instructions are printed when you run `run_all.sh`).

## 6. Monitor

Run this **on Trillium**:

```bash
squeue -u aniketrd                    # check job status
tail -f $SCRATCH/generative-quantum-states/logs/train_JOBID.out   # watch training
```

## 7. Pull results back

Run this **on your Mac**:

```bash
rsync -avz aniketrd@trillium-gpu.scinet.utoronto.ca:\$SCRATCH/generative-quantum-states/results/ ./results/
```

## 8. Clean up and disconnect

Run this **on Trillium** before closing your SSH session:

```bash
# Check what's running under your name
squeue -u aniketrd

# Cancel ALL your jobs (pending + running)
scancel -u aniketrd

# Or cancel a specific job
scancel JOBID

# Verify nothing is left
squeue -u aniketrd
```

Once `squeue` shows no jobs, you can safely close the SSH tab (`exit` or Ctrl+D). Closing the SSH session does NOT cancel submitted/running jobs — they keep running on the cluster. So always `scancel` first if you want to stop everything.

## Notes

- The `--account` flag was removed from SLURM scripts (Trillium may use your default allocation). If jobs fail with an account error, check your allocation with `sacctmgr show associations user=aniketrd` and add `#SBATCH --account=YOUR_ALLOCATION` to each script.
- GPU jobs request 1 GPU + 4 CPUs (the cluster guideline is 1 GPU per 4 CPUs).
- CPU-only jobs (data gen, eval, plot) request a full node (192 cores).
- Each step can be submitted individually with env vars, e.g. `ROWS=3 COLS=3 sbatch slurm/01_generate_data.sh`.
- Steps 3-5 require `RESULTS_DIR` to be set to the training output directory.
- All job output goes to `$SCRATCH/generative-quantum-states/logs/`.
- "User not known to scheduler" error means your account hasn't propagated yet — can take up to a day after getting access.
