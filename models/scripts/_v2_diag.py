import numpy as np, torch
from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model
from fermionic_pipeline.eval.regressor_eval import predict_signal_matrix
from fermionic_pipeline.eval.omega_source import OmegaOpSource

base='results/fermionic_pipeline/regression'
ck=f'{base}/h2o_regress_v2_orb_s42_model/regressor.pt'
data=f'{base}/h2o_regress_v2/regression_targets.h5'
dev='cuda'
model,payload=load_checkpoint_model(ck,device=dev)
h=RegressionDatasetHandle(data)
src=OmegaOpSource("train-interp",handle=h,payload=payload)

def pear(Dm,De):
    ps=[]
    for i in range(De.shape[0]):
        se,sm=np.std(De[i]),np.std(Dm[i])
        if se<1e-12 or sm<1e-12: continue
        ps.append(np.corrcoef(De[i],Dm[i])[0,1])
    return float(np.nanmean(ps)) if ps else float('nan')

Rv=h.R_values
# sample a handful across the box
idxs=[int(np.argmin(np.abs(Rv-t))) for t in (0.75,1.0,1.2,1.35,1.5)]
print("R     omega_oracle omega_tinterp | pear_oracle rr_oracle | pear_tinterp rr_tinterp")
for ri in idxs:
    R=float(Rv[ri]); orb=h.hf_orbital_energies[ri] if h.hf_orbital_energies is not None else None
    oo=float(h.omega_op[ri]) if h.omega_op is not None else None
    ot=src.value(r_idx=ri)
    De=h.expectations[ri].T
    Dm_o=predict_signal_matrix(model,R,h.times,dev,orb_energies=orb,omega_op=oo)
    Dm_t=predict_signal_matrix(model,R,h.times,dev,orb_energies=orb,omega_op=ot)
    rr_o=np.nanmean([np.std(Dm_o[i])/np.std(De[i]) for i in range(De.shape[0]) if np.std(De[i])>1e-12])
    rr_t=np.nanmean([np.std(Dm_t[i])/np.std(De[i]) for i in range(De.shape[0]) if np.std(De[i])>1e-12])
    print("%.2f   %8.4f   %8.4f   | %8.3f %8.3f | %8.3f %8.3f"%(R,oo,ot,pear(Dm_o,De),rr_o,pear(Dm_t,De),rr_t))
