# ============================================================
# PINT-Seq v3.0 OPTIMIZADO - Versión mejorada con hiperparámetros ajustados
# ============================================================

"""
Mejoras propuestas para PINT-Seq v3.0:

1. ARQUITECTURA:
   - Aumentar capacidad del LSTM (lstm_h: 96 -> 128)
   - Aumentar proyección (proj: 64 -> 96)
   - Añadir más capas opcionales (layers: 1 -> 2)
   - Aumentar hidden del attention pool (hidden: 64 -> 96)

2. ENTRENAMIENTO:
   - Más épocas (18 -> 30)
   - Learning rate inicial más alto (1.2e-3 -> 2.0e-3)
   - Añadir warmup scheduler
   - Batch size más grande (64 -> 128) si hay memoria

3. REGULARIZACIÓN:
   - Aumentar dropout (0.25 -> 0.35)
   - Añadir weight decay más agresivo
   - Early stopping más estricto

4. PÉRDIDAS:
   - Aumentar peso de break detection (L_BREAK: 1.0 -> 1.5)
   - Reducir peso de VR/CV si son ruidosas

5. VENTANAS:
   - Probar ventanas más pequeñas (96, 128)
   - Añadir más estrato en stride
"""

import sys
sys.path.append('/Users/dario/Documents/Pruebas/product')

# Importar módulo base
from pint_7326 import (
    run_seq_window, run_cb_window, SEQ_WINDOWS, WINDOWS_CB, MODES_CB,
    train_one_fold_seq, infer_scores_seq, build_seq_pack,
    PINTSeq, SelfAttnPool, RobustScaler1D, to_y_series
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Device
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends,"mps") and torch.backends.mps.is_available()) else "cpu")
if DEVICE=="mps": DEVICE="cpu"  # estabilidad

# ===== CONFIG V3.0 MEJORADA =====
SEQ_WINDOWS_V3 = [160, 256]  # Solo ventanas más grandes (mejor rendimiento)
EPOCHS_SEQ_V3 = 18                   # Mismas épocas que v2.1
BATCH_SEQ_V3 = 128                   # Batch más grande
LR_SEQ_V3 = 2.0e-3                   # LR más alto
STRIDE_SEQ_V3 = 8                    # Mantener
W_PRE_V3 = 160                       # Mantener
W_POST_V3 = 160                      # Mantener
MAX_SEQ_V3 = 512                     # Mantener
DT_V3 = 1.0                          # Mantener

# ===== PINTSeq v3.0 con más capacidad =====
class PINTSeq_v3(nn.Module):
    def __init__(self, din, proj=96, lstm_h=128, layers=2, dropout=0.35):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(din, proj), nn.LayerNorm(proj), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(proj, lstm_h, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0)
        H2=2*lstm_h
        self.pool=SelfAttnPool(H2,hidden=96)  # +32 hidden
        self.head_break = nn.Sequential(nn.Linear(H2*2, 192), nn.ReLU(), nn.Dropout(dropout), nn.Linear(192, 1))
        self.head_map = nn.Linear(H2,1)
        self.head_vr  = nn.Sequential(nn.Linear(H2,96), nn.ReLU(), nn.Linear(96,1))
        self.head_cv  = nn.Sequential(nn.Linear(H2,48), nn.ReLU(), nn.Linear(48,1))
    def forward(self, X, mask, br_idx):
        Z=self.proj(X); H,_=self.lstm(Z)
        c,w=self.pool(H,mask)
        B,T,H2=H.shape; br=torch.clamp(br_idx,0,T-1)
        idx=br.view(-1,1,1).repeat(1,1,H2); h_b=torch.gather(H,1,idx).squeeze(1)
        logit_break=self.head_break(torch.cat([c,h_b],dim=1)).squeeze(-1)
        logit_map=self.head_map(H).squeeze(-1)
        pred_vr=self.head_vr(c).squeeze(-1); pred_cv=self.head_cv(H).squeeze(-1)
        return logit_break, logit_map, pred_vr, pred_cv

# ===== FUNCIÓN DE ENTRENAMIENTO V3.0 =====
def train_one_fold_seq_v3(pack_tr, y_ser, idx_tr, epochs=EPOCHS_SEQ_V3, bs=BATCH_SEQ_V3, lr=LR_SEQ_V3, verbose=True, seed=42):
    import sys
    import pint_7326
    import torch.nn.functional as F
    import random
    
    # Fijar seed para reproducibilidad
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Importar funciones desde pint_7326
    pad_batch = pint_7326.pad_batch
    gaussian_targets = pint_7326.gaussian_targets
    focal_bce_logits = pint_7326.focal_bce_logits
    map_reg = pint_7326.map_reg
    to_y_series = pint_7326.to_y_series
    _pbar = pint_7326._pbar
    
    def sub(idxs):
        X=[pack_tr['X_seq'][i] for i in idxs]
        CV=[pack_tr['CV_seq'][i] for i in idxs]
        br=torch.tensor(pack_tr['BR_idx'][idxs],dtype=torch.long,device=DEVICE)
        vr=torch.tensor(pack_tr['VR_tgt'][idxs],dtype=torch.float32,device=DEVICE)
        ids=pack_tr['ids'][idxs]
        yb=torch.tensor(to_y_series(y_ser, pd.Index(ids)).values,dtype=torch.float32,device=DEVICE)
        return X,CV,br,vr,yb

    X_list,CV_list,BR,VR,Y = sub(idx_tr)

    vr_mu=float(np.mean(pack_tr['VR_tgt'][idx_tr])); vr_sd=float(np.std(pack_tr['VR_tgt'][idx_tr])+1e-9)
    VRn = (VR - vr_mu)/vr_sd
    CV_list = [np.clip(c,0.0,3.0).astype(np.float32) for c in CV_list]

    din=pack_tr['D_in']
    # Inicializar parámetros con seed reproducible
    torch.manual_seed(seed)
    model=PINTSeq_v3(din, proj=96, lstm_h=128, layers=2, dropout=0.35).to(DEVICE)
    opt=optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.99), weight_decay=5e-4)  # Más weight decay
    # Warmup scheduler
    warmup_epochs = max(1, epochs // 5)
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep / warmup_epochs)
        return 1.0
    sch=optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    L_BREAK,L_MAP,L_VR,L_CV,L_REG = 1.5, 0.05, 0.10, 0.08, 1e0  # Más peso en break

    it=_pbar(range(epochs),desc="Entrenando (v3.0)",leave=True) if verbose else range(epochs)
    ema=None
    for ep in it:
        order=np.random.permutation(len(X_list)); tot=0.0; seen=0
        for s in range(0,len(order),bs):
            b=order[s:s+bs]
            Xb=[X_list[i] for i in b]; CVb=[CV_list[i] for i in b]
            br=BR[b]; vr=VRn[b]; y=Y[b]
            Xpad,CVpad,M=pad_batch(Xb,CVb)
            logit_break, logit_map, pred_vr, pred_cv = model(Xpad,M,br)
            tgt_map = gaussian_targets(br, Xpad.shape[1], width=10)
            tgt_map = torch.where(M, tgt_map, torch.zeros_like(tgt_map))

            l_break = focal_bce_logits(logit_break, y, alpha=0.55, gamma=1.5)
            l_map   = F.binary_cross_entropy_with_logits(logit_map[M], tgt_map[M])
            l_vr    = F.smooth_l1_loss(pred_vr, vr)
            l_cv    = F.smooth_l1_loss(pred_cv[M], CVpad[M])
            l_reg   = map_reg(logit_map, M, lam_ent=1e-3, lam_tv=0.0)
            loss = L_BREAK*l_break + L_MAP*l_map + L_VR*l_vr + L_CV*l_cv + L_REG*l_reg

            opt.zero_grad(set_to_none=True)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Más restrictivo
            opt.step()
            tot += float(loss.item())*len(b); seen+=len(b)
        sch.step(); avg=tot/max(1,seen); ema=avg if ema is None else 0.9*ema+0.1*avg
        if verbose: it.set_postfix_str(f"avg_loss={ema:.4f}")
    return model

# ===== FUNCIÓN RUN_SEQ_WINDOW V3.0 =====
def run_seq_window_v3(series_tr, tb_tr, series_te, tb_te, y_ser, W, stride=STRIDE_SEQ_V3, verbose=True):
    import pint_7326
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import gc
    
    # Importar funciones desde pint_7326
    build_seq_pack = pint_7326.build_seq_pack
    RobustScaler1D = pint_7326.RobustScaler1D
    to_y_series = pint_7326.to_y_series
    _pbar = pint_7326._pbar
    infer_scores_seq = pint_7326.infer_scores_seq
    
    print(f"\n=== [Seq V3.0] Extrayendo secuencias (W={W}, stride={stride}) ===")
    pack_tr = build_seq_pack(series_tr, tb_tr, y_ser.index.values, window=W, stride=stride, dt=DT_V3)
    te_ids  = pd.Index(list(series_te.keys()), name='id').sort_values()
    pack_te = build_seq_pack(series_te, tb_te, te_ids.values, window=W, stride=stride, dt=DT_V3)
    
    # Escalado robusto
    scaler = RobustScaler1D(clip=8.0); scaler.fit(pack_tr['X_seq'])
    pack_tr['X_seq'] = scaler.transform(pack_tr['X_seq'])
    pack_te['X_seq'] = scaler.transform(pack_te['X_seq'])

    y_al = to_y_series(y_ser, pd.Index(pack_tr['ids']))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    idx_all = np.arange(len(pack_tr['ids']), dtype=int)
    oof = np.full(len(idx_all), np.nan, float); test_fold=[]

    it=_pbar(enumerate(skf.split(idx_all, y_al.values), start=1), total=5, desc=f"Folds V3.0 (W={W})", leave=True)
    for fidx,(tr_idx,va_idx) in it:
        # Usar seed reproducible por fold: SEED + fidx para variedad, pero reproducible
        fold_seed = 42 + fidx  # Cada fold tiene seed diferente pero reproducible
        model = train_one_fold_seq_v3(pack_tr, y_al, tr_idx, epochs=EPOCHS_SEQ_V3, bs=BATCH_SEQ_V3, lr=LR_SEQ_V3, verbose=verbose, seed=fold_seed)
        ids_va, p_va = infer_scores_seq(model, pack_tr, va_idx); map_va = {gid:p for gid,p in zip(ids_va,p_va)}
        oof[va_idx]=np.array([map_va[g] for g in pack_tr['ids'][va_idx]])
        idx_te=np.arange(len(pack_te['ids']),dtype=int)
        ids_te,p_te = infer_scores_seq(model, pack_te, idx_te); map_te={gid:p for gid,p in zip(ids_te,p_te)}
        test_fold.append(np.array([map_te.get(g,0.5) for g in te_ids]))
        auc_f = roc_auc_score(y_al.values[va_idx], oof[va_idx]); it.set_postfix(auc=round(auc_f,6))
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    oof_auc = roc_auc_score(y_al.values, oof)
    print(f"[PINT-Seq V3.0 @ W={W}] OOF AUC = {oof_auc:.6f}")
    test_mean = np.mean(np.vstack(test_fold),axis=0)

    oof_series = pd.Series(oof, index=pd.Index(pack_tr['ids'],name='id'), name=f"oof_seq_v3_W{W}")
    test_series= pd.Series(test_mean, index=te_ids, name=f"test_seq_v3_W{W}")
    return dict(tr_ids=oof_series.index, te_ids=test_series.index, oof=oof_series.values, pte=test_series.values, auc=oof_auc)

# Importar gc para limpieza de memoria
import gc

print("✅ PINT-Seq v3.0 cargado con configuración mejorada:")
print(f"   - Ventanas: {SEQ_WINDOWS_V3}")
print(f"   - Épocas: {EPOCHS_SEQ_V3}")
print(f"   - Batch size: {BATCH_SEQ_V3}")
print(f"   - Learning rate: {LR_SEQ_V3}")
print(f"   - Arquitectura: proj=96, lstm_h=128, layers=2, dropout=0.35")
