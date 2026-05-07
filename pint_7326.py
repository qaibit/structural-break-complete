# ============================================================
# Comité HÍBRIDO = Feature-only CatBoost (PINT-CV) + PINT-Seq v2.1
# - K-Fold, Isotonic calibration por modelo
# - Rank-blend, Calibrated-mean y Meta-Stacker (LR) -> elige el mejor
# - Forense FP/FN
# - Entrada: X_train, y_train, X_test, y_test en memoria
# - Salida: submission_committee_meta_hybrid.csv
# ============================================================

import os, gc, math, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"]=str(SEED)

# tqdm opcional
try:
    from tqdm import tqdm; USE_TQDM=True
except:
    USE_TQDM=False
def _pbar(it, **kw): return tqdm(it, **kw) if USE_TQDM else it

# ----------------- Config -----------------
# ⚠️  IMPORTANT: All proprietary hyperparameters have been removed.
#     Supply your own fine-tuned parameters before running.
#     The full optimized CONDOR engine is available at: https://condor.qaibit.com
#     For licensing inquiries: condor@qaibit.com
#
#     © 2026 Qaibit Technologies S.L. — All rights reserved.

FOLDS        = 5
DT           = 1.0

# CatBoost — supply your own fine-tuned parameters
WINDOWS_CB   = None  # Supply your own window sizes (list of ints)
MODES_CB     = None  # Supply your own feature extraction modes (list of strings)
CAT_PARAMS   = None  # Supply your own fine-tuned CatBoost params (dict)

# PINT-Seq — supply your own fine-tuned parameters
RUN_SEQ      = True
SEQ_WINDOWS  = None  # Supply your own sequence window sizes (list of ints)
EPOCHS_SEQ   = None  # Supply your own training epochs (int)
BATCH_SEQ    = None  # Supply your own batch size (int)
LR_SEQ       = None  # Supply your own learning rate (float)
STRIDE_SEQ   = None  # Supply your own stride (int)
W_PRE=W_POST = None  # Supply your own pre/post normalization window (int)
MAX_SEQ      = 512

# ============================================================
# Utils básicos de datos
# ============================================================
def to_y_series(y, index):
    if isinstance(y, pd.DataFrame): y = y.iloc[:,0]
    y = y.astype(int)
    if isinstance(y.index, pd.MultiIndex):
        if 'id' in y.index.names:
            y = y.reset_index('id').set_index('id').iloc[:,0]
        else:
            raise ValueError("y_* tiene MultiIndex sin nivel 'id'.")
    return y.reindex(index)

def boundary_index_from_period(period_values):
    pv = np.asarray(period_values, dtype=int)
    w = np.flatnonzero(pv==1)
    return int(w[0]) if len(w) else None

def to_series_dict(X_mi: pd.DataFrame):
    assert isinstance(X_mi.index, pd.MultiIndex), "X_* debe tener MultiIndex (id,time)"
    assert {'value','period'}.issubset(set(X_mi.columns)), "X_* debe contener 'value' y 'period'"
    X_mi = X_mi.sort_index(level=[0,1])
    series, tbreak = {}, {}
    for gid, g in X_mi.groupby(level='id', sort=False):
        v = g['value'].to_numpy(); p = g['period'].to_numpy()
        tb = boundary_index_from_period(p)
        if tb is None: continue
        series[gid] = v; tbreak[gid] = int(tb)
    return series, tbreak

# ============================================================
# Estadística/PINT helpers (compartidos por CB y Seq)
# ============================================================
def _mad_sigma(x):
    x = np.asarray(x, float)
    if len(x)==0: return 0.0
    med = np.median(x)
    return 1.4826 * float(np.median(np.abs(x - med)))

def robust_std(x):
    x = np.asarray(x, float)
    if len(x) < 8: return float(np.std(x)+1e-12)
    s_mad = _mad_sigma(x); s_std = float(np.std(x) + 1e-12)
    return float(0.7*s_mad + 0.3*s_std)

def iqr(x):
    x = np.asarray(x, float)
    if len(x)==0: return 0.0
    q1, q3 = np.percentile(x, [25, 75]); return float(q3 - q1)

def energy_derivative(x):
    x = np.asarray(x, float)
    if len(x) < 2: return 0.0
    dx = np.diff(x); return float(np.mean(dx*dx))

def dominant_freq(x, dt=1.0):
    x = np.asarray(x, float)
    if len(x) < 8: return 0.0
    x = x - np.mean(x)
    w = np.hanning(len(x))
    X = np.fft.rfft(x*w)
    freqs = np.fft.rfftfreq(len(x), d=dt)
    if len(freqs) < 2: return 0.0
    mag = np.abs(X); mag[0]=0.0
    k = np.argmax(mag)
    return float(2*np.pi*freqs[k])

def cusum_stat(x):
    x = np.asarray(x, float)
    if len(x)==0: return 0.0
    x = x - np.mean(x); c = np.cumsum(x)
    return float(np.max(np.abs(c)))

def robust_cv(x):
    x = np.asarray(x, float)
    if len(x)==0: return 0.0
    mad = _mad_sigma(x)
    mean_abs = np.mean(np.abs(x)) + 1e-9
    return float(mad / mean_abs)

def cqv(x):
    x = np.asarray(x, float)
    if len(x)==0: return 0.0
    q1, q3 = np.percentile(x, [25,75])
    denom = (q3 + q1)
    return float((q3 - q1) / (denom + 1e-12))

def dfa_alpha(x, scales=(8,12,16,24,32,48)):
    x = np.asarray(x, float)
    n = len(x)
    if n < 16: return 0.0
    y = np.cumsum(x - np.mean(x))
    Fs, Ls = [], []
    for s in scales:
        if s*2 > n: continue
        m = n // s
        if m < 2: continue
        F2 = 0.0
        for k in range(m):
            seg = y[k*s:(k+1)*s]
            t = np.arange(s)
            A = np.vstack([t, np.ones_like(t)]).T
            a, b = np.linalg.lstsq(A, seg, rcond=None)[0]
            trend = a*t + b
            F2 += np.mean((seg - trend)**2)
        F = math.sqrt(F2 / max(1,m))
        Fs.append(F + 1e-12); Ls.append(s)
    if len(Fs)<2: return 0.0
    Ls = np.array(Ls, float); Fs = np.array(Fs, float)
    X = np.vstack([np.log(Ls), np.ones_like(Ls)]).T
    slope = np.linalg.lstsq(X, np.log(Fs), rcond=None)[0][0]
    return float(slope)

def psd_slope(x, dt=1.0):
    x = np.asarray(x, float)
    if len(x) < 16: return 0.0
    x = x - np.mean(x)
    X = np.fft.rfft(x*np.hanning(len(x)))
    f = np.fft.rfftfreq(len(x), d=dt)
    S = (np.abs(X)**2) + 1e-12
    mask = (f > 0) & (f < 0.99*np.max(f))
    if mask.sum() < 10: return 0.0
    xf = np.log(f[mask]); yf = np.log(S[mask])
    A = np.vstack([xf, np.ones_like(xf)]).T
    beta = np.linalg.lstsq(A, yf, rcond=None)[0][0]
    return float(beta)

def allan_vars(x, taus=(2,4,8,16)):
    x = np.asarray(x, float)
    out = {}
    for tau in taus:
        if len(x) < 2*tau:
            out[f"avar_{tau}"] = 0.0; continue
        m = len(x) // tau
        y = x[:m*tau].reshape(m, tau).mean(axis=1)
        dy = np.diff(y)
        avar = 0.5 * np.mean(dy*dy)
        out[f"avar_{tau}"] = float(avar)
    return out

def structure_functions(x, qs=(1,2,3), taus=(1,2,4,8)):
    x = np.asarray(x, float)
    out = {}
    ln_abs_d = []
    for tau in taus:
        if len(x) <= tau:
            for q in qs: out[f"S{q}_tau{tau}"] = 0.0
            continue
        dx = np.abs(x[tau:] - x[:-tau]) + 1e-12
        for q in qs:
            out[f"S{q}_tau{tau}"] = float(np.mean(dx**q))
        ln_abs_d.append(np.log(dx))
    out["intermittency_l2"] = float(np.var(np.concatenate(ln_abs_d))) if ln_abs_d else 0.0
    return out

def amplitude_cv(x, win=16):
    x = np.asarray(x, float)
    n = len(x)
    if n < win*2: return 0.0
    k = np.ones(win)/win
    e = np.convolve(x*x, k, mode='valid')
    rms = np.sqrt(np.maximum(e, 1e-12))
    return float((np.std(rms)+1e-12) / (np.mean(np.abs(rms))+1e-12))

def haar_energies_even(x, levels=3):
    x = np.asarray(x, float)
    a = x.copy()
    Es = []
    for _ in range(levels):
        if len(a) < 2:
            Es.append(0.0); continue
        if len(a) % 2 == 1:
            a = a[:-1]
        approx = (a[0::2] + a[1::2]) * 0.5
        detail = (a[0::2] - a[1::2]) * 0.5
        Es.append(float(np.mean(detail**2)))
        a = approx
    while len(Es) < levels: Es.append(0.0)
    return Es[0], Es[1], Es[2]

# ============================================================
# Feature-only extractor (para CatBoost)
# ============================================================
def extract_features_for_ids(series, tbreaks, ids, w_pre, w_post, mode="base", dt=DT):
    feats = []
    for gid in ids:
        if gid not in series or gid not in tbreaks:
            feats.append(pd.Series({}, name=gid)); continue
        x = np.asarray(series[gid], float); tb = int(tbreaks[gid])

        if mode == "base":
            a = max(0, tb-w_pre); b = min(len(x), tb+w_post)
        elif mode == "pre2":
            a = max(0, tb-2*w_pre); b = min(len(x), tb+w_post)
        elif mode == "post2":
            a = max(0, tb-w_pre); b = min(len(x), tb+2*w_post)
        else:
            raise ValueError("mode desconocido")

        pre  = x[a:tb]; post = x[tb:b]
        if len(pre)<max(20,int(0.5*w_pre)) or len(post)<max(20,int(0.5*w_post)):
            feats.append(pd.Series({}, name=gid)); continue

        # Normalización por PRE
        mu = np.mean(pre); sd = np.std(pre) + 1e-9
        pre_n  = (pre - mu)/sd
        post_n = (post - mu)/sd
        whole  = (x   - mu)/sd

        def block_stats(z):
            return dict(
                mean=float(np.mean(z)),
                std=float(np.std(z)+1e-12),
                mad=_mad_sigma(z),
                iqr=iqr(z),
                energy=energy_derivative(z),
                domw=dominant_freq(z, dt=dt),
                cus=cusum_stat(z),
                q25=float(np.percentile(z,25)),
                q50=float(np.percentile(z,50)),
                q75=float(np.percentile(z,75)),
            )
        S_pre  = block_stats(pre_n)
        S_post = block_stats(post_n)
        E1,E2,E3 = haar_energies_even(whole, levels=3)
        G_basic = dict(
            g_std = robust_std(whole),
            g_mad = _mad_sigma(whole),
            g_iqr = iqr(whole),
            g_energy = energy_derivative(whole),
            g_cus = cusum_stat(whole),
            g_domw = dominant_freq(whole, dt=dt),
            g_hE1=E1, g_hE2=E2, g_hE3=E3,
        )

        # ----- PINT-CV & familia -----
        cv_pre  = robust_cv(pre_n);   cv_post  = robust_cv(post_n);   cv_glob = robust_cv(whole)
        cqv_pre = cqv(pre_n);         cqv_post = cqv(post_n);         cqv_glob = cqv(whole)
        dfa_pre  = dfa_alpha(pre_n);  dfa_post = dfa_alpha(post_n);   dfa_glob = dfa_alpha(whole)
        beta_pre = psd_slope(pre_n, dt=dt); beta_post = psd_slope(post_n, dt=dt); beta_glob = psd_slope(whole, dt=dt)
        av_pre  = allan_vars(pre_n,  taus=(2,4,8,16))
        av_post = allan_vars(post_n, taus=(2,4,8,16))
        av_glob = allan_vars(whole,  taus=(2,4,8,16))
        sf_pre  = structure_functions(pre_n,  qs=(1,2,3), taus=(1,2,4,8))
        sf_post = structure_functions(post_n, qs=(1,2,3), taus=(1,2,4,8))
        sf_glob = structure_functions(whole,  qs=(1,2,3), taus=(1,2,4,8))
        amcv_pre  = amplitude_cv(pre_n,  win=16)
        amcv_post = amplitude_cv(post_n, win=16)
        amcv_glob = amplitude_cv(whole,  win=16)

        R = dict(
            r_rcv = (cv_post / (cv_pre + 1e-9)),
            r_cqv = (cqv_post / (cqv_pre + 1e-9)),
            r_dfa = (dfa_post - dfa_pre),
            r_beta = (beta_post - beta_pre),
            r_amcv = (amcv_post / (amcv_pre + 1e-9)),
            r_S1_tau1 = (sf_post["S1_tau1"] / (sf_pre.get("S1_tau1",1e-9)+1e-9)),
            r_S2_tau1 = (sf_post["S2_tau1"] / (sf_pre.get("S2_tau1",1e-9)+1e-9)),
            r_S3_tau1 = (sf_post["S3_tau1"] / (sf_pre.get("S3_tau1",1e-9)+1e-9)),
            r_avar2 = (av_post["avar_2"] / (av_pre["avar_2"]+1e-9)),
            r_avar8 = (av_post["avar_8"] / (av_pre["avar_8"]+1e-9)),
            r_intermittency = (sf_post["intermittency_l2"] - sf_pre["intermittency_l2"]),
        )

        row = {}
        for k,v in S_pre.items():   row[f"pre_{k}"]=v
        for k,v in S_post.items():  row[f"post_{k}"]=v
        for k,v in G_basic.items(): row[k]=v
        row.update({
            "pre_rcv":cv_pre, "post_rcv":cv_post, "g_rcv":cv_glob,
            "pre_cqv":cqv_pre, "post_cqv":cqv_post, "g_cqv":cqv_glob,
            "pre_dfa":dfa_pre, "post_dfa":dfa_post, "g_dfa":dfa_glob,
            "pre_beta":beta_pre, "post_beta":beta_post, "g_beta":beta_glob,
            "pre_amcv":amcv_pre, "post_amcv":amcv_post, "g_amcv":amcv_glob,
            "pre_intermit":sf_pre["intermittency_l2"],
            "post_intermit":sf_post["intermittency_l2"],
            "g_intermit":sf_glob["intermittency_l2"],
        })
        for d, tag in [(av_pre,"pre"),(av_post,"post"),(av_glob,"g")]:
            for kk,vv in d.items(): row[f"{tag}_{kk}"]=vv
        for d, tag in [(sf_pre,"pre"),(sf_post,"post"),(sf_glob,"g")]:
            for kk,vv in d.items():
                if kk!="intermittency_l2": row[f"{tag}_{kk}"]=vv
        for k,v in R.items(): row[k]=v

        feats.append(pd.Series(row, name=gid))

    F = pd.DataFrame(feats)
    F = F.fillna(F.mean(numeric_only=True))
    return F

# ============================================================
# CatBoost CV
# ============================================================
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
try:
    from catboost import CatBoostClassifier
except Exception as e:
    raise RuntimeError("Necesitas instalar catboost: pip install catboost") from e

def train_catboost_cv(F_tr, y_ser, F_te, folds=5, params=None, desc=""):
    if params is None: params = CAT_PARAMS
    ids_tr = F_tr.index
    y_al = to_y_series(y_ser, ids_tr)
    if y_al.isna().any(): raise ValueError("y_train no cubre todos los ids de F_tr.")
    X = F_tr.values; y = y_al.values
    Xte = F_te.reindex(F_te.index).values

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.full(len(X), np.nan, float); te_fold = []; aucs=[]
    for fidx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        cb = CatBoostClassifier(**params)
        cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), use_best_model=False)
        p_va = cb.predict_proba(X[va_idx])[:,1]
        p_te = cb.predict_proba(Xte)[:,1]
        oof[va_idx] = p_va; te_fold.append(p_te)
        auc_f = roc_auc_score(y[va_idx], p_va); aucs.append(auc_f)
        print(f"[Fold {fidx}] {desc}  AUC={auc_f:.6f}")

    oof_auc = roc_auc_score(y, oof)
    test_mean = np.mean(np.vstack(te_fold), axis=0)
    return oof, test_mean, oof_auc, np.array(aucs)

# ============================================================
# PINT-Seq v2.1 (resumen)
# ============================================================
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends,"mps") and torch.backends.mps.is_available()) else "cpu")
if DEVICE=="mps": DEVICE="cpu"  # estabilidad

def robust_var_ratio(pre, post):
    s1 = _mad_sigma(pre); s2 = _mad_sigma(post)
    return float((s2**2)/(s1**2+1e-12))

def sliding_features(x, window, stride, dt=1.0):
    x = np.asarray(x, float); n=len(x)
    if n<window: return None,None,None
    F=[]; centers=[]; cv_path=[]
    for a in range(0, n-window+1, stride):
        b=a+window; seg=x[a:b]
        mu=float(np.mean(seg)); sd=float(np.std(seg)+1e-12)
        md=_mad_sigma(seg); cv=float(md/(abs(mu)+1e-9))
        # extras rápidos
        w0=dominant_freq(seg,dt); en=energy_derivative(seg); cus=cusum_stat(seg)
        # un pequeño set robusto
        F.append([mu,sd,md,cv,w0,en,cus])
        centers.append(a+window//2); cv_path.append(cv)
    F=np.array(F,np.float32); centers=np.array(centers,int); cv_path=np.array(cv_path,np.float32)
    dF=np.zeros_like(F); dF[1:]=F[1:]-F[:-1]
    F_aug=np.concatenate([F,dF],axis=1)  # [F | dF]
    return F_aug, centers, cv_path

def downsample_time(F, centers, cv_path, max_len=MAX_SEQ):
    T=F.shape[0]
    if T<=max_len: return F,centers,cv_path
    idx=np.linspace(0,T-1,max_len).round().astype(int)
    return F[idx], centers[idx], cv_path[idx]

def build_seq_pack(series_dict, tbreak_dict, ids, window=160, stride=8, dt=DT):
    X_seq, CV_seq, BR_idx, VR_tgt, IDS, LEN=[],[],[],[],[],[]
    D_ref=None
    for gid in ids:
        if gid not in series_dict or gid not in tbreak_dict: continue
        x=np.asarray(series_dict[gid],float); tb=int(tbreak_dict[gid])
        a=max(0,tb-W_PRE); b=min(len(x), tb+W_POST)
        pre=x[a:tb]; post=x[tb:b]
        if len(pre)<max(20,W_PRE//2) or len(post)<max(20,W_POST//2): continue
        mu_pre=float(np.mean(pre)); sd_pre=float(np.std(pre)+1e-9)
        x_n=(x-mu_pre)/sd_pre

        F, centers, cvp = sliding_features(x_n, window=window, stride=stride, dt=dt)
        if F is None or len(centers)==0: continue
        F, centers, cvp = downsample_time(F, centers, cvp, MAX_SEQ)
        k=int(np.argmin(np.abs(centers - tb)))
        vratio=robust_var_ratio((pre-mu_pre)/sd_pre, (post-mu_pre)/sd_pre)
        log_vr=float(np.log(vratio+1e-12))

        if D_ref is None: D_ref=F.shape[1]
        if F.shape[1]!=D_ref: continue

        X_seq.append(F); CV_seq.append(cvp); BR_idx.append(k); VR_tgt.append(log_vr); IDS.append(gid); LEN.append(F.shape[0])

    return dict(
        X_seq=X_seq, CV_seq=CV_seq, BR_idx=np.array(BR_idx,int),
        VR_tgt=np.array(VR_tgt,np.float32), ids=np.array(IDS), lens=np.array(LEN,int),
        D_in=(X_seq[0].shape[1] if X_seq else 0)
    )

class RobustScaler1D:
    def __init__(self, clip=8.0): self.med=None; self.iqr=None; self.clip=clip
    def fit(self, X_list):
        X = np.concatenate([x for x in X_list], axis=0)
        self.med = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0); q25 = np.percentile(X, 25, axis=0)
        self.iqr = q75 - q25; self.iqr[self.iqr<1e-6]=1e-6
    def transform(self, X_list):
        Y=[]
        for x in X_list:
            y=(x - self.med)/self.iqr; y=np.clip(y, -self.clip, self.clip)
            Y.append(y.astype(np.float32))
        return Y

def pad_batch(X_list, cv_list):
    B=len(X_list); Tmax=max([x.shape[0] for x in X_list]); D=X_list[0].shape[1]
    X=torch.zeros(B,Tmax,D,dtype=torch.float32,device=DEVICE)
    CV=torch.zeros(B,Tmax,dtype=torch.float32,device=DEVICE)
    M=torch.zeros(B,Tmax,dtype=torch.bool,device=DEVICE)
    for i,(xi,cvi) in enumerate(zip(X_list,cv_list)):
        t=xi.shape[0]; X[i,:t,:]=torch.tensor(xi,device=DEVICE); CV[i,:t]=torch.tensor(cvi,device=DEVICE); M[i,:t]=True
    return X,CV,M

class SelfAttnPool(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__(); self.lin1=nn.Linear(dim, hidden); self.lin2=nn.Linear(hidden,1)
    def forward(self,H,mask):
        a=torch.tanh(self.lin1(H)); e=self.lin2(a).squeeze(-1)
        e=e.masked_fill(~mask, -1e9); w=torch.softmax(e, dim=1)
        c=torch.bmm(w.unsqueeze(1),H).squeeze(1); return c,w

class PINTSeq(nn.Module):
    # Supply your own fine-tuned architecture dimensions
    def __init__(self, din, proj=None, lstm_h=None, layers=None, dropout=None):
        assert all(v is not None for v in [proj, lstm_h, layers, dropout]), \
            "Supply your own fine-tuned architecture parameters (proj, lstm_h, layers, dropout)"
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(din, proj), nn.LayerNorm(proj), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(proj, lstm_h, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.0)
        H2=2*lstm_h
        self.pool=SelfAttnPool(H2,hidden=None)  # Supply your own fine-tuned attention hidden dim
        self.head_break = nn.Sequential(nn.Linear(H2*2, None), nn.ReLU(), nn.Dropout(dropout), nn.Linear(None, 1))  # Supply your own fine-tuned head dims
        self.head_map = nn.Linear(H2,1)
        self.head_vr  = nn.Sequential(nn.Linear(H2,64), nn.ReLU(), nn.Linear(64,1))
        self.head_cv  = nn.Sequential(nn.Linear(H2,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, X, mask, br_idx):
        Z=self.proj(X); H,_=self.lstm(Z)
        c,w=self.pool(H,mask)
        B,T,H2=H.shape; br=torch.clamp(br_idx,0,T-1)
        idx=br.view(-1,1,1).repeat(1,1,H2); h_b=torch.gather(H,1,idx).squeeze(1)
        logit_break=self.head_break(torch.cat([c,h_b],dim=1)).squeeze(-1)
        logit_map=self.head_map(H).squeeze(-1)
        pred_vr=self.head_vr(c).squeeze(-1); pred_cv=self.head_cv(H).squeeze(-1)
        return logit_break, logit_map, pred_vr, pred_cv

def gaussian_targets(idx,T,width=10):
    t=torch.arange(T,device=idx.device).float().unsqueeze(0)
    mu=idx.float().unsqueeze(1)
    g=torch.exp(-0.5*((t-mu)/(width+1e-6))**2)
    return g/(g.max(dim=1,keepdim=True).values+1e-9)

def focal_bce_logits(logits, targets, alpha=0.55, gamma=1.5):
    bce=F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p=torch.sigmoid(logits)
    pt=torch.where(targets>0.5, p, 1-p)
    return (alpha*(1-pt)**gamma * bce).mean()

def map_reg(logit_map, mask, lam_ent=1e-3, lam_tv=0.0):
    logits=logit_map.masked_fill(~mask, -1e9)
    prob=torch.softmax(logits, dim=1)
    ent=-(prob*torch.clamp(prob,1e-9).log()).sum(dim=1).mean()
    reg=lam_ent*ent
    if lam_tv>0:
        diff=(logit_map[:,1:]-logit_map[:,:-1]); m2=mask[:,1:]&mask[:,:-1]
        reg += lam_tv*torch.mean(torch.abs(diff[m2]))
    return reg

def train_one_fold_seq(pack_tr, y_ser, idx_tr, epochs=EPOCHS_SEQ, bs=BATCH_SEQ, lr=LR_SEQ, verbose=True):
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
    # Supply your own fine-tuned architecture and training parameters
    model=PINTSeq(din, proj=None, lstm_h=None, layers=None, dropout=None).to(DEVICE)
    opt=optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.99), weight_decay=None)  # Supply your own weight_decay
    sch=optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,epochs))
    # Supply your own fine-tuned loss weights
    L_BREAK,L_MAP,L_VR,L_CV,L_REG = None, None, None, None, None

    it=_pbar(range(epochs),desc="Entrenando (épocas)",leave=True) if verbose else range(epochs)
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
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())*len(b); seen+=len(b)
        sch.step(); avg=tot/max(1,seen); ema=avg if ema is None else 0.9*ema+0.1*avg
        if verbose: it.set_postfix_str(f"avg_loss={ema:.4f}")
    return model

@torch.no_grad()
def infer_scores_seq(model, pack, idxs):
    ids=pack['ids'][idxs]
    X_list=[pack['X_seq'][i] for i in idxs]
    CV_list=[pack['CV_seq'][i] for i in idxs]
    br=torch.tensor(pack['BR_idx'][idxs],dtype=torch.long,device=DEVICE)
    scores=[]
    step=256
    for s in range(0,len(idxs),step):
        b=idxs[s:s+step]
        Xb=X_list[s:s+step]; CVb=CV_list[s:s+step]
        Xpad,CVpad,M=pad_batch(Xb,CVb)
        logit_break, logit_map, pred_vr, pred_cv = model(Xpad,M,br[s:s+step])
        p=torch.sigmoid(logit_break).detach().cpu().numpy().astype(np.float32)
        scores.append(p)
    return ids, np.concatenate(scores,axis=0)

def run_seq_window(series_tr, tb_tr, series_te, tb_te, y_ser, W, stride=STRIDE_SEQ, verbose=True):
    if not RUN_SEQ:
        return None
    print(f"\n=== [Seq] Extrayendo secuencias (W={W}, stride={stride}) ===")
    pack_tr = build_seq_pack(series_tr, tb_tr, y_ser.index.values, window=W, stride=stride, dt=DT)
    te_ids  = pd.Index(list(series_te.keys()), name='id').sort_values()
    pack_te = build_seq_pack(series_te, tb_te, te_ids.values, window=W, stride=stride, dt=DT)
    # Escalado robusto
    scaler = RobustScaler1D(clip=8.0); scaler.fit(pack_tr['X_seq'])
    pack_tr['X_seq'] = scaler.transform(pack_tr['X_seq'])
    pack_te['X_seq'] = scaler.transform(pack_te['X_seq'])

    y_al = to_y_series(y_ser, pd.Index(pack_tr['ids']))
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    idx_all = np.arange(len(pack_tr['ids']), dtype=int)
    oof = np.full(len(idx_all), np.nan, float); test_fold=[]

    it=_pbar(enumerate(skf.split(idx_all, y_al.values), start=1), total=FOLDS, desc=f"Folds (W={W})", leave=True)
    for fidx,(tr_idx,va_idx) in it:
        model = train_one_fold_seq(pack_tr, y_al, tr_idx, epochs=EPOCHS_SEQ, bs=BATCH_SEQ, lr=LR_SEQ, verbose=verbose)
        ids_va, p_va = infer_scores_seq(model, pack_tr, va_idx); map_va = {gid:p for gid,p in zip(ids_va,p_va)}
        oof[va_idx]=np.array([map_va[g] for g in pack_tr['ids'][va_idx]])
        idx_te=np.arange(len(pack_te['ids']),dtype=int)
        ids_te,p_te = infer_scores_seq(model, pack_te, idx_te); map_te={gid:p for gid,p in zip(ids_te,p_te)}
        test_fold.append(np.array([map_te.get(g,0.5) for g in te_ids]))
        auc_f = roc_auc_score(y_al.values[va_idx], oof[va_idx]); it.set_postfix(auc=round(auc_f,6))
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    oof_auc = roc_auc_score(y_al.values, oof)
    print(f"[PINT-Seq @ W={W}] OOF AUC = {oof_auc:.6f}")
    test_mean = np.mean(np.vstack(test_fold),axis=0)

    oof_series = pd.Series(oof, index=pd.Index(pack_tr['ids'],name='id'), name=f"oof_seq_W{W}")
    test_series= pd.Series(test_mean, index=te_ids, name=f"test_seq_W{W}")
    return dict(tr_ids=oof_series.index, te_ids=test_series.index, oof=oof_series.values, pte=test_series.values, auc=oof_auc)

# ============================================================
# Blends / Calibración / Forense
# ============================================================
from sklearn.metrics import roc_curve

def youden_threshold(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr; k = np.argmax(j); return float(thr[k])

def print_forensics(tag, y_true, y_score, ids, save_dir="forensics_out"):
    os.makedirs(save_dir, exist_ok=True)
    thr = youden_threshold(y_true, y_score)
    pred = (y_score >= thr).astype(int)
    TP = int(((pred==1)&(y_true==1)).sum())
    FP = int(((pred==1)&(y_true==0)).sum())
    FN = int(((pred==0)&(y_true==1)).sum())
    TN = int(((pred==0)&(y_true==0)).sum())
    prec = TP / max(1, (TP+FP)); rec  = TP / max(1, (TP+FN))
    print(f"[FORENSE:{tag}] Umbral={thr:.4f} | TP={TP} FP={FP} FN={FN} TN={TN} | Precision={prec:.3f} Recall={rec:.3f}")

    df = pd.DataFrame({'id': ids, 'score': y_score}).set_index('id')
    top_fp = df.loc[y_true==0].sort_values('score', ascending=False).head(15)
    top_fn = df.loc[y_true==1].sort_values('score', ascending=True).head(15)
    print(f"[FORENSE:{tag}] Top-15 FP:\n{top_fp}\n[FORENSE:{tag}] Top-15 FN:\n{top_fn}")
    top_fp.to_csv(os.path.join(save_dir, f"{tag}_fp.csv"))
    top_fn.to_csv(os.path.join(save_dir, f"{tag}_fn.csv"))
    return thr

def isotonic_calibrate_per_model(results_dict, y_ser):
    """
    results_dict: {name: {tr_ids, te_ids, oof, pte, auc}}
    Devuelve matrices calibradas O_cal/T_cal alineadas a intersección de ids.
    """
    keys = list(results_dict.keys())
    tr_ids = results_dict[keys[0]]["tr_ids"]
    for k in keys[1:]: tr_ids = tr_ids.intersection(results_dict[k]["tr_ids"])
    tr_ids = tr_ids.sort_values()

    te_ids = results_dict[keys[0]]["te_ids"]
    for k in keys[1:]: te_ids = te_ids.intersection(results_dict[k]["te_ids"])
    te_ids = te_ids.sort_values()

    Y = to_y_series(y_ser, tr_ids).values
    O_cal, T_cal = [], []

    for k in keys:
        o = pd.Series(results_dict[k]["oof"], index=results_dict[k]["tr_ids"]).reindex(tr_ids).values
        t = pd.Series(results_dict[k]["pte"], index=results_dict[k]["te_ids"]).reindex(te_ids).values
        # normaliza a [0,1] antes de IR
        o_n = (o - o.min())/(o.ptp()+1e-12)
        ir = IsotonicRegression(out_of_bounds='clip'); ir.fit(o_n, Y)
        o_c = ir.transform(o_n)
        t_n = (t - t.min())/(t.ptp()+1e-12)
        t_c = ir.transform(t_n)
        O_cal.append(o_c); T_cal.append(t_c)
    return keys, tr_ids, te_ids, np.vstack(O_cal), np.vstack(T_cal)

def calibrated_mean_blend(O_cal, T_cal, y_true):
    aucs = np.array([roc_auc_score(y_true, O_cal[m]) for m in range(O_cal.shape[0])])
    w = np.exp((aucs - aucs.mean())/max(1e-6, aucs.std())); w = w / w.sum()
    oof = (O_cal * w.reshape(-1,1)).sum(axis=0)
    te  = (T_cal * w.reshape(-1,1)).sum(axis=0)
    return oof, te, float(roc_auc_score(y_true, oof)), dict(enumerate(w))

def meta_stacker_lr(keys, tr_ids, te_ids, O_cal, T_cal, y_true, folds=FOLDS):
    M, N = O_cal.shape
    X_base = O_cal.T
    # metafeatures
    mean = X_base.mean(axis=1, keepdims=True); std  = X_base.std(axis=1, keepdims=True)
    mn   = X_base.min(axis=1, keepdims=True); mx   = X_base.max(axis=1, keepdims=True)
    q25  = np.quantile(X_base, 0.25, axis=1, keepdims=True); q75  = np.quantile(X_base, 0.75, axis=1, keepdims=True)
    iqr_ = q75 - q25
    ranks = np.argsort(np.argsort(X_base, axis=1), axis=1) / (M-1+1e-9)
    mean_rank = ranks.mean(axis=1, keepdims=True)
    X_meta = np.hstack([X_base, mean, std, mn, mx, iqr_, mean_rank])
    # test meta
    Xb_te = T_cal.T
    mean_te = Xb_te.mean(axis=1, keepdims=True); std_te  = Xb_te.std(axis=1, keepdims=True)
    mn_te   = Xb_te.min(axis=1, keepdims=True); mx_te   = Xb_te.max(axis=1, keepdims=True)
    q25_te  = np.quantile(Xb_te, 0.25, axis=1, keepdims=True); q75_te  = np.quantile(Xb_te, 0.75, axis=1, keepdims=True)
    iqr_te  = q75_te - q25_te
    ranks_te = np.argsort(np.argsort(Xb_te, axis=1), axis=1) / (M-1+1e-9)
    mean_rank_te = ranks_te.mean(axis=1, keepdims=True)
    X_meta_te = np.hstack([Xb_te, mean_te, std_te, mn_te, mx_te, iqr_te, mean_rank_te])

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.full(X_meta.shape[0], np.nan); te_fold=[]
    for tr, va in skf.split(X_meta, y_true):
        lr = LogisticRegression(penalty="l2", C=2.0, solver="liblinear", max_iter=2000, random_state=SEED)
        lr.fit(X_meta[tr], y_true[tr])
        oof[va] = lr.predict_proba(X_meta[va])[:,1]
        te_fold.append(lr.predict_proba(X_meta_te)[:,1])
    te_mean = np.mean(np.vstack(te_fold), axis=0)
    auc = roc_auc_score(y_true, oof)
    return oof, te_mean, auc

# ============================================================
# Runners
# ============================================================
def run_cb_window(series_tr, tb_tr, series_te, tb_te, y_ser, w, mode):
    tr_ids = pd.Index(list(series_tr.keys()), name='id').sort_values()
    te_ids = pd.Index(list(series_te.keys()),  name='id').sort_values()
    F_tr = extract_features_for_ids(series_tr, tb_tr, tr_ids, w, w, mode=mode, dt=DT)
    F_te = extract_features_for_ids(series_te, tb_te, te_ids, w, w, mode=mode, dt=DT)
    F_tr = F_tr.fillna(F_tr.mean(numeric_only=True)); F_te = F_te.fillna(F_tr.mean(numeric_only=True))
    desc = f"w=({w},{w}) {mode}"
    oof, pte, auc_oof, _ = train_catboost_cv(F_tr, y_ser, F_te, folds=FOLDS, params=CAT_PARAMS, desc=desc)
    print(f"[Feature+CatBoost @ W={w} mode={mode}] OOF AUC = {auc_oof:.6f}")
    return dict(tr_ids=tr_ids, te_ids=te_ids, oof=oof, pte=pte, auc=auc_oof)

def rank_blend(results, y_ser):
    keys = list(results.keys())
    aucs = np.array([results[k]["auc"] for k in keys], float)
    w = np.exp( (aucs - aucs.mean()) / (max(1e-6, aucs.std())) ); w = w / w.sum()

    all_ids = results[keys[0]]["tr_ids"]
    for k in keys[1:]: all_ids = all_ids.union(results[k]["tr_ids"])
    all_ids = all_ids.sort_values()

    rank_mat = []
    for k in keys:
        ids_k = results[k]["tr_ids"]
        oof_k = pd.Series(results[k]["oof"], index=ids_k).reindex(all_ids)
        r = oof_k.rank(method="average").values; rank_mat.append(r)
    rank_mat = np.vstack(rank_mat); w_col = w.reshape(-1,1)
    rank_blended = (rank_mat * w_col).sum(axis=0) / w.sum()
    oof_blend = (rank_blended - rank_blended.min()) / (rank_blended.ptp() + 1e-12)

    te_ids = results[keys[0]]["te_ids"]
    for k in keys[1:]: te_ids = te_ids.intersection(results[k]["te_ids"])
    te_ids = te_ids.sort_values()

    rank_te = []
    for k in keys:
        ids_k = results[k]["te_ids"]
        pte_k = pd.Series(results[k]["pte"], index=ids_k).reindex(te_ids)
        r = pte_k.rank(method="average").values; rank_te.append(r)
    rank_te = np.vstack(rank_te)
    te_blended = (rank_te * w_col).sum(axis=0) / w.sum()
    test_blend = (te_blended - te_blended.min()) / (te_blended.ptp() + 1e-12)

    y_full = to_y_series(y_ser, all_ids).values
    auc_blend = roc_auc_score(y_full, oof_blend)
    return all_ids, te_ids, oof_blend, test_blend, float(auc_blend), dict(zip(keys, w))

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Comprobaciones
    for nm in ["X_train","y_train","X_test","y_test"]:
        if nm not in globals(): raise RuntimeError(f"{nm} no está en memoria.")
    if not isinstance(X_train.index, pd.MultiIndex) or not isinstance(X_test.index, pd.MultiIndex):
        raise ValueError("X_train / X_test deben tener MultiIndex (id,time).")
    if not {'value','period'}.issubset(set(X_train.columns)) or not {'value','period'}.issubset(set(X_test.columns)):
        raise ValueError("X_* deben contener columnas 'value' y 'period'.")

    # dicts
    tr_series, tr_tb = to_series_dict(X_train)
    te_series, te_tb = to_series_dict(X_test)
    train_ids = pd.Index(list(tr_series.keys()), name='id').sort_values()
    test_ids  = pd.Index(list(te_series.keys()),  name='id').sort_values()
    y_ser = to_y_series(y_train, train_ids)
    if y_ser.isna().any(): raise ValueError("y_train no cubre todos los ids presentes en X_train.")
    print(f"DEVICE: {DEVICE}  (CUDA:{torch.cuda.is_available()}  MPS:{hasattr(torch.backends,'mps') and torch.backends.mps.is_available()})")

    # ===== 1) Entrenar CatBoosts =====
    results = {}
    for w in WINDOWS_CB:
        print(f"\n=== Ejecutando ventana {w} (CatBoost) ===")
        for mode in MODES_CB:
            key = f"CB_W{w}_{mode}"
            res = run_cb_window(tr_series, tr_tb, te_series, te_tb, y_ser, w, mode)
            results[key] = res; gc.collect()

    # ===== 2) Entrenar PINT-Seq (opcional) =====
    if RUN_SEQ:
        for w in SEQ_WINDOWS:
            key = f"SEQ_W{w}"
            res = run_seq_window(tr_series, tr_tb, te_series, te_tb, y_ser, w, stride=STRIDE_SEQ, verbose=True)
            if res is not None:
                results[key] = res; gc.collect()

    # ===== 3) Rank-Blend (referencia)
    all_ids_rk, te_ids_rk, oof_rk, test_rk, auc_rk, weights_rk = rank_blend(results, y_ser)
    print(f"\n[Rank-Blend] OOF AUC (comité) = {auc_rk:.6f} | #models={len(results)}")

    # ===== 4) Isotonic calibration por modelo y Media ponderada
    keys, tr_ids_cal, te_ids_cal, O_cal, T_cal = isotonic_calibrate_per_model(results, y_ser)
    y_true_cal = to_y_series(y_train, tr_ids_cal).values
    oof_cm, te_cm, auc_cm, w_idx = calibrated_mean_blend(O_cal, T_cal, y_true_cal)
    print(f"[Calibrated-Mean] OOF AUC = {auc_cm:.6f}")

    # ===== 5) Meta-Stacker (LR)
    oof_ms, te_ms, auc_ms = meta_stacker_lr(keys, tr_ids_cal, te_ids_cal, O_cal, T_cal, y_true_cal, folds=FOLDS)
    print(f"[Meta-LR] OOF AUC = {auc_ms:.6f}")

    # ===== 6) Elegir mejor ensemble por OOF (desempate por TEST si disponible)
    from sklearn.metrics import roc_auc_score
    # calculamos TEST para cada método si tenemos y_test
    test_auc_rk = test_auc_cm = test_auc_ms = None
    y_te_rk = to_y_series(y_test, te_ids_rk)
    if not y_te_rk.isna().any(): test_auc_rk = roc_auc_score(y_te_rk.values, test_rk)
    y_te_cm = to_y_series(y_test, te_ids_cal)
    if not y_te_cm.isna().any(): test_auc_cm = roc_auc_score(y_te_cm.values, te_cm)
    if not y_te_cm.isna().any(): test_auc_ms = roc_auc_score(y_te_cm.values, te_ms)

    candidates = [
        ("rank", auc_rk, test_auc_rk, (all_ids_rk, te_ids_rk, oof_rk, test_rk)),
        ("cmean", auc_cm, test_auc_cm, (tr_ids_cal, te_ids_cal, oof_cm, te_cm)),
        ("metalr", auc_ms, test_auc_ms, (tr_ids_cal, te_ids_cal, oof_ms, te_ms)),
    ]
    candidates_sorted = sorted(candidates, key=lambda z: (z[1], z[2] if z[2] is not None else -1), reverse=True)
    best_name, best_oof_auc, best_test_auc, (best_tr_ids, best_te_ids, best_oof, best_test) = candidates_sorted[0]
    print(f"\n[Committee FINAL] Método={best_name} | OOF AUC = {best_oof_auc:.6f} | TEST AUC = {best_test_auc if best_test_auc is not None else None:.6f}")

    # ===== 7) Forense y submission
    thr_oof = print_forensics("oof", to_y_series(y_train, best_tr_ids).values, best_oof, best_tr_ids.values)
    y_te_best = to_y_series(y_test, best_te_ids)
    if not y_te_best.isna().any(): print_forensics("test", y_te_best.values, best_test, best_te_ids.values)

    SUB_NAME = "submission_committee_meta_hybrid.csv"
    pd.Series(best_test, index=best_te_ids, name="break_score").to_csv(SUB_NAME, header=True)
    print("Guardado:", SUB_NAME)

    # Summary
    print("\n=== SUMMARY ===")
    print("oof_auc:", best_oof_auc)
    print("test_auc:", best_test_auc if best_test_auc is not None else None)
    print("oof: shape=", (len(best_tr_ids),))
    print("test_pred: shape=", (len(best_te_ids),))
