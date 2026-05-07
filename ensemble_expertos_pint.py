#!/usr/bin/env python3
"""
Wrapper para ejecutar expertos_8642.py + pint_7326.py juntos
Desde notebook: importar este archivo y ejecutar run_all_combined
"""

import sys

# Importar ambos módulos
from expertos_8642 import (
    to_series_dict, build_features_impl2, foldwise_mi_select,
    ks_shift_filter_train_only, oof_hgb_with_test, oof_dist_hgb_with_test,
    oof_catboost_multi, oof_xgb_impl3_with_test, oof_xgb_on_matrix,
    optimize_rank_blend_dirichlet, optimize_rank_blend_slsqp, meta_stacker_lr,
    build_curriculum_and_pseudo, to_y_series, build_impl3_feature_tables,
    create_comprehensive_features_pd, oof_pint_hybrid_with_test, build_features_pint,
    SUBMISSION_NAME, SEED, FOLDS, HAS_CATBOOST, HAS_XGB, HAS_TORCH
)

from pint_7326 import (
    run_seq_window, run_cb_window,
    rank_blend, isotonic_calibrate_per_model, calibrated_mean_blend,
    meta_stacker_lr,
    SEQ_WINDOWS, WINDOWS_CB, MODES_CB, RUN_SEQ as USE_PINT_SEQ
)

# Opcional: cargar versión v3 optimizada de PINT-Seq
try:
    from pint_seq_v3_optimized import run_seq_window_v3, SEQ_WINDOWS_V3, EPOCHS_SEQ_V3, BATCH_SEQ_V3, LR_SEQ_V3
    USE_PINT_SEQ_V3 = True
    print("✅ PINT-Seq v3.0 cargado (optimizado)")
except ImportError as e:
    USE_PINT_SEQ_V3 = False
    print(f"⚠️ PINT-Seq v3.0 no disponible: {e}")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

# Verificar si scipy.optimize está disponible
try:
    from scipy.optimize import minimize
    HAS_SLSQP = True
except Exception:
    HAS_SLSQP = False

def run_all_combined(
    X_train,
    y_train,
    X_test,
    y_test=None,
    use_pint=True,
    use_pint_hybrid=True,
    allowed_models=None,
    use_meta_in_mix=True,
    alpha_fixed=None,
):
    """
    Ejecuta ensemble completo con expertos_8642 + PINT-Seq
    
    Entrada:
        X_train, X_test: DataFrame con MultiIndex (id, time) y columnas [value, period]
        y_train: Series con índice de id
        y_test: Series con índice de id (opcional)
    
    Salida:
        dict con resultados completos
    """
    
    print("="*80)
    print("🚀 ENSEMBLE EXPERTO 8642 + PINT-SEQ v2.1")
    print("="*80)
    
    # ===== 1. EXPERTOS 8642: Features y modelos base =====
    print("\n📊 FASE 1: Expertos 8642 (features y modelos base)...")
    
    # Impl3 features
    Ftr_impl3, Fte_impl3 = build_impl3_feature_tables(X_train, X_test)
    print(f"   ✅ Impl3: {Ftr_impl3.shape}")
    
    # Comprehensive features
    try:
        df_long_tr = X_train.reset_index()[['id', 'period', 'value']]
        df_long_te = X_test.reset_index()[['id', 'period', 'value']]
        feat_train_cf = create_comprehensive_features_pd(df_long_tr).set_index('id')
        feat_test_cf = create_comprehensive_features_pd(df_long_te).set_index('id')
        print(f"   ✅ Comprehensive features: {feat_train_cf.shape}")
    except Exception as e:
        print(f"   ⚠️ Comprehensive features falló: {e}")
        feat_train_cf = pd.DataFrame(index=Ftr_impl3.index)
        feat_test_cf = pd.DataFrame(index=Fte_impl3.index)
    
    # Impl2 features
    print("   📊 Construyendo Impl2...")
    # Generar features PINT por id (como en la implementación original) para inyectarlas en Xfull
    pint_tr_df = pd.DataFrame(index=Ftr_impl3.index)
    pint_te_df = pd.DataFrame(index=Fte_impl3.index)
    try:
        tr_series_tmp, tr_tb_tmp = to_series_dict(X_train)
        te_series_tmp, te_tb_tmp = to_series_dict(X_test)
        if HAS_TORCH:
            pint_tr_df = build_features_pint(tr_series_tmp, tr_tb_tmp, desc="PINT (train)")
            # Nota: para replicar fielmente, se podría reutilizar el modelo entrenado; aquí calculamos también en test
            pint_te_df = build_features_pint(te_series_tmp, te_tb_tmp, desc="PINT (test)")
            # Alinear índices
            pint_tr_df = pint_tr_df.reindex(Ftr_impl3.index)
            pint_te_df = pint_te_df.reindex(Fte_impl3.index)
    except Exception as e:
        print(f"   ⚠️ Features PINT no disponibles para inyección en Impl2: {e}")
    (Xfull, Xfull_t, Xdist, Xdist_t,
     (tr_series, tr_tb), (te_series, te_tb), sig_tr_df) = build_features_impl2(
        X_train, X_test, extra_impl3_feats=None,
        extra_pint_feats_tr=pint_tr_df.join(feat_train_cf, how='left'),
        extra_pint_feats_te=pint_te_df.join(feat_test_cf,  how='left')
    )
    print(f"   ✅ Impl2: {Xfull.shape}")
    
    y_vec = to_y_series(y_train, Xfull.index).astype(int).values.ravel()
    
    # Filtro KS
    print("   🧹 Aplicando filtro KS-shift...")
    keep_cols, drop_cols = ks_shift_filter_train_only(Xfull, y_vec, frac=0.05, seed=SEED)
    Xks = Xfull[keep_cols].copy()
    Xks_t = Xfull_t[keep_cols].copy()
    print(f"   ✅ Eliminadas {len(drop_cols)} cols, quedan {len(keep_cols)}")
    
    # MI selection
    print("   🔍 Selección MI fold-wise...")
    Xmi, keep_mi = foldwise_mi_select(Xks, y_vec, folds=FOLDS, topk=420, seed=SEED)
    Xmi_t = Xks_t.reindex(columns=keep_mi).fillna(Xks[keep_mi].median(numeric_only=True))
    print(f"   ✅ MI: {Xmi.shape}")
    
    # Modelos base (teacher)
    print("\n🎯 FASE 2: Modelos base (teacher)...")
    
    preds_train0 = {}
    preds_test0 = {}
    
    # HGB
    print("   📈 HGB...")
    oof_hgb0, te_hgb0, auc_hgb0 = oof_hgb_with_test(Xmi, y_vec, Xmi_t)
    preds_train0["hgb"] = oof_hgb0
    preds_test0["hgb"] = te_hgb0
    print(f"      AUC: {auc_hgb0:.4f}")
    
    # Dist
    print("   📈 Dist...")
    oof_dist0, te_dist0, auc_dist0 = oof_dist_hgb_with_test(Xdist, y_vec, Xdist_t)
    preds_train0["dist"] = oof_dist0
    preds_test0["dist"] = te_dist0
    print(f"      AUC: {auc_dist0:.4f}")
    
    # CatBoost A
    if HAS_CATBOOST:
        print("   📈 CatBoost A...")
        cbA0 = oof_catboost_multi(Xmi, y_vec, Xmi_t, seeds=(42, 1337, 2027), 
                                   params=dict(loss_function="Logloss", eval_metric="AUC",
                                               auto_class_weights="Balanced", iterations=3200,
                                               learning_rate=0.028, depth=7, l2_leaf_reg=12.0,
                                               verbose=False, thread_count=1, rsm=0.92,
                                               border_count=128, bootstrap_type="Bayesian",
                                               bagging_temperature=0.5, random_strength=0.7,
                                               leaf_estimation_iterations=6),
                                   feat_fraction=0.80, label="A")
        if cbA0 is not None:
            oA, tA, _ = cbA0["avg"]
            preds_train0["cbA"] = oA
            preds_test0["cbA"] = tA
            print(f"      AUC: {roc_auc_score(y_vec, oA):.4f}")
    
    # CatBoost B
    if HAS_CATBOOST:
        print("   📈 CatBoost B...")
        cbB0 = oof_catboost_multi(Xmi, y_vec, Xmi_t, seeds=(3329, 5153),
                                   params=dict(loss_function="Logloss", eval_metric="AUC",
                                               auto_class_weights="Balanced", iterations=4000,
                                               learning_rate=0.022, depth=8, l2_leaf_reg=9.0,
                                               verbose=False, thread_count=1, rsm=0.95,
                                               border_count=128, bootstrap_type="Bernoulli",
                                               subsample=0.72, random_strength=0.8,
                                               leaf_estimation_iterations=6),
                                   feat_fraction=0.70, label="B")
        if cbB0 is not None:
            oB, tB, _ = cbB0["avg"]
            preds_train0["cbB"] = oB
            preds_test0["cbB"] = tB
            print(f"      AUC: {roc_auc_score(y_vec, oB):.4f}")
    
    # XGBoost Impl3
    if HAS_XGB:
        print("   📈 XGB Impl3...")
        xgb_oof0, xgb_te0, auc_xgb0 = oof_xgb_impl3_with_test(Ftr_impl3, to_y_series(y_train, Ftr_impl3.index), Fte_impl3)
        preds_train0["xgb_raw"] = xgb_oof0
        preds_test0["xgb_raw"] = xgb_te0
        print(f"      AUC: {auc_xgb0:.4f}")
    
    # XGBoost Xmi
    if HAS_XGB:
        print("   📈 XGB Xmi...")
        xgb_xmi_oof, xgb_xmi_te, auc_xgb_xmi = oof_xgb_on_matrix(Xmi, y_vec, Xmi_t)
        preds_train0["xgb_xmi"] = xgb_xmi_oof
        preds_test0["xgb_xmi"] = xgb_xmi_te
        print(f"      AUC: {auc_xgb_xmi:.4f}")
    
    # PINT-Hybrid (como en la implementación original)
    if use_pint_hybrid and HAS_TORCH:
        try:
            print("   📈 PINT-Hybrid...")
            cbP_params = dict(
                loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
                iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,  # Tune: these values
                verbose=False, thread_count=1, rsm=0.85, border_count=128,
                bootstrap_type="Bayesian", bagging_temperature=1.0,
                random_strength=0.5, leaf_estimation_iterations=4
            )
            oof_ph, te_ph, auc_ph = oof_pint_hybrid_with_test(
                Xmi, y_vec, Xmi_t,
                series_source_tr=X_train, series_source_te=X_test,
                n_folds=FOLDS, cb_params_pint=cbP_params
            )
            preds_train0["pint_hybrid"] = oof_ph
            preds_test0["pint_hybrid"] = te_ph
            print(f"      AUC: {auc_ph:.4f}")
        except Exception as e:
            print(f"      ⚠️ PINT-Hybrid falló: {e}")
    
    # ===== 3. PINT-SEQ: Modelos adicionales =====
    print(f"\n🧠 FASE 3: PINT-Seq ({'v3.0 optimizado' if USE_PINT_SEQ_V3 else 'v2.1'})...")
    
    # PINT-Seq por ventana
    pint_seq_results = {}
    if USE_PINT_SEQ and HAS_TORCH and use_pint:
        # Usar versión v3 si está disponible
        windows_to_use = SEQ_WINDOWS_V3 if USE_PINT_SEQ_V3 else SEQ_WINDOWS
        run_func = run_seq_window_v3 if USE_PINT_SEQ_V3 else run_seq_window
        
        for w in windows_to_use:
            print(f"   🧠 PINT-Seq W={w} ({'v3.0' if USE_PINT_SEQ_V3 else 'v2.1'})...")
            try:
                res = run_func(tr_series, tr_tb, te_series, te_tb, y_train, w)
                if res is not None:
                    key = f"pint_seq_{w}"
                    pint_seq_results[key] = res
                    # Integrar como modelos base adicionales
                    oof_series = pd.Series(res["oof"], index=res["tr_ids"])
                    te_series_k = pd.Series(res["pte"], index=res["te_ids"])
                    preds_train0[key] = oof_series.reindex(Xfull.index).fillna(0.5).values
                    preds_test0[key] = te_series_k.reindex(Xmi_t.index).fillna(0.5).values
                    print(f"      AUC: {res['auc']:.4f}")
                else:
                    print(f"      ⚠️ Sin resultados")
            except Exception as e:
                print(f"      ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
    elif not use_pint:
        print("   ⚠️ PINT-Seq deshabilitado por parámetro (use_pint=False)")
    
    # CatBoost con ventanas múltiples
    if use_pint:
        print("   📈 CatBoost multi-ventana...")
        for w in WINDOWS_CB:
            for mode in MODES_CB:
                key = f"cb_{w}_{mode}"
                print(f"   📈 CB W={w} mode={mode}...")
                res = run_cb_window(tr_series, tr_tb, te_series, te_tb, y_train, w, mode)
                if res is not None:
                    preds_train0[key] = pd.Series(res["oof"], index=res["tr_ids"]).reindex(Xfull.index).fillna(0.5).values
                    preds_test0[key] = pd.Series(res["pte"], index=res["te_ids"]).reindex(Xmi_t.index).fillna(0.5).values
                    print(f"      AUC: {res['auc']:.4f}")
    else:
        print("   ⚠️ CatBoost multi-ventana deshabilitado (use_pint=False)")
    
    # ===== 4. BLENDING =====
    print("\n🎯 FASE 4: Blending...")
    
    # Limitar modelos al conjunto permitido (por defecto, el de la implementación original)
    if allowed_models is not None:
        allowed_set = set(allowed_models)
        preds_train0 = {k: v for k, v in preds_train0.items() if k in allowed_set}
        preds_test0  = {k: v for k, v in preds_test0.items()  if k in allowed_set}
    
    # Rank-blend Dirichlet
    auc_rb, w0 = optimize_rank_blend_dirichlet(preds_train0, y_vec)
    print(f"   📊 Rank-Blend (Dirichlet): AUC={auc_rb:.4f}")
    
    # SLSQP refine (si disponible)
    if HAS_SLSQP:
        keys = list(w0.keys())
        # Rank normalizado manualmente
        def rank01_arr(arr):
            ranks = np.argsort(np.argsort(arr))
            return ranks / (len(ranks) - 1 + 1e-12)
        
        R = np.column_stack([rank01_arr(preds_train0[k]) for k in keys])
        yv = np.asarray(y_vec, int)
        
        def loss(w):
            w = np.clip(w, 0, 1)
            if w.sum() <= 0: return 1.0
            w = w / w.sum()
            s = (R @ w.reshape(-1,1)).ravel()
            return 1.0 - roc_auc_score(yv, s)
        
        cons = [{'type':'eq','fun':lambda w: np.sum(np.clip(w,0,1)) - 1.0}]
        bnds = [(0.0,1.0)] * len(keys)
        w0_arr = np.array([w0.get(k, 1.0/len(keys)) for k in keys], float)
        
        try:
            res = minimize(loss, w0_arr, method='SLSQP', bounds=bnds, constraints=cons, 
                          options={'maxiter':200, 'ftol':1e-9, 'disp':False})
            if res.success:
                w_slsqp = np.clip(res.x, 0, 1); w_slsqp = w_slsqp / w_slsqp.sum()
                w_dict = {k: float(wi) for k,wi in zip(keys, w_slsqp)}
                auc_slsqp = 1.0 - res.fun
                if auc_slsqp > auc_rb:
                    w_final = w_dict
                    auc_final = auc_slsqp
                    print(f"   ✅ SLSQP mejoró: {auc_final:.4f}")
                else:
                    w_final = w0
                    auc_final = auc_rb
            else:
                w_final = w0
                auc_final = auc_rb
        except:
            w_final = w0
            auc_final = auc_rb
    else:
        w_final = w0
        auc_final = auc_rb
    
    feature_importance = None
    oof_meta = te_meta = None
    auc_meta = 0.0
    if use_meta_in_mix:
        print("   📊 Meta-stacker LR (optimizado)...")
        meta_keys = list(preds_train0.keys())
        oof_meta_arr = np.column_stack([preds_train0[k] for k in meta_keys])
        test_meta_arr = np.column_stack([preds_test0[k] for k in meta_keys])
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
        best_auc = 0
        best_oof = None
        best_te = None
        configs = [
            {'solver': 'lbfgs', 'C': 1.0, 'max_iter': 2000},
            {'solver': 'lbfgs', 'C': 0.5, 'max_iter': 2000},
        ]
        for config in configs:
            oof_temp = np.zeros(len(y_vec), float)
            te_folds = []
            for tr, va in skf.split(oof_meta_arr, y_vec):
                lr = LogisticRegression(random_state=42, class_weight='balanced', **config)
                lr.fit(oof_meta_arr[tr], y_vec[tr])
                oof_temp[va] = lr.predict_proba(oof_meta_arr[va])[:, 1]
                te_folds.append(lr.predict_proba(test_meta_arr)[:, 1])
            auc_temp = roc_auc_score(y_vec, oof_temp)
            if auc_temp > best_auc:
                best_auc = auc_temp
                best_oof = oof_temp
                best_te = np.mean(np.vstack(te_folds), axis=0)
                print(f"      Config {config}: AUC={auc_temp:.4f}")
        oof_meta, te_meta, auc_meta = best_oof, best_te, best_auc
        oof_meta_norm = (oof_meta_arr - oof_meta_arr.mean(axis=0)) / (oof_meta_arr.std(axis=0) + 1e-9)
        lr = LogisticRegression(random_state=42, class_weight='balanced')
        lr.fit(oof_meta_norm, y_vec)
        feature_importance = dict(zip(meta_keys, np.abs(lr.coef_[0])))
        feature_importance_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\n   📊 Feature Importance (Meta-Stacker):")
        for model, importance in feature_importance_sorted[:10]:
            print(f"      {model}: {importance:.6f}")
        print(f"      AUC: {auc_meta:.4f}")
    
    # Final blend: Rank-blend + Meta-stacker
    keys_rb = list(w_final.keys())
    Wf = np.array([w_final[k] for k in keys_rb], float); Wf = Wf / Wf.sum()
    
    # Rank normalizado manualmente
    def rank01(arr):
        ranks = np.argsort(np.argsort(arr))
        return ranks / (len(ranks) - 1 + 1e-12)
    
    Rtr_rb = np.column_stack([rank01(preds_train0[k]) for k in keys_rb])
    Rte_rb = np.column_stack([rank01(preds_test0[k]) for k in keys_rb])
    # Multiplicar pesos (filas) por rankings (columnas)
    s_tr_rb = (Rtr_rb @ Wf.reshape(-1,1)).ravel()
    s_te_rb = (Rte_rb @ Wf.reshape(-1,1)).ravel()
    
    if use_meta_in_mix and (oof_meta is not None) and (te_meta is not None):
        Rtr_all = np.column_stack([s_tr_rb, rank01(oof_meta)])
        Rte_all = np.column_stack([s_te_rb, rank01(te_meta)])
        if alpha_fixed is not None:
            best_alpha = float(alpha_fixed)
            best_auc_mix = roc_auc_score(y_vec, best_alpha * Rtr_all[:,0] + (1-best_alpha) * Rtr_all[:,1])
        else:
            best_auc_mix = -1.0; best_alpha = 0.5
            for a in np.linspace(0.0, 1.0, 11):
                s_mix = a * Rtr_all[:, 0] + (1 - a) * Rtr_all[:, 1]
                auc_mix = roc_auc_score(y_vec, s_mix)
                if auc_mix > best_auc_mix:
                    best_auc_mix = auc_mix; best_alpha = float(a)
        print(f"   ✅ Blend final (alpha={best_alpha:.2f}): {best_auc_mix:.4f}")
        s_test = best_alpha * Rte_all[:, 0] + (1 - best_alpha) * Rte_all[:, 1]
        auc_final = best_auc_mix
    else:
        if alpha_fixed is None:
            alpha_fixed = 1.0
        print(f"   ✅ Blend final solo Rank-Blend (alpha={float(alpha_fixed):.2f})")
        s_test = s_te_rb
        auc_final = roc_auc_score(y_vec, s_tr_rb)
    preds_test_df = pd.Series(s_test, index=Xmi_t.index, name="break_score").sort_index()
    preds_test_df.to_csv(SUBMISSION_NAME, header=True)
    print(f"\n💾 Guardado: {SUBMISSION_NAME}")
    
    # TEST AUC si disponible
    test_auc = None
    if y_test is not None:
        y_te = to_y_series(y_test, preds_test_df.index).values
        test_auc = roc_auc_score(y_te, preds_test_df.values)
        print(f"📊 TEST AUC: {test_auc:.6f}")

    # ===== Rama paralela: SOLO modelos que NO usan global_cv_var =====
    # Candidatos: 'dist', 'xgb_raw' (Impl3), 'pint_seq_*', 'cb_*' (multi-ventana)
    keys_no_gcv = [k for k in preds_train0.keys() if (k == 'dist' or k == 'xgb_raw' or k.startswith('pint_seq_') or k.startswith('cb_'))]
    oof_meta_mix_nogcv = None; test_auc_nogcv = None
    if len(keys_no_gcv) >= 2:
        print("\n🎯 Blend paralelo (sin global_cv_var): usando modelos ", keys_no_gcv)
        # Rank-blend en subset
        preds_tr_ng = {k: preds_train0[k] for k in keys_no_gcv}
        auc_rb_ng, w0_ng = optimize_rank_blend_dirichlet(preds_tr_ng, y_vec)

        # Meta-stacker (optimizado) en subset
        oof_meta_arr2 = np.column_stack([preds_train0[k] for k in keys_no_gcv])
        test_meta_arr2 = np.column_stack([preds_test0[k] for k in keys_no_gcv])
        skf2 = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
        best_auc2 = 0.0; best_oof2 = None; best_te2 = None
        for config in [
            {'solver': 'lbfgs', 'C': 1.0, 'max_iter': 2000},
            {'solver': 'lbfgs', 'C': 5.0, 'max_iter': 2000},
            {'solver': 'liblinear', 'C': 1.0, 'max_iter': 2000},
        ]:
            oof_tmp = np.zeros(len(y_vec), float); te_f = []
            for tr, va in skf2.split(oof_meta_arr2, y_vec):
                lr2 = LogisticRegression(random_state=42, class_weight='balanced', **config)
                lr2.fit(oof_meta_arr2[tr], y_vec[tr])
                oof_tmp[va] = lr2.predict_proba(oof_meta_arr2[va])[:,1]
                te_f.append(lr2.predict_proba(test_meta_arr2)[:,1])
            auc_tmp = roc_auc_score(y_vec, oof_tmp)
            if auc_tmp > best_auc2:
                best_auc2 = auc_tmp; best_oof2 = oof_tmp; best_te2 = np.mean(np.vstack(te_f), axis=0)

        # Combinar por alpha (igual que rama principal)
        # Rank-blend para subset
        def rank01_(arr):
            r = np.argsort(np.argsort(arr)); return r / (len(r) - 1 + 1e-12)
        keys_rb2 = list(w0_ng.keys())
        Wf2 = np.array([w0_ng[k] for k in keys_rb2], float); Wf2 = Wf2 / Wf2.sum()
        Rtr_rb2 = np.column_stack([rank01_(preds_train0[k]) for k in keys_rb2])
        Rte_rb2 = np.column_stack([rank01_(preds_test0[k]) for k in keys_rb2])
        s_tr_rb2 = (Rtr_rb2 @ Wf2.reshape(-1,1)).ravel()
        s_te_rb2 = (Rte_rb2 @ Wf2.reshape(-1,1)).ravel()
        Rtr_all2 = np.column_stack([s_tr_rb2, rank01_(best_oof2)])
        Rte_all2 = np.column_stack([s_te_rb2, rank01_(best_te2)])
        best_auc_mix2 = -1.0; best_alpha2 = 0.5
        for a in np.linspace(0.0, 1.0, 11):
            s_mix2 = a * Rtr_all2[:,0] + (1-a) * Rtr_all2[:,1]
            auc_mix2 = roc_auc_score(y_vec, s_mix2)
            if auc_mix2 > best_auc_mix2:
                best_auc_mix2 = auc_mix2; best_alpha2 = float(a)
        print(f"   ✅ Blend final (sin gcv, alpha={best_alpha2:.2f}): {best_auc_mix2:.4f}")
        oof_meta_mix_nogcv = float(best_auc_mix2)

        # Importancias de la rama sin gcv (coeficientes LR en subset)
        oof_meta_norm2 = (oof_meta_arr2 - oof_meta_arr2.mean(axis=0)) / (oof_meta_arr2.std(axis=0) + 1e-9)
        lr_ng = LogisticRegression(random_state=42, class_weight='balanced', solver='lbfgs', max_iter=2000)
        lr_ng.fit(oof_meta_norm2, y_vec)
        feature_importance_no_gcv = {k: float(abs(c)) for k, c in zip(keys_no_gcv, lr_ng.coef_[0])}
        print("\n   📊 Feature Importance (Meta-Stacker sin gcv):")
        for model, imp in sorted(feature_importance_no_gcv.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"      {model}: {imp:.6f}")

        # Predicciones test paralelas
        s_test2 = best_alpha2 * Rte_all2[:,0] + (1-best_alpha2) * Rte_all2[:,1]
        preds_test_df2 = pd.Series(s_test2, index=Xmi_t.index, name="break_score").sort_index()
        sub2 = SUBMISSION_NAME[:-4] + "_nogcv.csv" if SUBMISSION_NAME.endswith('.csv') else SUBMISSION_NAME + "_nogcv.csv"
        preds_test_df2.to_csv(sub2, header=True)
        print(f"💾 Guardado (sin gcv): {sub2}")
        if y_test is not None:
            y_te2 = to_y_series(y_test, preds_test_df2.index).values
            test_auc_nogcv = roc_auc_score(y_te2, preds_test_df2.values)
            print(f"📊 TEST AUC (sin gcv): {test_auc_nogcv:.6f}")
    
    # Summary
    summary = dict(
        keep_cols=keep_mi,
        removed_by_shift=drop_cols,
        oof={k: float(roc_auc_score(y_vec, v)) for k, v in preds_train0.items()},
        oof_blend=float(auc_final),
        oof_meta=float(auc_meta),
        oof_meta_mix=float(best_auc_mix) if 'best_auc_mix' in locals() else float(auc_final),
        weights=w_final,
        feature_importance=feature_importance,
        test_auc=float(test_auc) if test_auc is not None else None,
        oof_meta_mix_no_gcv=float(oof_meta_mix_nogcv) if oof_meta_mix_nogcv is not None else None,
        test_auc_no_gcv=float(test_auc_nogcv) if test_auc_nogcv is not None else None,
        feature_importance_no_gcv=feature_importance_no_gcv if 'feature_importance_no_gcv' in locals() else None
    )
    
    return summary


if __name__ == "__main__":
    # Para uso directo
    try:
        summary = run_all_combined(X_train, y_train, X_test, y_test)
        print("\n=== SUMMARY ===")
        print(f"Ensemble AUC: {summary['oof_meta_mix']:.4f}")
        print(f"\n📊 Top 10 Modelos por Importancia:")
        for model, importance in sorted(summary['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {model}: {importance:.6f}")
        print(f"\n📊 TEST AUC: {summary['test_auc']:.4f if summary['test_auc'] else 'N/A'}")
    except NameError:
        print("⚠️ X_train, y_train, X_test, y_test no están en memoria")
        print("   Usa este script desde tu notebook donde tengas los datos cargados")
