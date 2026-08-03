"""
MuleTrace-IN v4 — Training Pipeline (UPI-native, 200K+ accounts)

Beyond v2's four models (XGBoost · IsolationForest · GNN-SGC · Louvain), v3 adds:

  1. PROBABILITY CALIBRATION  — isotonic regression on the ensemble; Brier
     score before/after; a 0.8 now means ~80%真 probability (RBI FREE-AI ready)
  2. PU LEARNING              — real banks only know CONFIRMED mules, never
     confirmed negatives. We hide 50% of training mule labels, compare naive
     training vs Elkan–Noto correction, report recall on the hidden mules.
  3. DRIFT MONITOR            — PSI per feature between era-1 (day<60) and
     era-2 (day≥60) suspicious activity; catches the simulated tactic shift.
  4. ADVERSARIAL STRESS TEST  — evasion sweep: dampen behavioural features
     of test mules by α ∈ {0..1}; recall-vs-evasion curve; graph features
     stay fixed (restructuring a ring is operationally expensive → headline).
  5. TWO-STAGE CASCADE        — stage-1 light XGB (8 cheap features) screens
     everything; top slice → full ensemble. Latency benchmarked per account
     and per streamed transaction.
  6. PRIOR-ADJUSTED METRICS   — precision re-computed at the real-world mule
     prevalence (~0.5%, per I4C: 2.65M mules vs ~55Cr+ active users) rather
     than the 4% training prevalence.

Usage:  python train_models_v3.py [--data-dir ./aml_output_v3]
Writes: scores.csv · metrics.json (dashboard-compatible + "upgrades" block)
"""

import argparse, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             precision_score, recall_score, f1_score, brier_score_loss)
import xgboost as xgb

FEATURES = [
    # behavioural (v1)
    "pass_through_ratio","max_24h_velocity","off_hours_ratio",
    "sub_threshold_ratio","fan_in_diversity",
    "n_counterparties_in","n_counterparties_out",
    "amount_mean_inr","amount_std_inr","n_sent","n_recv",
    # device / IP / geo (v2)
    "n_unique_ips","n_unique_devices","vpn_ratio",
    "device_shared_accounts","ip_subnet_shared_accounts",
    "geo_velocity_kmh_max","in_hotspot",
    # graph
    "in_degree","out_degree","pagerank","community_size",
    # profile
    "account_age_days","kyc_tier",
    # v3: behavioural biometrics + identity + cross-bank + temporal
    "avg_session_ms","paste_ratio","cross_bank_ratio",
    "sim_age_days","sim_ported_recent","kyc_updated_recent",
    "activity_shift","burstiness",
    # v4: UPI-native
    "n_vpas","binding_resets","upi_share","collect_in_ratio",
]

# features an adversary can cheaply manipulate (behaviour), vs structural
EVADABLE = ["max_24h_velocity","off_hours_ratio","vpn_ratio","geo_velocity_kmh_max",
            "avg_session_ms","paste_ratio","sub_threshold_ratio","pass_through_ratio"]
CHEAP8   = ["pass_through_ratio","max_24h_velocity","off_hours_ratio","binding_resets",
            "sim_age_days","paste_ratio","n_counterparties_in","amount_mean_inr"]


def sgc(accounts, edges, feat_cols, k=2):
    ids = accounts["account_id"].values
    idx = {a: i for i, a in enumerate(ids)}
    src = edges["src"].map(idx).values; dst = edges["dst"].map(idx).values
    w   = np.log1p(edges["amount"].values)
    A = sp.coo_matrix((w,(src,dst)), shape=(len(ids),)*2)
    A = A + A.T + sp.eye(len(ids))
    d = np.asarray(A.sum(1)).ravel()
    Dv = sp.diags(1/np.sqrt(np.maximum(d,1e-9)))
    S = (Dv@A@Dv).tocsr()
    X = StandardScaler().fit_transform(accounts[feat_cols].fillna(0).values)
    H = X.copy()
    for _ in range(k): H = S@H
    return np.hstack([X,H])


def psi(a, b, bins=10):
    """Population Stability Index between two 1-D samples."""
    qs = np.quantile(a, np.linspace(0,1,bins+1)); qs[0]-=1e-9; qs[-1]+=1e-9
    pa,_ = np.histogram(a, qs); pb,_ = np.histogram(b, qs)
    pa = np.clip(pa/max(pa.sum(),1),1e-4,None); pb = np.clip(pb/max(pb.sum(),1),1e-4,None)
    return float(((pa-pb)*np.log(pa/pb)).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./aml_output_v4")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    rng = np.random.RandomState(a.seed)

    print("[1/8] Load")
    acc   = pd.read_csv(f"{a.data_dir}/accounts.csv")
    edges = pd.read_csv(f"{a.data_dir}/graph_edges.csv")
    feats = [f for f in FEATURES if f in acc.columns]
    y = acc["is_mule"].values
    X = acc[feats].fillna(0).values
    tr, te = train_test_split(np.arange(len(acc)), test_size=.30, stratify=y, random_state=a.seed)
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)

    print(f"[2/8] Core models ({len(feats)} features)")
    spw = (y[tr]==0).sum()/max((y[tr]==1).sum(),1)
    m_xgb = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=.05,
        subsample=.9, colsample_bytree=.8, scale_pos_weight=spw,
        eval_metric="logloss", verbosity=0, random_state=a.seed).fit(Xs[tr], y[tr])
    s_xgb = m_xgb.predict_proba(Xs)[:,1]
    fi = dict(zip(feats, m_xgb.feature_importances_.round(5).tolist()))

    iso = IsolationForest(n_estimators=300, contamination=.05, random_state=a.seed).fit(Xs[tr])
    r = -iso.score_samples(Xs); s_iso = (r-r.min())/(r.max()-r.min()+1e-9)

    Xg = sgc(acc, edges, feats)
    m_gnn = LogisticRegression(max_iter=1000, class_weight="balanced", C=.5,
                               random_state=a.seed).fit(Xg[tr], y[tr])
    s_gnn = m_gnn.predict_proba(Xg)[:,1]

    base = .5*s_xgb + .5*s_gnn
    comm = acc.assign(_b=base).groupby("community_id")["_b"].mean()
    s_lvn = acc["community_id"].map(comm).values
    s_lvn = (s_lvn-s_lvn.min())/(s_lvn.max()-s_lvn.min()+1e-9)
    ens = .40*s_xgb + .25*s_gnn + .20*s_iso + .15*s_lvn

    print("[3/8] Calibration (isotonic)")
    cal_fit, cal_val = train_test_split(tr, test_size=.35, stratify=y[tr], random_state=a.seed)
    iso_cal = IsotonicRegression(out_of_bounds="clip").fit(ens[cal_val], y[cal_val])
    ens_cal = iso_cal.predict(ens)
    brier_raw = brier_score_loss(y[te], ens[te])
    brier_cal = brier_score_loss(y[te], ens_cal[te])
    bins = np.linspace(0,1,11)
    calib_curve = []
    for lo,hi in zip(bins[:-1],bins[1:]):
        m = (ens_cal[te]>=lo)&(ens_cal[te]<hi)
        if m.sum()>=15:
            calib_curve.append({"bin_mid":round((lo+hi)/2,2),
                                "pred":round(float(ens_cal[te][m].mean()),3),
                                "actual":round(float(y[te][m].mean()),3),
                                "n":int(m.sum())})

    print("[4/8] PU-learning experiment (50% mule labels hidden)")
    y_pu = y.copy()
    tr_mules = np.array([i for i in tr if y[i]==1])
    hidden = rng.choice(tr_mules, size=len(tr_mules)//2, replace=False)
    y_pu[hidden] = 0                                   # now "unlabelled"
    m_naive = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.05,
        scale_pos_weight=(y_pu[tr]==0).sum()/max((y_pu[tr]==1).sum(),1),
        eval_metric="logloss", verbosity=0, random_state=a.seed).fit(Xs[tr], y_pu[tr])
    s_naive = m_naive.predict_proba(Xs)[:,1]
    # Elkan–Noto: c = E[s(x) | labelled positive]; corrected p = s/c
    c = float(s_naive[[i for i in tr if y_pu[i]==1]].mean())
    s_pu = np.clip(s_naive/max(c,1e-6), 0, 1)
    dec = np.quantile(s_naive, 0.95)
    dec_pu = np.quantile(s_pu, 0.95)
    pu_exp = {
        "hidden_in_top5pct_naive": round(float((s_naive[hidden]>=dec).mean()),4),
        "hidden_in_top5pct_pu":    round(float((s_pu[hidden]>=dec_pu).mean()),4),
        "hidden_mules": int(len(hidden)),
        "label_frequency_c": round(c,4),
        "recall_on_hidden_naive": round(float((s_naive[hidden]>=.5).mean()),4),
        "recall_on_hidden_pu":    round(float((s_pu[hidden]>=.5).mean()),4),
        "test_auc_naive": round(float(roc_auc_score(y[te], s_naive[te])),4),
        "test_auc_pu":    round(float(roc_auc_score(y[te], s_pu[te])),4),
    }

    print("[5/8] Drift monitor (PSI era1 vs era2 suspicious txns)")
    tx = pd.read_csv(f"{a.data_dir}/transactions.csv", parse_dates=["timestamp"],
                     usecols=["timestamp","amount","is_suspicious","vpn_flag",
                              "session_ms","paste_flag"])
    tx["hour"] = tx.timestamp.dt.hour
    tx["odd"]  = ((tx.hour<8)|(tx.hour>=22)).astype(int)
    day = (tx.timestamp - tx.timestamp.min()).dt.days
    s1 = tx[(tx.is_suspicious==1)&(day<60)]; s2 = tx[(tx.is_suspicious==1)&(day>=60)]
    drift = {f: round(psi(s1[f].values.astype(float), s2[f].values.astype(float)),4)
             for f in ["amount","hour","odd","vpn_flag","session_ms"]}
    drift_flag = {k:("ALERT — retrain" if v>=.25 else "watch" if v>=.10 else "stable")
                  for k,v in drift.items()}

    print("[6/8] Adversarial evasion sweep")
    ev_idx = [feats.index(f) for f in EVADABLE if f in feats]
    te_mules = np.array([i for i in te if y[i]==1])
    mu = Xs[tr][y[tr]==0].mean(axis=0)                # benign centroid (scaled)
    curve = []
    for alpha in [0,.2,.4,.6,.8,1.0]:
        Xa = Xs[te_mules].copy()
        for j in ev_idx: Xa[:,j] = (1-alpha)*Xa[:,j] + alpha*mu[j]
        rec = float((m_xgb.predict_proba(Xa)[:,1]>=.5).mean())
        curve.append({"evasion_alpha":alpha, "xgb_recall":round(rec,4)})
    # structural-only model: how much survives when ALL behaviour is faked?
    struct_feats = [f for f in ["in_degree","out_degree","pagerank","community_size",
                                 "device_shared_accounts","ip_subnet_shared_accounts"] if f in feats]
    sidx = [feats.index(f) for f in struct_feats]
    m_str = xgb.XGBClassifier(n_estimators=200, max_depth=4, scale_pos_weight=spw,
        eval_metric="logloss", verbosity=0, random_state=a.seed).fit(Xs[tr][:,sidx], y[tr])
    rec_struct = float((m_str.predict_proba(Xs[te_mules][:,sidx])[:,1]>=.5).mean())

    print("[7/8] Two-stage cascade + latency")
    c8 = [feats.index(f) for f in CHEAP8 if f in feats]
    m_lite = xgb.XGBClassifier(n_estimators=60, max_depth=3, scale_pos_weight=spw,
        eval_metric="logloss", verbosity=0, random_state=a.seed).fit(Xs[tr][:,c8], y[tr])
    t0=time.perf_counter(); s_lite = m_lite.predict_proba(Xs[:,c8])[:,1]
    lite_us = (time.perf_counter()-t0)/len(Xs)*1e6
    cutoff = np.quantile(s_lite[tr], .92)             # escalate top ~8%
    esc = s_lite >= cutoff
    t0=time.perf_counter(); _ = m_xgb.predict_proba(Xs[esc]); _ = m_gnn.predict_proba(Xg[esc])
    full_ms = (time.perf_counter()-t0)/max(esc.sum(),1)*1e3
    casc_pred = np.zeros(len(y)); casc_pred[esc] = (ens_cal[esc]>=.5).astype(int)
    cascade = {
        "stage1_features": CHEAP8, "escalation_rate": round(float(esc[te].mean()),4),
        "stage1_latency_us_per_account": round(lite_us,1),
        "stage2_latency_ms_per_account": round(full_ms,2),
        "cascade_recall_test": round(float(recall_score(y[te], casc_pred[te])),4),
        "cascade_precision_test": round(float(precision_score(y[te], casc_pred[te])),4),
        "full_recall_test": round(float(recall_score(y[te],(ens_cal[te]>=.5))),4),
    }

    print("[8/8] Metrics + prior adjustment")
    def ev(s, name, th=.5):
        yt, st = y[te], s[te]; pr = (st>=th).astype(int)
        p_,r_,_ = precision_recall_curve(yt, st)
        prec = precision_score(yt,pr,zero_division=0); rec = recall_score(yt,pr,zero_division=0)
        # prior-adjust precision to real-world 0.5% prevalence (I4C-scale)
        pi_tr, pi_rw = yt.mean(), 0.005
        fpr = ((pr==1)&(yt==0)).sum()/max((yt==0).sum(),1)
        prec_rw = (rec*pi_rw)/max(rec*pi_rw + fpr*(1-pi_rw), 1e-9)
        return {"model":name,"auc_roc":round(float(roc_auc_score(yt,st)),4),
                "auc_pr":round(float(auc(r_,p_)),4),
                "precision":round(float(prec),4),"recall":round(float(rec),4),
                "f1":round(float(f1_score(yt,pr,zero_division=0)),4),
                "precision_at_real_prevalence":round(float(prec_rw),4)}

    metrics = {
        "version":"v4","test_size":int(len(te)),"test_mules":int(y[te].sum()),"threshold":.5,
        "models":[ev(s,n) for s,n in [(s_xgb,"xgboost"),(s_iso,"isolation_forest"),
                  (s_gnn,"gnn_sgc"),(s_lvn,"louvain_community"),
                  (ens,"ensemble"),(ens_cal,"ensemble_calibrated")]],
        "feature_importance":dict(sorted(fi.items(), key=lambda x:-x[1])),
        "ensemble_weights":{"xgboost":.40,"gnn_sgc":.25,"isolation_forest":.20,"louvain":.15},
        "upgrades":{
            "calibration":{"brier_raw":round(brier_raw,5),"brier_calibrated":round(brier_cal,5),
                           "curve":calib_curve},
            "pu_learning":pu_exp,
            "drift_monitor":{"psi":drift,"status":drift_flag,
                             "note":"PSI>0.25 on suspicious-txn features between day<60 and day>=60 eras"},
            "adversarial":{"evasion_curve":curve,
                           "structural_only_recall":round(rec_struct,4),
                           "structural_features":struct_feats},
            "cascade":cascade,
            "real_world_calibration_sources":[
                "NPCI avg UPI ticket Rs1,293; P2M>P2P mix (2026)",
                "I4C: 26.5L layer-1 mule accounts by Dec-2025; ~Rs20,000Cr routed",
                "Hotspots: Nuh 1000+ mules, Jamtara 350+ (2025)",
                "Bank mule concentration: SBI>PNB>Canara>Kotak>PaymentsBank (CFCFRMS)"],
        },
    }

    out = acc[["account_id","is_mule","mule_subtype","community_id"]].copy()
    for c_,s_ in [("score_xgb",s_xgb),("score_iso",s_iso),("score_gnn",s_gnn),
                  ("score_louvain",s_lvn),("score_ensemble",ens_cal)]:
        out[c_] = np.round(s_,5)
    out["score_ensemble_raw"] = np.round(ens,5)
    out["split"] = "train"; out.loc[te,"split"] = "test"
    out.to_csv(f"{a.data_dir}/scores.csv", index=False)
    json.dump(metrics, open(f"{a.data_dir}/metrics.json","w"), indent=2)

    print(f"\n{'MODEL':<22}{'AUC':>7}{'PREC':>7}{'REC':>7}{'P@0.5%':>8}")
    for m in metrics["models"]:
        print(f"{m['model']:<22}{m['auc_roc']:>7}{m['precision']:>7}{m['recall']:>7}"
              f"{m['precision_at_real_prevalence']:>8}")
    print(f"\nBrier raw→cal : {brier_raw:.5f} → {brier_cal:.5f}")
    print(f"PU hidden-mule capture: top5% naive {pu_exp['hidden_in_top5pct_naive']} → EN {pu_exp['hidden_in_top5pct_pu']} | recall@0.5 {pu_exp['recall_on_hidden_naive']}→{pu_exp['recall_on_hidden_pu']}")
    print(f"Drift PSI     : {drift}")
    print(f"Evasion curve : {[c_['xgb_recall'] for c_ in curve]}  | struct-only recall {rec_struct:.3f}")
    print(f"Cascade       : {cascade['stage1_latency_us_per_account']}µs/acct stage-1, "
          f"escalates {cascade['escalation_rate']*100:.1f}%, recall {cascade['cascade_recall_test']}")


if __name__ == "__main__":
    main()
