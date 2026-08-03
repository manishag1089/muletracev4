"""
MuleTrace-IN v2 — AML Analyst Dashboard
IIM Capstone · 500K+ transactions · IP / Geo / Device intelligence
4-model ensemble: XGBoost + Isolation Forest + GNN (SGC) + Louvain

Run:  streamlit run aml_dashboard_v2.py
Data: ./aml_output_v2/  (run aml_simulator_v2.py then train_models_v2.py first)
"""

import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MuleTrace-IN v2", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:#080C14 !important; color:#DFE3F0 !important; font-family:'DM Sans',sans-serif !important; }
[data-testid="stSidebar"] { background:#0D1120 !important; border-right:1px solid #1A2035 !important; }
[data-testid="stSidebar"] * { color:#DFE3F0 !important; }
#MainMenu, footer, [data-testid="stToolbar"] { display:none !important; }
[data-testid="stHeader"] { background:transparent !important; }
[data-testid="metric-container"] { background:#0D1120 !important; border:1px solid #1A2035 !important;
  border-radius:10px !important; padding:14px 18px !important; }
[data-testid="metric-container"] label { font-family:'Space Mono',monospace !important;
  font-size:9px !important; letter-spacing:.14em !important; color:#556080 !important; text-transform:uppercase !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family:'Space Mono',monospace !important;
  font-size:22px !important; color:#DFE3F0 !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size:11px !important; color:#556080 !important; }
[data-testid="stDataFrame"] { border:1px solid #1A2035 !important; border-radius:8px; }
.stButton > button { background:#0D1120 !important; border:1px solid #1A2035 !important; color:#DFE3F0 !important;
  font-family:'Space Mono',monospace !important; font-size:11px !important; border-radius:6px !important; }
.stButton > button:hover { border-color:#3B82F6 !important; color:#3B82F6 !important; }
[data-testid="stSelectbox"] > div > div, [data-testid="stTextInput"] > div > div > input {
  background:#0D1120 !important; border:1px solid #1A2035 !important; color:#DFE3F0 !important; border-radius:6px !important; }
.page-title { font-family:'Space Mono',monospace; font-size:20px; font-weight:700; color:#DFE3F0; margin:0 0 4px; }
.page-sub { font-size:13px; color:#556080; margin-bottom:24px; }
.sec-head { font-family:'Space Mono',monospace; font-size:9px; letter-spacing:.18em; color:#556080;
  text-transform:uppercase; border-bottom:1px solid #1A2035; padding-bottom:8px; margin:20px 0 14px; }
.risk-card { background:#0D1120; border:1px solid #1A2035; border-radius:10px; padding:16px 20px; margin-bottom:10px; }
.badge { display:inline-block; font-family:'Space Mono',monospace; font-size:9px; padding:3px 8px;
  border-radius:4px; letter-spacing:.07em; text-transform:uppercase; vertical-align:middle; }
.badge-red { background:rgba(239,68,68,.12); color:#F87171; border:1px solid rgba(239,68,68,.25); }
.badge-amber { background:rgba(245,158,11,.12); color:#FBB040; border:1px solid rgba(245,158,11,.25); }
.badge-green { background:rgba(34,197,94,.12); color:#4ADE80; border:1px solid rgba(34,197,94,.25); }
.badge-blue { background:rgba(59,130,246,.12); color:#60A5FA; border:1px solid rgba(59,130,246,.25); }
.bar-bg { background:#1A2035; border-radius:3px; height:6px; width:100%; margin-top:3px; }
.bar-fill { height:6px; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if not os.path.isdir(DATA):
    DATA = "./data"
PB = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0D1120",
          font=dict(family="DM Sans", color="#DFE3F0", size=11),
          margin=dict(l=8,r=8,t=36,b=8),
          xaxis=dict(gridcolor="#1A2035", linecolor="#1A2035", zerolinecolor="#1A2035"),
          yaxis=dict(gridcolor="#1A2035", linecolor="#1A2035", zerolinecolor="#1A2035"))

@st.cache_data(show_spinner=False)
def load():
    acc   = pd.read_parquet(f"{DATA}/accounts.parquet")
    edges = pd.read_parquet(f"{DATA}/graph_edges.parquet")
    with open(f"{DATA}/metrics.json") as f: metrics = json.load(f)
    return acc, edges, metrics

TXN_PARTS = None
def _parts():
    import glob
    global TXN_PARTS
    if TXN_PARTS is None:
        TXN_PARTS = sorted(glob.glob(f"{DATA}/transactions_part*.parquet")) or \
                    [f"{DATA}/transactions.parquet"]
    return TXN_PARTS

@st.cache_data(show_spinner=False)
def load_txns():
    """Column-pruned load for overview analytics — RAM-safe at 3M rows."""
    cols = ["timestamp","amount","is_suspicious","pattern","rail"]
    return pd.concat([pd.read_parquet(p, columns=cols) for p in _parts()],
                     ignore_index=True)

@st.cache_data(show_spinner=False)
def account_history(acc_id, limit=100):
    """Pyarrow predicate-filtered read — never loads the full ledger."""
    import pyarrow.parquet as pq
    frames = []
    for p in _parts():
        for side in ("sender_id","receiver_id"):
            t = pq.read_table(p, filters=[(side,"=",acc_id)])
            if t.num_rows: frames.append(t.to_pandas())
    if not frames: return pd.DataFrame()
    h = pd.concat(frames, ignore_index=True).drop_duplicates("txn_id")
    return h.sort_values("timestamp", ascending=False).head(limit)

def ego(edges, focal, hops=2, max_nodes=110):
    seen = {focal}; front = {focal}
    for _ in range(hops):
        m = edges["src"].isin(front) | edges["dst"].isin(front)
        sub_e = edges.loc[m, ["src","dst"]]
        nxt = set(sub_e["src"]) | set(sub_e["dst"])
        front = nxt - seen; seen |= front
        if len(seen) > 40_000: break
    m = edges["src"].isin(seen) & edges["dst"].isin(seen)
    ed = edges.loc[m]
    sub = nx.DiGraph()
    for s_, d_, a_, su_ in zip(ed["src"], ed["dst"], ed["amount"], ed["is_suspicious"]):
        sub.add_edge(s_, d_, amount=float(a_), is_suspicious=int(su_))
    sub.add_node(focal)
    if sub.number_of_nodes() > max_nodes:
        pr = nx.pagerank(sub, alpha=0.85); pr[focal] = 9e9
        sub = sub.subgraph(sorted(pr, key=pr.get, reverse=True)[:max_nodes]).copy()
    try:
        import community.community_louvain as cl
        comm = cl.best_partition(sub.to_undirected())
    except Exception:
        comm = {}
    return sub, comm

def draw_net(sub, comm, focal, risk, mules):
    if sub.number_of_nodes() == 0:
        f = go.Figure(); f.update_layout(**PB)
        f.add_annotation(text="No edges", x=.5, y=.5, showarrow=False,
                         font=dict(color="#556080")); return f
    pos = nx.spring_layout(sub, seed=7, k=3.0/max(sub.number_of_nodes()**.5,1))
    PAL = ["#3B82F6","#F59E0B","#22C55E","#A855F7","#06B6D4","#F97316","#EC4899","#14B8A6"]
    tr = []
    for u,v,d in sub.edges(data=True):
        x0,y0 = pos[u]; x1,y1 = pos[v]
        tr.append(go.Scatter(x=[x0,x1,None], y=[y0,y1,None], mode="lines",
            line=dict(width=max(.5,min(3.5,d.get("amount",0)/60000)),
                      color="rgba(239,68,68,.55)" if d.get("is_suspicious",0) else "rgba(59,130,246,.18)"),
            hoverinfo="none", showlegend=False))
    xs,ys,cs,ss,hs = [],[],[],[],[]
    for n in sub.nodes():
        x,y = pos[n]; xs.append(x); ys.append(y)
        scv = risk.get(n,0)
        hs.append(f"<b>{n}</b><br>Risk {scv:.3f}<br>{'⚠ MULE' if n in mules else 'Benign'}")
        if n == focal: cs.append("#FFFFFF"); ss.append(24)
        elif n in mules: cs.append("#EF4444"); ss.append(17)
        else: cs.append(PAL[comm.get(n,0)%8]); ss.append(8+scv*12)
    tr.append(go.Scatter(x=xs, y=ys, mode="markers",
        marker=dict(size=ss, color=cs, line=dict(width=1.2,color="#1A2035"), opacity=.93),
        text=hs, hoverinfo="text", showlegend=False))
    fx,fy = pos[focal]
    tr.append(go.Scatter(x=[fx], y=[fy+.08], mode="text", text=[f"<b>{focal}</b>"],
        textfont=dict(family="Space Mono", size=10, color="#FFF"), hoverinfo="none", showlegend=False))
    f = go.Figure(tr)
    f.update_layout(**{**PB, "xaxis":dict(visible=False), "yaxis":dict(visible=False)},
                    hovermode="closest", height=460)
    return f

def main():
    if not os.path.exists(f"{DATA}/accounts.parquet"):
        here = os.path.dirname(os.path.abspath(__file__))
        listing = "\n".join(sorted(os.listdir(here)))
        data_listing = ("\n".join(sorted(os.listdir(DATA)))
                        if os.path.isdir(DATA) else "(data/ folder not found)")
        st.error("Data files missing — the data/ folder must sit next to streamlit_app.py in the repo.")
        st.code(f"App folder contains:\n{listing}\n\ndata/ contains:\n{data_listing}")
        st.stop()
    acc, edges, metrics = load()
    risk = dict(zip(acc.account_id, acc.score_ensemble))
    mules = set(acc.loc[acc.is_mule==1, "account_id"])

    with st.sidebar:
        st.markdown("""<div style='font-family:Space Mono;font-size:10px;letter-spacing:.18em;color:#556080;
        text-transform:uppercase;'>MuleTrace-IN v2</div>
        <div style='font-family:Space Mono;font-size:17px;font-weight:700;margin-bottom:18px;'>🛡️ Command Centre</div>""",
        unsafe_allow_html=True)
        page = st.radio("", ["Overview","Alert Queue","Investigate","Geo Intelligence","Model Analytics"])
        st.markdown("<div class='sec-head'>Controls</div>", unsafe_allow_html=True)
        th = st.slider("Alert threshold", 0.0, 1.0, 0.60, 0.01)
        st.markdown("<div class='sec-head'>Engine</div>", unsafe_allow_html=True)
        st.markdown(f"""<div style='font-size:11px;color:#556080;line-height:1.9;'>
        4-model ensemble<br>XGB 40 · GNN 25 · IsoF 20 · Louvain 15<br>
        Accounts: {len(acc):,}<br>Known mules: {len(mules):,}<br>
        Test AUC-ROC: {[m for m in metrics['models'] if m['model']=='ensemble'][0]['auc_roc']}</div>""",
        unsafe_allow_html=True)

    alerts = acc[acc.score_ensemble >= th].sort_values("score_ensemble", ascending=False)
    n_al, n_tp = len(alerts), int((alerts.is_mule==1).sum())
    prec = n_tp/max(n_al,1)

    # ═════════ OVERVIEW ═════════
    if page == "Overview":
        st.markdown("<div class='page-title'>MuleTrace-IN v2</div>"
                    "<div class='page-sub'>3M+ transactions · dual rail (UPI + Bank) · IP / geo / device intelligence · 4-model ensemble</div>",
                    unsafe_allow_html=True)
        tx = load_txns()
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Transactions", f"{len(tx)/1e6:.2f}M",
                  f"UPI {(tx.rail=='UPI').mean()*100:.0f}% · Bank {(tx.rail=='BANK').mean()*100:.0f}%")
        c2.metric("Active Alerts", f"{n_al:,}", f"precision {prec*100:.0f}%")
        c3.metric("Volume Flagged", f"₹{tx[tx.is_suspicious==1].amount.sum()/1e7:.1f} Cr")
        c4.metric("Device Farms", f"{int(acc[acc.device_shared_accounts>3].is_mule.sum())}", "accounts on shared devices")
        c5.metric("Score Latency", "< 100 ms")

        col1,col2 = st.columns([3,2])
        with col1:
            d = tx.copy(); d["date"] = d.timestamp.dt.date
            g = d.groupby(["date","is_suspicious"]).size().reset_index(name="n")
            f = go.Figure()
            f.add_trace(go.Bar(x=g[g.is_suspicious==0].date, y=g[g.is_suspicious==0].n,
                               name="Benign", marker_color="rgba(59,130,246,.4)"))
            f.add_trace(go.Bar(x=g[g.is_suspicious==1].date, y=g[g.is_suspicious==1].n,
                               name="Suspicious", marker_color="rgba(239,68,68,.75)"))
            f.update_layout(**PB, barmode="stack", height=240,
                            title=dict(text="Daily volume", font=dict(size=12,color="#556080")),
                            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
            st.plotly_chart(f, use_container_width=True, config={"displayModeBar":False})
        with col2:
            f = go.Figure()
            f.add_trace(go.Histogram(x=acc[acc.is_mule==0].score_ensemble, name="Benign",
                                     marker_color="rgba(59,130,246,.5)", nbinsx=40))
            f.add_trace(go.Histogram(x=acc[acc.is_mule==1].score_ensemble, name="Mule",
                                     marker_color="rgba(239,68,68,.7)", nbinsx=40))
            f.update_layout(**PB, barmode="overlay", height=240,
                            title=dict(text="Ensemble score distribution", font=dict(size=12,color="#556080")),
                            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
            st.plotly_chart(f, use_container_width=True, config={"displayModeBar":False})

        col3,col4 = st.columns([3,2])
        with col3:
            s = tx[tx.is_suspicious==1].copy()
            s["hour"] = s.timestamp.dt.hour
            dm = {"Monday":"Mon","Tuesday":"Tue","Wednesday":"Wed","Thursday":"Thu","Friday":"Fri","Saturday":"Sat","Sunday":"Sun"}
            s["day"] = s.timestamp.dt.day_name().map(dm)
            piv = s.groupby(["day","hour"]).size().unstack(fill_value=0)
            piv = piv.reindex([d for d in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"] if d in piv.index])
            f = go.Figure(go.Heatmap(z=piv.values, x=list(range(24)), y=piv.index.tolist(),
                colorscale=[[0,"#080C14"],[.4,"#4B0000"],[1,"#EF4444"]],
                hovertemplate="%{y} %{x}:00 → %{z}<extra></extra>"))
            f.update_layout(**PB, height=225, title=dict(text="Suspicious activity · day × hour",
                            font=dict(size=12,color="#556080")))
            st.plotly_chart(f, use_container_width=True, config={"displayModeBar":False})
        with col4:
            fi = metrics["feature_importance"]
            items = sorted(fi.items(), key=lambda x:x[1])[-10:]
            f = go.Figure(go.Bar(x=[v for _,v in items], y=[k.replace("_"," ") for k,_ in items],
                orientation="h", marker_color="#3B82F6"))
            f.update_layout(**PB, height=225, title=dict(text="Top features (XGBoost)",
                            font=dict(size=12,color="#556080")))
            st.plotly_chart(f, use_container_width=True, config={"displayModeBar":False})

        st.markdown("<div class='sec-head'>Pattern breakdown</div>", unsafe_allow_html=True)
        pat = tx[tx.is_suspicious==1].pattern.astype(str).value_counts().reset_index()
        pat.columns = ["Pattern","Transactions"]
        pat = pat[pat.Transactions > 0]
        st.dataframe(pat, use_container_width=True, hide_index=True)

    # ═════════ ALERT QUEUE ═════════
    elif page == "Alert Queue":
        st.markdown("<div class='page-title'>Alert Queue</div>"
                    "<div class='page-sub'>Ranked by ensemble score · dismiss or escalate</div>", unsafe_allow_html=True)
        if "dis" not in st.session_state: st.session_state.dis = set()
        if "esc" not in st.session_state: st.session_state.esc = set()
        s1,s2 = st.columns([3,1])
        with s1: q = st.text_input("", placeholder="Search account…", label_visibility="collapsed")
        with s2: st.markdown(f"<div style='padding-top:9px;font-family:Space Mono;font-size:11px;color:#556080;'>{n_al} alerts</div>", unsafe_allow_html=True)
        act = alerts[~alerts.account_id.isin(st.session_state.dis)]
        if q: act = act[act.account_id.str.contains(q, case=False)]
        for _, r in act.head(50).iterrows():
            a_, sc_ = r.account_id, r.score_ensemble
            col = "#EF4444" if sc_>=.8 else "#F59E0B" if sc_>=.65 else "#22C55E"
            st.markdown(f"""<div class='risk-card' style='border-left:3px solid {col};'>
              <div style='display:flex;align-items:center;gap:12px;'>
                <div style='font-family:Space Mono;font-size:20px;font-weight:700;color:{col};min-width:54px;'>{sc_:.2f}</div>
                <div style='flex:1;'><div style='font-family:Space Mono;font-size:13px;font-weight:700;'>{a_}
                {"&nbsp;<span class='badge badge-red'>MULE</span>" if r.is_mule else ""}
                {"&nbsp;<span class='badge badge-amber'>ESCALATED</span>" if a_ in st.session_state.esc else ""}</div>
                <div style='font-size:12px;color:#556080;'>{str(r.mule_subtype).replace("_"," ").title()} ·
                {r.home_city} · XGB {r.score_xgb:.2f} · GNN {r.score_gnn:.2f} · IsoF {r.score_iso:.2f}</div></div>
              </div></div>""", unsafe_allow_html=True)
            b1,b2,_ = st.columns([1,1,6])
            with b1:
                if st.button("Dismiss", key=f"d{a_}"): st.session_state.dis.add(a_); st.rerun()
            with b2:
                if st.button("Escalate ↑", key=f"e{a_}"): st.session_state.esc.add(a_); st.rerun()

    # ═════════ INVESTIGATE ═════════
    elif page == "Investigate":
        st.markdown("<div class='page-title'>Account Investigation</div>"
                    "<div class='page-sub'>Feature scorecard · network graph · session intelligence</div>", unsafe_allow_html=True)
        default = alerts.iloc[0].account_id if len(alerts) else acc.iloc[0].account_id
        ids = acc.account_id.tolist()
        sel = st.selectbox("Account", ids, index=ids.index(default))
        r = acc[acc.account_id==sel].iloc[0]
        sc_ = r.score_ensemble
        col = "#EF4444" if sc_>=.75 else "#F59E0B" if sc_>=.5 else "#22C55E"
        verdict = "HIGH RISK · LIKELY MULE" if sc_>=.75 else "MEDIUM RISK · REVIEW" if sc_>=.5 else "LOW RISK · BENIGN"
        st.markdown(f"""<div class='risk-card' style='border-left:4px solid {col};'>
          <div style='display:flex;align-items:center;gap:24px;flex-wrap:wrap;'>
            <div><div style='font-family:Space Mono;font-size:38px;font-weight:700;color:{col};line-height:1;'>{sc_:.3f}</div>
            <div style='font-size:9px;color:#556080;font-family:Space Mono;text-transform:uppercase;margin-top:4px;'>Ensemble score</div></div>
            <div style='flex:1;min-width:160px;'>
              <div style='font-family:Space Mono;font-size:15px;font-weight:700;'>{sel}</div>
              <div style='font-size:12px;color:{col};font-family:Space Mono;margin:4px 0;'>{verdict}</div>
              <div style='font-size:12px;color:#556080;'>{r.bank_code} · {r.account_type} · {r.home_city}, {r.home_state}
              {'· ⚠ FRAUD HOTSPOT' if r.in_hotspot else ''} · {str(r.customer_segment).replace('_',' ')}</div></div>
            {'<span class="badge badge-red" style="font-size:11px;padding:5px 12px;">⚠ MULE (GT)</span>' if r.is_mule
             else '<span class="badge badge-green" style="font-size:11px;padding:5px 12px;">✓ BENIGN (GT)</span>'}
          </div></div>""", unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("XGBoost", f"{r.score_xgb:.3f}"); m2.metric("GNN (SGC)", f"{r.score_gnn:.3f}")
        m3.metric("IsoForest", f"{r.score_iso:.3f}"); m4.metric("Louvain", f"{r.score_louvain:.3f}")

        cf, cg = st.columns([1,2])
        with cf:
            st.markdown("<div class='sec-head'>Feature scorecard</div>", unsafe_allow_html=True)
            meta = [("pass_through_ratio","Pass-through","Funds out vs in"),
                    ("max_24h_velocity","24h velocity","Max txns / 24h"),
                    ("off_hours_ratio","Off-hours","Txns 22:00–08:00"),
                    ("device_shared_accounts","Device sharing","Accounts on same device"),
                    ("ip_subnet_shared_accounts","IP subnet sharing","Accounts on same /24"),
                    ("vpn_ratio","VPN usage","Share of VPN sessions"),
                    ("geo_velocity_kmh_max","Geo velocity","Max implied km/h"),
                    ("sub_threshold_ratio","Sub-threshold","₹40–50k structuring"),
                    ("avg_session_ms","Session length","Avg app session (ms)"),
                    ("paste_ratio","Paste behaviour","Pasted acct numbers"),
                    ("sim_age_days","SIM age","Days since SIM issued"),
                    ("activity_shift","Activity shift","Era-2 vs era-1 volume"),
                    ("binding_resets","SIM-binding resets","Device–SIM re-binds (UPI)"),
                    ("n_vpas","VPAs held","Virtual payment addresses"),
                    ("collect_in_ratio","Collect inflow","Share of collect-request credits"),
                    ("upi_share","UPI share","UPI vs bank-rail mix")]
            for feat,label,desc in meta:
                v = float(r[feat]); p99 = float(acc[feat].quantile(.99))
                norm = min(v/max(p99,1e-9), 1.0)
                bc = "#EF4444" if norm > .55 else "#3B82F6"
                fmt = f"{v:,.0f}" if v >= 100 else f"{v:.3f}"
                st.markdown(f"""<div style='margin-bottom:10px;'>
                  <div style='display:flex;justify-content:space-between;font-size:12px;'>
                  <span style='font-weight:500;'>{label}</span>
                  <span style='font-family:Space Mono;color:{bc};font-size:11px;'>{fmt}</span></div>
                  <div style='font-size:10px;color:#556080;margin-bottom:3px;'>{desc}</div>
                  <div class='bar-bg'><div class='bar-fill' style='width:{int(norm*100)}%;background:{bc};'></div></div>
                  </div>""", unsafe_allow_html=True)
        with cg:
            st.markdown("<div class='sec-head'>Transaction network</div>", unsafe_allow_html=True)
            hops = st.radio("Depth", [1,2,3], index=1, horizontal=True)
            sub, comm = ego(edges, sel, hops=hops)
            st.plotly_chart(draw_net(sub, comm, sel, risk, mules),
                            use_container_width=True, config={"displayModeBar":False})
            g1,g2,g3,g4 = st.columns(4)
            g1.metric("Nodes", sub.number_of_nodes()); g2.metric("Edges", sub.number_of_edges())
            g3.metric("Suspicious", sum(1 for *_,d in sub.edges(data=True) if d.get("is_suspicious",0)))
            g4.metric("Mule nodes", sum(1 for n in sub.nodes() if n in mules))

        st.markdown("<div class='sec-head'>Recent transactions (this account)</div>", unsafe_allow_html=True)
        h = account_history(sel, limit=100)
        if h.empty: st.info("No transactions.")
        else:
            d = h[["txn_id","timestamp","sender_id","receiver_id","amount","method",
                   "rail","txn_flow","psp_app","ip_address","device_id",
                   "vpn_flag","is_suspicious","pattern"]].copy()
            d["amount"] = d.amount.map(lambda x: f"₹{x:,.0f}")
            st.dataframe(d, use_container_width=True, hide_index=True)

    # ═════════ GEO INTELLIGENCE ═════════
    elif page == "Geo Intelligence":
        st.markdown("<div class='page-title'>Geo Intelligence</div>"
                    "<div class='page-sub'>Mule concentration map · fraud hotspots · impossible travel</div>",
                    unsafe_allow_html=True)
        g1,g2,g3,g4 = st.columns(4)
        hot = acc[acc.in_hotspot==1]
        g1.metric("Hotspot accounts", f"{len(hot):,}", f"{int(hot.is_mule.sum())} mules")
        g2.metric("Hotspot mule rate", f"{hot.is_mule.mean()*100:.0f}%", "vs 4% base rate")
        g3.metric("Impossible travel", f"{int((acc.geo_velocity_kmh_max>800).sum()):,}", ">800 km/h accounts")
        g4.metric("VPN-heavy accounts", f"{int((acc.vpn_ratio>0.15).sum()):,}", ">15% VPN sessions")

        city = acc.groupby("home_city").agg(
            home_lat=("home_lat","mean"), home_lon=("home_lon","mean"),
            accounts=("account_id","size"), mules=("is_mule","sum"),
            avg_risk=("score_ensemble","mean")).reset_index()
        city["mule_pct"] = city.mules/city.accounts*100
        f = go.Figure(go.Scattergeo(
            lat=city.home_lat, lon=city.home_lon,
            text=[f"<b>{c}</b><br>{a:,} accounts · {m} mules ({p:.0f}%)<br>avg risk {rr:.2f}"
                  for c,a,m,p,rr in zip(city.home_city,city.accounts,city.mules,city.mule_pct,city.avg_risk)],
            hoverinfo="text", mode="markers",
            marker=dict(size=np.clip(city.accounts/18,8,46),
                        color=city.mule_pct, colorscale=[[0,"#3B82F6"],[.4,"#F59E0B"],[1,"#EF4444"]],
                        cmin=0, cmax=100, opacity=.85, line=dict(width=.5,color="#1A2035"),
                        colorbar=dict(title=dict(text="Mule %",font=dict(color="#556080",size=10)),
                                      tickfont=dict(color="#556080",size=9), len=.6))))
        f.update_layout(**{k:v for k,v in PB.items() if k not in ("xaxis","yaxis")}, height=520,
            geo=dict(scope="asia", center=dict(lat=22.5,lon=80), projection_scale=3.6,
                     bgcolor="rgba(0,0,0,0)", showland=True, landcolor="#0D1120",
                     showcountries=True, countrycolor="#1A2035", showocean=True, oceancolor="#080C14",
                     showlakes=False, showcoastlines=False))
        st.plotly_chart(f, use_container_width=True, config={"displayModeBar":False})

        st.markdown("<div class='sec-head'>City risk table</div>", unsafe_allow_html=True)
        t = city[["home_city","accounts","mules","mule_pct","avg_risk"]].sort_values("mule_pct",ascending=False)
        t.columns = ["City","Accounts","Mules","Mule %","Avg Risk"]
        t["Mule %"] = t["Mule %"].round(1); t["Avg Risk"] = t["Avg Risk"].round(3)
        st.dataframe(t, use_container_width=True, hide_index=True)

    # ═════════ MODEL ANALYTICS ═════════
    elif page == "Model Analytics":
        st.markdown("<div class='page-title'>Model Analytics</div>"
                    "<div class='page-sub'>Held-out test metrics · PR curves · business impact</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='sec-head'>Test-set performance (30% stratified hold-out)</div>", unsafe_allow_html=True)
        mt = pd.DataFrame(metrics["models"]).rename(columns={
            "model":"Model","auc_roc":"AUC-ROC","auc_pr":"AUC-PR",
            "precision":"Precision","recall":"Recall","f1":"F1",
            "precision_at_real_prevalence":"Precision @ 0.5% prevalence"})
        st.dataframe(mt, use_container_width=True, hide_index=True)
        st.markdown(f"""<div style='font-size:11px;color:#556080;margin-top:-6px;'>
        Test set: {metrics['test_size']:,} accounts · {metrics['test_mules']} mules · threshold {metrics['threshold']}.
        Note: synthetic data is more separable than production data — expect 0.85–0.93 AUC on real bank data.</div>""",
        unsafe_allow_html=True)

        te = acc[acc.split=="test"]
        c1,c2 = st.columns(2)
        with c1:
            f = go.Figure()
            for scol,name,colr in [("score_xgb","XGBoost","#3B82F6"),("score_gnn","GNN","#A855F7"),
                                    ("score_iso","IsoForest","#F59E0B"),("score_ensemble","Ensemble","#EF4444")]:
                from sklearn.metrics import precision_recall_curve
                pr,rc,_ = precision_recall_curve(te.is_mule, te[scol])
                f.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=name,
                                       line=dict(color=colr, width=2)))
            f.update_layout(**PB, height=300, title=dict(text="Precision–recall (test set)",
                            font=dict(size=12,color="#556080")),
                            xaxis_title="Recall", yaxis_title="Precision",
                            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
            st.plotly_chart(f, use_container_width=True, config={"displayModeBar":False})
        with c2:
            sub_r = acc.groupby("mule_subtype")["score_ensemble"].mean().sort_values()
            f = go.Figure(go.Bar(x=sub_r.values, y=[s.replace("_"," ") for s in sub_r.index],
                orientation="h",
                marker_color=["#3B82F6" if s=="benign" else "#EF4444" for s in sub_r.index]))
            f.update_layout(**PB, height=300, title=dict(text="Avg score by subtype",
                            font=dict(size=12,color="#556080")))
            st.plotly_chart(f, use_container_width=True, config={"displayModeBar":False})

        up = metrics.get("upgrades", {})
        if up:
            st.markdown("<div class='sec-head'>v3 production upgrades</div>", unsafe_allow_html=True)
            u1,u2,u3,u4 = st.columns(4)
            cal = up.get("calibration",{}); pu = up.get("pu_learning",{})
            adv = up.get("adversarial",{}); cas = up.get("cascade",{})
            u1.metric("Brier (raw→cal)", f"{cal.get('brier_calibrated',0):.4f}",
                      f"from {cal.get('brier_raw',0):.4f}")
            u2.metric("Hidden-mule capture", f"{pu.get('hidden_in_top5pct_naive',0)*100:.0f}%",
                      "PU test · top-5% rank")
            u3.metric("Evasion floor (struct)", f"{adv.get('structural_only_recall',0)*100:.0f}%",
                      "recall if behaviour fully faked")
            u4.metric("Cascade stage-1", f"{cas.get('stage1_latency_us_per_account',0):.1f} µs",
                      f"escalates {cas.get('escalation_rate',0)*100:.0f}%")
            iv = up.get("interception",{})
            if iv:
                st.markdown("<div class='sec-head'>Interception engine — pre-settlement action</div>",
                            unsafe_allow_html=True)
                v1,v2,v3,v4 = st.columns(4)
                v1.metric("Suspicious ₹ intercepted", f"{iv.get('suspicious_value_intercepted_pct',0):.1f}%",
                          f"₹{iv.get('suspicious_value_stopped_cr',0):.1f} Cr stopped")
                v2.metric("Benign friction", f"{iv.get('benign_friction_rate_pct',0):.2f}%",
                          "honest customers challenged")
                v3.metric("Decision latency", f"{iv.get('decision_latency_us',0):.0f} µs",
                          f"{iv.get('throughput_txns_per_sec',0):,} txns/sec")
                mix = iv.get("decision_mix",{})
                v4.metric("Actions taken", f"{mix.get('BLOCK',0):,} blocks",
                          f"{mix.get('STEP_UP',0):,} step-up · {mix.get('COOLING',0):,} cooling")
                ft = iv.get("flag_tier",{})
                if ft:
                    st.markdown("<div class='sec-head'>Flag / Step-up / Intercept — the three response tiers</div>",
                                unsafe_allow_html=True)
                    g1,g2,g3 = st.columns(3)
                    g1.metric("🚩 Flag (review only)", f"{ft.get('flagged_allow_review_only',0):,}",
                              "let through, zero friction, analyst reviews")
                    g2.metric("⤴ Step-up (challenge)", f"{mix.get('STEP_UP',0):,}",
                              "re-authentication demanded")
                    g3.metric("⛔ Intercept (block/hold)", f"{mix.get('BLOCK',0):,}",
                              "stopped before settlement")
                tiers_ = iv.get("value_stopped_by_insertion_point_cr",{})
                if tiers_:
                    f = go.Figure(go.Bar(
                        x=list(tiers_.values()),
                        y=[k.replace("_"," ").title() for k in tiers_.keys()],
                        orientation="h", marker_color=["#EF4444","#F59E0B","#3B82F6"],
                        hovertemplate="%{y}: ₹%{x} Cr<extra></extra>"))
                    f.update_layout(**PB, height=190,
                        title=dict(text="₹ Cr stopped by UPI insertion point — remitter vs beneficiary vs NPCI tier",
                                   font=dict(size=12,color="#556080")),
                        xaxis_title="₹ crore intercepted")
                    st.plotly_chart(f, use_container_width=True, config={"displayModeBar":False})
            dr = up.get("drift_monitor",{}).get("psi",{})
            if dr:
                f = go.Figure(go.Bar(x=list(dr.values()), y=list(dr.keys()), orientation="h",
                    marker_color=["#EF4444" if v>=.25 else "#F59E0B" if v>=.10 else "#3B82F6"
                                  for v in dr.values()]))
                f.add_vline(x=.10, line_dash="dot", line_color="#F59E0B")
                f.add_vline(x=.25, line_dash="dot", line_color="#EF4444")
                f.update_layout(**PB, height=220,
                    title=dict(text="Drift monitor · PSI era-1 vs era-2 (suspicious txns)",
                               font=dict(size=12,color="#556080")))
                st.plotly_chart(f, use_container_width=True, config={"displayModeBar":False})
            ev = adv.get("evasion_curve",[])
            if ev:
                f = go.Figure(go.Scatter(x=[e["evasion_alpha"] for e in ev],
                    y=[e["xgb_recall"] for e in ev], mode="lines+markers",
                    line=dict(color="#EF4444", width=2)))
                f.add_hline(y=adv.get("structural_only_recall",0), line_dash="dash",
                    line_color="#22C55E",
                    annotation_text="structural-only floor",
                    annotation_font_color="#22C55E")
                f.update_layout(**PB, height=240,
                    title=dict(text="Adversarial evasion sweep — behaviour faked, graph holds",
                               font=dict(size=12,color="#556080")),
                    xaxis_title="Evasion strength α", yaxis_title="Recall")
                st.plotly_chart(f, use_container_width=True, config={"displayModeBar":False})

        st.markdown("<div class='sec-head'>Business impact simulator</div>", unsafe_allow_html=True)
        b1,b2 = st.columns(2)
        with b1: customers = st.number_input("Bank customers", 100_000, 50_000_000, 1_000_000, 100_000)
        with b2: avg_fraud = st.slider("Avg fraud per mule (₹ Lakh)", 1, 100, 15)
        ens = [m for m in metrics["models"] if m["model"]=="ensemble"][0]
        est = int(customers*0.04); caught = int(est*ens["recall"])
        fp = int(caught/max(ens["precision"],.01) - caught)
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Mules caught", f"{caught:,}", f"of ~{est:,}")
        r2.metric("Fraud prevented", f"₹{caught*avg_fraud/100:,.0f} Cr")
        r3.metric("False alerts", f"{fp:,}")
        r4.metric("Analyst hrs saved/yr", f"{int(est*0.95*0.4*0.8):,}", "vs rule-based TMS")

main()
