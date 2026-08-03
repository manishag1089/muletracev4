# MuleTrace-IN v4 — UPI-Native · 205,000 accounts · 3,000,711 transactions

## Deploy as a NEW Streamlit app (recommended)
1. New public GitHub repo (e.g. muletrace-v4)
2. Upload the 6 root files (streamlit_app.py, requirements.txt, the three
   .py pipeline files, interception_log_sample.csv) to the repo root
3. Create the data/ folder (Add file → Create new file → type `data/x.txt`),
   then upload ALL EIGHT data files into it — every file is < 25 MB:
   accounts.parquet · graph_edges.parquet · metrics.json ·
   transactions_part1..5.parquet
4. share.streamlit.io → Create app → repo/main/streamlit_app.py → Deploy

## What v4 adds over v3
- Dual rail: UPI 57.4% / Bank 42.6% of 3.0M transactions
- UPI layer: VPAs per account, PSP app (PhonePe/GPay/Paytm…), device–SIM
  binding resets, flows (push / qr_p2m / collect_request / autopay /
  circle_delegate), NPCI Rs2,000 collect cap
- 2 new fraud patterns: collect-request scam bursts, UPI Circle delegation abuse
- Engine: collect-decline rule, binding-reset step-up, and PER-INSERTION-POINT
  attribution (remitter bank ₹245.5 Cr · beneficiary bank ₹74.3 Cr · NPCI tier)
- Dashboard: rail-split KPI, 4 UPI scorecard rows, insertion-point chart;
  memory-safe loaders (45 MB pruned ledger, pyarrow-filtered history)

## Headline numbers (30% hold-out · 61,500 test accounts, 1,845 mules)
- Calibrated ensemble: precision 0.998 · recall 0.996 · P@0.5% prevalence 0.990
- Interception: 95.97% of suspicious value (₹319.8 Cr of ₹333.2 Cr) held
  pre-settlement · benign friction 1.70% · 14.15 µs · 70,674 txns/sec
- Adversarial structural floor: 91.8% recall under full behavioural mimicry
