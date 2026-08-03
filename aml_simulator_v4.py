"""
AML & Mule Account Detection — Synthetic Dataset Simulator v4 (UPI-NATIVE)
SCALE: 200,000+ accounts · ~2,000,000 transactions · dual rail (UPI + BANK)
NEW UPI LAYER:
  - VPAs per account (name@psp), PSP app by market share (PhonePe/GPay/Paytm/…)
  - Device–SIM binding resets (binding break = top-tier mule signal)
  - Transaction flows: push / qr_p2m / collect_request / autopay / circle_delegate
  - NPCI collect-request cap (Rs 2,000) enforced on collect flows
  - Two NEW fraud patterns: collect-request scam bursts, UPI Circle delegation abuse
ENGINEERING FOR SCALE: chunked txn accumulation (capped RAM), scipy power-iteration
PageRank, igraph multilevel communities (C-speed at 200K nodes)
(v3 base description follows)
Simulator v3 lineage
CALIBRATED against real published statistics (Jan–Jun 2026):
  - Avg UPI ticket ~Rs 1,293 (NPCI) → benign amount distribution recalibrated
  - P2M now exceeds P2P in UPI mix → purpose weights updated
  - Mule bank concentration (I4C/CFCFRMS): SBI > PNB > Canara > Kotak > PaymentsBank
  - Hotspots: Nuh 1000+ mules (2025), Jamtara 350+ (I4C)
NEW v3 SIGNALS:
  - Behavioural biometrics per txn: session_ms, paste_flag
  - Identity layer per account: sim_age_days, sim_ported_recent,
    kyc_updated_recent, sold_account signature (dormant→KYC update→burst)
  - Cross-bank ratio (counterparty bank diversity)
  - TEMPORAL DRIFT: after day 60, surviving mule farms ADAPT
    (slower windows, fewer odd hours) → enables drift-monitor demo
(v2 base description follows)
IIM Capstone Project · MuleTrace-IN

v2 upgrades over v1:
  - Scale: ~500,000 transactions (16,000 accounts, 90 days)
  - NEW signals per transaction: IP address, device ID, geolocation (lat/lon),
    VPN flag, session channel (mobile/web/USSD)
  - NEW mule behaviours: device farms (shared devices/IPs across mule accounts),
    impossible-travel geo velocity, fraud-hotspot districts, SIM churn
  - Vectorised feature engineering (v1 loop would take hours at this scale)
  - Graph features: PageRank, degrees + Louvain community assignment baked in

Outputs (./aml_output_v2/):
  transactions.csv   ~500K rows with ip, device_id, lat, lon, vpn, channel
  accounts.csv       16K rows · 30+ engineered features + is_mule label
  graph_edges.csv    edge list for GNN
  graph_nodes.csv    node features + community id + label
  eda_summary.txt    stats for the IIM deck

Usage:
  python aml_simulator_v2.py [--accounts 16000] [--days 90] [--mule-ratio 0.04] [--seed 42]
"""

import argparse, math, random
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

# ─────────────────────────────────────────────────────────────
#  GEOGRAPHY — Indian cities + known fraud-hotspot districts
# ─────────────────────────────────────────────────────────────
CITIES = {
    # name: (lat, lon, state, weight, is_hotspot)
    "Mumbai":      (19.076, 72.877, "MH", 10, 0),
    "Delhi":       (28.704, 77.102, "DL", 10, 0),
    "Bengaluru":   (12.972, 77.594, "KA", 9,  0),
    "Hyderabad":   (17.385, 78.487, "TG", 7,  0),
    "Chennai":     (13.083, 80.270, "TN", 7,  0),
    "Kolkata":     (22.573, 88.364, "WB", 7,  0),
    "Pune":        (18.520, 73.857, "MH", 6,  0),
    "Ahmedabad":   (23.023, 72.572, "GJ", 6,  0),
    "Jaipur":      (26.912, 75.787, "RJ", 5,  0),
    "Lucknow":     (26.847, 80.947, "UP", 5,  0),
    "Surat":       (21.170, 72.831, "GJ", 4,  0),
    "Kanpur":      (26.450, 80.332, "UP", 4,  0),
    "Nagpur":      (21.146, 79.088, "MH", 3,  0),
    "Indore":      (22.720, 75.858, "MP", 3,  0),
    "Patna":       (25.594, 85.138, "BR", 3,  0),
    "Bhopal":      (23.260, 77.413, "MP", 3,  0),
    "Kochi":       (9.932,  76.267, "KL", 3,  0),
    "Guwahati":    (26.145, 91.736, "AS", 2,  0),
    "Siliguri":    (26.727, 88.395, "WB", 2,  0),
    "Coimbatore":  (11.017, 76.956, "TN", 2,  0),
    # Fraud-hotspot districts (per NCRB / news reports on mule concentration)
    "Nuh":         (28.108, 77.001, "HR", 1,  1),
    "Jamtara":     (23.963, 86.803, "JH", 1,  1),
    "Bharatpur":   (27.217, 77.490, "RJ", 1,  1),
    "Alwar":       (27.553, 76.635, "RJ", 1,  1),
    "Deoghar":     (24.483, 86.697, "JH", 1,  1),
}
CITY_NAMES   = list(CITIES.keys())
CITY_WEIGHTS = np.array([CITIES[c][3] for c in CITY_NAMES], dtype=float)
CITY_WEIGHTS /= CITY_WEIGHTS.sum()
HOTSPOTS     = [c for c in CITY_NAMES if CITIES[c][4] == 1]

BANKS      = ["HDFC","ICICI","SBI","AXIS","KOTAK","PNB","BOB","UNION","CANARA","AIRTELPB"]
MULE_BANK_W= [4, 4, 40, 4, 6, 10, 3, 3, 7, 5]      # I4C concentration: SBI>>PNB>Canara>Kotak>PB
SEGMENTS   = ["retail","sme","student","senior","gig_worker"]
ACC_TYPES  = ["savings","current","salary"]
CHANNELS   = ["mobile_app","web","ussd"]
METHODS    = ["UPI","IMPS","NEFT","RTGS"]
PURPOSES   = ["P2P","P2M","bill_pay","salary","investment","rent"]
PURPOSE_W  = [30, 45, 12, 5, 4, 4]                  # P2M > P2P (NPCI 2026)
MULE_TYPES = ["fan_in_out","smurfing","round_trip","dormant_active","rapid_passthrough",
              "collect_scam","circle_abuse"]
PSP_APPS   = ["phonepe","gpay","paytm","navi","cred","bhim"]
PSP_W      = [46, 34, 10, 4, 3, 3]                   # NPCI app market-share shape
COLLECT_CAP = 2_000                                   # NPCI cap on collect requests
UPI_CAP     = 100_000                                 # standard UPI per-txn limit


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def rand_ip(pool_id=None):
    """Generate an IPv4. Same pool_id → same /24 subnet (device farm signal)."""
    if pool_id is not None:
        return f"103.{40 + pool_id % 60}.{pool_id % 250}.{random.randint(2, 254)}"
    return f"{random.choice([49,59,101,106,115,122,157,182,203])}." \
           f"{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(2,254)}"


# ─────────────────────────────────────────────────────────────
#  ACCOUNTS
# ─────────────────────────────────────────────────────────────
def generate_accounts(n, mule_ratio):
    n_mules = int(n * mule_ratio)
    rows = []

    # Device farms: groups of mules share device+IP pools
    n_farms = max(4, n_mules // 25)

    for i in range(n):
        is_mule = i < n_mules
        subtype = MULE_TYPES[i % len(MULE_TYPES)] if is_mule else "benign"

        if is_mule:
            careful = random.random() < 0.30               # 30% hide their fingerprint
            if careful:
                city = random.choice(HOTSPOTS) if random.random() < 0.15 \
                       else np.random.choice(CITY_NAMES, p=CITY_WEIGHTS)
                farm_id, ip_pool_id = -1, None
                device_ids = [f"DEV_{i:06d}_0"]            # own clean device
                vpn_base, geo_mult, travel_prob = 0.05, 0.15, 0.0
            else:
                city = random.choice(HOTSPOTS) if random.random() < 0.60 \
                       else np.random.choice(CITY_NAMES, p=CITY_WEIGHTS)
                farm_id  = random.randint(0, n_farms - 1)
                device_ids = [f"DEV_FARM_{farm_id}_{random.randint(0, 3)}"]
                ip_pool_id = farm_id
                vpn_base, geo_mult, travel_prob = 0.25, 1.0, 0.0
        else:
            persona = random.choices(["normal","family","traveler","vpn_user"],
                                      weights=[80, 8, 7, 5])[0]
            city = np.random.choice(CITY_NAMES, p=CITY_WEIGHTS)
            farm_id, ip_pool_id = -1, None
            if persona == "family":                        # shared household device
                fam = random.randint(0, max(1, n // 55))
                device_ids = [f"DEV_FAM_{fam:05d}"]
            else:
                nd = random.choices([1, 2, 3], weights=[70, 25, 5])[0]
                device_ids = [f"DEV_{i:06d}_{d}" for d in range(nd)]
            vpn_base    = 0.30 if persona == "vpn_user" else 0.02
            travel_prob = 0.05 if persona == "traveler" else 0.0
            geo_mult    = 1.0

        lat, lon, state, _, hotspot = CITIES[city]
        # identity-layer signals (with realistic overlap)
        if is_mule:
            if random.random() < 0.35:                     # bought AGED account/SIM
                sim_age = random.randint(200, 2500)
                sim_port, kyc_upd = int(random.random()<0.10), int(random.random()<0.55)
            else:                                          # fresh SIM operation
                sim_age = random.randint(3, 150)
                sim_port, kyc_upd = int(random.random()<0.30), int(random.random()<0.40)
            sold_acct = int(subtype == "dormant_active" and random.random() < 0.70)
            bank      = random.choices(BANKS, weights=MULE_BANK_W)[0]
            session_mu, paste_base = (10.3, 0.35) if random.random()<0.30 else (9.6, 0.65)
        else:
            if random.random() < 0.25:                     # genuinely new customers
                sim_age = random.randint(10, 180)
                sim_port, kyc_upd = int(random.random()<0.10), int(random.random()<0.15)
            else:
                sim_age = random.randint(180, 4000)
                sim_port, kyc_upd = int(random.random()<0.04), int(random.random()<0.06)
            sold_acct = 0
            bank      = random.choice(BANKS)
            r_ = random.random()                           # session/paste personas
            if r_ < 0.15:   session_mu, paste_base = 9.8, 0.15   # quick payers
            elif r_ < 0.35: session_mu, paste_base = 10.6, 0.40  # biz users paste a lot
            else:           session_mu, paste_base = 10.9, 0.10
        psp = random.choices(PSP_APPS, weights=PSP_W)[0]
        if is_mule:
            n_vpas = random.choices([1,2,3,4], weights=[20,35,30,15])[0]
            binding_resets = random.choices([0,1,2,3], weights=[15,35,30,20])[0]
        else:
            n_vpas = random.choices([1,2,3], weights=[75,20,5])[0]
            binding_resets = random.choices([0,1,2], weights=[88,10,2])[0]
        rows.append({
            "account_id":       f"ACC{i:06d}",
            "psp_app":          psp,
            "n_vpas":           n_vpas,
            "binding_resets":   binding_resets,
            "bank_code":        bank,
            "sim_age_days":     sim_age,
            "sim_ported_recent": sim_port,
            "kyc_updated_recent": kyc_upd,
            "sold_account":     sold_acct,
            "session_mu":       session_mu,
            "paste_base":       paste_base,
            "account_type":     random.choice(ACC_TYPES),
            "customer_segment": random.choice(SEGMENTS),
            "home_city":        city,
            "home_state":       state,
            "home_lat":         lat + random.uniform(-0.05, 0.05),
            "home_lon":         lon + random.uniform(-0.05, 0.05),
            "in_hotspot":       hotspot,
            "account_age_days": random.randint(5, 30) if (is_mule and subtype=="dormant_active")
                                 else random.randint(30, 1800),
            "kyc_tier":         random.choices([1,2,3], weights=[50,35,15])[0] if is_mule
                                 else random.choices([1,2,3], weights=[20,45,35])[0],
            "activation_delay": random.randint(30,60) if (is_mule and subtype=="dormant_active") else 0,
            "device_ids":       device_ids,
            "ip_pool_id":       ip_pool_id,
            "farm_id":          farm_id,
            "vpn_base":         vpn_base,
            "geo_mult":         geo_mult,
            "travel_prob":      travel_prob,
            "is_mule":          int(is_mule),
            "mule_subtype":     subtype,
        })
    df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────
#  TRANSACTION CONTEXT (ip / device / geo per txn)
# ─────────────────────────────────────────────────────────────
def txn_context(acc, odd_geo_prob=0.0):
    """Return device, ip, lat, lon, vpn, channel for one txn by this account."""
    device = random.choice(acc["device_ids"])
    pid = acc["ip_pool_id"]
    vpn = int(random.random() < acc.get("vpn_base", 0.02))
    if pid is not None and not (isinstance(pid, float) and math.isnan(pid)):  # mule farm subnet
        ip = rand_ip(int(pid))
    else:
        ip = rand_ip()

    odd_eff = odd_geo_prob * acc.get("geo_mult", 1.0) + acc.get("travel_prob", 0.0)
    if random.random() < odd_eff:                          # impossible travel
        far = np.random.choice(CITY_NAMES, p=CITY_WEIGHTS)
        lat, lon = CITIES[far][0], CITIES[far][1]
    else:
        lat = acc["home_lat"] + random.uniform(-0.15, 0.15)
        lon = acc["home_lon"] + random.uniform(-0.15, 0.15)

    channel = random.choices(CHANNELS, weights=[75, 20, 5])[0]
    # behavioural biometrics drawn from per-account personas (overlapping)
    session_ms = int(np.random.lognormal(acc.get("session_mu", 10.9), 0.65))
    paste      = int(random.random() < acc.get("paste_base", 0.10))
    return device, ip, round(lat,4), round(lon,4), vpn, channel, session_ms, paste


SIM_START = datetime(2024, 1, 1)
ADAPT_DAY = 60          # mule farms adapt tactics after this day (drift demo)

_txn_n = 0
def txn_id():
    global _txn_n; _txn_n += 1
    return f"TXN{_txn_n:09d}"


ACC_REG = {}          # account_id -> record, set in main before generation

def make_txn(ts, sender, receiver, amount, method, purpose, susp, pattern,
             device, ip, lat, lon, vpn, channel, session_ms=30000, paste=0,
             flow=None):
    rail = "UPI" if method == "UPI" else "BANK"
    s_rec = ACC_REG.get(sender, {})
    psp = s_rec.get("psp_app", "other") if rail == "UPI" else "na"
    if rail == "UPI":
        amount = min(amount, UPI_CAP)
        if flow is None:
            flow = "qr_p2m" if purpose == "P2M" else                    ("autopay" if purpose in ("bill_pay","rent") and random.random() < 0.35
                    else "push")
    else:
        flow = "bank_transfer"
    return dict(txn_id=txn_id(), timestamp=ts, sender_id=sender, receiver_id=receiver,
                amount=round(amount,2), method=method, purpose=purpose,
                is_suspicious=susp, pattern=pattern,
                device_id=device, ip_address=ip, lat=lat, lon=lon,
                vpn_flag=vpn, channel=channel, session_ms=session_ms, paste_flag=paste,
                rail=rail, txn_flow=flow, psp_app=psp)


# ─────────────────────────────────────────────────────────────
#  PATTERN GENERATORS
# ─────────────────────────────────────────────────────────────
def gen_benign(acc, other_recs, start, days, n_lo=4, n_hi=14):
    txns = []
    is_biz = acc["customer_segment"] in ("sme", "gig_worker")
    n_txn  = random.randint(n_lo, n_hi)
    if is_biz:
        n_txn = int(n_txn * random.uniform(1.3, 2.2))      # businesses transact more
    for _ in range(n_txn):
        base = start + timedelta(days=random.randint(0, days-1))
        if random.random() < (0.18 if is_biz else 0.10):    # night owls exist
            hour = random.randint(0, 23)
        else:
            hour = int(np.clip(np.random.normal(13, 3.5), 7, 21))
        ts   = base.replace(hour=hour, minute=random.randint(0,59), second=random.randint(0,59))
        cp   = random.choice(other_recs)
        amt  = min(np.random.lognormal(6.65, 1.35), 100_000)   # mean~Rs1.9K, median~Rs775 (NPCI avg ticket Rs1,293)
        if random.random() < 0.5:
            s, r, sender_rec = acc["account_id"], cp["account_id"], acc
        else:
            s, r, sender_rec = cp["account_id"], acc["account_id"], cp
        d,ip,la,lo,v,ch,sm,pf = txn_context(sender_rec, odd_geo_prob=0.002)
        txns.append(make_txn(ts, s, r, amt,
                    random.choices(METHODS, weights=[62,23,10,5])[0],
                    random.choices(PURPOSES, weights=PURPOSE_W)[0], 0, "benign", d, ip, la, lo, v, ch, sm, pf))

    # SME distributor pattern: receive payment, pay suppliers same day
    # (a benign hard-negative that mimics pass-through behaviour)
    if is_biz and random.random() < 0.55:
        for _ in range(random.randint(2, 6)):
            t0  = start + timedelta(days=random.randint(0, days-2),
                                     hours=random.randint(8, 18))
            amt = random.uniform(20_000, 150_000)
            src = random.choice(other_recs)
            d,ip,la,lo,v,ch,sm,pf = txn_context(src, odd_geo_prob=0.002)
            txns.append(make_txn(t0, src["account_id"], acc["account_id"], amt,
                                 "IMPS", "P2M", 0, "benign", d, ip, la, lo, v, ch, sm, pf))
            remaining = amt * random.uniform(0.85, 0.99)
            for _ in range(random.randint(1, 3)):
                dst = random.choice(other_recs)
                d,ip,la,lo,v,ch,sm,pf = txn_context(acc, odd_geo_prob=0.002)
                txns.append(make_txn(t0 + timedelta(hours=random.uniform(0.5, 20)),
                                     acc["account_id"], dst["account_id"],
                                     remaining * random.uniform(0.3, 0.7),
                                     random.choice(["UPI","IMPS"]), "P2M", 0,
                                     "benign", d, ip, la, lo, v, ch, sm, pf))
    return txns


def odd_hour(base):
    """Era-aware: pre-adaptation 60% night; post-day-60 adapted mules 35% night."""
    p_night = 0.35 if (base - SIM_START).days >= ADAPT_DAY else 0.60
    if random.random() < p_night:
        h = random.choice(list(range(0,6)) + [22,23])
    else:
        h = random.randint(9, 20)
    return base.replace(hour=h, minute=random.randint(0,59), second=random.randint(0,59))


def gen_fan_in_out(acc, feeder_recs, drain, start, days):
    txns, total = [], 0.0
    w0 = start + timedelta(days=random.randint(0, days-7))
    adapted = (w0 - SIM_START).days >= ADAPT_DAY
    w1 = w0 + timedelta(hours=random.randint(24, 96) if adapted else random.randint(6, 48))
    for f in random.choices(feeder_recs, k=random.randint(8, 20) if adapted else random.randint(10, 28)):
        amt = random.uniform(2_000, 15_000); total += amt
        ts  = w0 + (w1 - w0) * random.random()
        d,ip,la,lo,v,ch,sm,pf = txn_context(f, odd_geo_prob=0.01)
        txns.append(make_txn(ts, f["account_id"], acc["account_id"], amt, "UPI", "P2P", 1,
                             "fan_in_out_in", d, ip, la, lo, v, ch, sm, pf))
    for k in range(random.randint(1, 3)):
        ts  = w1 + timedelta(hours=random.uniform(0.5, 6))
        amt = total / max(k+1,1) * random.uniform(0.85, 0.98)
        d,ip,la,lo,v,ch,sm,pf = txn_context(acc, odd_geo_prob=0.15)
        txns.append(make_txn(odd_hour(ts), acc["account_id"], drain, amt,
                             random.choice(["IMPS","NEFT"]), "P2P", 1,
                             "fan_in_out_out", d, ip, la, lo, v, ch, sm, pf))
    return txns


def gen_smurfing(acc, src, dests, start, days):
    txns = []
    big  = random.uniform(150_000, 900_000)
    n    = math.ceil(big / 49_999)
    per  = big / n * random.uniform(0.88, 0.97)
    w0   = start + timedelta(days=random.randint(0, days-3))
    for i, dst in enumerate(random.choices(dests, k=n)):
        ts = w0 + timedelta(hours=i * random.uniform(0.5, 4))
        d,ip,la,lo,v,ch,sm,pf = txn_context(acc, odd_geo_prob=0.10)
        txns.append(make_txn(ts, acc["account_id"], dst,
                             per + random.uniform(-200, 200), "UPI", "P2P", 1,
                             "smurfing", d, ip, la, lo, v, ch, sm, pf))
    return txns


def gen_round_trip(ring_accs, start, days):
    txns = []
    amt  = random.uniform(30_000, 200_000)
    w0   = start + timedelta(days=random.randint(0, days-5))
    hop  = random.uniform(2, 12)
    for i in range(len(ring_accs)):
        a, b = ring_accs[i], ring_accs[(i+1) % len(ring_accs)]
        ts   = w0 + timedelta(hours=i*hop)
        d,ip,la,lo,v,ch,sm,pf = txn_context(a, odd_geo_prob=0.20)
        txns.append(make_txn(ts, a["account_id"], b["account_id"],
                             amt * (0.97**i), random.choice(["UPI","IMPS"]),
                             "P2P", 1, "round_trip", d, ip, la, lo, v, ch, sm, pf))
    return txns


def gen_dormant(acc, feeder_recs, drain, start, days):
    txns = []
    a0 = start + timedelta(days=min(acc["activation_delay"], days-10))
    a1 = start + timedelta(days=days)
    for _ in range(random.randint(18, 45)):
        ts = odd_hour(a0 + (a1 - a0) * random.random())
        amt = random.uniform(5_000, 80_000)
        if random.random() < 0.5:
            f = random.choice(feeder_recs)
            s, r, sender_rec, ogp = f["account_id"], acc["account_id"], f, 0.01
        else:
            s, r, sender_rec, ogp = acc["account_id"], drain, acc, 0.15
        d,ip,la,lo,v,ch,sm,pf = txn_context(sender_rec, odd_geo_prob=ogp)
        txns.append(make_txn(ts, s, r, amt, random.choice(["UPI","IMPS"]),
                             "P2P", 1, "dormant_active", d, ip, la, lo, v, ch, sm, pf))
    return txns


def gen_passthrough(acc, src_rec, dst, start, days):
    txns = []
    w0 = start + timedelta(days=random.randint(0, days-14))
    for i in range(random.randint(4, 12)):
        c0  = w0 + timedelta(days=i * random.uniform(0.5, 3))
        amt = random.uniform(10_000, 100_000)
        d,ip,la,lo,v,ch,sm,pf = txn_context(src_rec, odd_geo_prob=0.12)
        txns.append(make_txn(c0, src_rec["account_id"], acc["account_id"], amt, "UPI", "P2P", 1,
                             "rapid_passthrough_in", d, ip, la, lo, v, ch, sm, pf))
        d,ip,la,lo,v,ch,sm,pf = txn_context(acc, odd_geo_prob=0.12)
        txns.append(make_txn(c0 + timedelta(hours=random.uniform(0.25, 23)),
                             acc["account_id"], dst, amt * random.uniform(0.91, 0.99),
                             random.choice(["IMPS","NEFT"]), "P2P", 1,
                             "rapid_passthrough_out", d, ip, la, lo, v, ch, sm, pf))
    return txns


def gen_collect_scam(acc, victim_recs, start, days):
    """Mule blasts collect requests; a fraction of victims approve.
    NPCI collect cap Rs2,000 shapes amounts."""
    txns = []
    w0 = start + timedelta(days=random.randint(0, days-5))
    for v in random.choices(victim_recs, k=random.randint(15, 40)):
        ts = w0 + timedelta(hours=random.uniform(0, 96))
        amt = random.uniform(200, COLLECT_CAP)
        d,ip,la,lo,vp,ch,sm,pf = txn_context(v, odd_geo_prob=0.005)
        txns.append(make_txn(ts, v["account_id"], acc["account_id"], amt,
                             "UPI", "P2P", 1, "collect_scam",
                             d, ip, la, lo, vp, ch, sm, pf, flow="collect_request"))
    # drain approvals onward
    total = sum(t["amount"] for t in txns)
    drain = random.choice(victim_recs)["account_id"]
    d,ip,la,lo,vp,ch,sm,pf = txn_context(acc, odd_geo_prob=0.10)
    txns.append(make_txn(w0 + timedelta(hours=100), acc["account_id"], drain,
                         total*random.uniform(0.85,0.97), "IMPS", "P2P", 1,
                         "collect_scam_out", d, ip, la, lo, vp, ch, sm, pf))
    return txns


def gen_circle_abuse(acc, victim_recs, drain, start, days):
    """UPI Circle delegation abuse: fraudster's device initiates debits FROM
    victim accounts (delegated payments) routed to the mule, then drained."""
    txns = []
    fr_dev = acc["device_ids"][0]                    # fraudster device on victims
    w0 = start + timedelta(days=random.randint(0, days-4))
    victims = random.sample(victim_recs, min(6, len(victim_recs)))
    total = 0.0
    for v in victims:
        for _ in range(random.randint(2, 5)):
            ts = w0 + timedelta(hours=random.uniform(0, 72))
            amt = random.uniform(1_000, 15_000); total += amt
            _,ip,la,lo,vp,ch,sm,pf = txn_context(acc, odd_geo_prob=0.05)
            txns.append(make_txn(ts, v["account_id"], acc["account_id"], amt,
                                 "UPI", "P2P", 1, "circle_abuse",
                                 fr_dev, ip, la, lo, vp, ch, sm, pf,
                                 flow="circle_delegate"))
    d,ip,la,lo,vp,ch,sm,pf = txn_context(acc, odd_geo_prob=0.12)
    txns.append(make_txn(w0 + timedelta(hours=80), acc["account_id"], drain,
                         total*random.uniform(0.88,0.97), "IMPS", "P2P", 1,
                         "circle_abuse_out", d, ip, la, lo, vp, ch, sm, pf))
    return txns


# ─────────────────────────────────────────────────────────────
#  VECTORISED FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(acc_df, tx):
    print("      · aggregating amounts & counterparties")
    sent = tx.groupby("sender_id").agg(
        total_sent_inr=("amount","sum"), n_sent=("amount","size"),
        n_counterparties_out=("receiver_id","nunique")).rename_axis("account_id")
    recv = tx.groupby("receiver_id").agg(
        total_recv_inr=("amount","sum"), n_recv=("amount","size"),
        n_counterparties_in=("sender_id","nunique")).rename_axis("account_id")

    # Per-account combined stream (long format)
    print("      · building combined per-account stream")
    s = tx[["sender_id","timestamp","amount","ip_address","device_id","lat","lon","vpn_flag"]]\
          .rename(columns={"sender_id":"account_id"})
    r = tx[["receiver_id","timestamp","amount","ip_address","device_id","lat","lon","vpn_flag"]]\
          .rename(columns={"receiver_id":"account_id"})
    # ip/device/geo of receiver-side rows belong to the *sender's* session; keep only
    # sender rows for device/ip/geo features, both for amounts/velocity
    both = pd.concat([s, r], ignore_index=True)
    both["hour"] = both["timestamp"].dt.hour

    print("      · amount stats, off-hours, sub-threshold")
    amt_stats = both.groupby("account_id")["amount"].agg(
        amount_mean_inr="mean", amount_std_inr="std").fillna(0)
    off = both.assign(off=(both["hour"].lt(8) | both["hour"].ge(22)).astype(int))\
              .groupby("account_id")["off"].mean().rename("off_hours_ratio")
    sub = both.assign(st=both["amount"].between(40_000, 49_999).astype(int))\
              .groupby("account_id")["st"].mean().rename("sub_threshold_ratio")

    print("      · 24h max velocity (searchsorted per account)")
    vel = {}
    for aid, grp in both.groupby("account_id")["timestamp"]:
        t = np.sort(grp.values.astype("datetime64[s]").astype(np.int64))
        j = np.searchsorted(t, t + 86_400, side="right")
        vel[aid] = int((j - np.arange(len(t))).max())
    vel = pd.Series(vel, name="max_24h_velocity").rename_axis("account_id")

    print("      · IP / device intelligence")
    sonly = s.copy()
    ip_div  = sonly.groupby("account_id")["ip_address"].nunique().rename("n_unique_ips")
    dev_div = sonly.groupby("account_id")["device_id"].nunique().rename("n_unique_devices")
    vpn_r   = sonly.groupby("account_id")["vpn_flag"].mean().rename("vpn_ratio")
    # how many DISTINCT accounts share this account's devices (farm signal)
    dev_acc = tx.groupby("device_id")["sender_id"].nunique()
    sonly["dev_shared"] = sonly["device_id"].map(dev_acc)
    dev_shared = sonly.groupby("account_id")["dev_shared"].max()\
                      .rename("device_shared_accounts")
    # accounts per /24 subnet
    sonly["subnet"] = sonly["ip_address"].str.rsplit(".", n=1).str[0]
    sub_acc = sonly.groupby("subnet")["account_id"].nunique()
    sonly["ip_shared"] = sonly["subnet"].map(sub_acc)
    ip_shared = sonly.groupby("account_id")["ip_shared"].max()\
                     .rename("ip_subnet_shared_accounts")

    print("      · geo velocity (impossible travel)")
    geo = {}
    for aid, grp in sonly.sort_values("timestamp").groupby("account_id"):
        if len(grp) < 2:
            geo[aid] = 0.0; continue
        la, lo = grp["lat"].values, grp["lon"].values
        dt_h = np.diff(grp["timestamp"].values.astype("datetime64[s]").astype(np.int64)) / 3600
        dk   = haversine_km(la[:-1], lo[:-1], la[1:], lo[1:])
        kmh  = np.where(dt_h > 0.05, dk/np.maximum(dt_h, 0.05), 0)
        geo[aid] = float(np.nanmax(kmh)) if len(kmh) else 0.0
    geo = pd.Series(geo, name="geo_velocity_kmh_max").rename_axis("account_id")

    out = acc_df.set_index("account_id")\
        .join([sent, recv, amt_stats, off, sub, vel,
               ip_div, dev_div, vpn_r, dev_shared, ip_shared, geo])\
        .fillna(0)

    # v3: behavioural + cross-bank aggregates (sender side)
    print("      · behavioural biometrics & cross-bank")
    s3 = tx[["sender_id","receiver_id","session_ms","paste_flag","timestamp"]]\
           .rename(columns={"sender_id":"account_id"})
    beh = s3.groupby("account_id").agg(
        avg_session_ms=("session_ms","mean"), paste_ratio=("paste_flag","mean"))
    bank_map = dict(zip(acc_df["account_id"], acc_df["bank_code"]))
    s3b = s3.assign(sb=s3["account_id"].map(bank_map),
                     rb=s3["receiver_id"].map(bank_map))
    xb = s3b.assign(x=(s3b.sb != s3b.rb).astype(int)).groupby("account_id")["x"]\
            .mean().rename("cross_bank_ratio")
    # temporal drift features: activity shift + burstiness
    s3["day"] = (s3["timestamp"] - s3["timestamp"].min()).dt.days
    def _shift(g):
        e1 = (g < ADAPT_DAY).sum(); e2 = (g >= ADAPT_DAY).sum()
        return e2 / (e1 * (90-ADAPT_DAY)/ADAPT_DAY + 1)
    act = s3.groupby("account_id")["day"].apply(_shift).rename("activity_shift")
    daily = s3.groupby(["account_id","day"]).size()
    burst = daily.groupby("account_id").agg(["std","mean"])
    burst = (burst["std"].fillna(0) / burst["mean"].clip(lower=1)).rename("burstiness")
    # UPI-native aggregates
    upi_sh = tx.assign(u=(tx["rail"]=="UPI").astype(int))\
               .rename(columns={"sender_id":"account_id"})\
               .groupby("account_id")["u"].mean().rename("upi_share")
    coll = tx[tx["txn_flow"]=="collect_request"].groupby("receiver_id").size()\
             .rename("collect_inflows").rename_axis("account_id")
    out = out.join([beh, xb, act, burst, upi_sh, coll]).fillna(0)
    out["collect_in_ratio"] = (out["collect_inflows"] /
                                out["n_recv"].clip(lower=1)).clip(upper=1)

    out["pass_through_ratio"] = np.minimum(
        out["total_sent_inr"] / out["total_recv_inr"].replace(0, np.nan), 1.0).fillna(0)
    out["fan_in_diversity"] = (out["n_counterparties_in"] /
                                out["n_recv"].replace(0, np.nan)).fillna(0)
    return out.reset_index()


# ─────────────────────────────────────────────────────────────
#  GRAPH DATA
# ─────────────────────────────────────────────────────────────
def build_graph(tx, acc_df):
    print("      · building digraph")
    agg = tx.groupby(["sender_id","receiver_id"]).agg(
        amount=("amount","sum"), n=("amount","size"),
        is_suspicious=("is_suspicious","max")).reset_index()
    G = nx.from_pandas_edgelist(agg, "sender_id", "receiver_id",
                                 edge_attr=["amount","n","is_suspicious"],
                                 create_using=nx.DiGraph())
    print(f"      · pagerank (scipy) on {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")
    import scipy.sparse as sp
    nodes_list = list(G.nodes()); nidx = {n:i for i,n in enumerate(nodes_list)}
    r_, c_, w_ = [], [], []
    for u,v,dd in G.edges(data=True):
        r_.append(nidx[u]); c_.append(nidx[v]); w_.append(np.log1p(dd.get("amount",1)))
    A = sp.coo_matrix((w_,(r_,c_)), shape=(len(nodes_list),)*2).tocsr()
    outsum = np.asarray(A.sum(1)).ravel(); outsum[outsum==0]=1
    M = sp.diags(1/outsum) @ A
    pr_v = np.full(len(nodes_list), 1/len(nodes_list))
    for _ in range(50):
        pr_v = 0.15/len(nodes_list) + 0.85*(M.T @ pr_v)
    pr = {n: float(pr_v[i]) for n,i in nidx.items()}
    ind = dict(G.in_degree()); outd = dict(G.out_degree())

    print("      · communities (igraph multilevel)")
    try:
        import igraph as ig
        gU = ig.Graph(n=len(nodes_list),
                      edges=[(r_[k], c_[k]) for k in range(len(r_))], directed=False)
        memb = gU.community_multilevel().membership
        part = {n: memb[i] for n,i in nidx.items()}
    except Exception:
        import community.community_louvain as cl
        part = cl.best_partition(G.to_undirected(), random_state=42)

    lbl = dict(zip(acc_df["account_id"], acc_df["is_mule"]))
    nodes = pd.DataFrame([{
        "account_id": n, "in_degree": ind.get(n,0), "out_degree": outd.get(n,0),
        "pagerank": round(pr.get(n,0), 8), "community_id": part.get(n,-1),
        "is_mule": lbl.get(n, 0),
    } for n in G.nodes()])

    # community risk = mule fraction per community (ground-truth aid for eval;
    # models must NOT use is_mule — they use structure + score propagation)
    comm_size = nodes.groupby("community_id")["account_id"].size().rename("community_size")
    nodes = nodes.merge(comm_size, on="community_id")
    return agg.rename(columns={"sender_id":"src","receiver_id":"dst"}), nodes


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--accounts",   type=int,   default=205000)
    p.add_argument("--days",       type=int,   default=90)
    p.add_argument("--mule-ratio", type=float, default=0.03)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--output-dir", type=str,   default="./aml_output_v4")
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed)
    import os; os.makedirs(a.output_dir, exist_ok=True)
    start = datetime(2024, 1, 1)

    print(f"[1/6] Accounts: {a.accounts:,} ({a.mule_ratio*100:.0f}% mules, device farms on)")
    acc_df = generate_accounts(a.accounts, a.mule_ratio)
    acc_recs = acc_df.to_dict("records")
    by_id    = {r["account_id"]: r for r in acc_recs}
    mules    = [r for r in acc_recs if r["is_mule"]]
    benign   = [r for r in acc_recs if not r["is_mule"]]
    benign_ids = [r["account_id"] for r in benign]

    global ACC_REG
    ACC_REG = by_id

    print(f"[2/6] Benign transactions for {len(benign):,} accounts …")
    frames, txns, total_n = [], [], 0
    def flush():
        nonlocal txns, total_n
        if txns:
            frames.append(pd.DataFrame(txns)); total_n += len(txns); txns = []
    for i, r in enumerate(benign):
        others = random.sample(benign, 25)
        txns.extend(gen_benign(r, others, start, a.days))
        if len(txns) >= 300_000: flush()
        if (i+1) % 20000 == 0: print(f"      · {i+1:,} accounts, {total_n+len(txns):,} txns")

    print(f"[3/6] Cover + mule patterns for {len(mules):,} mule accounts …")
    for r in mules:
        others = random.sample(benign, 20)
        cover = gen_benign(r, others, start, a.days, n_lo=4, n_hi=10)
        txns.extend(cover)
        feeders = random.sample(benign, 30)
        drain   = random.choice(benign_ids)
        st = r["mule_subtype"]
        if st == "fan_in_out":
            txns.extend(gen_fan_in_out(r, feeders, drain, start, a.days))
        elif st == "smurfing":
            txns.extend(gen_smurfing(r, random.choice(mules)["account_id"],
                                      random.sample(benign_ids, 20), start, a.days))
        elif st == "round_trip":
            ring = [r] + random.sample([m for m in mules if m is not r],
                                        min(3, len(mules)-1))
            txns.extend(gen_round_trip(ring, start, a.days))
        elif st == "dormant_active":
            txns.extend(gen_dormant(r, feeders, drain, start, a.days))
        elif st == "rapid_passthrough":
            txns.extend(gen_passthrough(r, random.choice(mules), drain, start, a.days))
        elif st == "collect_scam":
            txns.extend(gen_collect_scam(r, feeders, start, a.days))
        elif st == "circle_abuse":
            txns.extend(gen_circle_abuse(r, feeders, drain, start, a.days))

    flush()
    tx = pd.concat(frames, ignore_index=True); frames.clear()
    print(f"[4/6] DataFrame: {len(tx):,} transactions → engineering features")
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx = tx[tx["sender_id"] != tx["receiver_id"]].sort_values("timestamp").reset_index(drop=True)
    feat_df = engineer_features(acc_df, tx)

    print("[5/6] Graph features + Louvain")
    edges, nodes = build_graph(tx, acc_df)
    feat_df = feat_df.merge(
        nodes[["account_id","in_degree","out_degree","pagerank",
               "community_id","community_size"]], on="account_id", how="left").fillna(0)

    print("[6/6] Writing outputs")
    feat_df = feat_df.drop(columns=["device_ids","ip_pool_id","vpn_base",
                                     "geo_mult","travel_prob","farm_id",
                                     "session_mu","paste_base"], errors="ignore")
    tx.to_csv(f"{a.output_dir}/transactions.csv", index=False)
    feat_df.to_csv(f"{a.output_dir}/accounts.csv", index=False)
    edges.to_csv(f"{a.output_dir}/graph_edges.csv", index=False)
    nodes.to_csv(f"{a.output_dir}/graph_nodes.csv", index=False)

    mu, be = feat_df[feat_df.is_mule==1], feat_df[feat_df.is_mule==0]
    lines = [
        "="*62, "AML SYNTHETIC DATASET v4 — EDA SUMMARY (UPI-native · dual rail)",
        "MuleTrace-IN · IIM Capstone", "="*62, "",
        f"Accounts            : {len(feat_df):,}  (mules {len(mu):,} = {len(mu)/len(feat_df)*100:.1f}%)",
        f"Transactions        : {len(tx):,}",
        f"Suspicious txns     : {tx.is_suspicious.sum():,} ({tx.is_suspicious.mean()*100:.1f}%)",
        f"Total volume        : ₹{tx.amount.sum():,.0f}",
        f"Simulation window   : {a.days} days", "",
        "NEW v2 SIGNALS (mule mean vs benign mean)",
        f"  {'feature':<30}{'mule':>12}{'benign':>12}",
        f"  {'-'*54}",
    ]
    for f in ["device_shared_accounts","ip_subnet_shared_accounts","vpn_ratio",
              "geo_velocity_kmh_max","in_hotspot","pass_through_ratio",
              "max_24h_velocity","off_hours_ratio","pagerank",
              "avg_session_ms","paste_ratio","cross_bank_ratio",
              "sim_age_days","activity_shift","burstiness",
              "n_vpas","binding_resets","upi_share","collect_in_ratio"]:
        lines.append(f"  {f:<30}{mu[f].mean():>12.4f}{be[f].mean():>12.4f}")
    lines += ["", "PATTERN BREAKDOWN"]
    for pat, c in tx[tx.is_suspicious==1]["pattern"].value_counts().items():
        lines.append(f"  {pat:<28}: {c:>7,}")
    lines += ["", "FILES", "  transactions.csv / accounts.csv / graph_edges.csv / graph_nodes.csv", "="*62]
    summary = "\n".join(lines)
    with open(f"{a.output_dir}/eda_summary.txt","w") as f: f.write(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
