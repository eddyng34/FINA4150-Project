# main.py
# ETH OTC Accumulator – FP XXX% with kx=3 for LV (Smoothed for Stability)
# Live Deribit data, cubic IV & LV, robust MC

import requests
import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings("ignore")

# ========================================
# 1. FETCH DERIBIT DATA
# ========================================
def fetch_eth_options_with_prices():
    url_instr = "https://www.deribit.com/api/v2/public/get_instruments"
    params = {"currency": "ETH", "kind": "option", "expired": "false"}
    resp = requests.get(url_instr, params=params, timeout=15).json()
    if 'result' not in resp: raise ValueError(f"API error: {resp}")
    df = pd.DataFrame(resp['result'])

    url_price = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    resp_price = requests.get(url_price, params=params, timeout=15).json()
    if 'result' not in resp_price: raise ValueError(f"Price API error: {resp_price}")
    price_df = pd.DataFrame(resp_price["result"])[["instrument_name", "mark_price", "mark_iv"]]

    df = df.merge(price_df, on="instrument_name", how="left")

    strike_col = 'strike_price' if 'strike_price' in df.columns else 'strike'
    if strike_col not in df.columns: raise KeyError("No strike column")

    df = df.rename(columns={strike_col: 'K', 'mark_price': 'mid', 'expiration_timestamp': 'expiry_ts'})
    df['expiry'] = pd.to_datetime(df['expiry_ts'], unit='ms', utc=True)
    df = df.drop(columns=['expiry_ts'], errors='ignore')

    df = df[df['mid'] > 1e-6]
    today = pd.Timestamp.now(tz='UTC').normalize()
    df['T'] = (df['expiry'] - today).dt.total_seconds() / (365.25 * 24 * 3600)
    df = df[(df['T'] > 0.01) & (df['T'] <= 1.0)]

    try:
        S0 = requests.get("https://www.deribit.com/api/v2/public/get_index_price?index_name=eth_usd").json()["result"]["index_price"]
    except:
        S0 = 3500.0
    S0 = float(S0)

    print(f"Fetched {len(df)} liquid ETH options. Spot = {S0:,.2f} USD")
    return df, S0

# ========================================
# 2. BLACK-SCHOLES & IV
# ========================================
def bs_call(S, K, T, r, sigma, q=0):
    if T <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Note: implied_vol function removed as mark_iv is used directly

# ========================================
# 3. CUBIC IV SURFACE (kx=3)
# ========================================
def build_iv_surface(df, S0, r=0):
    # Use mark_iv directly (convert % to decimal)
    df['IV'] = df['mark_iv'] / 100
    df = df.dropna(subset=["IV"]).copy()
    df["log_m"] = np.log(df["K"] / S0)

    T_grid = np.linspace(max(df["T"].min(), 0.02), min(df["T"].max(), 0.5), 14)
    m_grid = np.linspace(df["log_m"].min(), df["log_m"].max(), 45)
    T_grid, m_grid = np.unique(T_grid), np.unique(m_grid)

    iv_grid = np.full((len(T_grid), len(m_grid)), np.nan)
    for i, T in enumerate(T_grid):
        sub = df[np.abs(df["T"] - T) < 0.08]
        if len(sub) < 3: continue
        iv_grid[i] = np.interp(m_grid, sub['log_m'], sub['IV'], left=np.nan, right=np.nan)

    for i in range(iv_grid.shape[0]):
        nan = np.isnan(iv_grid[i])
        if np.all(nan): continue
        valid_x = m_grid[~nan]
        valid_y = iv_grid[i][~nan]
        if len(valid_x) > 1:
            iv_grid[i][nan] = np.interp(m_grid[nan], valid_x, valid_y)

    median_iv = np.nanmedian(iv_grid)
    iv_grid = np.nan_to_num(iv_grid, nan=median_iv)

    return RectBivariateSpline(T_grid, m_grid, iv_grid, kx=3, ky=3, s=0)

# ========================================
# 4. CUBIC LV SURFACE (kx=3) – SMOOTHED FOR STABILITY
# ========================================
def calibrate_lv(iv_surf, S0, r=0, T_max=0.3, N_T=40, N_K=80):  # Finer defaults
    T_grid = np.linspace(0.02, T_max, N_T)
    K_grid = np.logspace(np.log(S0*0.65), np.log(S0*1.75), N_K)
    T_grid, K_grid = np.unique(T_grid), np.unique(K_grid)

    lv = np.zeros((len(T_grid), len(K_grid)))
    dT = T_grid[1] - T_grid[0]

    for i in range(1, len(T_grid)):
        T = T_grid[i]
        for j in range(1, len(K_grid)-1):
            K = K_grid[j]
            m = np.log(K / S0)
            sigma_imp = iv_surf(T, m, grid=False)
            if np.isnan(sigma_imp): sigma_imp = 0.3

            C  = bs_call(S0, K, T, r, sigma_imp)
            C_T = (bs_call(S0, K, T+dT, r, iv_surf(T+dT, m, grid=False) if T+dT <= T_grid[-1] else sigma_imp) - C) / dT

            dK  = K_grid[j+1] - K_grid[j-1]
            C_Kp = bs_call(S0, K_grid[j+1], T, r, iv_surf(T, np.log(K_grid[j+1]/S0), grid=False))
            C_Km = bs_call(S0, K_grid[j-1], T, r, iv_surf(T, np.log(K_grid[j-1]/S0), grid=False))
            C_K  = (C_Kp - C_Km) / dK
            C_KK = (C_Kp - 2*C + C_Km) / (dK/2)**2

            num = C_T
            den = 0.5 * K**2 * C_KK
            lv[i, j] = np.sqrt(max(num / den, 0)) if den > 1e-10 else sigma_imp

    # FIXED: Fill borders with interior median before smoothing
    interior_mask = (lv > 0.05)  # Exclude clipped artifacts
    if np.any(interior_mask):
        border_fill = np.median(lv[interior_mask])
        lv[0, :] = border_fill      # First T row
        lv[-1, :] = border_fill     # Last T row
        lv[:, 0] = border_fill      # First K col
        lv[:, -1] = border_fill     # Last K col

    # SMOOTH WITH MEDIAN FILTER TO PREVENT OSCILLATIONS
    lv = median_filter(lv, size=3)
    
    # CLIP TO PREVENT NEGATIVES/EXPLOSIONS
    lv = np.clip(lv, 0.05, 4.0)
    
    # FILL REMAINING NANS
    lv = np.nan_to_num(lv, nan=np.nanmedian(lv))

    lv_surf = RectBivariateSpline(T_grid, K_grid, lv, kx=3, ky=3, s=0)

    # DIAGNOSTIC: Check sample eval (remove after testing)
    sample_t, sample_k = 0.1, S0
    sample_vol = lv_surf(sample_t, sample_k, grid=False)
    print(f"Sample LV (T=0.1, K={S0}): {sample_vol:.4f} (should ≈ ATM IV ~0.6-0.7)")

    return lv_surf

# ========================================
# 5. MONTE CARLO
# ========================================
def price_accumulator(fp_pct, lv_surf, S0, r=0, N=50000, seed=42):
    np.random.seed(seed)
    fp = fp_pct * S0
    ko = 1.10 * S0
    weeks, dt = 13, 7/365.25
    sqrt_dt = np.sqrt(dt)

    T_min, T_max = lv_surf.get_knots()[0][[0, -1]]
    K_min, K_max = lv_surf.get_knots()[1][[0, -1]]

    S = np.full((N, weeks+1), S0)
    payoff = np.zeros(N)  # Now tracks discounted total payoff per path
    active = np.ones(N, dtype=bool)

    for w in range(weeks):
        t = w * dt
        t_c = np.clip(t, T_min, T_max)
        S_c = np.clip(S[active, w], K_min, K_max)

        vol = lv_surf(t_c, S_c, grid=False)
        print(vol)
        vol = np.clip(vol, 0.05, 4.0)
        vol = np.nan_to_num(vol, nan=0.3)
        
        dW = np.random.normal(0, sqrt_dt, size=vol.shape)
        S[active, w+1] = S[active, w] * np.exp((r-0.5*vol**2)*dt + vol*dW)

        settle = S[active, w+1]
        weekly_shares = np.where((settle >= fp) & (settle < ko), 1.0,
                                 np.where(settle < fp, 2.0, 0.0)) * 7
        weekly_payoff = weekly_shares * (settle - fp)
        t_week = (w + 1) * dt
        disc_factor = np.exp(-r * t_week)
        payoff[active] += weekly_payoff * disc_factor

        ko_hit = settle >= ko
        if ko_hit.any():
            # FIXED: Explicit indices for bonus addition
            active_idx = np.where(active)[0]
            ko_idx = active_idx[ko_hit]
            bonus_weeks = max(0, 3 - (w + 1))
            if bonus_weeks > 0:
                bonus_shares = bonus_weeks * 7 * 1.0  # Min 1 share/day * 7 days/week
                bonus_settle = settle[ko_hit]
                bonus_payoff = bonus_shares * (bonus_settle - fp)
                bonus_disc = np.exp(-r * t_week)  # Discount from KO time
                payoff[ko_idx] += bonus_payoff * bonus_disc
            active[active] = ~ko_hit  # Deactivate KO paths
        if not active.any():
            break

    return np.mean(payoff)

# ========================================
# 6. SOLVE FP XXX%
# ========================================
# Define r = 0.0275
r = 0.0275
def solve_fp_xxx(S0, iv_surf, r):
    lv_surf = calibrate_lv(iv_surf, S0, r)
    print(f"lv_surf is{lv_surf}")
    def obj(fp_pct):
        val = price_accumulator(fp_pct, lv_surf, S0, r)
        return val

    try:
        fp_pct = brentq(obj, 0.80, 1.20, xtol=1e-5)
        return fp_pct * 100
    
    except Exception as e:
        print(f"brentq failed ({e}) → grid search")
        grid = np.linspace(0.85, 1.15, 61)
        vals = [obj(p) for p in grid]
        return grid[np.argmin(np.abs(vals))] * 100

# ========================================
# MAIN
# ========================================
if __name__ == "__main__":
    print("ETH OTC Accumulator Pricer (Cubic LV kx=3)")
    print("="*60)

    df, S0 = fetch_eth_options_with_prices()
    print("Building IV surface...")
    iv_surf = build_iv_surface(df, S0, r)

    print("Calibrating cubic LV (kx=3) & solving for FP XXX%...")
    xxx = solve_fp_xxx(S0, iv_surf, r)

    print("\n" + "="*60)
    print(f" RESULT: Forward Price (FP) = {xxx:.3f}% of Initial Spot")
    print(f"         FP Strike = {xxx/100*S0:,.2f} USD")
    print(f"         Initial Spot = {S0:,.2f} USD")
    print("\nFill in PDF:")
    print(f"    Forward Price (FP) or strike: {xxx:.3f}%")
    print("="*60)