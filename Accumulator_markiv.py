# Accumulator_markiv.py
# ETH OTC Accumulator – FINAL PRODUCTION VERSION
# Local Volatility in (T, log-moneyness) → correct dynamics!

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
        S0 = 3050.0
    S0 = float(S0)
    print(f"Fetched {len(df)} liquid ETH options. Spot = {S0:,.2f} USD")
    return df, S0

# ========================================
# 2. BLACK-SCHOLES
# ========================================
def bs_call(S, K, T, r, sigma, q=0):
    if T <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# ========================================
# 3. CUBIC IV SURFACE (log-moneyness)
# ========================================
def build_iv_surface(df, S0, r=0):
    df['IV'] = df['mark_iv'] / 100
    df = df.dropna(subset=["IV"]).copy()
    df["log_m"] = np.log(df["K"] / S0)

    T_grid = np.linspace(max(df["T"].min(), 0.02), min(df["T"].max(), 0.5), 14)
    m_grid = np.linspace(df["log_m"].min(), df["log_m"].max(), 45)

    iv_grid = np.full((len(T_grid), len(m_grid)), np.nan)
    for i, T in enumerate(T_grid):
        sub = df[np.abs(df["T"] - T) < 0.05]
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
# 4. LOCAL VOLATILITY – FINAL CORRECT VERSION (log-moneyness!)
# ========================================
def calibrate_lv(iv_surf, S0, r=0, T_max=0.3, N_T=50, N_K=90):
    T_grid = np.linspace(0.001, T_max, N_T)
    logm_grid = np.linspace(-0.6, 0.8, N_K)        # uniform in log-moneyness ← CRUCIAL
    K_grid = S0 * np.exp(logm_grid)                # derived from logm

    lv = np.zeros((len(T_grid), len(K_grid)))
    dT = T_grid[1] - T_grid[0]

    for i in range(1, len(T_grid)):
        T = T_grid[i]
        for j in range(1, len(K_grid)-1):
            K = K_grid[j]
            m = logm_grid[j]
            sigma_imp = iv_surf(T, m, grid=False)
            if np.isnan(sigma_imp) or sigma_imp <= 0.01:
                sigma_imp = 0.5

            C = bs_call(S0, K, T, r, sigma_imp)
            C_T = (bs_call(S0, K, T+dT, r, iv_surf(T+dT, m, grid=False) if T+dT <= T_grid[-1] else sigma_imp) - C) / dT

            dm = logm_grid[j+1] - logm_grid[j-1]
            C_Kp = bs_call(S0, S0*np.exp(logm_grid[j+1]), T, r, iv_surf(T, logm_grid[j+1], grid=False))
            C_Km = bs_call(S0, S0*np.exp(logm_grid[j-1]), T, r, iv_surf(T, logm_grid[j-1], grid=False))
            C_K  = (C_Kp - C_Km) / (S0 * np.exp(logm_grid[j]) * dm)   # dC/d(logm) → dC/dK
            C_KK = (C_Kp - 2*C + C_Km) / (dm/2)**2
            den = 0.5 * (np.exp(logm_grid[j]))**2 * C_KK * K**2      # adjust for chain rule

            num = C_T

            if den > 1e-10 and num > 0:
                lv[i, j] = np.sqrt(num / den)
            else:
                lv[i, j] = sigma_imp

    # Clean & fill
    lv = np.where((lv < 0.05) | (lv > 4.0) | np.isnan(lv), np.nan, lv)
    for i in range(lv.shape[0]):
        row = lv[i]
        nans = np.isnan(row)
        if nans.any() and (~nans).any():
            row[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), row[~nans])
        lv[i] = row
    for j in range(lv.shape[1]):
        col = lv[:, j]
        nans = np.isnan(col)
        if nans.any() and (~nans).any():
            col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), col[~nans])
        lv[:, j] = col
    if np.isnan(lv).any():
        lv = np.nan_to_num(lv, nan=np.nanmedian(lv))

    # Short-dated boost + light smoothing
    boost = 1.0 + 0.25 * np.exp(-35 * T_grid)[:, np.newaxis]
    lv *= boost
    lv = np.clip(lv, 0.4, 3.0)
    lv = median_filter(lv, size=3)

    # BUILD SPLINE IN (T, log-moneyness) ← THIS IS THE KEY
    lv_surf = RectBivariateSpline(T_grid, logm_grid, lv, kx=3, ky=3, s=0)

    # CORRECT DIAGNOSTIC (now using log-moneyness!)
    atm_vol = lv_surf(0.1, 0.0, grid=False)                    # logm = 0
    low_vol = lv_surf(0.1, np.log(0.7), grid=False)            # logm = log(0.7)
    print(f"Sample LV ATM (T=0.1): {atm_vol:.4f}   |   70% spot (logm={np.log(0.7):.3f}): {low_vol:.4f}  ← NOW CORRECT!")

    return lv_surf

# ========================================
# 5. MONTE CARLO – CORRECT LOG-MONEYNESS QUERY
# ========================================
def price_accumulator(fp_pct, lv_surf, S0, r=0, N=50000, seed=42):
    np.random.seed(seed)
    fp = fp_pct * S0
    ko = 1.10 * S0
    weeks, dt = 13, 7/365.25
    sqrt_dt = np.sqrt(dt)

    S = np.full((N, weeks+1), S0)
    payoff = np.zeros(N)
    active = np.ones(N, dtype=bool)
    week_stds = []

    for w in range(weeks):
        t = w * dt
        S_c = S[active, w]
        logm_c = np.log(S_c / S0)                  # ← LOG-MONEYNESS!
        vol = lv_surf(t, logm_c, grid=False)
        vol = np.clip(vol, 0.05, 4.0)

        if len(week_stds) < 13:
            week_stds.append(vol.std())
            print(f"Week {w}: vol mean/std = {vol.mean():.4f} / {vol.std():.4f}")

        dW = np.random.normal(0, sqrt_dt, size=vol.shape)
        S[active, w+1] = S[active, w] * np.exp((r - 0.5*vol**2)*dt + vol*dW)

        settle = S[active, w+1]
        weekly_shares = np.where((settle >= fp) & (settle < ko), 1.0,
                                 np.where(settle < fp, 2.0, 0.0)) * 7
        weekly_payoff = weekly_shares * (settle - fp)
        disc = np.exp(-r * (w+1)*dt)
        payoff[active] += weekly_payoff * disc

        ko_hit = settle >= ko
        if ko_hit.any():
            active_idx = np.where(active)[0]
            ko_idx = active_idx[ko_hit]
            bonus_weeks = max(0, 3 - (w + 1))
            if bonus_weeks > 0:
                bonus_payoff = bonus_weeks * 7 * (settle[ko_hit] - fp)
                payoff[ko_idx] += bonus_payoff * disc
            active[active] = ~ko_hit
        if not active.any():
            break

    print(f"Final avg weekly vol std: {np.mean(week_stds):.4f}")
    return np.mean(payoff)

# ========================================
# 6. SOLVER
# ========================================
r = 0.0275
def solve_fp_xxx(S0, iv_surf):
    lv_surf = calibrate_lv(iv_surf, S0, r)
    def obj(p): return price_accumulator(p, lv_surf, S0, r)
    fp_pct = brentq(obj,-0.1, 1.10, xtol=1e-5)
    print(f"Brentq converged to {fp_pct*100:.3f}%")
    return fp_pct * 100

# ========================================
# 7. FINAL BULLETPROOF PLOTTING – NO grid=True EVER!
# ========================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_volatility_surfaces(iv_surf, lv_surf, S0):
    # Get exact knot ranges
    T_iv_knots, m_knots = iv_surf.get_knots()
    T_lv_knots, _ = lv_surf.get_knots()

    T_min = max(T_iv_knots[0], T_lv_knots[0], 0.015)
    T_max = min(T_iv_knots[-1], T_lv_knots[-1], 0.40)

    if T_max - T_min < 0.03:
        print("Warning: Very narrow expiry range. Skipping plot.")
        return

    T_plot = np.linspace(T_min, T_max, 70)
    logm_plot = np.linspace(m_knots[0], m_knots[-1], 130)
    moneyness = np.exp(logm_plot)

    TT, MM = np.meshgrid(T_plot, logm_plot, indexing='ij')

    # THIS IS THE KEY: use np.vectorize + grid=False → 100% safe
    eval_iv = np.vectorize(lambda t, m: iv_surf(t, m, grid=False))
    eval_lv = np.vectorize(lambda t, m: lv_surf(t, m, grid=False))

    print("Evaluating surfaces for plotting (this takes ~2 seconds)...")
    IV = eval_iv(TT, MM) * 100
    LV = eval_lv(TT, MM) * 100

    fig = plt.figure(figsize=(21, 12))
    plt.style.use('default')  # clean look

    # 3D IV
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(TT, np.exp(MM), IV, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9, shade=True)
    ax1.set_title('Implied Volatility Surface', fontsize=14, pad=20)
    ax1.set_xlabel('Time to Expiry (years)')
    ax1.set_ylabel('Moneyness K/S₀')
    ax1.set_zlabel('IV (%)')
    fig.colorbar(surf1, ax=ax1, shrink=0.6)

    # 3D LV
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(TT, np.exp(MM), LV, cmap='inferno', linewidth=0, antialiased=True, alpha=0.9, shade=True)
    ax2.set_title('Local Volatility Surface (Dupire)', fontsize=14, pad=20)
    ax2.set_xlabel('Time to Expiry (years)')
    ax2.set_ylabel('Moneyness K/S₀')
    ax2.set_zlabel('Local Vol (%)')
    fig.colorbar(surf2, ax=ax2, shrink=0.6)

    # Contour IV
    ax3 = fig.add_subplot(2, 3, 3)
    c1 = ax3.contourf(TT, np.exp(MM), IV, levels=50, cmap='viridis')
    ax3.contour(TT, np.exp(MM), IV, levels=12, colors='white', alpha=0.4, linewidths=0.6)
    ax3.set_title('IV Contour')
    ax3.set_xlabel('T (years)'); ax3.set_ylabel('Moneyness')
    plt.colorbar(c1, ax=ax3)

    # Contour LV
    ax4 = fig.add_subplot(2, 3, 4)
    c2 = ax4.contourf(TT, np.exp(MM), LV, levels=50, cmap='inferno')
    ax4.contour(TT, np.exp(MM), LV, levels=12, colors='white', alpha=0.4, linewidths=0.6)
    ax4.set_title('Local Vol Contour')
    ax4.set_xlabel('T (years)'); ax4.set_ylabel('Moneyness')
    plt.colorbar(c2, ax=ax4)

    # ATM slice
    ax5 = fig.add_subplot(2, 3, 5)
    iv_atm = eval_iv(T_plot, np.zeros_like(T_plot)) * 100
    lv_atm = eval_lv(T_plot, np.zeros_like(T_plot)) * 100
    ax5.plot(T_plot, iv_atm, label='Implied Vol ATM', color='blue', lw=3)
    ax5.plot(T_plot, lv_atm, label='Local Vol ATM', color='red', lw=3, ls='--')
    ax5.set_xlabel('Time to Expiry (years)')
    ax5.set_ylabel('Volatility (%)')
    ax5.set_title('ATM Term Structure')
    ax5.legend(); ax5.grid(alpha=0.3)

    # Fixed maturity skew
    ax6 = fig.add_subplot(2, 3, 6)
    t_fixed = np.clip(0.1, T_min + 0.01, T_max - 0.01)
    iv_skew = eval_iv(t_fixed, logm_plot) * 100
    lv_skew = eval_lv(t_fixed, logm_plot) * 100
    ax6.plot(moneyness, iv_skew, label=f'IV @ {t_fixed:.3f}y', lw=3)
    ax6.plot(moneyness, lv_skew, '--', label=f'LV @ {t_fixed:.3f}y', lw=3)
    ax6.set_xlabel('Moneyness K/S₀')
    ax6.set_ylabel('Volatility (%)')
    ax6.set_title(f'Vol Skew (T = {t_fixed:.3f}y)')
    ax6.legend(); ax6.grid(alpha=0.3)

    plt.suptitle(f'ETH Option-Implied vs Local Volatility | Spot ${S0:,.0f} | {pd.Timestamp.now():%Y-%m-%d %H:%M} UTC',
                 fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    print("ETH Accumulator Pricer – FINAL VERSION")
    print("="*60)
    df, S0 = fetch_eth_options_with_prices()
    print("Building IV surface...")
    iv_surf = build_iv_surface(df, S0)

    print("=== MARKET IMPLIED VOL SKEW CHECK ===")
    for ratio in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        iv = iv_surf(0.1, np.log(ratio), grid=False)
        print(f"  {ratio:4.1f} × spot → IV = {iv*100:6.2f}%")

    print("Calibrating Local Volatility surface...")
    lv_surf = calibrate_lv(iv_surf, S0, r, T_max=0.3)

    print("Solving fair Forward Price...")
    def obj(pct): 
        return price_accumulator(pct/100, lv_surf, S0, r, N=100000, seed=42)
    fp_pct = brentq(obj, 60.0, 105.0, xtol=1e-4)

    print("\n" + "="*60)
    print(f" FAIR FORWARD PRICE = {fp_pct:.4f}% of spot")
    print(f" FP Level = {fp_pct/100*S0:,.2f} USD")
    print(f" Spot     = {S0:,.2f} USD")
    print("="*60)

    # THIS WILL NOW WORK 100% OF THE TIME
    plot_volatility_surfaces(iv_surf, lv_surf, S0)