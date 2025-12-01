import re
import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings("ignore")
from scipy.ndimage import gaussian_filter

# ========================================
# 1. Load Options Data from CSV
# ========================================
def fetch_eth_options_from_excel(file_path, spot=3050.0):
    df = pd.read_excel(file_path)

    # Extract strike from instrument name, e.g., ETH-19NOV25-2800-C → 2800
    def extract_strike(instr):
        m = re.search(r'-(\d+)-[CP]$', instr)
        return float(m.group(1)) if m else np.nan

    df['K'] = df['Instrument'].apply(extract_strike)

    # Load bid and ask prices
    df['bid'] = pd.to_numeric(df['Bid'], errors='coerce')
    df['ask'] = pd.to_numeric(df['Ask'], errors='coerce')

    # IV: bid, ask (raw percentages)
    df['bid_iv'] = pd.to_numeric(df['IV Bid'], errors='coerce')
    df['ask_iv'] = pd.to_numeric(df['IV Ask'], errors='coerce')

    # Filter valid rows for bid and ask separately
    df_bid = df.dropna(subset=['K', 'bid', 'bid_iv']).copy()
    df_bid = df_bid[df_bid['bid'] > 1e-6]
    df_ask = df.dropna(subset=['K', 'ask', 'ask_iv']).copy()
    df_ask = df_ask[df_ask['ask'] > 1e-6]

    # Compute time to expiry T (extract from Instrument)
    # e.g., ETH-19NOV25-2800-C → expiry 2025-11-19
    def extract_expiry(instr):
        m = re.search(r'ETH-(\d{2}[A-Z]{3}\d{2})-\d+-[CP]$', instr)
        if not m: return pd.NaT
        date_str = m.group(1)
        return pd.to_datetime(date_str, format='%d%b%y', errors='coerce')

    df_bid['expiry'] = df_bid['Instrument'].apply(extract_expiry)
    today = pd.Timestamp.now().normalize()
    df_bid['T'] = (df_bid['expiry'] - today).dt.total_seconds() / (365.25*24*3600)
    df_bid = df_bid[(df_bid['T'] > 0.01) & (df_bid['T'] <= 1.0)]

    df_ask['expiry'] = df_ask['Instrument'].apply(extract_expiry)
    df_ask['T'] = (df_ask['expiry'] - today).dt.total_seconds() / (365.25*24*3600)
    df_ask = df_ask[(df_ask['T'] > 0.01) & (df_ask['T'] <= 1.0)]

    print(f"Loaded {len(df_bid)} ETH options (bid) from Excel. Spot = {spot:,.2f} USD")
    print(f"Loaded {len(df_ask)} ETH options (ask) from Excel. Spot = {spot:,.2f} USD")
    return df_bid, df_ask, spot

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
def build_iv_surface(df, S0, r=0, iv_col='bid_iv'):
    df['IV'] = df[iv_col] / 100
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
# 4. LOCAL VOLATILITY SURFACE (log-moneyness)
# ========================================
def calibrate_lv(iv_surf, S0, r=0.0, T_max=0.50, N_T=50, N_K=90, q=0.0):
    """
    Calibrate local volatility surface (Dupire) on a grid uniform in log-moneyness
    - iv_surf(t, m, grid=False) expects t = time, m = log(K/S0)
    - returns a RectBivariateSpline defined on (T_grid, logm_grid)
    """
    # build grids: use iv_surf knots to be safe if available
    try:
        _, m_knots = iv_surf.get_knots()
        m_min, m_max = m_knots[0], m_knots[-1]
    except Exception:
        m_min, m_max = -0.9, 0.9

    #T_grid = np.linspace(max(1e-4, 1e-3), min(T_max, 1.0), N_T)
    T_grid = np.linspace(0.01, T_max, N_T)
    logm_grid = np.linspace(m_min, m_max, N_K)
    K_grid = S0 * np.exp(logm_grid)

    lv = np.full((len(T_grid), len(logm_grid)), np.nan)

    # finite-difference steps
    # Use central differences where possible; fallback to forward/backward on edges
    # small thresholds
    eps_den = 1e-12
    min_vol = 0.02
    max_vol = 4.0

    for i, T in enumerate(T_grid):
        # choose dT relative to T
        dT = max(1e-4, (T_grid[1] - T_grid[0]) if len(T_grid) > 1 else 1e-3)
        for j, m in enumerate(logm_grid):
            K = K_grid[j]

            # get implied vols (central difference in m and T where possible)
            try:
                sig = float(iv_surf(T, m, grid=False))
            except Exception:
                sig = np.nan
            if np.isnan(sig) or sig <= 1e-3:
                # fallback to median-ish value
                sig = 0.5

            # --- sigma T derivative (central) ---
            # choose points safely within grid bounds
            T_plus = T + dT if T + dT <= T_grid[-1] else T
            T_minus = T - dT if T - dT >= T_grid[0] else T

            sig_plusT = float(iv_surf(T_plus, m, grid=False)) if T_plus != T else sig
            sig_minusT = float(iv_surf(T_minus, m, grid=False)) if T_minus != T else sig
            sig_T = (sig_plusT - sig_minusT) / (T_plus - T_minus) if (T_plus - T_minus) > 0 else 0.0

            # --- sigma m derivative (central) ---
            dm = logm_grid[1] - logm_grid[0] if len(logm_grid) > 1 else 1e-3
            m_plus = logm_grid[j+1] if j+1 < len(logm_grid) else m
            m_minus = logm_grid[j-1] if j-1 >= 0 else m
            sig_plusm = float(iv_surf(T, m_plus, grid=False)) if m_plus != m else sig
            sig_minusm = float(iv_surf(T, m_minus, grid=False)) if m_minus != m else sig
            # effective dm for edges
            dm_eff = (m_plus - m_minus) if (m_plus - m_minus) > 0 else dm
            sig_m = (sig_plusm - sig_minusm) / dm_eff if dm_eff > 0 else 0.0
            sig_mm = (sig_plusm - 2*sig + sig_minusm) / ( (dm_eff/2)**2 ) if dm_eff > 0 else 0.0

            # --- Black-Scholes quantities for C, vega, theta ---
            # compute d1,d2 for the current point (protect T==0)
            T_safe = max(T, 1e-8)
            sqrtT = np.sqrt(T_safe)
            # d1,d2 for implied vol 'sig'
            d1 = (np.log(S0 / K) + (r - q + 0.5 * sig**2) * T_safe) / (sig * sqrtT)
            d2 = d1 - sig * sqrtT
            # price and Greeks
            # call price
            C = bs_call(S0, K, T_safe, r, sig, q=q)
            # vega (BS vega wrt vol)
            vega = S0 * norm.pdf(d1) * sqrtT * np.exp(-q * T_safe)
            # theta (BS theta: derivative wrt T) - using standard closed form
            # Note: theta here is total derivative of BS call price wrt T (not annualised)
            theta = (-S0 * norm.pdf(d1) * sig * np.exp(-q * T_safe) / (2 * sqrtT)
                     - r * K * np.exp(-r * T_safe) * norm.cdf(d2)
                     + q * S0 * np.exp(-q * T_safe) * norm.cdf(d1))

            # stable C_T using theta + vega * d(sigma)/dT
            C_T = theta + vega * sig_T

            # --- compute C(m +/-) prices for finite diff in m ---
            # use implied vols at m+ and m- (we already have sig_plusm, sig_minusm)
            K_plus = S0 * np.exp(m_plus)
            K_minus = S0 * np.exp(m_minus)
            C_plus = bs_call(S0, K_plus, T_safe, r, sig_plusm, q=q)
            C_minus = bs_call(S0, K_minus, T_safe, r, sig_minusm, q=q)

            # derivatives wrt m
            # dC/dm (central)
            C_m = (C_plus - C_minus) / dm_eff if dm_eff > 0 else 0.0
            # d2C/dm2
            C_mm = (C_plus - 2*C + C_minus) / ( (dm_eff/2)**2 ) if dm_eff > 0 else 0.0

            # chain rule to strike derivatives
            # dC/dK = (1/K) * dC/dm
            C_K = C_m / K if K != 0 else 0.0
            # d2C/dK2 = (1/K^2) * (d2C/dm2 - dC/dm)
            C_KK = (C_mm - C_m) / (K**2) if K != 0 else 0.0

            # Dupire denominator: 0.5 * K^2 * C_KK (standard)
            den = 0.5 * (K**2) * C_KK

            # numerator: C_T
            num = C_T

            # stability checks: require den > small and num > 0
            if den > eps_den and num > 0:
                val = num / den
                if val > 0:
                    lv_val = np.sqrt(val)
                else:
                    lv_val = sig  # fallback
            else:
                lv_val = sig  # fallback to implied vol

            # safety clamp
            lv_val = np.clip(lv_val, min_vol, max_vol)
            lv[i, j] = lv_val

    # Fill any remaining NaNs by interpolation along each row/col
    for i in range(lv.shape[0]):
        row = lv[i, :]
        nans = np.isnan(row)
        if nans.any() and (~nans).any():
            valid_x = np.flatnonzero(~nans)
            valid_y = row[~nans]
            interp_idx = np.flatnonzero(nans)
            row[nans] = np.interp(interp_idx, valid_x, valid_y)
            lv[i, :] = row

    for j in range(lv.shape[1]):
        col = lv[:, j]
        nans = np.isnan(col)
        if nans.any() and (~nans).any():
            valid_x = np.flatnonzero(~nans)
            valid_y = col[~nans]
            interp_idx = np.flatnonzero(nans)
            col[nans] = np.interp(interp_idx, valid_x, valid_y)
            lv[:, j] = col

    # final fallback
    if np.isnan(lv).any():
        lv = np.nan_to_num(lv, nan=np.nanmedian(lv))

    # light smoothing, then clip to realistic bounds
    lv = median_filter(lv, size=(3, 3))
    # lv = np.clip(lv, 0.05, 3.0)

    lv = gaussian_filter(lv, sigma=(1.8, 2.5))

    # # stronger wing smoothing to kill spikes
    # for j in [0, 1, 2, 3, -3, -2, -1]:
    #     lv[:, j] = median_filter(lv[:, j], size=7)

    # build spline in (T, log-moneyness)
    lv_surf = RectBivariateSpline(T_grid, logm_grid, lv, kx=3, ky=3, s=0)

    # diagnostic: ATM and wings
    try:
        atm_vol = float(lv_surf(max(1e-4, min(0.1, T_grid[-1])), 0.0, grid=False))
        wing_vol = float(lv_surf(max(1e-4, min(0.1, T_grid[-1])), np.log(0.7), grid=False))
        print(f"Sample LV ATM (T~0.1): {atm_vol:.4f}   |   70% spot (logm={np.log(0.7):.3f}): {wing_vol:.4f}")
    except Exception:
        pass

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
        logm_c = np.log(S_c / S0)   # log-moneyness
        vol = lv_surf(t, logm_c, grid=False)
        vol = np.clip(vol, 0.05, 4.0)

        if len(week_stds) < 13:
            week_stds.append(vol.std())
            #print(f"Week {w}: vol mean/std = {vol.mean():.4f} / {vol.std():.4f}")

        dW = np.random.normal(0, sqrt_dt, size=vol.shape)
        S[active, w+1] = S[active, w] * np.exp((r - 0.5*vol**2)*dt + vol*dW)

        settle = S[active, w+1]
        weekly_shares = np.where(settle >= fp, 7.0, 14.0)
        weekly_payoff = weekly_shares * (settle - fp)
        disc = np.exp(-r * (w+1)*dt)
        payoff[active] += weekly_payoff * disc

        # Knock-Out logic: only active from week 4 onward (w starts from 0)
        if w >= 3:   # i.e., weeks 4 to 13 (w = 3,4,...,12)
            ko_hit = settle >= ko
            if ko_hit.any():
                # Paths that knock out: deactivate them (no more accumulation)
                active[active] = ~ko_hit
                if not active.any():
                    break
        # For weeks 0,1,2 (w=0,1,2): KO is completely ignored → nothing here

    #print(f"Final avg weekly vol std: {np.mean(week_stds):.4f}")
    return np.mean(payoff)

# ========================================
# 6. SOLVER
# ========================================
r = 0 # Justified by getcsv_proveRis0.py
def solve_fp_xxx(S0, iv_surf, label):
    lv_surf = calibrate_lv(iv_surf, S0, r)
    def obj(p): return price_accumulator(p, lv_surf, S0, r)
    fp_pct = brentq(obj,-0.1, 1.10, xtol=1e-5)
    print(f"Brentq converged to {fp_pct*100:.3f}% ({label})")
    return fp_pct * 100

# ========================================
# 7. PLOTTING
# ========================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_volatility_surfaces(iv_surf, lv_surf, S0):
    # Get exact knot ranges
    T_iv_knots, m_knots = iv_surf.get_knots()
    T_lv_knots, _ = lv_surf.get_knots()

    T_min = max(T_iv_knots[0], T_lv_knots[0], 0.015)
    T_max = min(T_iv_knots[-1], T_lv_knots[-1], 0.50)

    if T_max - T_min < 0.03:
        print("Warning: Very narrow expiry range. Skipping plot.")
        return

    T_plot = np.linspace(T_min, T_max, 70)
    logm_plot = np.linspace(m_knots[0], m_knots[-1], 130)
    moneyness = np.exp(logm_plot)

    TT, MM = np.meshgrid(T_plot, logm_plot, indexing='ij')

    # use np.vectorize + grid=False to be 100% safe
    eval_iv = np.vectorize(lambda t, m: iv_surf(t, m, grid=False))
    eval_lv = np.vectorize(lambda t, m: lv_surf(t, m, grid=False))

    print("Evaluating surfaces for plotting...")
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

    # # Contour IV
    # ax3 = fig.add_subplot(2, 3, 3)
    # c1 = ax3.contourf(TT, np.exp(MM), IV, levels=50, cmap='viridis')
    # ax3.contour(TT, np.exp(MM), IV, levels=12, colors='white', alpha=0.4, linewidths=0.6)
    # ax3.set_title('IV Contour')
    # ax3.set_xlabel('T (years)'); ax3.set_ylabel('Moneyness')
    # plt.colorbar(c1, ax=ax3)

    # # Contour LV
    # ax4 = fig.add_subplot(2, 3, 4)
    # c2 = ax4.contourf(TT, np.exp(MM), LV, levels=50, cmap='inferno')
    # ax4.contour(TT, np.exp(MM), LV, levels=12, colors='white', alpha=0.4, linewidths=0.6)
    # ax4.set_title('Local Vol Contour')
    # ax4.set_xlabel('T (years)'); ax4.set_ylabel('Moneyness')
    # plt.colorbar(c2, ax=ax4)

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

    print("ETH Accumulator Pricer")
    print("="*60)

    # Load CSV
    csv_file = "ETH-Option Data.xlsx"
    df_bid, df_ask, S0 = fetch_eth_options_from_excel(csv_file, spot=3050.0)

    print("Building bid IV surface...")
    iv_surf_bid = build_iv_surface(df_bid, S0, iv_col='bid_iv')

    print("=== MARKET IMPLIED VOL SKEW CHECK (Bid IV) ===")
    for ratio in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
         iv = iv_surf_bid(0.1, np.log(ratio), grid=False)
         print(f"  {ratio:4.1f} × spot → IV = {iv*100:6.2f}%")

    print("Building ask IV surface...")
    iv_surf_ask = build_iv_surface(df_ask, S0, iv_col='ask_iv')

    print("=== MARKET IMPLIED VOL SKEW CHECK (Ask IV) ===")
    for ratio in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
         iv = iv_surf_ask(0.1, np.log(ratio), grid=False)
         print(f"  {ratio:4.1f} × spot → IV = {iv*100:6.2f}%")

    print("Calibrating Local Volatility surface (bid)...")
    lv_surf_bid = calibrate_lv(iv_surf_bid, S0, r, T_max=0.3)

    print("Calibrating Local Volatility surface (ask)...")
    lv_surf_ask = calibrate_lv(iv_surf_ask, S0, r, T_max=0.3)

    print("Solving fair Forward Price (bid)...")
    def obj_bid(pct): 
        return price_accumulator(pct/100, lv_surf_bid, S0, r, N=100000, seed=42)
    fp_bid_pct = brentq(obj_bid, 60.0, 105.0, xtol=1e-4)

    print("Solving fair Forward Price (ask)...")
    def obj_ask(pct): 
        return price_accumulator(pct/100, lv_surf_ask, S0, r, N=100000, seed=42)
    fp_ask_pct = brentq(obj_ask, 60.0, 105.0, xtol=1e-4)

    print("\n" + "="*60)
    print(f" BID FORWARD PRICE (bid price & bid_iv) = {fp_bid_pct:.4f}% of spot")
    print(f" ASK FORWARD PRICE (ask price & ask_iv) = {fp_ask_pct:.4f}% of spot")
    print(f" Bid-Ask Spread (bps)                   = {(fp_ask_pct - fp_bid_pct)*10000:.0f}")
    print(f" FP Level (Bid)                         = {fp_bid_pct/100*S0:,.2f} USD")
    print(f" FP Level (Ask)                         = {fp_ask_pct/100*S0:,.2f} USD")
    print(f" Spot                                   = {S0:,.2f} USD")
    print("="*60)

    # Plot IV and LV surfaces (using bid as example)
    plot_volatility_surfaces(iv_surf_ask, lv_surf_ask, S0)