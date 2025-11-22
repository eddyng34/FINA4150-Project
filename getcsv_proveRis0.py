import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")


def fetch_eth_options_with_prices():
    url_instr = "https://www.deribit.com/api/v2/public/get_instruments"
    params = {"currency": "ETH", "kind": "option", "expired": "false"}
    resp = requests.get(url_instr, params=params, timeout=15).json()
    if 'result' not in resp: raise ValueError(f"API error: {resp}")
    df = pd.DataFrame(resp['result'])

    url_price = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    resp_price = requests.get(url_price, params=params, timeout=15).json()
    if 'result' not in resp_price: raise ValueError(f"Price API error: {resp_price}")
    price_df = pd.DataFrame(resp_price["result"])

    # Save the raw url_price data to CSV (as before)
    csv_filename = 'eth_prices.csv'
    price_df.to_csv(csv_filename, index=False)
    print(f"Saved {len(price_df)} rows from url_price to {csv_filename}")

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


def parse_instrument(name):
    parts = name.split('-')
    if len(parts) < 4 or not parts[0] == 'ETH':
        raise ValueError(f"Invalid instrument name: {name}")
    expiry_str = parts[1]  # e.g., '28NOV25' or '5DEC25'
    strike = float(parts[2])
    opt_type = parts[3][0]  # 'C' or 'P'
    
    # Parse expiry_str: day (1-2 digits), month (3 letters), year (2 digits)
    i = 0
    day_str = ''
    while i < len(expiry_str) and expiry_str[i].isdigit():
        day_str += expiry_str[i]
        i += 1
    if not day_str:
        raise ValueError(f"No day in expiry: {expiry_str}")
    day = int(day_str)
    
    if i + 3 > len(expiry_str):
        raise ValueError(f"Invalid month in expiry: {expiry_str}")
    month_str = expiry_str[i:i+3].upper()
    i += 3
    
    year_str = expiry_str[i:]
    if len(year_str) != 2 or not year_str.isdigit():
        raise ValueError(f"Invalid year in expiry: {expiry_str}")
    year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
    
    months = {'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6,
              'JUL':7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
    month = months.get(month_str, None)
    if month is None:
        raise ValueError(f"Invalid month '{month_str}' in expiry: {expiry_str}")
    
    expiry = pd.Timestamp(year=year, month=month, day=day)
    return expiry, strike, opt_type


def black76_price(F, K, T, r, sigma, option_type):
    if T <= 0:
        if option_type == 'C':
            return max(F - K, 0)
        else:
            return max(K - F, 0)
    d1 = (np.log(F / K) + (sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'C':
        forward = F * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        forward = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    return np.exp(-r * T) * forward


def implied_r(F, K, T, sigma, target, option_type):
    def objective(r):
        return black76_price(F, K, T, r, sigma, option_type) - target
    try:
        r = brentq(objective, -2, 2)  # Search in [-200%, 200%]
        return r
    except ValueError:
        # If no root in bounds, try wider or return NaN
        try:
            r = brentq(objective, -10, 10)
            return r
        except:
            return np.nan


# Fetch and process the price data
params = {"currency": "ETH", "kind": "option", "expired": "false"}
url_price = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
resp_price = requests.get(url_price, params=params, timeout=15).json()
if 'result' not in resp_price: raise ValueError(f"Price API error: {resp_price}")
price_df = pd.DataFrame(resp_price["result"])

# Parse instruments and compute T (using provided date: Nov 22, 2025)
today = pd.Timestamp('2025-11-22')
price_df['expiry'], price_df['strike'], price_df['option_type'] = zip(*[parse_instrument(name) for name in price_df['instrument_name']])
price_df['T'] = (price_df['expiry'] - today).dt.total_seconds() / (365.25 * 24 * 3600)
price_df = price_df[(price_df['T'] > 0) & (price_df['mark_iv'] > 0) & (price_df['underlying_price'] > 0)]  # Filter valid

# Compute implied r for each
def compute_r(row):
    F = row['underlying_price']
    K = row['strike']
    T = row['T']
    sigma = row['mark_iv'] / 100
    premium_eth = row['mark_price']
    target = premium_eth * F
    opt_type = row['option_type']
    return implied_r(F, K, T, sigma, target, opt_type)

print("Computing implied r for all options... (this may take a minute)")
price_df['implied_r'] = price_df.apply(compute_r, axis=1)

# Save with new column
output_csv = 'eth_options_with_r.csv'
price_df.to_csv(output_csv, index=False)
print(f"\nSaved {len(price_df)} rows (with implied_r column) to {output_csv}")

# Summary
valid_rs = price_df['implied_r'].dropna()
print(f"\nImplied r summary (for {len(valid_rs)} valid options):")
print(valid_rs.describe())
print("\nSample rows:")
print(price_df[['instrument_name', 'option_type', 'strike', 'underlying_price', 'mark_price', 'mark_iv', 'T', 'implied_r']].head(10))


# Run the computation
if __name__ == "__main__":
    # fetch_eth_options_with_prices()  # Optional: if you want the merged df too
    pass