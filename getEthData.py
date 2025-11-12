import requests
import pandas as pd

# Deribit: Get ETH options (adapt for your date)
url = "https://www.deribit.com/api/v2/public/get_instruments?currency=ETH&kind=option&expired=false"
response = requests.get(url).json()
df = pd.DataFrame(response['result'])
df['expiration_timestamp'] = pd.to_datetime(df['expiration_timestamp'], unit='ms')
df = df[df['expiration_timestamp'] > pd.Timestamp.now()]  # Future expiries
df.to_csv('eth_options.csv', index=False)  # Save chain