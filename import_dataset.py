import yfinance as yf
import pandas as pd

# Download free data for SPY (S&P 500 ETF)
data = yf.download("SPY", start="2010-01-01", end="2025-11-12", progress=False)

# Display the first few rows
print(data.head())

# Save to CSV
data.to_csv("SPY_data.csv")

print("Dataset saved as SPY_data.csv")
