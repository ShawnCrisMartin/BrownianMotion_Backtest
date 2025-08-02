import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from browniondot import generate_brownian_motion

ticker = "AAPL"
period = "2y"
simulation_days = 504
S0 = 100
num_paths = 1

stock_data = yf.download(ticker, period=period)
close_prices = stock_data['Close']
stock_returns = np.log(close_prices / close_prices.shift(1)).dropna()
real_volatility = stock_returns.rolling(window=21).std() * np.sqrt(252)

bm_paths = generate_brownian_motion(S0, simulation_days, num_paths)
simulated_path = bm_paths[0]

simulated_returns = np.log(simulated_path[1:] / simulated_path[:-1])
sim_volatility = np.convolve(simulated_returns**2, np.ones(21)/21, mode='valid') ** 0.5
sim_volatility_annualized = sim_volatility * np.sqrt(252)

plt.figure(figsize=(12, 6))
plt.plot(real_volatility.values, label=f"{ticker} Realized Volatility (21-day rolling)", color='blue')
plt.plot(sim_volatility_annualized, label="Simulated Brownian Volatility", color='orange', linestyle='--')
plt.title(f"Volatility Comparison: {ticker} vs Simulated Brownian Motion")
plt.xlabel("Days")
plt.ylabel("Annualized Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
