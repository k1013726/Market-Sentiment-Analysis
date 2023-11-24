import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pandas as pd

# google stock price is daily, so
seasonal_period=7
df = pd.read_csv('WekaData/ETH-N2Y.csv')
#df['Close'] = np.log(df['Close'])
# Decompose the 'Close' column #multiplicative or additive
decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=seasonal_period)

# Extract the components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the components
fig, axes = plt.subplots(4, 1, figsize=(12, 10))  # Set the figure size

# Plot the original series
axes[0].plot(df['Close'], linewidth=2)  # Set the line width (point size)
axes[0].set_title('Original')

# Plot the trend component
axes[1].plot(trend, linewidth=2)
axes[1].set_title('Trend')

# Plot the seasonal component
axes[2].plot(seasonal, linewidth=2)
axes[2].set_title('Seasonal')

# Plot the residual component
axes[3].scatter(residual.index, residual, s=8)
axes[3].axhline(y=0, color='k', linestyle='-', linewidth=2)  # Horizontal line at y=0
axes[3].set_title('Residual')

plt.tight_layout()
plt.show()