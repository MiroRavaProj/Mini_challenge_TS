# %% [markdown]
# # Stationarity Tests for Time Series Analysis

# %% [markdown]
# ## Import Required Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Stationarity Test Functions

# %%
def adf_test(series, title='', print_results=True):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    
    Parameters:
    -----------
    series : pd.Series or array-like
        Time series data to test
    title : str, optional
        Title for the series being tested
    print_results : bool, optional
        Whether to print results
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    # Format the results
    output = {
        'test_statistic': result[0],
        'p_value': result[1],
        'lags': result[2],
        'n_observations': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05
    }
    
    if print_results:
        print(f"Augmented Dickey-Fuller Test on {title}")
        print("--------------------------------------")
        print(f'ADF Test Statistic: {output["test_statistic"]:.4f}')
        print(f'p-value: {output["p_value"]:.4f}')
        print(f'Number of lags used: {output["lags"]}')
        print(f'Number of observations: {output["n_observations"]}')
        
        print("\nCritical Values:")
        for key, value in output["critical_values"].items():
            print(f'\t{key}: {value:.4f}')
            
        print("\nResult:")
        if output["is_stationary"]:
            print(f"The series '{title}' is stationary (reject H0)")
        else:
            print(f"The series '{title}' is non-stationary (fail to reject H0)")
        print("\n")
        
    return output

def kpss_test(series, title='', print_results=True):
    """
    Perform KPSS test for stationarity
    
    Parameters:
    -----------
    series : pd.Series or array-like
        Time series data to test
    title : str, optional
        Title for the series being tested
    print_results : bool, optional
        Whether to print results
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    # KPSS test with null hypothesis: the series is stationary
    result = kpss(series.dropna(), regression='c', nlags='auto')
    
    # Format the results
    output = {
        'test_statistic': result[0],
        'p_value': result[1],
        'lags': result[2],
        'critical_values': result[3],
        'is_stationary': result[1] > 0.05  # Note: For KPSS, high p-value suggests stationarity
    }
    
    if print_results:
        print(f"KPSS Test on {title}")
        print("--------------------------------------")
        print(f'KPSS Test Statistic: {output["test_statistic"]:.4f}')
        print(f'p-value: {output["p_value"]:.4f}')
        print(f'Number of lags used: {output["lags"]}')
        
        print("\nCritical Values:")
        for key, value in output["critical_values"].items():
            print(f'\t{key}: {value:.4f}')
            
        print("\nResult:")
        if output["is_stationary"]:
            print(f"The series '{title}' is stationary (fail to reject H0)")
        else:
            print(f"The series '{title}' is non-stationary (reject H0)")
        print("\n")
        
    return output

# %%
def check_stationarity(series, title='', plot=True, print_results=True):
    """
    Comprehensive stationarity check using both ADF and KPSS tests
    
    Parameters:
    -----------
    series : pd.Series or array-like
        Time series data to test
    title : str, optional
        Title for the series being tested
    plot : bool, optional
        Whether to plot the series
    print_results : bool, optional
        Whether to print results
        
    Returns:
    --------
    tuple
        Tuple containing results from both tests
    """
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(series)
        plt.title(f'Time Series Plot: {title}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
        
        # Rolling statistics
        rolling_mean = series.rolling(window=12).mean()
        rolling_std = series.rolling(window=12).std()
        
        plt.figure(figsize=(12, 6))
        plt.plot(series, label='Original')
        plt.plot(rolling_mean, label='Rolling Mean')
        plt.plot(rolling_std, label='Rolling Std')
        plt.title(f'Rolling Statistics: {title}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Run both tests
    adf_results = adf_test(series, title, print_results)
    kpss_results = kpss_test(series, title, print_results)
    
    # Combined interpretation
    if print_results:
        print(f"Combined Interpretation for {title}")
        print("--------------------------------------")
        if adf_results['is_stationary'] and kpss_results['is_stationary']:
            print("Both tests indicate the series is STATIONARY")
        elif not adf_results['is_stationary'] and not kpss_results['is_stationary']:
            print("Both tests indicate the series is NON-STATIONARY")
        elif adf_results['is_stationary'] and not kpss_results['is_stationary']:
            print("ADF indicates STATIONARY but KPSS indicates NON-STATIONARY")
            print("This might suggest a trend-stationary series")
        else:
            print("ADF indicates NON-STATIONARY but KPSS indicates STATIONARY")
            print("This result is somewhat inconclusive and may require further investigation")
        print("\n")
    
    return adf_results, kpss_results

def apply_differencing(series, order=1, title='', plot=True):
    """
    Apply differencing to a time series
    
    Parameters:
    -----------
    series : pd.Series or array-like
        Time series data to difference
    order : int, optional
        Order of differencing
    title : str, optional
        Title for the series being differenced
    plot : bool, optional
        Whether to plot the differenced series
        
    Returns:
    --------
    pd.Series
        Differenced series
    """
    differenced = series.diff(order).dropna()
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(differenced)
        plt.title(f'Differenced ({order}) Series: {title}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
    
    return differenced

# %% [markdown]
# ## Example Usage with Synthetic Data

# %%
# Example with synthetic data
np.random.seed(42)

# Non-stationary series with trend
n = 1000
t = np.arange(n)
non_stationary = 0.1 * t + 5 * np.sin(t/100) + np.random.normal(0, 1, n)
non_stationary_series = pd.Series(non_stationary)

# Check stationarity
check_stationarity(non_stationary_series, 'Non-Stationary Series')

# Apply differencing and check again
diff_series = apply_differencing(non_stationary_series, title='Non-Stationary Series')
check_stationarity(diff_series, 'Differenced Series')

# %% [markdown]
# ## Usage with Hotels Dataset

# %%
# Load hotel data
import pickle
with open('hotels.pk', 'rb') as f:
    hotels = pickle.load(f)

# Filter for Chicago
city = "Chicago"
city_data = hotels[(hotels['Location'] == city)]

# Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(
    scaler.fit_transform(city_data[['Revenue', 'Demand', 'Occupancy']]), 
    columns=['Revenue', 'Demand', 'Occupancy'], 
    index=city_data.index
)

# %% [markdown]
# ## Test Stationarity for Each Column

# %%
# Check Revenue
check_stationarity(normalized_data['Revenue'], f'{city} Revenue')
diff_series = apply_differencing(normalized_data['Revenue'], title=f'{city} Revenue')
check_stationarity(diff_series, f'Differenced {city} Revenue')

# %%
# Check Demand
check_stationarity(normalized_data['Demand'], f'{city} Demand')
diff_series = apply_differencing(normalized_data['Demand'], title=f'{city} Demand')
check_stationarity(diff_series, f'Differenced {city} Demand')

# %%
# Check Occupancy
check_stationarity(normalized_data['Occupancy'], f'{city} Occupancy')
diff_series = apply_differencing(normalized_data['Occupancy'], title=f'{city} Occupancy')
check_stationarity(diff_series, f'Differenced {city} Occupancy') 