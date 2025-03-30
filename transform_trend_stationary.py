# %% [markdown]
# # Transformations for Trend-Stationary Series

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Load and Visualize the Data

# %%
# Load data
import pickle
with open('hotels.pk', 'rb') as f:
    hotels = pickle.load(f)

# Filter for Chicago
city = "Chicago"
city_data = hotels[hotels['Location'] == city]

# Plot the revenue
plt.figure(figsize=(12, 6))
plt.plot(city_data['Revenue'])
plt.title('Chicago Revenue')
plt.grid(True)
plt.show()

# %% [markdown]
# ## Quick Stationarity Test Function

# %%
def quick_stationarity_test(series, title):
    # ADF Test
    adf_result = adfuller(series.dropna(), autolag='AIC')
    adf_pvalue = adf_result[1]
    adf_stationary = adf_pvalue <= 0.05
    
    # KPSS Test
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    kpss_pvalue = kpss_result[1]
    kpss_stationary = kpss_pvalue > 0.05
    
    print(f"Stationarity Test Results for: {title}")
    print("-" * 50)
    print(f"ADF p-value: {adf_pvalue:.4f} ({'Stationary' if adf_stationary else 'Non-stationary'})")
    print(f"KPSS p-value: {kpss_pvalue:.4f} ({'Stationary' if kpss_stationary else 'Non-stationary'})")
    
    if adf_stationary and kpss_stationary:
        print("\n✓ CONCLUSION: Series is STATIONARY (both tests agree)")
    elif not adf_stationary and not kpss_stationary:
        print("\n✗ CONCLUSION: Series is NON-STATIONARY (both tests agree)")
    elif adf_stationary and not kpss_stationary:
        print("\n? CONCLUSION: Series may be TREND-STATIONARY (conflicting results)")
    else:
        print("\n? CONCLUSION: Inconclusive results (conflicting tests)")
    
    return adf_stationary and kpss_stationary

# %% [markdown]
# ## 2. Linear Trend Removal

# %%
def remove_linear_trend(series):
    # Create a time index
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Extract trend
    trend = model.predict(X)
    
    # Remove trend
    detrended = series - trend
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(311)
    plt.plot(series)
    plt.plot(series.index, trend)
    plt.title('Original Series with Fitted Trend')
    plt.grid(True)
    
    plt.subplot(312)
    plt.plot(series.index, trend)
    plt.title('Trend Component')
    plt.grid(True)
    
    plt.subplot(313)
    plt.plot(detrended)
    plt.title('Detrended Series')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return detrended, trend

# Apply linear detrending
detrended_linear, linear_trend = remove_linear_trend(city_data['Revenue'])

# Test stationarity of detrended series
quick_stationarity_test(detrended_linear, "Linear Detrended Chicago Revenue")

# %% [markdown]
# ## 3. Polynomial Trend Removal

# %%
def remove_polynomial_trend(series, degree=2):
    # Create a time index
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    
    # Create polynomial features
    X_poly = np.hstack([X ** i for i in range(1, degree + 1)])
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Extract trend
    trend = model.predict(X_poly)
    
    # Remove trend
    detrended = series - trend
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(311)
    plt.plot(series)
    plt.plot(series.index, trend)
    plt.title(f'Original Series with Fitted Polynomial Trend (degree={degree})')
    plt.grid(True)
    
    plt.subplot(312)
    plt.plot(series.index, trend)
    plt.title('Polynomial Trend Component')
    plt.grid(True)
    
    plt.subplot(313)
    plt.plot(detrended)
    plt.title('Detrended Series')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return detrended, trend

# Apply polynomial detrending
detrended_poly, poly_trend = remove_polynomial_trend(city_data['Revenue'], degree=3)

# Test stationarity
quick_stationarity_test(detrended_poly, "Polynomial Detrended Chicago Revenue")

# %% [markdown]
# ## 4. Moving Average Trend Removal

# %%
def remove_ma_trend(series, window=12):
    # Calculate moving average
    trend = series.rolling(window=window, center=True).mean()
    
    # Fill NAs at the beginning and end
    trend = trend.fillna(method='bfill').fillna(method='ffill')
    
    # Remove trend
    detrended = series - trend
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(311)
    plt.plot(series)
    plt.plot(trend)
    plt.title(f'Original Series with Moving Average Trend (window={window})')
    plt.grid(True)
    
    plt.subplot(312)
    plt.plot(trend)
    plt.title('Moving Average Trend Component')
    plt.grid(True)
    
    plt.subplot(313)
    plt.plot(detrended)
    plt.title('Detrended Series')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return detrended, trend

# Apply MA detrending with a monthly window
detrended_ma, ma_trend = remove_ma_trend(city_data['Revenue'], window=30)

# Test stationarity
quick_stationarity_test(detrended_ma, "Moving Average Detrended Chicago Revenue")

# %% [markdown]
# ## 5. Seasonal Decomposition

# %%
def apply_seasonal_decomposition(series, period=7):
    # Apply seasonal decomposition
    result = seasonal_decompose(series, model='additive', period=period)
    
    # Plot the decomposition
    plt.figure(figsize=(12, 10))
    
    plt.subplot(411)
    plt.plot(series)
    plt.title('Original Series')
    plt.grid(True)
    
    plt.subplot(412)
    plt.plot(result.trend)
    plt.title('Trend Component')
    plt.grid(True)
    
    plt.subplot(413)
    plt.plot(result.seasonal)
    plt.title('Seasonal Component')
    plt.grid(True)
    
    plt.subplot(414)
    plt.plot(result.resid)
    plt.title('Residual Component')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return result

# Weekly seasonality
decomposition_weekly = apply_seasonal_decomposition(city_data['Revenue'], period=7)

# Test stationarity of the residual component
residuals_weekly = decomposition_weekly.resid.dropna()
quick_stationarity_test(residuals_weekly, "Residuals after Weekly Seasonal Decomposition")

# Monthly seasonality (30 days)
decomposition_monthly = apply_seasonal_decomposition(city_data['Revenue'], period=30)
residuals_monthly = decomposition_monthly.resid.dropna()
quick_stationarity_test(residuals_monthly, "Residuals after Monthly Seasonal Decomposition")

# %% [markdown]
# ## 6. Compare All Transformations

# %%
def compare_transformations(transformations_dict):
    results = {}
    
    for name, series in transformations_dict.items():
        # ADF Test
        adf_result = adfuller(series.dropna(), autolag='AIC')
        adf_pvalue = adf_result[1]
        
        # KPSS Test
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
        kpss_pvalue = kpss_result[1]
        
        # Store results
        results[name] = {
            'adf_pvalue': adf_pvalue,
            'kpss_pvalue': kpss_pvalue,
            'combined_score': (adf_pvalue <= 0.05) + (kpss_pvalue > 0.05)  # 0, 1, or 2
        }
    
    # Plot comparison
    plt.figure(figsize=(14, 10))
    
    plt.subplot(211)
    bars = plt.bar(range(len(results)), [r['adf_pvalue'] for r in results.values()])
    plt.axhline(0.05, color='red', linestyle='--')
    plt.title('ADF p-value (lower is better)')
    plt.xticks(range(len(results)), list(results.keys()), rotation=45)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{list(results.values())[i]["adf_pvalue"]:.4f}',
                ha='center', va='bottom', rotation=0)
    
    plt.subplot(212)
    bars = plt.bar(range(len(results)), [r['kpss_pvalue'] for r in results.values()])
    plt.axhline(0.05, color='red', linestyle='--')
    plt.title('KPSS p-value (higher is better)')
    plt.xticks(range(len(results)), list(results.keys()), rotation=45)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{list(results.values())[i]["kpss_pvalue"]:.4f}',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    # Find best transformation
    best_transformation = max(results.items(), key=lambda x: x[1]['combined_score'])
    print(f"Best transformation: {best_transformation[0]} with score {best_transformation[1]['combined_score']}/2")
    
    return results

# Compare all transformations
transformations = {
    'Original': city_data['Revenue'],
    'Linear Detrended': detrended_linear,
    'Polynomial Detrended': detrended_poly,
    'MA Detrended': detrended_ma,
    'Weekly Seasonal Residuals': residuals_weekly,
    'Monthly Seasonal Residuals': residuals_monthly
}

comparison_results = compare_transformations(transformations)

# %% [markdown]
# ## 7. Prepare Best Transformation for Modeling

# %%
# Identify the best transformation from the comparison results
best_transform_name = max(comparison_results.items(), key=lambda x: x[1]['combined_score'])[0]
best_transform = transformations[best_transform_name]

print(f"Preparing {best_transform_name} for modeling...")

# Plot the best transformation
plt.figure(figsize=(12, 6))
plt.plot(best_transform)
plt.title(f'Best Transformation: {best_transform_name}')
plt.grid(True)
plt.show()

# Test for remaining autocorrelation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(12, 8))
plt.subplot(211)
plot_acf(best_transform.dropna(), lags=40, ax=plt.gca())
plt.title(f'Autocorrelation Function (ACF) - {best_transform_name}')

plt.subplot(212)
plot_pacf(best_transform.dropna(), lags=40, ax=plt.gca())
plt.title(f'Partial Autocorrelation Function (PACF) - {best_transform_name}')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Save Transformed Data for Further Modeling

# %%
# Save the transformed data for later use
transformed_data = pd.DataFrame({
    'original': city_data['Revenue'],
    'linear_detrended': detrended_linear,
    'poly_detrended': detrended_poly,
    'ma_detrended': detrended_ma,
    'weekly_residuals': residuals_weekly,
    'monthly_residuals': residuals_monthly
})

# Save to CSV
transformed_data.to_csv(f'{city}_transformed_revenue.csv')
print(f"Transformed data saved to {city}_transformed_revenue.csv")

# Print final recommendations
print("\nRecommendations for Modeling:")
print("-" * 30)
print(f"1. Use the {best_transform_name} transformation for modeling")
print("2. Consider ARIMA or SARIMA models based on the ACF/PACF patterns")
print("3. For forecasting, remember to reapply the trend after generating predictions") 