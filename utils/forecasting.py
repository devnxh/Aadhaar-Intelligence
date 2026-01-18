"""
Forecasting utilities for Aadhaar Intelligence Platform.

This module provides functions for time-series forecasting and
predictive modeling of enrollment and update patterns.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


def train_test_split_timeseries(ts, test_size=6):
    """
    Split time series into train and test sets.
    
    Uses last N months as hold-out test data for proper evaluation.
    This ensures we evaluate forecasting ability on unseen future data.
    
    Parameters:
    -----------
    ts : pd.Series
        Time series data with datetime index
    test_size : int
        Number of months to use as test data (default: 6)
        
    Returns:
    --------
    tuple: (train_series, test_series)
    """
    if len(ts) <= test_size:
        # If data too small, use 80-20 split
        split_point = int(len(ts) * 0.8)
    else:
        split_point = len(ts) - test_size
    
    train = ts.iloc[:split_point]
    test = ts.iloc[split_point:]
    
    print(f"✓ Train/Test Split:")
    print(f"  - Training: {len(train)} months ({train.index[0]} to {train.index[-1]})")
    print(f"  - Testing: {len(test)} months ({test.index[0]} to {test.index[-1]})")
    
    return train, test


def calculate_smape(actual, forecast):
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    
    sMAPE is preferred over MAPE because:
    - Handles zero values safely (no division by zero)
    - Symmetric (treats over/under-prediction equally)
    - Bounded between 0-200% (more interpretable)
    - Accepted in forecasting literature
    
    Formula: sMAPE = mean(|A - F| / ((|A| + |F|)/2)) × 100
    
    Parameters:
    -----------
    actual : array-like
        Actual observed values
    forecast : array-like
        Forecasted values
        
    Returns:
    --------
    float: sMAPE percentage (0-200%)
    """
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    # Calculate sMAPE with safe division
    # Ignore rows where both actual and forecast are zero
    smape_values = []
    for a, f in zip(actual, forecast):
        denominator = (abs(a) + abs(f)) / 2.0
        if denominator > 1e-10:  # Avoid near-zero denominators
            smape_values.append(abs(a - f) / denominator * 100.0)
    
    if len(smape_values) == 0:
        return 0.0
    
    return np.mean(smape_values)


def prepare_timeseries(df, value_col, date_col='date', freq='M', agg_func='sum'):
    """
    Prepare time series data for forecasting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    value_col : str
        Column to forecast
    date_col : str
        Date column name
    freq : str
        Frequency for resampling ('D', 'W', 'M', 'Q', 'Y')
    agg_func : str
        Aggregation function ('sum', 'mean', 'median')
        
    Returns:
    --------
    pd.Series
        Time series ready for forecasting
    """
    df = df.copy()
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Set date as index
    df = df.set_index(date_col)
    
    # Resample to specified frequency
    if agg_func == 'sum':
        ts = df[value_col].resample(freq).sum()
    elif agg_func == 'mean':
        ts = df[value_col].resample(freq).mean()
    elif agg_func == 'median':
        ts = df[value_col].resample(freq).median()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")
    
    # Remove any NaN values
    ts = ts.fillna(method='ffill').fillna(method='bfill')
    
    print(f"✓ Prepared time series for {value_col}")
    print(f"  - Frequency: {freq}")
    print(f"  - Data points: {len(ts)}")
    print(f"  - Date range: {ts.index.min()} to {ts.index.max()}")
    
    return ts


def decompose_timeseries(ts, model='additive', period=None):
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Parameters:
    -----------
    ts : pd.Series
        Time series data
    model : str
        'additive' or 'multiplicative'
    period : int, optional
        Seasonal period
        
    Returns:
    --------
    statsmodels.DecomposeResult
        Decomposition results
    """
    if not STATSMODELS_AVAILABLE:
        print("⚠ Decomposition requires statsmodels")
        return None
    
    if period is None:
        # Auto-detect period based on frequency
        freq = ts.index.freq or pd.infer_freq(ts.index)
        if freq and 'M' in str(freq):
            period = 12
        elif freq and 'Q' in str(freq):
            period = 4
        else:
            period = 7
    
    decomposition = seasonal_decompose(ts, model=model, period=period)
    
    print(f"✓ Decomposed time series")
    print(f"  - Model: {model}")
    print(f"  - Period: {period}")
    
    return decomposition


def fit_linear_trend(ts, forecast_periods=6):
    """
    Fit a simple linear trend model and generate forecasts.
    
    Parameters:
    -----------
    ts : pd.Series
        Time series data
    forecast_periods : int
        Number of periods to forecast ahead
        
    Returns:
    --------
    dict
        Dictionary with model, forecasts, and metrics
    """
    # Prepare data for linear regression
    X = np.arange(len(ts)).reshape(-1, 1)
    y = ts.values
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # In-sample predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    # Generate forecasts
    future_X = np.arange(len(ts), len(ts) + forecast_periods).reshape(-1, 1)
    forecasts = model.predict(future_X)
    
    # Create future dates
    last_date = ts.index[-1]
    freq = ts.index.freq or pd.infer_freq(ts.index)
    future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=freq)[1:]
    
    # Calculate confidence intervals (simple approach)
    residuals = y - y_pred
    std_error = np.std(residuals)
    ci_lower = forecasts - 1.96 * std_error
    ci_upper = forecasts + 1.96 * std_error
    
    print(f"✓ Fitted Linear Trend model")
    print(f"  - MAE: {mae:.2f}")
    print(f"  - RMSE: {rmse:.2f}")
    print(f"  - R²: {r2:.4f}")
    
    return {
        'model': model,
        'forecasts': pd.Series(forecasts, index=future_dates),
        'ci_lower': pd.Series(ci_lower, index=future_dates),
        'ci_upper': pd.Series(ci_upper, index=future_dates),
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'fitted_values': pd.Series(y_pred, index=ts.index)
    }




def fit_arima_model(ts, order=(1, 1, 1), seasonal_order=None, forecast_periods=12):
    """
    Fit ARIMA/SARIMA model with proper hold-out validation.
    
    **Evaluation Protocol:**
    - Last 6 months used as test data (hold-out validation)
    - Metrics calculated ONLY on test data (no training data leakage)
    - This ensures realistic forecast accuracy estimates
    
    Parameters:
    -----------
    ts : pd.Series
        Time series with datetime index
    order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple, optional
        Seasonal order (P, D, Q, s) for SARIMA
    forecast_periods : int
        Number of periods to forecast into future
        
    Returns:
    --------
    dict: Contains forecasts, confidence intervals, metrics, and fitted values
    """
    # Split into train and test (last 6 months as test)
    train, test = train_test_split_timeseries(ts, test_size=6)
    
    # Fit model on training data ONLY
    try:
        if seasonal_order is not None:
            # SARIMA Model
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_name = f"SARIMA{order}x{seasonal_order}"
        else:
            # ARIMA Model
            model = ARIMA(train, order=order)
            model_name = f"ARIMA{order}"
        
        fitted_model = model.fit()
        
        # Generate predictions on TEST data (proper validation)
        test_predictions = fitted_model.forecast(steps=len(test))
        
        # Calculate metrics on TEST data ONLY (not training data!)
        mae = np.mean(np.abs(test.values - test_predictions.values))
        rmse = np.sqrt(np.mean((test.values - test_predictions.values) ** 2))
        smape = calculate_smape(test.values, test_predictions.values)
        
        # Calculate R² on test data
        ss_res = np.sum((test.values - test_predictions.values) ** 2)
        ss_tot = np.sum((test.values - np.mean(test.values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"✓ Fitted {model_name}")
        print(f"  - Order: {order}")
        if seasonal_order:
            print(f"  - Seasonal: {seasonal_order}")
        print(f"  **Evaluation on TEST data (last 6 months):**")
        print(f"  - MAE: {mae:.2f}")
        print(f"  - RMSE: {rmse:.2f}")
        print(f"  - sMAPE: {smape:.2f}%")
        print(f"  - R²: {r2:.4f}")
        
        # Now forecast future periods beyond the data
        # Refit on ALL data for final forecast
        if seasonal_order is not None:
            final_model = SARIMAX(
                ts,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            final_model = ARIMA(ts, order=order)
        
        final_fitted = final_model.fit()
        
        # Generate future forecasts
        forecasts = final_fitted.forecast(steps=forecast_periods)
        
        # Get confidence intervals (95%)
        forecast_result = final_fitted.get_forecast(steps=forecast_periods)
        ci = forecast_result.conf_int()
        
        # Ensure no negative predictions (enrollment/updates can't be negative)
        forecasts = forecasts.clip(lower=0)
        ci_lower = ci.iloc[:, 0].clip(lower=0)
        ci_upper = ci.iloc[:, 1].clip(lower=0)
        
        # Get fitted values for the full series
        fitted_values = final_fitted.fittedvalues
        
        return {
            'forecasts': forecasts,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'fitted': fitted_values,
            'method': model_name,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'smape': smape,
                'r2': r2
            }
        }
        
    except Exception as e:
        print(f"⚠ {model_name} fitting failed: {str(e)}")
        print(f"  Falling back to Linear Trend model")
        return fit_linear_trend(ts, forecast_periods)


def auto_select_arima_order(ts, max_p=3, max_d=2, max_q=3):
    """
    Automatically select best ARIMA order using AIC.
    
    Parameters:
    -----------
    ts : pd.Series
        Time series data
    max_p, max_d, max_q : int
        Maximum values for p, d, q parameters
        
    Returns:
    --------
    tuple
        Best (p, d, q) order
    """
    if not STATSMODELS_AVAILABLE:
        return (1, 1, 1)
    
    best_aic = np.inf
    best_order = (1, 1, 1)
    
    print("Searching for best ARIMA order...")
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted = model.fit()
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    print(f"✓ Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
    
    return best_order


def generate_forecasts(df, target_col, forecast_periods=12, method='arima', 
                      date_col='date', freq='M'):
    """
    Generate forecasts for a target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Column to forecast
    forecast_periods : int
        Number of periods to forecast
    method : str
        'arima', 'sarima', or 'linear'
    date_col : str
        Date column name
    freq : str
        Frequency for resampling
        
    Returns:
    --------
    dict
        Forecast results
    """
    print("=" * 60)
    print(f"FORECASTING: {target_col}")
    print("=" * 60)
    
    # Prepare time series
    ts = prepare_timeseries(df, target_col, date_col, freq)
    
    # Generate forecasts based on method
    if method == 'linear':
        results = fit_linear_trend(ts, forecast_periods)
    elif method == 'arima':
        order = auto_select_arima_order(ts) if STATSMODELS_AVAILABLE else (1, 1, 1)
        results = fit_arima_model(ts, order=order, forecast_periods=forecast_periods)
    elif method == 'sarima':
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12) if freq == 'M' else (1, 1, 1, 4)
        results = fit_arima_model(ts, order=order, seasonal_order=seasonal_order, 
                                 forecast_periods=forecast_periods)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add original time series to results
    results['original'] = ts
    results['method'] = method
    results['target'] = target_col
    
    print("=" * 60)
    print("FORECASTING COMPLETE")
    print("=" * 60)
    
    return results


def evaluate_forecast_accuracy(actual, predicted):
    """
    Evaluate forecast accuracy using multiple metrics.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of accuracy metrics
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }


def create_forecast_dataframe(results):
    """
    Create a formatted dataframe from forecast results.
    
    Parameters:
    -----------
    results : dict
        Forecast results from generate_forecasts()
        
    Returns:
    --------
    pd.DataFrame
        Formatted forecast dataframe
    """
    forecast_df = pd.DataFrame({
        'forecast': results['forecasts'],
        'lower_ci': results['ci_lower'],
        'upper_ci': results['ci_upper']
    })
    
    forecast_df['method'] = results['method']
    forecast_df['target'] = results['target']
    
    return forecast_df


if __name__ == "__main__":
    # Example usage
    print("Forecasting module loaded successfully")
    print("Available functions:")
    print("  - prepare_timeseries()")
    print("  - fit_linear_trend()")
    print("  - fit_arima_model()")
    print("  - generate_forecasts()")
    print("  - decompose_timeseries()")
    print("  - auto_select_arima_order()")
