"""
Enhanced forecasting utilities with advanced models.

Adds Prophet and ensemble methods for improved predictions.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠ Prophet not available. Install with: pip install prophet")


def fit_prophet_model(ts, forecast_periods=6, seasonality_mode='additive'):
    """
    Fit Facebook Prophet model for time-series forecasting.
    
    Prophet handles trends, seasonality, and holidays automatically.
    """
    if not PROPHET_AVAILABLE:
        print("⚠ Prophet not available, falling back to ARIMA")
        from utils.forecasting import fit_arima_model
        return fit_arima_model(ts, forecast_periods=forecast_periods)
    
    # Prepare data for Prophet (needs 'ds' and 'y' columns)
    df_prophet = pd.DataFrame({
        'ds': ts.index,
        'y': ts.values
    })
    
    # Initialize and fit model
    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(df_prophet)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_periods, freq='M')
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Extract forecasts and confidence intervals
    forecast_future = forecast.tail(forecast_periods)
    
    forecasts = pd.Series(
        forecast_future['yhat'].values,
        index=pd.date_range(start=ts.index[-1], periods=forecast_periods + 1, freq='M')[1:]
    )
    
    ci_lower = pd.Series(
        forecast_future['yhat_lower'].values,
        index=forecasts.index
    )
    
    ci_upper = pd.Series(
        forecast_future['yhat_upper'].values,
        index=forecasts.index
    )
    
    # Calculate metrics on in-sample data
    fitted_values = forecast.head(len(ts))['yhat'].values
    mae = mean_absolute_error(ts.values, fitted_values)
    rmse = np.sqrt(mean_squared_error(ts.values, fitted_values))
    r2 = r2_score(ts.values, fitted_values)
    
    print(f"✓ Fitted Prophet model")
    print(f"  - MAE: {mae:.2f}")
    print(f"  - RMSE: {rmse:.2f}")
    print(f"  - R²: {r2:.4f}")
    
    return {
        'model': model,
        'forecasts': forecasts,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'fitted_values': pd.Series(fitted_values, index=ts.index),
        'full_forecast': forecast
    }


def fit_exponential_smoothing(ts, forecast_periods=6, seasonal_periods=12):
    """
    Fit Exponential Smoothing model (Holt-Winters).
    """
    if not STATSMODELS_AVAILABLE:
        print("⚠ Statsmodels not available")
        return None
    
    try:
        model = ExponentialSmoothing(
            ts,
            seasonal_periods=seasonal_periods,
            trend='add',
            seasonal='add',
            use_boxcox=False
        )
        
        fitted_model = model.fit()
        
        # Generate forecasts
        forecasts = fitted_model.forecast(steps=forecast_periods)
        
        # Calculate metrics
        fitted_values = fitted_model.fittedvalues
        y = ts.values[len(ts) - len(fitted_values):]
        y_pred = fitted_values.values
        
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Simple confidence intervals (±1.96 * std of residuals)
        residuals = ts - fitted_values
        std_error = np.std(residuals.dropna())
        
        ci_lower = forecasts - 1.96 * std_error
        ci_upper = forecasts + 1.96 * std_error
        
        print(f"✓ Fitted Exponential Smoothing model")
        print(f"  - MAE: {mae:.2f}")
        print(f"  - RMSE: {rmse:.2f}")
        
        return {
            'model': fitted_model,
            'forecasts': forecasts,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'metrics': {'mae': mae, 'rmse': rmse},
            'fitted_values': fitted_values
        }
    
    except Exception as e:
        print(f"⚠ Exponential Smoothing failed: {str(e)}")
        return None


def fit_ml_ensemble(ts, forecast_periods=6):
    """
    Fit ensemble of machine learning models (Random Forest + Gradient Boosting).
    """
    # Create features from time series
    df = pd.DataFrame({'value': ts.values}, index=ts.index)
    
    # Feature engineering
    df['lag_1'] = df['value'].shift(1)
    df['lag_2'] = df['value'].shift(2)
    df['lag_3'] = df['value'].shift(3)
    df['rolling_mean_3'] = df['value'].rolling(window=3).mean()
    df['rolling_std_3'] = df['value'].rolling(window=3).std()
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    
    # Drop NaN rows
    df = df.dropna()
    
    # Split features and target
    X = df[['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'month', 'quarter']]
    y = df['value']
    
    # Train models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    
    rf_model.fit(X, y)
    gb_model.fit(X, y)
    
    # In-sample predictions
    rf_pred = rf_model.predict(X)
    gb_pred = gb_model.predict(X)
    ensemble_pred = (rf_pred + gb_pred) / 2
    
    # Calculate metrics
    mae = mean_absolute_error(y, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
    r2 = r2_score(y, ensemble_pred)
    
    # Generate future forecasts
    forecasts = []
    last_values = ts.tail(3).values.tolist()
    
    for i in range(forecast_periods):
        # Create features for next prediction
        next_features = pd.DataFrame({
            'lag_1': [last_values[-1]],
            'lag_2': [last_values[-2]],
            'lag_3': [last_values[-3]],
            'rolling_mean_3': [np.mean(last_values)],
            'rolling_std_3': [np.std(last_values)],
            'month': [(ts.index[-1] + pd.DateOffset(months=i+1)).month],
            'quarter': [(ts.index[-1] + pd.DateOffset(months=i+1)).quarter]
        })
        
        # Predict
        rf_next = rf_model.predict(next_features)[0]
        gb_next = gb_model.predict(next_features)[0]
        ensemble_next = (rf_next + gb_next) / 2
        
        forecasts.append(ensemble_next)
        last_values.append(ensemble_next)
        last_values.pop(0)
    
    # Create forecast series
    future_dates = pd.date_range(start=ts.index[-1], periods=forecast_periods + 1, freq='M')[1:]
    forecast_series = pd.Series(forecasts, index=future_dates)
    
    # Simple confidence intervals
    std_error = np.std(y - ensemble_pred)
    ci_lower = forecast_series - 1.96 * std_error
    ci_upper = forecast_series + 1.96 * std_error
    
    print(f"✓ Fitted ML Ensemble model")
    print(f"  - MAE: {mae:.2f}")
    print(f"  - RMSE: {rmse:.2f}")
    print(f"  - R²: {r2:.4f}")
    
    return {
        'models': {'rf': rf_model, 'gb': gb_model},
        'forecasts': forecast_series,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'fitted_values': pd.Series(ensemble_pred, index=y.index)
    }


def generate_enhanced_forecasts(df, target_col, forecast_periods=12, 
                               methods=['prophet', 'exponential', 'ml_ensemble'],
                               date_col='date', freq='M'):
    """
    Generate forecasts using multiple advanced methods and return best model.
    """
    print("=" * 60)
    print(f"ENHANCED FORECASTING: {target_col}")
    print("=" * 60)
    
    # Import from base forecasting module
    from utils.forecasting import prepare_timeseries, fit_arima_model
    
    # Prepare time series
    ts = prepare_timeseries(df, target_col, date_col, freq)
    
    results = {}
    
    # Try each method
    for method in methods:
        print(f"\n[{method.upper()}]")
        try:
            if method == 'prophet' and PROPHET_AVAILABLE:
                results[method] = fit_prophet_model(ts, forecast_periods)
            elif method == 'exponential' and STATSMODELS_AVAILABLE:
                result = fit_exponential_smoothing(ts, forecast_periods)
                if result:
                    results[method] = result
            elif method == 'ml_ensemble':
                results[method] = fit_ml_ensemble(ts, forecast_periods)
            elif method == 'arima' and STATSMODELS_AVAILABLE:
                results[method] = fit_arima_model(ts, forecast_periods=forecast_periods)
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
    
    # Select best model based on MAE
    if results:
        best_method = min(results.keys(), key=lambda k: results[k]['metrics']['mae'])
        best_result = results[best_method]
        
        print("\n" + "=" * 60)
        print(f"BEST MODEL: {best_method.upper()}")
        print(f"MAE: {best_result['metrics']['mae']:.2f}")
        print("=" * 60)
        
        best_result['original'] = ts
        best_result['method'] = best_method
        best_result['target'] = target_col
        best_result['all_results'] = results
        
        return best_result
    else:
        # Fallback to basic linear regression
        print("\n⚠ All advanced methods failed, using Linear Regression")
        from utils.forecasting import fit_linear_trend
        result = fit_linear_trend(ts, forecast_periods)
        result['original'] = ts
        result['method'] = 'linear'
        result['target'] = target_col
        return result


if __name__ == "__main__":
    print("Enhanced forecasting module loaded")
    print("Available methods:")
    print("  - Prophet (Facebook)")
    print("  - Exponential Smoothing (Holt-Winters)")
    print("  - ML Ensemble (Random Forest + Gradient Boosting)")
    print("  - ARIMA/SARIMA")
