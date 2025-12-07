# src/cointegration.py

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from arch.unitroot import VarianceRatio

from src.data import load_cointegration_data


def calculate_hurst_exponent(ts: np.ndarray) -> float:
    """Calculate Hurst exponent without nolds dependency."""
    # Your existing function from the notebook
    try:
        ts = np.array(ts).flatten()
        N = len(ts)
        if N < 100:
            return 0.5
        
        min_lag = 10
        max_lag = N // 10
        if max_lag <= min_lag:
            return 0.5
        
        lags = np.logspace(np.log10(min_lag), np.log10(max_lag), num=20).astype(int)
        lags = np.unique(lags)
        
        RS = []
        for lag in lags:
            subseries = [ts[i:i+lag] for i in range(0, N-lag+1, lag)]
            rs_values = []
            
            for sub in subseries:
                if len(sub) < lag:
                    continue
                mean_sub = np.mean(sub)
                Y = np.cumsum(sub - mean_sub)
                R = np.max(Y) - np.min(Y)
                S = np.std(sub, ddof=1)
                if S > 0 and R > 0:
                    rs_values.append(R/S)
            
            if rs_values:
                RS.append(np.mean(rs_values))
        
        RS = np.array(RS)
        valid_mask = ~np.isnan(RS)
        lags = lags[:len(RS)][valid_mask]
        RS = RS[valid_mask]
        
        if len(RS) < 3:
            return 0.5
        
        H = np.polyfit(np.log(lags), np.log(RS), 1)[0]
        return np.clip(H, 0, 1)
    except:
        return 0.5


def run_stationarity_tests(spread: pd.Series, significance_level: float = 0.05) -> dict:
    """
    Run comprehensive stationarity tests on a spread.
    
    Extracted from your notebook's comprehensive_stationarity_tests function.
    """
    results = {}
    
    # 1. ADF Test
    adf_stat, adf_p, _, _, _, _ = adfuller(spread)
    results['adf_stat'] = adf_stat
    results['adf_pvalue'] = adf_p
    results['adf_stationary'] = adf_p < significance_level
    
    # 2. KPSS Test
    kpss_stat, kpss_p, _, _ = kpss(spread, regression='c', nlags='auto')
    results['kpss_stat'] = kpss_stat
    results['kpss_pvalue'] = kpss_p
    results['kpss_stationary'] = kpss_p > significance_level
    
    # 3. Variance Ratio Test
    vr = VarianceRatio(spread, lags=8)
    results['variance_ratio'] = vr.stat
    results['variance_ratio_pvalue'] = vr.pvalue
    results['variance_ratio_mr'] = vr.stat < 1
    
    # 4. Hurst Exponent
    H = calculate_hurst_exponent(spread.values)
    results['hurst'] = H
    results['hurst_mr'] = H < 0.5
    
    # 5. Half-Life via OLS
    spread_diff = np.diff(spread)
    spread_lagged = spread.iloc[:-1].values
    ols_model = sm.OLS(spread_diff, sm.add_constant(spread_lagged)).fit()
    lambda_param = ols_model.params[1]
    half_life = -np.log(2) / lambda_param if lambda_param < 0 else np.inf
    results['half_life'] = half_life
    results['lambda'] = lambda_param
    
    return results


def run_cointegration_analysis(
    tickers: list,
    lookback_days: int = 504,
    significance_level: float = 0.05,
    min_half_life: float = 5,
    max_half_life: float = 30,
    data_source: str = 'alpaca',
    verbose: bool = False,
    start_date: str = None,
    end_date: str = None
) -> dict:
    """
    Run full cointegration analysis on a pair of assets.
    
    This is the main function that wraps all your notebook logic.
    
    Parameters
    ----------
    tickers : list
        List of two ticker symbols, e.g., ['XLI', 'XLB']
    lookback_days : int
        Number of days of historical data
    significance_level : float
        Significance level for statistical tests (default 0.05)
    min_half_life : float
        Minimum acceptable half-life in days
    max_half_life : float
        Maximum acceptable half-life in days
    data_source : str
        'yfinance' or 'alpaca'
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    dict
        Dictionary containing all analysis results
    """
    
    results = {
        'tickers': tickers,
        'lookback_days': lookback_days,
        'success': False,
        'error': None
    }
    
    try:
        # ============================================================
        # STEP 1: Load and prepare data
        # ============================================================
        if verbose:
            print(f"Loading data for {tickers}...")
        
        log_prices_df, validation_results = load_cointegration_data(
            tickers, 
            lookback_days, 
            source=data_source,
            start_date=start_date,
            end_date=end_date
        )   
        if verbose:
            print(f"  DEBUG: Received DataFrame with {len(log_prices_df)} rows")
         
        
        if log_prices_df is None or len(log_prices_df) < 50:
            results['error'] = "Insufficient data"
            return results
        
        results['n_observations'] = len(log_prices_df)
        results['date_start'] = str(log_prices_df.index[0].date())
        results['date_end'] = str(log_prices_df.index[-1].date())
        



        n_assets = len(tickers)
        
        # Add this after loading data (around line 125)
        correlation = log_prices_df.corr().iloc[0, 1]
        results['correlation'] = correlation

        if abs(correlation) < 0.3:
            print(f"Warning: Low correlation ({correlation:.2f}) - pair may not be suitable")


        # ============================================================
        # STEP 2: Lag selection
        # ============================================================
        model_var = VAR(log_prices_df)
        lag_selection = model_var.select_order(maxlags=10)
        k_ar_diff = max(1, lag_selection.aic)  # At least 1 lag
        results['optimal_lag'] = k_ar_diff
        
        # ============================================================
        # STEP 3: Johansen cointegration test
        # ============================================================
        if verbose:
            print("Running Johansen test...")
        
        johansen_result = coint_johansen(log_prices_df, det_order=0, k_ar_diff=k_ar_diff)
        
        trace_stats = johansen_result.lr1
        trace_crit = johansen_result.cvt
        eigen_stats = johansen_result.lr2
        eigen_crit = johansen_result.cvm
        
        # Count cointegrating vectors (with break on first non-rejection)
        n_coint_trace = 0
        for i in range(n_assets):
            if trace_stats[i] > trace_crit[i, 1]:
                n_coint_trace = i + 1
            else:
                break
        
        n_coint_eigen = 0
        for i in range(n_assets):
            if eigen_stats[i] > eigen_crit[i, 1]:
                n_coint_eigen = i + 1
            else:
                break
        
        cointegration_exists = n_coint_trace > 0 or n_coint_eigen > 0
        
        results['n_coint_trace'] = n_coint_trace
        results['n_coint_eigen'] = n_coint_eigen
        results['cointegration_exists'] = cointegration_exists
        results['trace_stat'] = trace_stats[0]
        results['trace_crit'] = trace_crit[0, 1]
        results['eigen_stat'] = eigen_stats[0]
        results['eigen_crit'] = eigen_crit[0, 1]
        
        # Johansen hedge ratios (first eigenvector)
        johansen_vector = johansen_result.evec[:, 0]
        johansen_normalized = johansen_vector / johansen_vector[0]
        results['johansen_hedge_ratios'] = dict(zip(tickers, johansen_normalized.tolist()))
        
        # ============================================================
        # STEP 4: VAR eigenvalue decomposition
        # ============================================================
        if verbose:
            print("Running VAR decomposition...")
        
        var_fit = model_var.fit(maxlags=1)
        B_matrix = var_fit.params.iloc[1:n_assets+1, :].values.T
        
        I_matrix = np.eye(n_assets)
        kappa_matrix = (I_matrix - B_matrix)
        
        eigenvalues, eigenvectors = np.linalg.eig(kappa_matrix)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        
        # Sort by magnitude (fastest mean-reversion first)
        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]
        
        optimal_kappa = eigenvalues_sorted[0]
        var_hedge_ratios = eigenvectors_sorted[:, 0]
        var_normalized = var_hedge_ratios / var_hedge_ratios[0]
        
        results['var_kappa'] = optimal_kappa
        results['var_hedge_ratios'] = dict(zip(tickers, var_normalized.tolist()))
        
        # ============================================================
        # STEP 5: Construct spreads and run stationarity tests
        # ============================================================
        if verbose:
            print("Running stationarity tests...")
        
        # Johansen spread
        spread_johansen = log_prices_df @ johansen_vector
        johansen_tests = run_stationarity_tests(spread_johansen, significance_level)
        
        # VAR spread
        spread_var = log_prices_df @ var_hedge_ratios
        var_tests = run_stationarity_tests(spread_var, significance_level)
        
        # ============================================================
        # STEP 6: Determine winner
        # ============================================================
        def calc_score(tests):
            return sum([
                tests['adf_stationary'],
                tests['kpss_stationary'],
                tests['variance_ratio_mr'],
                tests['hurst_mr'],
                min_half_life < tests['half_life'] < max_half_life
            ])
        
        johansen_score = calc_score(johansen_tests)
        var_score = calc_score(var_tests)
        
        if var_score > johansen_score:
            best_method = 'VAR'
            best_tests = var_tests
            best_hedge_ratios = results['var_hedge_ratios']
        else:
            best_method = 'Johansen'
            best_tests = johansen_tests
            best_hedge_ratios = results['johansen_hedge_ratios']
        
        # ============================================================
        # STEP 7: Compile final results
        # ============================================================
        results['best_method'] = best_method
        results['hedge_ratios'] = best_hedge_ratios
        results['johansen_score'] = johansen_score
        results['var_score'] = var_score
        
        # Key metrics (from best method)
        results['half_life'] = best_tests['half_life']
        results['adf_pvalue'] = best_tests['adf_pvalue']
        results['adf_stationary'] = best_tests['adf_stationary']
        results['kpss_stationary'] = best_tests['kpss_stationary']
        results['hurst'] = best_tests['hurst']
        results['variance_ratio'] = best_tests['variance_ratio']
        
        # Tradability assessment
        hl = results['half_life']
        results['half_life_tradable'] = min_half_life < hl < max_half_life
        
        results['success'] = True
        
        if verbose:
            print(f"Done. Half-life: {hl:.1f} days, Cointegration: {cointegration_exists}")
        
    except Exception as e:
        results['error'] = str(e)
        results['success'] = False
    
    return results