# src/strategy.py (UPDATED)

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
# Assuming a generic Backtester is also defined here or imported

# --- NEW CLASS: Mean Reversion Strategy (Bollinger Bands) ---

class MultiAssetMeanReversionStrategy:
    """
    Generalized mean-reversion strategy for N assets (pairs or baskets).
    
    Uses Johansen eigenvector to construct a stationary spread, then
    trades deviations from the mean using Bollinger Bands.
    """
    
    def __init__(
        self,
        hedge_ratios: Dict[str, float],  # Changed from single float to dict
        z_score_lookback: int = 252,
        entry_std: float = 2.0,
        exit_std: float = 0.5,
        tickers: List[str] = None
    ):
        """
        Initialize strategy parameters.
        
        Args:
            hedge_ratios: Dictionary of {ticker: weight} from cointegration
            z_score_lookback: Window for mean/std calculation
            entry_std: Entry threshold (standard deviations)
            exit_std: Exit threshold (standard deviations)
            tickers: List of tickers (for ordering)
        """
        self.hedge_ratios = hedge_ratios
        self.lookback = z_score_lookback
        self.entry_std = entry_std
        self.exit_std = exit_std
        self.tickers = tickers or list(hedge_ratios.keys())
        
        # Normalize weights so first asset = 1.0
        self.normalized_weights = self._normalize_weights()
    
    def _normalize_weights(self) -> Dict[str, float]:
        """Normalize weights so first ticker has weight 1.0"""
        first_ticker = self.tickers[0]
        first_weight = self.hedge_ratios[first_ticker]
        
        normalized = {}
        for ticker in self.tickers:
            normalized[ticker] = self.hedge_ratios[ticker] / first_weight
        
        return normalized
    
    def generate_signals(self, log_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for N assets.
        
        Args:
            log_prices: DataFrame of log-prices with tickers as columns
            
        Returns:
            DataFrame of position weights for each asset
        """
        # Ensure column order matches tickers
        log_prices = log_prices[self.tickers].copy()
        
        # 1. Construct the spread as weighted sum
        # Spread = w1*Asset1 + w2*Asset2 + ... + wN*AssetN
        spread = pd.Series(0.0, index=log_prices.index)
        for ticker in self.tickers:
            spread += self.normalized_weights[ticker] * log_prices[ticker]
        
        # 2. Calculate Z-Score
        roll_mean = spread.rolling(window=self.lookback).mean()
        roll_std = spread.rolling(window=self.lookback).std()
        z_score = (spread - roll_mean) / roll_std
        
        # 3. Generate spread position signals
        signals = pd.Series(0, index=log_prices.index, dtype=float)
        current_position = 0
        
        for i in range(self.lookback, len(z_score)):
            score = z_score.iloc[i]
            
            if current_position == 0:
                if score > self.entry_std:
                    current_position = -1  # Short spread
                elif score < -self.entry_std:
                    current_position = 1   # Long spread
            
            elif current_position == 1:  # Long spread
                if score > -self.exit_std:
                    current_position = 0
            
            elif current_position == -1:  # Short spread
                if score < self.exit_std:
                    current_position = 0
            
            signals.iloc[i] = current_position
        
        # 4. Convert spread position to asset positions
        # If spread position = 1 (long spread):
        #   Each asset position = weight × spread_position
        # Since spread = sum(weights × assets), we want:
        #   Buy the spread → buy each asset proportional to its weight
        
        signals_df = pd.DataFrame(index=log_prices.index)
        
        for ticker in self.tickers:
            # Position = spread_position × normalized_weight
            signals_df[ticker] = signals * self.normalized_weights[ticker]
        
        # 5. Normalize total exposure to 1.0
        total_abs_exposure = signals_df.abs().sum(axis=1)
        signals_df = signals_df.div(total_abs_exposure.replace(0, 1), axis=0)
        signals_df = signals_df.fillna(0)
        
        # 6. Shift to avoid look-ahead bias
        return signals_df.shift(1).fillna(0)

class PairsBollingerStrategy:
    """
    Implements a classic pairs trading strategy based on Bollinger Bands 
    (Z-Score) applied to the spread, using a STATIC hedge ratio.
    """
    
    def __init__(
        self,
        hedge_ratio: float,
        z_score_lookback: int = 252,
        entry_std: float = 2.0,
        exit_std: float = 0.5,
        target_pair: List[str] = None # For logging/tracking
    ):
        """
        Initialize the strategy parameters.

        Args:
            hedge_ratio: The static beta (slope) derived from cointegration/VAR.
            z_score_lookback: Window size (in days) for mean and std calculation.
            entry_std: Number of standard deviations to trigger an OPEN trade.
            exit_std: Number of standard deviations to trigger a CLOSE trade.
        """
        self.hedge_ratio = hedge_ratio
        self.lookback = z_score_lookback
        self.entry_std = entry_std
        self.exit_std = exit_std
        self.tickers = target_pair
        
        # Log-prices are required for the hedge ratio calculation
        self.log_price_col = f'log_{target_pair[0]}'
        self.log_hedge_col = f'log_{target_pair[1]}'

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals for the pair (long/short both assets).

        Signals are:
        +1 (long the spread): Short the overvalued asset (Asset 0), Long the undervalued asset (Asset 1).
        -1 (short the spread): Long the overvalued asset (Asset 0), Short the undervalued asset (Asset 1).

        Args:
            prices: DataFrame of log-prices for the two assets.

        Returns:
            DataFrame with 'position' column (-1, 0, 1) representing the spread position.
        """
        if prices.shape[1] != 2:
            raise ValueError("Input DataFrame must contain exactly two assets (log-prices).")
            
        prices.columns = self.tickers
        
        # 1. Calculate the spread (The residual of the cointegrating relationship)
        # Spread = log(P_A) - beta * log(P_B)
        spread = prices[self.tickers[0]] - self.hedge_ratio * prices[self.tickers[1]]
        
        # 2. Calculate the rolling mean and standard deviation of the spread
        roll_mean = spread.rolling(window=self.lookback).mean()
        roll_std = spread.rolling(window=self.lookback).std()
        
        # 3. Calculate the Z-Score
        z_score = (spread - roll_mean) / roll_std
        
        # 4. Generate the Trading Signals
        signals = pd.Series(0, index=prices.index, dtype=float)
        
        # Initialize current position: 1=Long Spread, -1=Short Spread, 0=Flat
        current_position = 0
        
        # Trading logic is applied iteratively to respect path dependency
        for i in range(self.lookback, len(z_score)):
            score = z_score.iloc[i]

            if current_position == 0:
                # ENTRY: Short the Spread (Long Z-Score)
                if score > self.entry_std:
                    current_position = -1 # Spread is overextended (SELL/SHORT the spread)
                # ENTRY: Long the Spread (Short Z-Score)
                elif score < -self.entry_std:
                    current_position = 1 # Spread is under-extended (BUY/LONG the spread)
            
            elif current_position == 1: # Currently Long the Spread
                # EXIT: Crosses mean or is overextended on the other side
                if score > -self.exit_std:
                    current_position = 0 # Close trade
                
            elif current_position == -1: # Currently Short the Spread
                # EXIT: Crosses mean or is overextended on the other side
                if score < self.exit_std:
                    current_position = 0 # Close trade

            signals.iloc[i] = current_position
            
        # The final 'signals' Series contains the SPREAD position (-1, 0, 1).
        # We need to transform this into the position for the underlying assets (A and B).
        # Position for Asset 0 (P_A): -1 * Spread Position
        # Position for Asset 1 (P_B): +Hedge Ratio * Spread Position
        
        signals_df = pd.DataFrame(index=prices.index)
        signals_df[self.tickers[0]] = -signals
        signals_df[self.tickers[1]] = signals * self.hedge_ratio
        
        # Normalize weights to ensure total exposure doesn't exceed 1 (or is scaled to desired leverage)
        total_abs_exposure = signals_df.abs().sum(axis=1)
        signals_df = signals_df.div(total_abs_exposure.replace(0, 1), axis=0) # Normalize to +/-1 exposure

        # Replace NaN/inf with 0
        signals_df = signals_df.fillna(0)
        
        return signals_df.shift(1).fillna(0) # Shift by 1 day to ensure no look-ahead bias

class Backtester:
    """
    A generic engine to run a backtest for any daily-rebalanced,
    dollar-weighted strategy (like Statistical Arbitrage).
    """

    def __init__(self, transaction_cost: float = 0.0001):
        """
        Initialize the Backtester.
        
        Args:
            transaction_cost: Percentage transaction cost applied per dollar traded (one-way).
        """
        # 1 basis point (0.01%) is a common estimate for liquid ETFs/stocks
        self.transaction_cost = transaction_cost

    def calculate_metrics(self, returns: pd.Series) -> dict:
        """Calculates standard risk and return metrics."""
        if returns.empty:
            return {}
        
        # Calculate Cumulative Returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Annualized Metrics (assuming 252 trading days)
        ann_factor = np.sqrt(252)
        total_days = len(returns)
        
        # --- NEW: Calculate Win Rate ---
        # Note: Win Rate is calculated as the percentage of days with positive returns.
        winning_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = winning_days / total_days if total_days > 0 else 0


        # Simple Annualized Return
        total_return = cumulative_returns.iloc[-1] - 1
        ann_return = (1 + total_return) ** (252 / total_days) - 1
        
        # Volatility & Sharpe
        ann_vol = returns.std() * ann_factor
        sharpe_ratio = ann_return / ann_vol if ann_vol != 0 else 0
        
        # Max Drawdown
        high_watermark = cumulative_returns.cummax()
        drawdown = cumulative_returns / high_watermark - 1
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate
        }

    def run(self, raw_prices: pd.DataFrame, signals: pd.DataFrame) -> dict:
        """
        Runs the backtest using raw prices and dollar-weighted signals.
        
        Args:
            raw_prices: Daily Adjusted Close prices for the assets.
            signals: Daily dollar-weighted positions (e.g., [-1.0, 1.0]). 
                     *Must be shifted by 1 already (no look-ahead bias).*
                     
        Returns:
            A dictionary containing returns, metrics, and final signals.
        """
        # Ensure signals and prices are aligned
        prices = raw_prices.reindex(signals.index).ffill()
        
        # 1. Calculate Asset Returns
        daily_returns = prices.pct_change(fill_method=None).fillna(0)
        
        # 2. Calculate Gross Strategy Returns
        # Gross Return = Sum(Position_t-1 * Asset_Return_t)
        gross_returns = (signals * daily_returns).sum(axis=1)
        
        # 3. Calculate Transaction Costs (CRUCIAL STEP for Stat Arb)
        # Cost = Sum(|Change in Position| * Cost Rate)
        # Position change is current day's position minus previous day's position
        position_change = (signals - signals.shift(1)).abs().sum(axis=1)
        transaction_costs = position_change * self.transaction_cost
        
        # 4. Calculate Net Portfolio Returns
        portfolio_returns = gross_returns - transaction_costs
        
        # 5. Calculate Metrics
        metrics = self.calculate_metrics(portfolio_returns)

        return {
            'portfolio_returns': portfolio_returns,
            'signals': signals,
            'metrics': metrics,
            'transaction_costs': transaction_costs
        }