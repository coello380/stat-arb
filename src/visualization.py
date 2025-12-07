"""
Visualization module for momentum strategy backtest results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional
import os


def set_style():
    """Set a clean, professional plotting style."""
    # Use a style that is available in standard matplotlib
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('bmh') # Fallback if seaborn style isn't found
        
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.dpi'] = 100


def plot_equity_curve(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    title: str = 'Strategy Performance',
    save_path: str = None
) -> plt.Figure:
    """Plot cumulative returns (equity curve)."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Strategy equity curve
    cumulative = (1 + portfolio_returns).cumprod()
    ax.plot(cumulative.index, cumulative.values, label='Momentum Strategy', linewidth=1.5, color='#2E86AB')
    
    # Benchmark if provided
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Align dates just in case
        common_idx = cumulative.index.intersection(benchmark_returns.index)
        if len(common_idx) > 0:
            bench_cumulative = (1 + benchmark_returns.loc[common_idx]).cumprod()
            ax.plot(bench_cumulative.index, bench_cumulative.values, label='Benchmark', 
                    linewidth=1.5, color='#A23B72', alpha=0.7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_drawdown(
    portfolio_returns: pd.Series,
    title: str = 'Strategy Drawdown',
    save_path: str = None
) -> plt.Figure:
    """Plot drawdown over time."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Calculate drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative / rolling_max - 1) * 100  # Convert to percentage
    
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='#E74C3C')
    ax.plot(drawdown.index, drawdown.values, color='#E74C3C', linewidth=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_rolling_sharpe(
    portfolio_returns: pd.Series,
    window: int = 252,  # 1 year of trading days
    title: str = 'Rolling Sharpe Ratio (1-Year)',
    save_path: str = None
) -> plt.Figure:
    """Plot rolling Sharpe ratio."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Calculate rolling Sharpe (daily data, annualized)
    rolling_return = portfolio_returns.rolling(window).mean() * 252
    rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / rolling_vol
    
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color='#27AE60', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=-1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_monthly_returns_heatmap(
    portfolio_returns: pd.Series,
    title: str = 'Monthly Returns (%)',
    save_path: str = None
) -> plt.Figure:
    """Plot monthly returns as a heatmap."""
    set_style()
    


    # Resample to monthly returns first
    monthly_returns = portfolio_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly = monthly_returns * 100
    
    # Create pivot table
    monthly_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values
    })
    
    pivot = monthly_df.pivot(index='year', columns='month', values='return')
    
    # Ensure all months are present
    for i in range(1, 13):
        if i not in pivot.columns:
            pivot[i] = np.nan
    pivot = pivot.reindex(columns=range(1, 13))
            
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig, ax = plt.subplots(figsize=(14, len(pivot) * 0.5 + 2))
    
    # Create heatmap
    cmap = plt.cm.RdYlGn
    # Handle NaN for display
    plot_data = pivot.fillna(0)
    im = ax.imshow(plot_data.values, cmap=cmap, aspect='auto', vmin=-10, vmax=10)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Return (%)')
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > 5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color=text_color, fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig

def plot_strategy_vs_assets(
    portfolio_returns: pd.Series,
    raw_prices: pd.DataFrame,
    title: str = 'Strategy vs Buy-and-Hold',
    save_path: str = None
) -> plt.Figure:
    """Compare strategy cumulative returns vs holding each asset."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Strategy cumulative returns
    strat_cum = (1 + portfolio_returns).cumprod()
    ax.plot(strat_cum.index, strat_cum.values, 
            label='Pairs Strategy', linewidth=2, color='#2E86AB')
    
    # Individual assets cumulative returns
    colors = ['#E74C3C', '#27AE60']
    for i, col in enumerate(raw_prices.columns):
        asset_returns = raw_prices[col].pct_change().fillna(0)
        asset_cum = (1 + asset_returns).cumprod()
        # Align to strategy dates
        asset_cum = asset_cum.reindex(strat_cum.index).ffill()
        ax.plot(asset_cum.index, asset_cum.values, 
                label=f'{col} (Buy & Hold)', linewidth=1.5, 
                color=colors[i], alpha=0.7, linestyle='--')
    
    ax.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_signal_distribution(
    signals: pd.DataFrame,
    title: str = 'Signal Distribution',
    save_path: str = None
) -> plt.Figure:
    """Plot distribution of trading signals."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Flatten all signals
    all_signals = signals.values.flatten()
    all_signals = all_signals[~np.isnan(all_signals)]
    
    ax.hist(all_signals, bins=50, density=True, alpha=0.7, color='#3498DB', edgecolor='white')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=all_signals.mean(), color='red', linestyle='--', linewidth=1, 
               label=f'Mean: {all_signals.mean():.2f}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Signal Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_full_report(
    results: dict,
    benchmark_returns: pd.Series = None,
    output_dir: str = 'results',
    raw_prices: pd.DataFrame = None
) -> None:
    """Create a full visual report of backtest results."""
    os.makedirs(output_dir, exist_ok=True)
    
    portfolio_returns = results['portfolio_returns']
    signals = results['signals']
    metrics = results['metrics']
    
    # Print metric
    
    # FIXED: Handle key name change safely
    obs = metrics.get('n_observations', metrics.get('n_months', 0))
    # Generate plots
    print("Generating visualizations...")
    
    plot_equity_curve(
        portfolio_returns, 
        benchmark_returns,
        save_path=os.path.join(output_dir, 'equity_curve.png')
    )
    
    plot_drawdown(
        portfolio_returns,
        save_path=os.path.join(output_dir, 'drawdown.png')
    )
    
    plot_rolling_sharpe(
        portfolio_returns,
        window=12,  # Explicitly use 12 months
        save_path=os.path.join(output_dir, 'rolling_sharpe.png')
    )
    
    plot_monthly_returns_heatmap(
        portfolio_returns,
        save_path=os.path.join(output_dir, 'monthly_returns.png')
    )
    
    plot_signal_distribution(
        signals,
        save_path=os.path.join(output_dir, 'signal_distribution.png')
    )
    

    if raw_prices is not None:
        plot_strategy_vs_assets(
            portfolio_returns,
            raw_prices,
            title='Strategy vs Buy-and-Hold',
            save_path=os.path.join(output_dir, 'strategy_vs_assets.png')
        )

    plt.close('all')
def plot_spread_and_zscore(
    raw_prices: pd.DataFrame,
    signals_df: pd.DataFrame,
    hedge_ratio: float,
    z_score_lb: int,
    entry_std: float,
    exit_std: float,
    title: str,
    save_path: str = None
) -> None:
    """Plots the spread and its Z-Score with entry/exit bands."""
    set_style() # Assuming set_style() is defined in your file
    
    # 1. Recalculate Spread and Z-Score for plotting
    prices = np.log(raw_prices.copy())
    
    # Assume first column is Asset 0, second is Asset 1 (based on main.py logic)
    spread = prices.iloc[:, 0] - hedge_ratio * prices.iloc[:, 1]
    
    roll_mean = spread.rolling(window=z_score_lb).mean()
    roll_std = spread.rolling(window=z_score_lb).std()
    z_score = (spread - roll_mean) / roll_std

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Spread and Trading Signals: {title}', fontsize=14)

    # --- Plot 1: The Spread ---
    ax1.plot(spread.index, spread.values, label='Spread (Log)', color='gray', linewidth=0.8)
    ax1.plot(roll_mean.index, roll_mean.values, label=f'Rolling Mean ({z_score_lb}d)', color='red', linestyle='--')
    ax1.set_title('Log Spread (Cointegrating Portfolio Value)', fontsize=10)
    ax1.legend()
    ax1.grid(True)
    
    # --- Plot 2: The Z-Score and Signals ---
    ax2.plot(z_score.index, z_score.values, label='Z-Score', color='#2E86AB', linewidth=1.2)
    
    # Plot Bollinger Bands (Entry/Exit)
    ax2.axhline(entry_std, color='orange', linestyle='-', label=f'Entry (+{entry_std:.1f}σ)')
    ax2.axhline(-entry_std, color='orange', linestyle='-')
    ax2.axhline(exit_std, color='green', linestyle=':', label=f'Exit ({exit_std:.1f}σ)')
    ax2.axhline(-exit_std, color='green', linestyle=':')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)

        # Highlight entry/exit points visually based on signals_df
    # NOTE: signals_df contains ASSET positions, not spread positions
    # Asset 0 position = -spread_position, so we need to invert the logic
    # When Asset 0 < 0: spread position is POSITIVE (Long Spread)
    # When Asset 0 > 0: spread position is NEGATIVE (Short Spread)
    
    long_entry_dates = z_score.index[signals_df.iloc[:, 0].shift(-1) < 0]  # Asset 0 negative → Long Spread
    short_entry_dates = z_score.index[signals_df.iloc[:, 0].shift(-1) > 0]  # Asset 0 positive → Short Spread

    # Plot entries with correct colors
    ax2.plot(short_entry_dates, z_score.loc[short_entry_dates], 'v', markersize=5, color='red', label='Short Spread Entry')
    ax2.plot(long_entry_dates, z_score.loc[long_entry_dates], '^', markersize=5, color='green', label='Long Spread Entry')
    ax2.set_title(f'Z-Score and Trading Bands', fontsize=10)
    ax2.set_ylabel('Standard Deviations (σ)')
    ax2.legend()
    ax2.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

if __name__ == '__main__':
    # Test with dummy data
    np.random.seed(42)
    dates = pd.date_range('2019-01-01', '2024-01-01', freq='ME')
    returns = pd.Series(np.random.randn(len(dates)) * 0.04, index=dates)
    
    signals = pd.DataFrame(
        np.random.randn(len(dates), 5).clip(-2, 2),
        index=dates,
        columns=[f'Asset_{i}' for i in range(5)]
    )
    
    results = {
        'portfolio_returns': returns,
        'signals': signals,
        'metrics': {
            'total_return': 0.35,
            'annualized_return': 0.08,
            'annualized_volatility': 0.12,
            'sharpe_ratio': 0.67,
            'max_drawdown': -0.15,
            'win_rate': 0.52,
            'calmar_ratio': 0.53,
            'n_observations': len(dates)
        }
    }
    
    create_full_report(results, output_dir='test_results')