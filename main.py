#!/usr/bin/env python3
"""
Statistical Arbitrage Strategy - Main Entry Point

This script runs a complete backtest for pairs trading:
1. Identifies the best pair and loads its cointegration data.
2. Implements a Bollinger Band (Z-Score) strategy.
3. Runs backtest with transaction costs.
4. Outputs performance metrics and visualizations.

Usage:
    python main.py
    python main.py --pair XLK XLE

Author: Eduardo Medina
"""

import argparse
import sys
import os
import json

# Add src to path
# Assuming the structure requires 'src'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
# Imports from your data and cointegration modules
from data import DataFetcher, load_pairs_data, OIL_PRODUCERS 
# Assuming a function/utility to load cointegration results
# For now, we'll hardcode the best pair and its hedge ratio for simplicity.
from cointegration import run_cointegration_analysis  # ← ACTUALLY USE IT!
from strategy import PairsBollingerStrategy, Backtester 
from visualization import create_full_report, plot_spread_and_zscore


# --- Temporary Function to Load Cointegration Results ---


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Pairs Trading strategy backtest (Bollinger Bands)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    
    parser.add_argument(
        '--pair', 
        nargs=2, 
        default=OIL_PRODUCERS, # Default to USO/XOP, your top pair
        help='The two tickers to use for pairs trading (e.g., USO XOP)'
    )

    parser.add_argument(
        '--output', 
        type=str, 
        default='output/stat_arb_bb', 
        help='Directory to save plots and results'
    )

        
    parser.add_argument(
            '--cost',
            type=float,
            default=0.001,
            help='Transaction cost as decimal (default: 0.001 = 10bps)'
    )
        
    parser.add_argument(
            '--source',
            choices=['yfinance', 'alpaca'],
            default='alpaca',
            help='Data source (default: yfinance)'
    )
        
    parser.add_argument(
            '--no-plots',
            action='store_true',
            help='Skip generating plots'
    )



    parser.add_argument(
        '--z-score-multiplier',
        type=float,
        default=1.5,
        help='Multiplier for half-life (default: 3.0)'
)

    parser.add_argument(
        '--entry-std',
        type=float,
        default=2.0,
        help='Entry threshold (default: 2.0σ)'
)

    parser.add_argument(
        '--exit-std',
        type=float,
        default=0.5,
        help='Exit threshold (default: 0.5σ)'
)



    parser.add_argument(
        '--train-start',
        type=str,
        required=True,
        help='Training period start date (YYYY-MM-DD)'
)

    parser.add_argument(
        '--train-end',
        type=str,
        required=True,
        help='Training period end date (YYYY-MM-DD)'
)

    parser.add_argument(
        '--test-start',
        type=str,
        required=True,
        help='Test period start date (YYYY-MM-DD)'
)

    parser.add_argument(
        '--test-end',
        type=str,
        default=None,
        help='Test period end date (YYYY-MM-DD), default: today'
) 
    
    return parser.parse_args()



def main():
    args = parse_args()
    
    print(f"Fetching prices for: {args.pair}")

    
    # --- Configuration ---
    coint_results = run_cointegration_analysis(
        tickers=args.pair,
        data_source=args.source,
        start_date=args.train_start,
        end_date=args.train_end
    )
    if not coint_results['success']:
        print(f"Cointegration failed: {coint_results.get('error')}")
        sys.exit(1)
    print(" ")
    print("Cointegration tests...")
    # Extract hedge ratios (dictionary: {ticker: weight})
    hedge_ratios_dict = coint_results['hedge_ratios']
    
    # For PairsBollingerStrategy, we need the ratio for asset 1 relative to asset 0
    # The hedge ratio represents: Spread = Asset0 - beta * Asset1
    # So beta = hedge_ratios_dict[Asset1] / hedge_ratios_dict[Asset0]
    
    # Normalize to Asset0 = 1.0
    # NEW:
    asset0_weight = hedge_ratios_dict[args.pair[0]]
    asset1_weight = hedge_ratios_dict[args.pair[1]]

    # Calculate the hedge ratio to match the cointegrating relationship
    # Johansen gives: weight0 * Asset0 + weight1 * Asset1 = spread
    # strategy does: Asset0 - hedge_ratio * Asset1 = spread
    # Therefore: hedge_ratio = -weight1 / weight0

    hedge_ratio = -asset1_weight / asset0_weight
    
    print(f"Method: {coint_results['best_method']}")
    print(f"Hedge Ratios: {args.pair[0]}: {asset0_weight:.4f}, {args.pair[1]}: {asset1_weight:.4f}")
    print(f"Beta (for strategy): {hedge_ratio:.4f}")
    
    # Extract half-life
    half_life = coint_results['half_life']
    print(f"Half-Life: {half_life:.1f} days")
    
    # Calculate z-score lookback from half-life
    z_score_lookback = int(half_life * args.z_score_multiplier)
    print(f"Z-Score Window: {z_score_lookback} days ({args.z_score_multiplier}× half-life)")
    
    # Additional stats
    print(f"Statistical Properties:")
    print(f"ADF p-value: {coint_results['adf_pvalue']:.4f}")
    print(f"Hurst exponent: {coint_results['hurst']:.3f}")
    print(f"Variance ratio: {coint_results['variance_ratio']:.3f}")
    print(" ")
    # BACKTEST OUT OF SAMPLE
    print("Bactesting...")
    print(f"Fetching data from: {args.test_start} to {args.test_end} or 'today'")
    
    try:



        # Load log prices for strategy
        fetcher = DataFetcher(source=args.source)
        test_log_prices = fetcher.get_log_prices(
            args.pair,
            args.test_start,
            args.test_end
        )
        
        print(f"Loaded prices: {test_log_prices.shape}")

        test_raw_prices = np.exp(test_log_prices)  # Convert log prices back to raw!


        print(f"Date range: {test_log_prices.index[0].date()} to {test_log_prices.index[-1].date()}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # 3. Initialize Strategy and Backtester
    strategy = PairsBollingerStrategy(
        hedge_ratio=hedge_ratio,
        z_score_lookback=z_score_lookback,
        entry_std=args.entry_std,
        exit_std=args.exit_std,
        target_pair=args.pair
    )
    
    print(f"Strategy: Bollinger Bands (Z-Score)")
    print(f"Entry threshold: ±{args.entry_std}σ")
    print(f"Exit threshold: ±{args.exit_std}σ")




    
    signals = strategy.generate_signals(test_log_prices)
    
    n_trades = (signals != signals.shift(1)).any(axis=1).sum()
    print(f"Signals generated: {len(signals)} days")
    print(f"Position changes: {n_trades}")
    print(f"Trading frequency: {n_trades / len(signals) * 100:.1f}% of days")


    backtester = Backtester(transaction_cost=0.0001)

    # NOTE: The Backtester needs the RAW prices for P&L calculation (P_t / P_{t-1} - 1)
    backtester = Backtester(transaction_cost=args.cost)
    results = backtester.run(test_raw_prices, signals)
    metrics = results['metrics']
    print("")
    print("Results...")
    print(f"Total Return:        {metrics['total_return']*100:.2f}%")
    print(f"Annualized Return:   {metrics['annualized_return']*100:.2f}%")
    print(f"Annualized Vol:      {metrics['annualized_volatility']*100:.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")
    print(f"Calmar Ratio:        {metrics['calmar_ratio']:.2f}")
    print(f"Win Rate:            {metrics['win_rate']*100:.1f}%")
    if not args.no_plots:
        print(" ")
        
        os.makedirs(args.output, exist_ok=True)
        
        try:
            create_full_report(
                results=results,
                benchmark_returns=None,
                raw_prices=test_raw_prices,
                output_dir=args.output

            )
            plot_spread_and_zscore(
                raw_prices=test_raw_prices,
                signals_df=signals,
                hedge_ratio=hedge_ratio,
                z_score_lb=z_score_lookback,
                entry_std=args.entry_std,
                exit_std=args.exit_std,
                title=f'{args.pair[0]}/{args.pair[1]}',
                save_path=os.path.join(args.output, 'spread_zscore.png')
)

        except Exception as e:
            print(f" Could not generate some plots: {e}")
    
    # ========================================================================
    # STEP 10: Save Results
    # ========================================================================
    

    
    # Save strategy configuration
    config = {
        'pair': args.pair,
        'cointegration': {
            'method': coint_results['best_method'],
            'hedge_ratio': hedge_ratio,
            'half_life': half_life,
            'adf_pvalue': coint_results['adf_pvalue'],
            'hurst': coint_results['hurst']
        },
        'strategy': {
            'z_score_lookback': z_score_lookback,
            'entry_std': args.entry_std,
            'exit_std': args.exit_std
        },
        'backtest': metrics
    }
    
    config_file = os.path.join(args.output, 'backtest_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved: {config_file}")
    
    # Save returns
    returns_file = os.path.join(args.output, 'returns.csv')
    results['portfolio_returns'].to_csv(returns_file)
    print(f"Returns saved: {returns_file}")
    
    # Save signals
    signals_file = os.path.join(args.output, 'signals.csv')
    signals.to_csv(signals_file)
    print(f"Signals saved: {signals_file}")
      
    return results


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
