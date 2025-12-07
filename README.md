# stat-arb
Statistical-arbitrage startegy on cointegrating spreads, a simple implementation

Cointegration research on pairs and baskets and statistical arbitrage backtesting framework built in Python. This system identifies cointegrated asset pairs or baskets, constructs mean-reverting spreads, and implements Bollinger Band-based trading strategies.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

### Core Capabilities
- **Robust Cointegration Testing**: Johansen and VAR-based methods with automatic selection
- **Multi-Method Stationarity Tests**: ADF, KPSS, Variance Ratio, and Hurst Exponent
- **Dynamic Hedge Ratio Calibration**: Automatically determines optimal spread construction
- **Graphs**: Publication-quality charts and analytics
- **Multi-Data Source Support**: Works with both yfinance (free) and Alpaca

### Statistical Tests Implemented
- **Augmented Dickey-Fuller (ADF)**: Tests for unit root (non-stationarity)
- **KPSS Test**: Confirms stationarity from a different null hypothesis
- **Variance Ratio Test**: Detects mean-reversion characteristics
- **Hurst Exponent**: Measures long-term memory and trending behavior
- **Half-Life Calculation**: Estimates mean-reversion speed via OLS

### Trading Strategy
- **Entry**: Z-score exceeds ±2σ (configurable)
- **Exit**: Z-score returns to ±0.5σ (configurable)
- **Position Sizing**: Normalized to unit exposure with proper hedge ratio
- **Risk Management**: Transaction costs, max drawdown tracking, win rate analysis

---

## Example Results

### Sample Output: XLK/XLE Pair (Tech vs Energy)
```
COINTEGRATION ANALYSIS
 Method: Johansen
 Hedge Ratio: 0.1566
 Half-Life: 56.2 days
 ADF p-value: 0.0001 (Stationary ✓)
 Hurst Exponent: 0.45 (Mean-reverting ✓)

BACKTEST RESULTS (Out-of-Sample)
Total Return:        -8.96%
Annualized Return:   -3.12%
Sharpe Ratio:        -0.86
Max Drawdown:        -12.34%
Win Rate:            48.2%
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stat-arb.git
cd stat-arb

# Set up environment variables (for Alpaca)
cp .env.example .env
# Edit .env with your Alpaca API keys
```

### Basic Usage

```bash
# Run a simple pairs backtest
python main.py \
  --pair XLK XLE \
  --train-start 2020-01-01 \
  --train-end 2022-12-31 \
  --test-start 2023-01-01 \
  --test-end 2025-11-30 \
  --source alpaca \
  --output output/xlk_xle

# Run with custom parameters
python main.py \
  --pair HD LOW \
  --entry-std 1.5 \
  --exit-std 0.3 \
  --cost 0.001 \
  --z-score-multiplier 2.0
```

---

## Project Structure

```
stat-arb/
├── src/
│   ├── cointegration.py      # Cointegration tests (Johansen, VAR, stationarity)
│   ├── data.py                # Data fetching (yfinance, Alpaca)
│   ├── strategy.py            # Pairs trading strategy (Bollinger Bands)
│   ├── visualization.py       # Plotting and analytics
├── main.py                    # Command-line entry point
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
└── README.md                  # This file
```

---

## How It Works

### 1. **Cointegration Detection**

The system tests for cointegration using two methods and automatically selects the best:

**Johansen Test**
- Identifies long-run equilibrium relationships
- Returns eigenvectors (hedge ratios) directly
- Standard approach in academic literature

**VAR Eigenvalue Decomposition**
- Based on mean-reversion speed (kappa eigenvalues)
- Often finds faster mean-reverting spreads
- Better for short-term trading

**Selection Criteria** (5-point scoring system):
```python
score = sum([
    adf_stationary,           # ADF p-value < 0.05
    kpss_stationary,          # KPSS p-value > 0.05
    variance_ratio < 1,       # Mean-reverting
    hurst < 0.5,              # Not trending
    5 < half_life < 30 days   # Tradable timeframe
])
```

### 2. **Spread Construction**

Given cointegration vector `[1.0, -0.85]`:

```python
# For opposite-sign eigenvector (traditional pairs)
Spread = log(Asset1) - 0.85 × log(Asset2)

# Long spread when Z < -2σ:
# → Long Asset1, Short Asset2

# Short spread when Z > +2σ:
# → Short Asset1, Long Asset2
```

### 3. **Signal Generation**

```python
Z-Score = (Spread - Rolling_Mean) / Rolling_StdDev

if Z > +2.0σ:
    → Short the spread (sell overvalued, buy undervalued)
elif Z < -2.0σ:
    → Long the spread (buy undervalued, sell overvalued)
    
# Exit when Z returns to ±0.5σ
```

### 4. **Walk-Forward Optimization**

To avoid look-ahead bias:

```
Time:  |---Estimation (504d)---|--Trade (2×HL)--|---Estimation---|--Trade--|...
       t-504                   t                t+HL                        ...
                               ↑
                        Calibrate hedge ratio
                        Calculate half-life
                        Trade out-of-sample
```

---

## Understanding the Eigenvector

### Same-Sign Eigenvector: Basket Trading

```python
Eigenvector: [1.0, 0.16]  (e.g., XLK/XLE)
Spread = XLK + 0.16×XLE   (both positive)

# When spread is high (Z > +2σ):
# → SHORT both assets (basket is overvalued)

# When spread is low (Z < -2σ):
# → LONG both assets (basket is undervalued)
```

**Interpretation**: Assets move together, trading the basket's deviation from equilibrium.

### Opposite-Sign Eigenvector: Traditional Pairs

```python
Eigenvector: [1.0, -3.37]  (e.g., GD/LOW)
Spread = GD - 3.37×LOW     (opposite signs)

# When spread is high (Z > +2σ):
# → SHORT GD, LONG LOW (GD relatively expensive)

# When spread is low (Z < -2σ):
# → LONG GD, SHORT LOW (GD relatively cheap)
```

**Interpretation**: One asset is expensive relative to the other, trading the relative value.

---

## ⚙️ Configuration Options

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--pair` | Two tickers to trade | `['USO', 'XOP']` |
| `--train-start` | Training period start | Required |
| `--train-end` | Training period end | Required |
| `--test-start` | Test period start | Required |
| `--test-end` | Test period end | Today |
| `--entry-std` | Entry threshold (σ) | `2.0` |
| `--exit-std` | Exit threshold (σ) | `0.5` |
| `--cost` | Transaction cost (decimal) | `0.001` |
| `--z-score-multiplier` | Z-score lookback = HL × multiplier | `1.5` |
| `--source` | Data source (`yfinance` or `alpaca`) | `alpaca` |
| `--output` | Output directory | `output/stat_arb_bb` |
| `--no-plots` | Skip generating plots | `False` |

### Environment Variables

Create a `.env` file:

```bash
# Alpaca API (https://alpaca.markets)
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

---

## Visualization Outputs

The system generates comprehensive analytics:

1. **Equity Curve**: Strategy cumulative returns vs buy-and-hold
2. **Drawdown Chart**: Underwater equity curve
3. **Rolling Sharpe**: 1-year rolling Sharpe ratio
4. **Monthly Returns Heatmap**: Calendar-based performance
5. **Signal Distribution**: Position weight histogram
6. **Spread & Z-Score**: Entry/exit points overlaid on spread
7. **Strategy vs Assets**: Compare pairs strategy to individual holdings

All saved to `output/{pair_name}/` as high-resolution PNGs.

---

## Finding Good components for a Spread

### What Makes a Good Pair?

 **Statistical Requirements**:
- ADF p-value < 0.05 (stationary spread)
- Hurst exponent < 0.5 (mean-reverting)
- Half-life: 5-60 days (tradable timeframe)
- Correlation: 0.5-0.8 (related but not identical)

**Economic Rationale**:
- Same industry/sector (e.g., Coke vs Pepsi)
- Substitutable products (e.g., oil producers)
- Complementary businesses (e.g., airlines vs oil)

 **Avoid**:
- Correlation > 0.95 (too similar, spread doesn't vary)
- Correlation < 0.3 (unrelated, no cointegration)
- Different market caps by 10x+ (liquidity mismatch)
- Companies with merger/acquisition risk

### Suggested Pairs to Test

**Consumer Staples**:
- `KO` / `PEP` (Coca-Cola vs Pepsi)
- `PG` / `CL` (Procter & Gamble vs Colgate)

**Energy**:
- `XOM` / `CVX` (Exxon vs Chevron)
- `USO` / `XOP` (Oil ETF vs Oil Producers ETF)

**Technology**:
- `AAPL` / `MSFT` (Apple vs Microsoft)
- `NVDA` / `AMD` (GPU manufacturers)

**Financials**:
- `JPM` / `BAC` (JPMorgan vs Bank of America)
- `GS` / `MS` (Goldman Sachs vs Morgan Stanley)

**Retail**:
- `HD` / `LOW` (Home Depot vs Lowe's) *
- `TGT` / `WMT` (Target vs Walmart)

\* *Note: HD/LOW shows high correlation but fails stationarity tests. Good example of why statistical tests matter!*

---

## Advanced Usage

### Python API

```python
from src.cointegration import run_cointegration_analysis
from src.strategy import PairsBollingerStrategy
from src.data import DataFetcher

# 1. Test for cointegration
results = run_cointegration_analysis(
    tickers=['AAPL', 'MSFT'],
    lookback_days=504,
    data_source='alpaca',
    verbose=True
)

if results['success'] and results['adf_stationary']:
    # 2. Load test data
    fetcher = DataFetcher(source='alpaca')
    log_prices = fetcher.get_log_prices(
        ['AAPL', 'MSFT'],
        '2023-01-01',
        '2024-12-31'
    )
    
    # 3. Generate signals
    strategy = PairsBollingerStrategy(
        hedge_ratio=results['hedge_ratios']['MSFT'] / results['hedge_ratios']['AAPL'],
        z_score_lookback=int(results['half_life'] * 1.5),
        entry_std=2.0,
        exit_std=0.5,
        target_pair=['AAPL', 'MSFT']
    )
    
    signals = strategy.generate_signals(log_prices)
    
    # 4. Backtest
    from src.strategy import Backtester
    backtester = Backtester(transaction_cost=0.001)
    raw_prices = np.exp(log_prices)
    backtest_results = backtester.run(raw_prices, signals)
    
    print(backtest_results['metrics'])
```

### Batch Testing Multiple Pairs

```python
from src.data import COKE_PEPSI, GOLD_MINERS, OIL_PRODUCERS

pairs_to_test = [
    COKE_PEPSI,
    GOLD_MINERS,
    OIL_PRODUCERS,
    ['AAPL', 'MSFT'],
    ['JPM', 'BAC']
]

results = {}
for pair in pairs_to_test:
    coint = run_cointegration_analysis(
        tickers=pair,
        lookback_days=504,
        verbose=False
    )
    results[f"{pair[0]}/{pair[1]}"] = coint

# Filter to tradable pairs
tradable = {
    name: res for name, res in results.items()
    if res['success'] and res['adf_stationary'] and res['half_life_tradable']
}

print(f"Found {len(tradable)} tradable pairs out of {len(pairs_to_test)}")
```

## References

### Papers
- Engle & Granger (1987) - "Co-integration and Error Correction"
- Johansen (1991) - "Estimation and Hypothesis Testing of Cointegration Vectors"
- Gatev, Goetzmann & Rouwenhorst (2006) - "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
---
### Areas for Contribution
- [ ] Support for 3+ asset baskets
- [ ] Additional cointegration tests (Phillips-Ouliaris, etc.)
- [ ] Machine learning-based entry/exit optimization
- [ ] Real-time trading integration
- [ ] Options-based pairs trading
- [ ] More sophisticated risk management (Kelly criterion, etc.)

---



---

**⭐ Star this repo if you find it useful!**
