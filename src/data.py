"""
Data fetching module for momentum and statistical arbitrage strategies.
Supports multiple data sources: yfinance (default) and Alpaca.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Union, Tuple
from dotenv import load_dotenv
from alpaca.data.enums import DataFeed
load_dotenv()
import os

# Primary: yfinance (no API key required)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Secondary: Alpaca (requires API key)
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


# ============================================================================
# UNIVERSE DEFINITIONS
# ============================================================================
# Sector etfs pairs
INDUSTRIALS_MATERIALS = ['XLI', 'XLB']
FINANCE_REGIONAL = ['XLF', 'KRE']
TECH_SEMIS = [ 'XLK', 'SMH']
ENERGY_OIL = ['XLE', 'USO']
# Commodity-related pairs
GOLD_MINERS = ['GLD', 'GDX']
OIL_PRODUCERS = ['USO', 'XOP']
# Classic equity pairs
COKE_PEPSI = ['KO', 'PEP']
VISA_MASTERCARD = ['V', 'MA']
HOME_DEPOT_LOWES = ['HD', 'LOW']

# Likely to fail
TECH_ENERGY = ['XLK', 'XLE']

# Default universe for backtesting
DEFAULT_UNIVERSE = COKE_PEPSI


class DataFetcher:
    """
    Unified interface for fetching historical price data.
    Supports yfinance (free) and Alpaca (requires API key).
    """
    
    def __init__(self, source: str = 'yfinance', alpaca_key: str = None, alpaca_secret: str = None):
        """
        Initialize the data fetcher.
        
        Args:
            source: 'yfinance' or 'alpaca'
            alpaca_key: Alpaca API key (optional, can use env var ALPACA_API_KEY)
            alpaca_secret: Alpaca secret key (optional, can use env var ALPACA_SECRET_KEY)
        """
        self.source = source.lower()
        
        if self.source == 'yfinance':
            if not YFINANCE_AVAILABLE:
                raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        elif self.source == 'alpaca':
            if not ALPACA_AVAILABLE:
                raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")
            
            # Get API keys from arguments or environment
            self.alpaca_key = alpaca_key or os.getenv('ALPACA_API_KEY')
            self.alpaca_secret = alpaca_secret or os.getenv('ALPACA_SECRET_KEY')
            
            if not self.alpaca_key or not self.alpaca_secret:
                raise ValueError(
                    "Alpaca API credentials required. Either pass them as arguments "
                    "or set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
                )
            
            self.alpaca_client = StockHistoricalDataClient(self.alpaca_key, self.alpaca_secret)
        
        else:
            raise ValueError(f"Unknown data source: {source}. Use 'yfinance' or 'alpaca'.")
    
    def get_prices(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        price_col: str = 'Adj Close'
    ) -> pd.DataFrame:
        """
        Fetch historical prices for a list of tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (str 'YYYY-MM-DD' or datetime)
            end_date: End date (default: today)
            price_col: Which price to use ('Adj Close', 'Close', 'Open', etc.)
        
        Returns:
            DataFrame with dates as index and tickers as columns
        """
        if end_date is None:
            end_date = datetime.now()
        
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if self.source == 'yfinance':
            return self._fetch_yfinance(tickers, start_date, end_date, price_col)
        elif self.source == 'alpaca':
            return self._fetch_alpaca(tickers, start_date, end_date)
    
    def _fetch_yfinance(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        price_col: str
    ) -> pd.DataFrame:
        """Fetch data using yfinance."""
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
        
        if len(tickers) == 1:
            # Single ticker returns different structure
            prices = data[[price_col]].copy()
            prices.columns = tickers
        else:
            prices = data[price_col].copy()
        
        return prices
    
    def _fetch_alpaca(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data using Alpaca API."""
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX
        )
        
        bars = self.alpaca_client.get_stock_bars(request_params)
        
        # Convert to DataFrame
        data_dict = {}
        for ticker in tickers:
            if ticker in bars.data:
                ticker_bars = bars.data[ticker]
                data_dict[ticker] = pd.Series(
                    [bar.close for bar in ticker_bars],
                    index=[bar.timestamp for bar in ticker_bars]
                )
        
        prices = pd.DataFrame(data_dict)
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        
        return prices
    
    # ========================================================================
    # STAT ARB SPECIFIC METHODS
    # ========================================================================
    
    def get_log_prices(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        price_col: str = 'Adj Close'
    ) -> pd.DataFrame:
        """
        Fetch historical prices and return log-transformed prices.
        
        Log prices are essential for cointegration analysis because:
        - Makes price series more normally distributed
        - Converts multiplicative relationships to additive
        - Log returns are time-additive: log(P2/P1) = log(P2) - log(P1)
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (str 'YYYY-MM-DD' or datetime)
            end_date: End date (default: today)
            price_col: Which price to use ('Adj Close', 'Close', etc.)
        
        Returns:
            DataFrame with log-transformed prices
        """
  
        
        # 1. Fetch raw prices
        prices = self.get_prices(tickers, start_date, end_date) # Assuming this calls Alpaca and returns a DataFrame
        
        if prices.empty:
            return prices
            
        # 2. Set Index to Datetime and set frequency ('B' for Business Day)
        # This is the step that satisfies statsmodels and suppresses the ValueWarning.
        prices.index = pd.to_datetime(prices.index).normalize()
        
        # Use reindex to align all data to a full business day calendar.
        full_index = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq='B')
        prices = prices.reindex(full_index)

        # 3. Log-transform and handle missing data.
        # Missing values (NaNs from reindex, holidays, or missing data) are filled
        # using the Forward Fill method (prevents data loss but is a data assumption).
        log_prices = np.log(prices).ffill()
        
        # Drop any rows with NaN after log transformation
        log_prices = log_prices.dropna()
        
        return log_prices
    
    def validate_data_quality(
        self,
        prices: pd.DataFrame,
        min_observations: int = 252,
        max_missing_pct: float = 0.05
    ) -> dict:
        """
        Validate data quality for statistical arbitrage.
        
        Args:
            prices: DataFrame of prices
            min_observations: Minimum required data points (default: 252 = 1 year)
            max_missing_pct: Maximum allowed missing data percentage
        
        Returns:
            dict with validation results
        """
        n_observations = len(prices)
        n_tickers = len(prices.columns)
        
        # Check for missing data
        missing_counts = prices.isnull().sum()
        missing_pct = missing_counts / n_observations
        
        # Check for sufficient data
        sufficient_data = n_observations >= min_observations
        
        # Check for excessive missing values
        clean_tickers = missing_pct[missing_pct <= max_missing_pct].index.tolist()
        removed_tickers = missing_pct[missing_pct > max_missing_pct].index.tolist()
        
        validation_results = {
            'n_observations': n_observations,
            'n_tickers': n_tickers,
            'sufficient_data': sufficient_data,
            'clean_tickers': clean_tickers,
            'removed_tickers': removed_tickers,
            'missing_pct': missing_pct.to_dict(),
            'passed': sufficient_data and len(clean_tickers) >= 2
        }
        
        return validation_results


def get_sp500_tickers() -> List[str]:
    """
    Fetch current S&P 500 constituents from Wikipedia.
    
    Returns:
        List of ticker symbols
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].tolist()
    # Clean up tickers (replace . with - for yfinance compatibility)
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_prices(
    tickers: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    source: str = 'alpaca'
) -> pd.DataFrame:
    """
    Quick helper to load price data with sensible defaults.
    
    Args:
        tickers: List of tickers (default: STAT_ARB_BASKET)
        start_date: Start date (default: 5 years ago)
        end_date: End date (default: today)
        source: Data source ('yfinance' or 'alpaca')
    
    Returns:
        DataFrame of adjusted close prices
    """
    if tickers is None:
        tickers = DEFAULT_UNIVERSE
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    fetcher = DataFetcher(source=source)
    return fetcher.get_prices(tickers, start_date, end_date)


def load_cointegration_data(
    tickers: List[str] = None,
    lookback_days: int = 504,  # ~2 years (recommended for cointegration)
    start_date: str = None,
    end_date: str = None,
    source: str = 'alpaca',
    validate: bool = True
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Load and prepare data specifically for cointegration analysis.
    
    Args:
        tickers: List of tickers (default: STAT_ARB_BASKET)
        lookback_days: Days of history (default: 504 = ~2 years)
        end_date: End date (default: today)
        source: Data source ('yfinance' or 'alpaca')
        validate: Whether to run data quality checks
    
    Returns:
        tuple: (log_prices_df, validation_results)
    """
    if tickers is None:
        tickers = STAT_ARB_BASKET
    
    if start_date and end_date:
    # Use explicit dates
        pass  # Just use them directly
    elif end_date:
        # Calculate start from lookback
        calendar_days = int(lookback_days * 1.6)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=calendar_days)
        start_date = start_dt.strftime('%Y-%m-%d')
    else:
        # Default behavior (existing code)
        end_date = datetime.now().strftime('%Y-%m-%d')
        calendar_days = int(lookback_days * 1.6)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=calendar_days)
        start_date = start_dt.strftime('%Y-%m-%d')

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch log prices
    fetcher = DataFetcher(source=source)
    log_prices = fetcher.get_log_prices(tickers, start_date, end_date)
    
    print(f"{len(log_prices)} observations")
    # Validate data quality
    validation_results = None
    if validate:
        validation_results = fetcher.validate_data_quality(log_prices)
        
        if validation_results['passed']:
            pass
        else:

            if not validation_results['sufficient_data']:
                print(f"  Insufficient data: {validation_results['n_observations']} < 252")
            if validation_results['removed_tickers']:
                print(f"  Removed tickers (too much missing data): {validation_results['removed_tickers']}")
        
        # Filter to clean tickers only
        if validation_results['removed_tickers']:
            log_prices = log_prices[validation_results['clean_tickers']]
            print(f" Filtered to {len(log_prices.columns)} clean tickers")
    
    print(f"\nFinal dataset:")
    print(f"  Shape: {log_prices.shape}")
    print(f"  Date range: {log_prices.index[0]} to {log_prices.index[-1]}")
    print(f"  Tickers: {list(log_prices.columns)}")
    
    return log_prices, validation_results


def load_pairs_data(
    pair: List[str],
    lookback_days: int = 504,
    source: str = 'alpaca'
) -> pd.DataFrame:
    """
    Quick helper to load data for a pair of assets.
    
    Args:
        pair: List of 2 ticker symbols
        lookback_days: Days of history (default: 504 = ~2 years)
        source: Data source ('yfinance' or 'alpaca')
    
    Returns:
        DataFrame of log prices
    """
    if len(pair) != 2:
        raise ValueError("Pair must contain exactly 2 tickers")
    
    log_prices, _ = load_cointegration_data(
        tickers=pair,
        lookback_days=lookback_days,
        source=source,
        validate=True
    )
    
    return log_prices


if __name__ == '__main__':
    # Example usage
    print("="*70)
    print("TESTING DATA FETCHER")
    print("="*70)
    
    # Test 1: Basic price loading
    print("\n1. Testing basic price loading...")
    prices = load_prices(tickers=['AAPL', 'MSFT'], start_date='2024-01-01')
    print(f"   ✓ Loaded {len(prices)} days")
    
    # Test 2: Cointegration data loading
    print("\n2. Testing cointegration data loading...")
    log_prices, validation = load_cointegration_data(
        tickers=['NVDA', 'AMD', 'INTC'],
        lookback_days=252
    )
    
    # Test 3: Pairs data loading
    print("\n3. Testing pairs data loading...")
    pair_prices = load_pairs_data(['PEP', 'KO'])
    print(pair_prices.head(10))
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)