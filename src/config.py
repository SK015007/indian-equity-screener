# Stock Screener Configuration — Default Thresholds

# === Technical Conditions ===
EMA_LONG = 200          # Long-term EMA period
EMA_SHORT = 50          # Short-term EMA period
CROSSOVER_LOOKBACK = 20 # Number of sessions to look back for 200 EMA crossover
VOLUME_MULTIPLIER = 1.5 # Minimum volume ratio vs 20-day average on breakout day
VOLUME_AVG_PERIOD = 20  # Period for average volume calculation
RSI_PERIOD = 14
RSI_LOWER = 50          # Minimum RSI
RSI_UPPER = 70          # Maximum RSI (avoid overbought)
MAX_ABOVE_EMA_PCT = 8.0 # Max % price can be above 200 EMA
ADX_PERIOD = 14         # ADX period
ADX_MIN = 20            # Minimum ADX for trend strength confirmation
ATR_PERIOD = 14         # ATR period for stop-loss calculation
ATR_SL_MULTIPLIER = 2.0 # Stop loss = entry - (ATR * multiplier)
NIFTY_REGIME_CHECK = True  # Check if Nifty 50 is above its 200 DMA

# === VCP (Volatility Contraction Pattern) ===
VCP_MAX_FROM_52W_HIGH_PCT = 25.0  # Price must be within 25% of 52W high
VCP_MIN_ABOVE_52W_LOW_PCT = 25.0  # Price must be at least 25% above 52W low
VCP_BASE_LENGTH = 90              # Days to look back for the consolidation base
VCP_MIN_CONTRACTIONS = 2          # Minimum number of tightening contractions
VCP_VOL_CONTRACTION_RATIO = 0.8   # Recent vol / prior vol must be below this
VCP_BREAKOUT_VOL_MULT = 1.5       # Min volume ratio on breakout vs 50-day avg
VCP_PIVOT_PROXIMITY_PCT = 3.0     # Max % below pivot high to qualify

# === Fundamental Conditions ===
MIN_MARKET_CAP_CR = 1000       # Minimum market cap in crores (₹)
MIN_SALES_GROWTH_PCT = 10.0    # Minimum YoY sales growth %
MIN_PROFIT_GROWTH_PCT = 10.0   # Minimum YoY profit growth %
MIN_ROE_PCT = 15.0             # Minimum Return on Equity %
MAX_DEBT_TO_EQUITY = 0.5       # Maximum Debt-to-Equity ratio
MIN_PROMOTER_HOLDING_PCT = 50.0  # Minimum promoter holding %
MAX_PLEDGED_PCT = 5.0          # Maximum pledged promoter shares %

# === Additional Filters ===
MIN_AVG_TRADED_VALUE_CR = 5.0  # Minimum avg daily traded value in crores (₹)

# === Data Settings ===
HISTORY_DAYS = 365             # Days of price history to fetch
CRORE = 1e7                    # 1 crore = 10 million
MAX_WORKERS = 10               # Parallel threads for fundamental data fetching
