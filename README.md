# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 19:38:12 2026

@author: Sagnik Chakraborty
"""

# ============================================================
# ALPHAREDGE CAPITAL — CAPSTONE PROJECT
# Banking Sector Stock Analysis
# Your Name: SAGNIK CHAKRABORTY__________-________________
# Date: 15-08-2026________________________________-
# ============================================================

import yfinance          as yf
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings

from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.cluster         import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (r2_score, mean_squared_error,
                                     accuracy_score, confusion_matrix,
                                     classification_report)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ── Your 5 bank stocks ──────────────────────────────────────
STOCKS  = ['HDFC Bank', 'ICICI Bank', 'Kotak', 'Axis Bank', 'SBI']
TICKERS = ['HDFCBANK.NS','ICICIBANK.NS','KOTAKBANK.NS',
           'AXISBANK.NS','SBIN.NS']
COLORS  = ['#185FA5','#D85A30','#1D9E75','#7F77DD','#BA7517']
RISK_FREE = 6.5   # Indian 10-year Gsec yield

# ── Download 2 years of data ────────────────────────────────
raw = yf.download(
    tickers=TICKERS,
    start='2024-08-14',
    end='2026-08-15',
    interval='1d',
    auto_adjust=True,
    progress=False
)
df     = raw['Close'].copy()
df.columns = STOCKS
df     = df.ffill()

volume = raw['Volume'].copy()
volume.columns = STOCKS

returns = df.pct_change() * 100
returns = returns.dropna()

# Pre-calculate annualised metrics
ann_ret = (returns.mean() * 252).round(2)
ann_vol = (returns.std()  * np.sqrt(252)).round(2)
sharpe  = ((ann_ret - RISK_FREE) / ann_vol).round(3)

print('Data ready:', df.shape[0], 'days,', df.shape[1], 'bank stocks')
print('Period:', df.index[0].date(), 'to', df.index[-1].date())

# ============================================================
# TASK A2 — DATASET INSPECTION
# ============================================================

print("\n================ DATA SHAPE ================")
print(df.shape)

print("\n================ FIRST 5 ROWS ================")
print(df.head(5))

print("\n================ DATA INFORMATION ================")
print(df.info())

print("\n================ DESCRIPTIVE STATISTICS ================")
print(df.describe())

# ============================================================
# TASK A3 — MISSING VALUE CHECK
# ============================================================

print("\n================ MISSING VALUES BEFORE CLEANING ================")
print(raw['Close'].isnull().sum())

print("\n================ MISSING VALUES AFTER FFILL ================")
print(df.isnull().sum())
# ============================================================
# TASK A3 — MISSING VALUE CHECK AND HANDLING
# ============================================================

# Check missing values BEFORE cleaning
print("\n================ MISSING VALUES BEFORE CLEANING ================")
print(df.isnull().sum())

# Handle missing values using forward fill
df = df.ffill()

# Check missing values AFTER cleaning
print("\n================ MISSING VALUES AFTER FFILL ================")
print(df.isnull().sum())

# ============================================================
# TASK A4 — DAILY RETURNS
# ============================================================

returns = df.pct_change() * 100
returns = returns.dropna()

print("\n================ DAILY RETURNS — FIRST 5 ROWS ================")
print(returns.head(5))

print("\n================ DAILY RETURN STATISTICS ================")
print(returns.describe())

# ============================================================
# PART B — EXPLORATORY DATA ANALYSIS
# B1 — PRICE HISTORY WITH MA20 AND MA50
# ============================================================

fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

for i, stock in enumerate(STOCKS):

    # Calculate moving averages
    ma20 = df[stock].rolling(window=20).mean()
    ma50 = df[stock].rolling(window=50).mean()

    # Plot closing price
    axes[i].plot(
        df.index,
        df[stock],
        label='Closing Price'
    )

    # Plot MA20
    axes[i].plot(
        df.index,
        ma20,
        label='MA20'
    )

    # Plot MA50
    axes[i].plot(
        df.index,
        ma50,
        label='MA50'
    )

    axes[i].set_title(f'{stock} — Price History with Moving Averages')
    axes[i].set_ylabel('Price (₹)')
    axes[i].legend()

axes[-1].set_xlabel('Date')

fig.suptitle(
    'Indian Banking Stocks — Price History and Moving Averages',
    fontsize=16,
    y=0.995
)

plt.tight_layout()

# Save chart
plt.savefig(
    'SagnikChakraborty_chart01_price_history.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# B2 — RETURN DISTRIBUTION
# ============================================================

plt.figure(figsize=(12, 7))

for stock in STOCKS:
    plt.hist(
        returns[stock],
        bins=40,
        alpha=0.45,
        label=stock
    )

# Zero-return reference line
plt.axvline(
    x=0,
    linestyle='--',
    linewidth=1.5,
    label='Zero Return'
)

plt.title(
    'Distribution of Daily Returns — Indian Banking Stocks',
    fontsize=15
)

plt.xlabel('Daily Return (%)')
plt.ylabel('Frequency')

plt.legend()

plt.tight_layout()

# Save using required naming convention
plt.savefig(
    'SagnikChakraborty_chart02_return_distribution.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# B3 — CORRELATION HEATMAP
# ============================================================

# Calculate correlation matrix using DAILY RETURNS
corr_matrix = returns.corr()

print("\n================ RETURN CORRELATION MATRIX ================")
print(corr_matrix)

# ------------------------------------------------------------
# Create heatmap
# ------------------------------------------------------------

plt.figure(figsize=(12, 9))

heatmap = plt.imshow(
    corr_matrix,
    interpolation='nearest',
    aspect='auto',
    cmap='RdYlGn',       # Red → Yellow → Green
    vmin=-1,
    vmax=1
)

# ------------------------------------------------------------
# Add colorbar
# ------------------------------------------------------------

cbar = plt.colorbar(heatmap)
cbar.set_label('Correlation', fontsize=12)

# ------------------------------------------------------------
# Axis labels
# ------------------------------------------------------------

plt.xticks(
    range(len(STOCKS)),
    STOCKS,
    rotation=45,
    ha='right'
)

plt.yticks(
    range(len(STOCKS)),
    STOCKS
)

# ------------------------------------------------------------
# Add correlation values inside each cell
# ------------------------------------------------------------

for i in range(len(STOCKS)):
    for j in range(len(STOCKS)):

        value = corr_matrix.iloc[i, j]

        # White text for strong positive correlations
        # Black text for weaker correlations
        if value >= 0.70:
            text_color = 'white'
        else:
            text_color = 'black'

        plt.text(
            j,
            i,
            f'{value:.2f}',
            ha='center',
            va='center',
            fontsize=14,
            fontweight='bold',
            color=text_color
        )

# ------------------------------------------------------------
# Title and labels
# ------------------------------------------------------------

plt.title(
    'Correlation Matrix of Daily Returns — Indian Banking Stocks',
    fontsize=18,
    fontweight='bold',
    pad=15
)

plt.xlabel('Bank', fontsize=13)
plt.ylabel('Bank', fontsize=13)

# ------------------------------------------------------------
# Add white borders between cells
# ------------------------------------------------------------

for i in range(len(STOCKS) + 1):
    plt.axhline(i - 0.5, linewidth=1, color='white')

    plt.axvline(i - 0.5, linewidth=1, color='white')

plt.tight_layout()

# ------------------------------------------------------------
# Save chart
# ------------------------------------------------------------

plt.savefig(
    'SagnikChakraborty_chart03_correlation_heatmap.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# B4 — ₹10,000 CUMULATIVE RETURN
# ============================================================

# Starting investment
initial_investment = 10000

# Calculate cumulative growth of ₹10,000
cumulative_value = (
    1 + returns / 100
).cumprod() * initial_investment

# ------------------------------------------------------------
# Create chart
# ------------------------------------------------------------

plt.figure(figsize=(14, 8))

for stock in STOCKS:
    plt.plot(
        cumulative_value.index,
        cumulative_value[stock],
        label=stock
    )

# Starting investment reference line
plt.axhline(
    y=initial_investment,
    linestyle='--',
    linewidth=1.5,
    label='Initial Investment (₹10,000)'
)

plt.title(
    'Growth of ₹10,000 Investment — Indian Banking Stocks',
    fontsize=16
)

plt.xlabel('Date')
plt.ylabel('Portfolio Value (₹)')

plt.legend()

# Format y-axis as rupees
plt.gca().yaxis.set_major_formatter(
    mticker.StrMethodFormatter('₹{x:,.0f}')
)

plt.tight_layout()

# ------------------------------------------------------------
# Save chart using required naming convention
# ------------------------------------------------------------

plt.savefig(
    'SagnikChakraborty_chart04_cumulative_return.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# ------------------------------------------------------------
# Final investment values
# ------------------------------------------------------------

final_values = cumulative_value.iloc[-1].round(2)

print("\n================ ₹10,000 FINAL VALUES ================")
print(final_values)

# Calculate cumulative percentage returns
cumulative_returns = (
    (final_values / initial_investment) - 1
) * 100

print("\n================ CUMULATIVE RETURNS (%) ================")
print(cumulative_returns.round(2))

# Identify highest cumulative return
best_stock = cumulative_returns.idxmax()

print("\nBest performing stock:", best_stock)
print(
    "Highest cumulative return:",
    round(cumulative_returns.max(), 2),
    "%"
)

# ============================================================
# B5 — RISK–RETURN ANALYSIS
# ============================================================

# ------------------------------------------------------------
# Create summary table
# ------------------------------------------------------------

risk_return = pd.DataFrame({
    'Annualised Return (%)': ann_ret,
    'Annualised Volatility (%)': ann_vol,
    'Sharpe Ratio': sharpe
})

print("\n================ RISK–RETURN SUMMARY ================")
print(risk_return)

# ------------------------------------------------------------
# Create Risk–Return Scatter Plot
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))

for stock in STOCKS:
    
    plt.scatter(
        ann_vol[stock],
        ann_ret[stock],
        s=150,
        label=stock
    )
    
    # Add stock name next to each point
    plt.annotate(
        stock,
        (ann_vol[stock], ann_ret[stock]),
        xytext=(8, 8),
        textcoords='offset points',
        fontsize=11
    )

# ------------------------------------------------------------
# Add risk-free return line
# ------------------------------------------------------------

plt.axhline(
    y=RISK_FREE,
    linestyle='--',
    linewidth=1.5,
    label=f'Risk-Free Rate ({RISK_FREE}%)'
)

# ------------------------------------------------------------
# Titles and labels
# ------------------------------------------------------------

plt.title(
    'Risk–Return Profile of Indian Banking Stocks',
    fontsize=16
)

plt.xlabel('Annualised Volatility (%)')
plt.ylabel('Annualised Return (%)')

plt.legend()

plt.tight_layout()

# ------------------------------------------------------------
# Save chart
# ------------------------------------------------------------

plt.savefig(
    'SagnikChakraborty_chart05_risk_return.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# SHARPE RATIO RANKING
# ============================================================

print("\n================ SHARPE RATIO RANKING ================")

sharpe_ranking = sharpe.sort_values(ascending=False)

print(sharpe_ranking)

print(
    "\nHighest Sharpe Ratio:",
    sharpe_ranking.index[0],
    "=",
    sharpe_ranking.iloc[0]
)

# ============================================================
# C1 — FEATURE ENGINEERING
# HDFC BANK
# ============================================================

# Select HDFC Bank price and returns
hdfc_price = df['HDFC Bank']
hdfc_returns = returns['HDFC Bank']

# ------------------------------------------------------------
# Create feature table
# ------------------------------------------------------------

features = pd.DataFrame(index=hdfc_returns.index)

# 1. Today's return
features['return_today'] = hdfc_returns

# 2. Previous day's return
features['return_1d_ago'] = hdfc_returns.shift(1)

# 3. Five-day return
features['return_5d'] = (
    hdfc_price.pct_change(periods=5) * 100
)

# 4. Price relative to 20-day moving average
ma20 = hdfc_price.rolling(window=20).mean()

features['price_vs_ma20'] = (
    (hdfc_price / ma20) - 1
) * 100

# 5. 30-day rolling volatility
features['volatility_30d'] = (
    hdfc_returns.rolling(window=30).std()
)

# ------------------------------------------------------------
# 6. Target: NEXT day's direction
# 1 = UP
# 0 = DOWN
# ------------------------------------------------------------

next_day_return = hdfc_returns.shift(-1)

features['target_direction'] = (
    next_day_return > 0
).astype(float)

# The final observation has no known next-day return.
# Set its target to missing.
features.loc[next_day_return.isna(), 'target_direction'] = np.nan

# ------------------------------------------------------------
# Remove rows with missing values
# ------------------------------------------------------------

features = features.dropna()

# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n================ C1 FEATURE TABLE — HDFC BANK ================")
print(features.head(3))

print("\n================ LAST 3 ROWS ================")
print(features.tail(3))

print("\n================ FEATURE TABLE INFORMATION ================")
print(features.info())

print("\nFeature table shape:", features.shape)

# ============================================================
# C2 — LINEAR REGRESSION
# HDFC BANK
# ============================================================

# ------------------------------------------------------------
# Define X and y
# ------------------------------------------------------------

X = features[
    [
        'return_1d_ago',
        'return_5d',
        'price_vs_ma20',
        'volatility_30d'
    ]
]

y = features['return_today']

# ------------------------------------------------------------
# Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------------------------
# Create and train Linear Regression model
# ------------------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

y_pred = linear_model.predict(X_test)

# ------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------

r2 = r2_score(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n================ C2 — LINEAR REGRESSION ================")

print("\nR² Score:", round(r2, 4))

print("RMSE:", round(rmse, 4))

# ------------------------------------------------------------
# Regression coefficients
# ------------------------------------------------------------

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': linear_model.coef_
})

print("\n================ REGRESSION COEFFICIENTS ================")
print(coefficients)

print("\nIntercept:", round(linear_model.intercept_, 4))

# ============================================================
# C3 — LOGISTIC REGRESSION
# HDFC BANK
# ============================================================

# ------------------------------------------------------------
# Define X and y
# ------------------------------------------------------------

X_logit = features[
    [
        'return_1d_ago',
        'return_5d',
        'price_vs_ma20',
        'volatility_30d'
    ]
]

y_logit = features['target_direction']

# ------------------------------------------------------------
# Train-test split
# ------------------------------------------------------------

X_train_logit, X_test_logit, y_train_logit, y_test_logit = train_test_split(
    X_logit,
    y_logit,
    test_size=0.20,
    random_state=42,
    stratify=y_logit
)

# ------------------------------------------------------------
# Standardise explanatory variables
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_logit)
X_test_scaled = scaler.transform(X_test_logit)

# ------------------------------------------------------------
# Create and train Logistic Regression model
# ------------------------------------------------------------

logit_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

logit_model.fit(
    X_train_scaled,
    y_train_logit
)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

y_pred_logit = logit_model.predict(X_test_scaled)

# Predicted probabilities
y_prob_logit = logit_model.predict_proba(X_test_scaled)[:, 1]

# ------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test_logit,
    y_pred_logit
)

cm = confusion_matrix(
    y_test_logit,
    y_pred_logit
)

print("\n================ C3 — LOGISTIC REGRESSION ================")

print("\nAccuracy:", round(accuracy, 4))

print("\n================ CONFUSION MATRIX ================")
print(cm)

print("\n================ CLASSIFICATION REPORT ================")
print(
    classification_report(
        y_test_logit,
        y_pred_logit,
        target_names=['DOWN', 'UP']
    )
)

# ------------------------------------------------------------
# Logistic regression coefficients
# ------------------------------------------------------------

logit_coefficients = pd.DataFrame({
    'Feature': X_logit.columns,
    'Coefficient': logit_model.coef_[0]
})

print("\n================ LOGISTIC COEFFICIENTS ================")
print(logit_coefficients)

print("\nIntercept:", round(logit_model.intercept_[0], 4))

# ============================================================
# C4 — K-MEANS CLUSTERING
# BANKING STOCKS
# ============================================================

# ------------------------------------------------------------
# Create clustering dataset
# ------------------------------------------------------------

cluster_data = pd.DataFrame({
    'Annualised_Return': ann_ret,
    'Annualised_Volatility': ann_vol
})

print("\n================ CLUSTERING DATA ================")
print(cluster_data)

# ------------------------------------------------------------
# Standardise variables
# ------------------------------------------------------------

cluster_scaler = StandardScaler()

cluster_scaled = cluster_scaler.fit_transform(
    cluster_data
)

# ------------------------------------------------------------
# Apply K-Means
# Using 2 clusters:
#   Cluster 0 = one risk-return group
#   Cluster 1 = another risk-return group
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(
    cluster_scaled
)

# Add cluster labels
cluster_data['Cluster'] = cluster_labels

print("\n================ K-MEANS RESULTS ================")
print(cluster_data.sort_values('Cluster'))

# ------------------------------------------------------------
# Cluster membership
# ------------------------------------------------------------

print("\n================ CLUSTER MEMBERS ================")

for cluster in sorted(cluster_data['Cluster'].unique()):

    members = cluster_data[
        cluster_data['Cluster'] == cluster
    ].index.tolist()

    print(
        f"Cluster {cluster}:",
        ", ".join(members)
    )

# ------------------------------------------------------------
# Plot clusters
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))

for cluster in sorted(cluster_data['Cluster'].unique()):

    cluster_points = cluster_data[
        cluster_data['Cluster'] == cluster
    ]

    plt.scatter(
        cluster_points['Annualised_Volatility'],
        cluster_points['Annualised_Return'],
        s=180,
        label=f'Cluster {cluster}'
    )

    # Add bank names
    for stock in cluster_points.index:

        plt.annotate(
            stock,
            (
                cluster_points.loc[stock, 'Annualised_Volatility'],
                cluster_points.loc[stock, 'Annualised_Return']
            ),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=11
        )

# ------------------------------------------------------------
# Titles and labels
# ------------------------------------------------------------

plt.title(
    'K-Means Clustering of Indian Banking Stocks',
    fontsize=16
)

plt.xlabel('Annualised Volatility (%)')
plt.ylabel('Annualised Return (%)')

plt.legend()

plt.tight_layout()

# ------------------------------------------------------------
# Save chart
# ------------------------------------------------------------

plt.savefig(
    'SagnikChakraborty_chart06_kmeans_clusters.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# C5 — FINAL INVESTMENT RECOMMENDATION
# ============================================================

# ------------------------------------------------------------
# Create final performance summary
# ------------------------------------------------------------

final_summary = pd.DataFrame({
    'Cumulative_Return_%': (
        ((df.iloc[-1] / df.iloc[0]) - 1) * 100
    ).round(2),

    'Annualised_Return_%': ann_ret,

    'Annualised_Volatility_%': ann_vol,

    'Sharpe_Ratio': sharpe
})

print("\n================ FINAL INVESTMENT SUMMARY ================")
print(final_summary.sort_values(
    'Sharpe_Ratio',
    ascending=False
))

# ------------------------------------------------------------
# Rank each stock
# ------------------------------------------------------------

final_summary['Return_Rank'] = (
    final_summary['Cumulative_Return_%']
    .rank(ascending=False, method='min')
)

final_summary['Sharpe_Rank'] = (
    final_summary['Sharpe_Ratio']
    .rank(ascending=False, method='min')
)

final_summary['Volatility_Rank'] = (
    final_summary['Annualised_Volatility_%']
    .rank(ascending=True, method='min')
)

# ------------------------------------------------------------
# Overall score
# Higher score = better
#
# Return and Sharpe are ranked positively.
# Lower volatility receives a better rank.
# ------------------------------------------------------------

final_summary['Overall_Score'] = (
    final_summary['Return_Rank'] +
    final_summary['Sharpe_Rank'] +
    final_summary['Volatility_Rank']
)

# Lower rank sum = better overall position

final_summary = final_summary.sort_values(
    'Overall_Score'
)

print("\n================ FINAL RANKING ================")
print(final_summary)

# ------------------------------------------------------------
# Best overall stock
# ------------------------------------------------------------

best_stock = final_summary.index[0]

print("\n================ FINAL RECOMMENDATION ================")

print(
    "Recommended Stock:",
    best_stock
)

print(
    "Cumulative Return:",
    final_summary.loc[
        best_stock,
        'Cumulative_Return_%'
    ],
    "%"
)

print(
    "Annualised Return:",
    final_summary.loc[
        best_stock,
        'Annualised_Return_%'
    ],
    "%"
)

print(
    "Annualised Volatility:",
    final_summary.loc[
        best_stock,
        'Annualised_Volatility_%'
    ],
    "%"
)

print(
    "Sharpe Ratio:",
    final_summary.loc[
        best_stock,
        'Sharpe_Ratio'
    ]
)

# ============================================================
# D1 — TRADING SIGNAL ANALYSIS
# ============================================================

# ------------------------------------------------------------
# Calculate moving averages
# ------------------------------------------------------------

ma20 = df.rolling(window=20).mean()
ma50 = df.rolling(window=50).mean()

# ------------------------------------------------------------
# Generate trading signals
# BUY  = MA20 crosses above MA50
# SELL = MA20 crosses below MA50
# HOLD = No crossover
# ------------------------------------------------------------

signals = pd.DataFrame(index=df.index)

for stock in STOCKS:

    signals[stock] = 'HOLD'

    # BUY signal
    buy_condition = (
        (ma20[stock] > ma50[stock]) &
        (ma20[stock].shift(1) <= ma50[stock].shift(1))
    )

    # SELL signal
    sell_condition = (
        (ma20[stock] < ma50[stock]) &
        (ma20[stock].shift(1) >= ma50[stock].shift(1))
    )

    signals.loc[buy_condition, stock] = 'BUY'
    signals.loc[sell_condition, stock] = 'SELL'


# ------------------------------------------------------------
# Latest trading signal
# ------------------------------------------------------------

latest_signals = signals.iloc[-1]

print("\n================ D1 — LATEST TRADING SIGNALS ================")
print(latest_signals)


# ------------------------------------------------------------
# Current moving-average position
# ------------------------------------------------------------

current_position = pd.DataFrame({
    'Current_Price': df.iloc[-1],
    'MA20': ma20.iloc[-1],
    'MA50': ma50.iloc[-1],
    'Signal': latest_signals
})

print("\n================ CURRENT MARKET POSITION ================")
print(current_position.round(2))


# ------------------------------------------------------------
# Count BUY / SELL signals over the entire period
# ------------------------------------------------------------

signal_counts = pd.DataFrame({
    'BUY': (signals == 'BUY').sum(),
    'SELL': (signals == 'SELL').sum(),
    'HOLD': (signals == 'HOLD').sum()
})

print("\n================ SIGNAL COUNTS ================")
print(signal_counts)


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

current_position.to_csv(
    'YourName_D1_current_trading_signals.csv'
)

signal_counts.to_csv(
    'YourName_D1_signal_counts.csv'
)

# ============================================================
# D2 — MOVING AVERAGE STRATEGY BACKTEST
# ============================================================

# ------------------------------------------------------------
# Create strategy position
# 1 = invested
# 0 = out of market
# ------------------------------------------------------------

position = (ma20 > ma50).astype(int)

# Shift position by 1 day to avoid look-ahead bias
position_lagged = position.shift(1)

# ------------------------------------------------------------
# Calculate daily strategy returns
# ------------------------------------------------------------

strategy_returns = returns * position_lagged

# Remove first missing observation
strategy_returns = strategy_returns.dropna()

# ------------------------------------------------------------
# Calculate cumulative strategy returns
# ------------------------------------------------------------

strategy_growth = (1 + strategy_returns / 100).cumprod()

buy_hold_growth = (1 + returns / 100).cumprod()

# ------------------------------------------------------------
# Calculate final values for ₹10,000 investment
# ------------------------------------------------------------

initial_investment = 10000

strategy_final = (
    strategy_growth.iloc[-1] * initial_investment
)

buy_hold_final = (
    buy_hold_growth.iloc[-1] * initial_investment
)

# ------------------------------------------------------------
# Calculate total returns
# ------------------------------------------------------------

strategy_total_return = (
    (strategy_growth.iloc[-1] - 1) * 100
)

buy_hold_total_return = (
    (buy_hold_growth.iloc[-1] - 1) * 100
)

# ------------------------------------------------------------
# Create comparison table
# ------------------------------------------------------------

backtest_summary = pd.DataFrame({
    'Strategy_Return_%': strategy_total_return.round(2),
    'Buy_Hold_Return_%': buy_hold_total_return.round(2),
    'Strategy_Final_Value': strategy_final.round(2),
    'Buy_Hold_Final_Value': buy_hold_final.round(2)
})

print("\n================ D2 — BACKTEST RESULTS ================")
print(backtest_summary)

# ------------------------------------------------------------
# Calculate outperformance
# ------------------------------------------------------------

backtest_summary['Outperformance_%'] = (
    backtest_summary['Strategy_Return_%'] -
    backtest_summary['Buy_Hold_Return_%']
).round(2)

print("\n================ STRATEGY OUTPERFORMANCE ================")
print(
    backtest_summary[
        ['Strategy_Return_%',
         'Buy_Hold_Return_%',
         'Outperformance_%']
    ]
)

# ------------------------------------------------------------
# Identify best strategy performer
# ------------------------------------------------------------

best_strategy_stock = (
    backtest_summary['Strategy_Return_%']
    .idxmax()
)

print(
    "\nBest stock under MA strategy:",
    best_strategy_stock
)

print(
    "Strategy return:",
    backtest_summary.loc[
        best_strategy_stock,
        'Strategy_Return_%'
    ],
    "%"
)

# ============================================================
# B6 — MAXIMUM DRAWDOWN
# ============================================================

# Calculate running maximum price for each stock
running_max = df.cummax()

# Calculate drawdown
drawdown = (df - running_max) / running_max * 100

# Maximum drawdown for each stock
max_drawdown = drawdown.min().round(2)

print("\n================ MAXIMUM DRAWDOWN ================")
print(max_drawdown)

# Identify stock with shallowest drawdown
shallowest_drawdown_stock = max_drawdown.idxmax()

print(
    "\nShallowest drawdown:",
    shallowest_drawdown_stock,
    "=",
    max_drawdown[shallowest_drawdown_stock],
    "%"
)

# ============================================================
# C5 — LOGISTIC REGRESSION FOR ALL 5 BANKS
# Predicting next-day UP / DOWN direction
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

c5_results = []

for stock in STOCKS:

    # --------------------------------------------------------
    # Create stock-specific feature table
    # --------------------------------------------------------
    
    stock_returns = returns[stock]

    features = pd.DataFrame(index=stock_returns.index)

    features['return_1d_ago'] = stock_returns.shift(1)
    features['return_5d'] = stock_returns.rolling(5).sum()

    # Price relative to MA20
    ma20 = df[stock].rolling(20).mean()
    features['price_vs_ma20'] = (df[stock] / ma20) - 1

    # 30-day volatility
    features['volatility_30d'] = stock_returns.rolling(30).std()

    # --------------------------------------------------------
    # Target: next-day direction
    # 1 = UP, 0 = DOWN
    # --------------------------------------------------------
    
    features['target_direction'] = (
        stock_returns.shift(-1) > 0
    ).astype(int)

    # Remove missing observations
    features = features.dropna()

    # --------------------------------------------------------
    # X and y
    # --------------------------------------------------------
    
    X = features[
        ['return_1d_ago',
         'return_5d',
         'price_vs_ma20',
         'volatility_30d']
    ]

    y = features['target_direction']

    # --------------------------------------------------------
    # Train-test split
    # Same approach used for HDFC model
    # --------------------------------------------------------
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Standardise features
    # --------------------------------------------------------
    
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------
    
    logit = LogisticRegression(random_state=42)

    logit.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = logit.predict(X_test_scaled)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Store result
    c5_results.append({
        'Bank': stock,
        'Accuracy (%)': round(accuracy * 100, 2)
    })


# ============================================================
# C5 RESULTS TABLE
# ============================================================

c5_results_df = pd.DataFrame(c5_results)

# Sort from highest to lowest accuracy
c5_results_df = c5_results_df.sort_values(
    'Accuracy (%)',
    ascending=False
).reset_index(drop=True)


print("\n================ C5 — LOGISTIC REGRESSION ALL BANKS ================")
print(c5_results_df.to_string(index=False))


# ============================================================
# MOST PREDICTABLE BANK
# ============================================================

best_bank = c5_results_df.iloc[0]

print("\n================ MOST PREDICTABLE BANK ================")
print(
    "Most predictable bank:",
    best_bank['Bank']
)

print(
    "Highest accuracy:",
    best_bank['Accuracy (%)'],
    "%"
)

# ============================================================
# PART C — MACHINE LEARNING MODELS
# ============================================================

# ============================================================
# C1 — FEATURE ENGINEERING
# HDFC BANK
# ============================================================

# Select HDFC Bank price and return series
hdfc_price = df['HDFC Bank']
hdfc_returns = returns['HDFC Bank']

# ------------------------------------------------------------
# Create feature table
# ------------------------------------------------------------

features_hdfc = pd.DataFrame(index=hdfc_returns.index)

# Feature 1 — Today's return
features_hdfc['return_today'] = hdfc_returns

# Feature 2 — Previous day's return
features_hdfc['return_1d_ago'] = hdfc_returns.shift(1)

# Feature 3 — Five-day return
features_hdfc['return_5d'] = (
    hdfc_price.pct_change(periods=5) * 100
)

# Feature 4 — Price relative to MA20
ma20_hdfc = hdfc_price.rolling(window=20).mean()

features_hdfc['price_vs_ma20'] = (
    (hdfc_price / ma20_hdfc) - 1
) * 100

# Feature 5 — 30-day rolling volatility
features_hdfc['volatility_30d'] = (
    hdfc_returns.rolling(window=30).std()
)

# ------------------------------------------------------------
# Target variable
# 1 = next day UP
# 0 = next day DOWN
# ------------------------------------------------------------

next_day_return = hdfc_returns.shift(-1)

features_hdfc['target_direction'] = np.where(
    next_day_return.isna(),
    np.nan,
    (next_day_return > 0).astype(int)
)

# Remove rows created by lagged/rolling calculations
features_hdfc = features_hdfc.dropna()

# ------------------------------------------------------------
# Print required feature table
# ------------------------------------------------------------

print("\n================ C1 — FEATURE TABLE ================")
print(features_hdfc.head(3))

print("\n================ C1 — FEATURE TABLE INFO ================")
print(features_hdfc.info())

print("\nFeature table shape:", features_hdfc.shape)


# ============================================================
# C2 — LINEAR REGRESSION
# PREDICT HDFC BANK PRICE TREND
# ============================================================

# ------------------------------------------------------------
# Create time index
# ------------------------------------------------------------

price_data = hdfc_price.dropna().copy()

X_time = np.arange(len(price_data)).reshape(-1, 1)
y_price = price_data.values

# ------------------------------------------------------------
# Time-series train/test split
# NO SHUFFLING
# ------------------------------------------------------------

split_point = int(len(price_data) * 0.80)

X_train_price = X_time[:split_point]
X_test_price = X_time[split_point:]

y_train_price = y_price[:split_point]
y_test_price = y_price[split_point:]

# ------------------------------------------------------------
# Train Linear Regression
# ------------------------------------------------------------

linear_price_model = LinearRegression()

linear_price_model.fit(
    X_train_price,
    y_train_price
)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

y_pred_price = linear_price_model.predict(X_test_price)

# ------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------

r2_price = r2_score(
    y_test_price,
    y_pred_price
)

rmse_price = np.sqrt(
    mean_squared_error(
        y_test_price,
        y_pred_price
    )
)

print("\n================ C2 — LINEAR REGRESSION ================")
print("R² Score:", round(r2_price, 4))
print("RMSE:", round(rmse_price, 4))

# ------------------------------------------------------------
# Trend direction
# ------------------------------------------------------------

trend_slope = linear_price_model.coef_[0]

if trend_slope > 0:
    trend_direction = "UPWARD"
elif trend_slope < 0:
    trend_direction = "DOWNWARD"
else:
    trend_direction = "FLAT"

print("Trend direction:", trend_direction)
print("Trend slope:", round(trend_slope, 4))

# ------------------------------------------------------------
# Plot actual price and predicted trend
# ------------------------------------------------------------

plt.figure(figsize=(14, 7))

# Actual price
plt.plot(
    price_data.index,
    y_price,
    label='Actual Price'
)

# Regression trend line
full_trend = linear_price_model.predict(
    X_time
)

plt.plot(
    price_data.index,
    full_trend,
    linestyle='--',
    linewidth=2,
    label='Linear Trend'
)

# Shade test-set region
plt.axvspan(
    price_data.index[split_point],
    price_data.index[-1],
    alpha=0.20,
    label='Test Set'
)

plt.title(
    'HDFC Bank — Actual Price and Linear Regression Trend',
    fontsize=15
)

plt.xlabel('Date')
plt.ylabel('Price (₹)')

plt.legend()

plt.tight_layout()

plt.savefig(
    'SagnikChakraborty_chart07_linear_regression.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()


# ============================================================
# C3 — LOGISTIC REGRESSION
# HDFC BANK NEXT-DAY DIRECTION
# ============================================================

# ------------------------------------------------------------
# Define X and y
# ------------------------------------------------------------

X_logit_hdfc = features_hdfc[
    [
        'return_1d_ago',
        'return_5d',
        'price_vs_ma20',
        'volatility_30d'
    ]
]

y_logit_hdfc = features_hdfc[
    'target_direction'
].astype(int)

# ------------------------------------------------------------
# Time-series train/test split
# NO SHUFFLING
# ------------------------------------------------------------

split_logit = int(len(X_logit_hdfc) * 0.80)

X_train_logit = X_logit_hdfc.iloc[:split_logit]
X_test_logit = X_logit_hdfc.iloc[split_logit:]

y_train_logit = y_logit_hdfc.iloc[:split_logit]
y_test_logit = y_logit_hdfc.iloc[split_logit:]

# ------------------------------------------------------------
# Standardise features
# Fit scaler ONLY on training data
# ------------------------------------------------------------

scaler_hdfc = StandardScaler()

X_train_scaled = scaler_hdfc.fit_transform(
    X_train_logit
)

X_test_scaled = scaler_hdfc.transform(
    X_test_logit
)

# ------------------------------------------------------------
# Train Logistic Regression
# ------------------------------------------------------------

logit_hdfc = LogisticRegression(
    random_state=42,
    max_iter=1000
)

logit_hdfc.fit(
    X_train_scaled,
    y_train_logit
)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

y_pred_logit = logit_hdfc.predict(
    X_test_scaled
)

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

accuracy_hdfc = accuracy_score(
    y_test_logit,
    y_pred_logit
)

cm_hdfc = confusion_matrix(
    y_test_logit,
    y_pred_logit
)

print("\n================ C3 — LOGISTIC REGRESSION ================")
print("Accuracy:", round(accuracy_hdfc, 4))

print("\n================ CONFUSION MATRIX ================")
print(cm_hdfc)

print("\n================ CLASSIFICATION REPORT ================")
print(
    classification_report(
        y_test_logit,
        y_pred_logit,
        target_names=['DOWN', 'UP']
    )
)

# ------------------------------------------------------------
# Coefficients
# ------------------------------------------------------------

logit_coefficients_hdfc = pd.DataFrame({
    'Feature': X_logit_hdfc.columns,
    'Coefficient': logit_hdfc.coef_[0]
})

print("\n================ LOGISTIC COEFFICIENTS ================")
print(logit_coefficients_hdfc)

print(
    "\nIntercept:",
    round(logit_hdfc.intercept_[0], 4)
)


# ============================================================
# C4 — K-MEANS CLUSTERING
# WITH ELBOW METHOD
# ============================================================

# ------------------------------------------------------------
# Create clustering data
# ------------------------------------------------------------

cluster_data = pd.DataFrame({
    'Annualised_Return': ann_ret,
    'Annualised_Volatility': ann_vol
})

print("\n================ C4 — CLUSTERING DATA ================")
print(cluster_data)

# ------------------------------------------------------------
# Standardise clustering variables
# ------------------------------------------------------------

cluster_scaler = StandardScaler()

cluster_scaled = cluster_scaler.fit_transform(
    cluster_data
)

# ------------------------------------------------------------
# Elbow Method
# ------------------------------------------------------------

k_range = range(
    1,
    len(STOCKS) + 1
)

inertias = []

for k in k_range:

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km.fit(cluster_scaled)

    inertias.append(km.inertia_)

# ------------------------------------------------------------
# Plot elbow curve
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    list(k_range),
    inertias,
    marker='o'
)

plt.title(
    'Elbow Method for K-Means Clustering',
    fontsize=15
)

plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (Inertia)')

plt.xticks(
    list(k_range)
)

plt.tight_layout()

plt.savefig(
    'SagnikChakraborty_chart08_kmeans_elbow.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

print("\n================ ELBOW METHOD ================")

for k, inertia in zip(k_range, inertias):
    print(
        "k =", k,
        "| Inertia =", round(inertia, 4)
    )

# ------------------------------------------------------------
# Select k = 2 based on the elbow
# ------------------------------------------------------------

chosen_k = 2

print(
    "\nChosen number of clusters:",
    chosen_k
)

# ------------------------------------------------------------
# Final K-Means Model
# ------------------------------------------------------------

kmeans_final = KMeans(
    n_clusters=chosen_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans_final.fit_predict(
    cluster_scaled
)

cluster_data['Cluster'] = cluster_labels

print("\n================ K-MEANS RESULTS ================")
print(
    cluster_data.sort_values(
        'Cluster'
    )
)

# ------------------------------------------------------------
# Print cluster members
# ------------------------------------------------------------

print("\n================ CLUSTER MEMBERS ================")

for cluster in sorted(
    cluster_data['Cluster'].unique()
):

    members = cluster_data[
        cluster_data['Cluster'] == cluster
    ].index.tolist()

    print(
        f"Cluster {cluster}:",
        ", ".join(members)
    )

# ------------------------------------------------------------
# Cluster scatter plot
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))

for cluster in sorted(
    cluster_data['Cluster'].unique()
):

    cluster_points = cluster_data[
        cluster_data['Cluster'] == cluster
    ]

    plt.scatter(
        cluster_points[
            'Annualised_Volatility'
        ],
        cluster_points[
            'Annualised_Return'
        ],
        s=180,
        label=f'Cluster {cluster}'
    )

    for stock in cluster_points.index:

        plt.annotate(
            stock,
            (
                cluster_points.loc[
                    stock,
                    'Annualised_Volatility'
                ],
                cluster_points.loc[
                    stock,
                    'Annualised_Return'
                ]
            ),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=11
        )

plt.title(
    'K-Means Clustering of Indian Banking Stocks',
    fontsize=16
)

plt.xlabel('Annualised Volatility (%)')
plt.ylabel('Annualised Return (%)')

plt.legend()

plt.tight_layout()

plt.savefig(
    'SagnikChakraborty_chart09_kmeans_clusters.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()


# ============================================================
# C5 — LOGISTIC REGRESSION FOR ALL FIVE BANKS
# ============================================================

c5_results = []

for stock in STOCKS:

    # --------------------------------------------------------
    # Stock-specific data
    # --------------------------------------------------------

    stock_price = df[stock]
    stock_returns = returns[stock]

    features_stock = pd.DataFrame(
        index=stock_returns.index
    )

    # Features
    features_stock['return_1d_ago'] = (
        stock_returns.shift(1)
    )

    features_stock['return_5d'] = (
        stock_price.pct_change(5) * 100
    )

    stock_ma20 = stock_price.rolling(
        window=20
    ).mean()

    features_stock['price_vs_ma20'] = (
        (stock_price / stock_ma20) - 1
    ) * 100

    features_stock['volatility_30d'] = (
        stock_returns.rolling(30).std()
    )

    # Target
    next_return = stock_returns.shift(-1)

    features_stock['target_direction'] = np.where(
        next_return.isna(),
        np.nan,
        (next_return > 0).astype(int)
    )

    features_stock = features_stock.dropna()

    # --------------------------------------------------------
    # X and y
    # --------------------------------------------------------

    X_stock = features_stock[
        [
            'return_1d_ago',
            'return_5d',
            'price_vs_ma20',
            'volatility_30d'
        ]
    ]

    y_stock = features_stock[
        'target_direction'
    ].astype(int)

    # --------------------------------------------------------
    # Time-series split
    # NO SHUFFLING
    # --------------------------------------------------------

    split_stock = int(
        len(X_stock) * 0.80
    )

    X_train_stock = X_stock.iloc[:split_stock]
    X_test_stock = X_stock.iloc[split_stock:]

    y_train_stock = y_stock.iloc[:split_stock]
    y_test_stock = y_stock.iloc[split_stock:]

    # --------------------------------------------------------
    # Standardise
    # --------------------------------------------------------

    scaler_stock = StandardScaler()

    X_train_stock_scaled = scaler_stock.fit_transform(
        X_train_stock
    )

    X_test_stock_scaled = scaler_stock.transform(
        X_test_stock
    )

    # --------------------------------------------------------
    # Logistic model
    # --------------------------------------------------------

    logit_stock = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    logit_stock.fit(
        X_train_stock_scaled,
        y_train_stock
    )

    # Prediction
    y_pred_stock = logit_stock.predict(
        X_test_stock_scaled
    )

    # Accuracy
    accuracy_stock = accuracy_score(
        y_test_stock,
        y_pred_stock
    )

    c5_results.append({
        'Bank': stock,
        'Accuracy (%)': round(
            accuracy_stock * 100,
            2
        )
    })


# ------------------------------------------------------------
# C5 results table
# ------------------------------------------------------------

c5_results_df = pd.DataFrame(
    c5_results
)

c5_results_df = c5_results_df.sort_values(
    'Accuracy (%)',
    ascending=False
).reset_index(drop=True)

print(
    "\n================ C5 — ALL BANKS ================"
)

print(
    c5_results_df.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Most predictable bank
# ------------------------------------------------------------

best_bank = c5_results_df.iloc[0]

print(
    "\nMost predictable bank:",
    best_bank['Bank']
)

print(
    "Highest accuracy:",
    best_bank['Accuracy (%)'],
    "%"
)

print(features_hdfc.head(3).round(4).to_string())

# ============================================================
# D1 — 52-WEEK LOW AND HIGH
# ============================================================

# Last 252 trading days ≈ 52 weeks
last_252_days = df.tail(252)

# Calculate 52-week low and high
week52_low = last_252_days.min().round(2)
week52_high = last_252_days.max().round(2)

print("\n================ 52-WEEK LOW ================")
print(week52_low)

print("\n================ 52-WEEK HIGH ================")
print(week52_high)

# ============================================================
# PART 4 — TIME-SERIES ANALYSIS
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Create folder for Section 4 charts
output_folder = 'Section_4_Charts'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Section 4 charts will be saved in:")
print(os.path.abspath(output_folder))


# ============================================================
# 4.1 MOVING AVERAGE ANALYSIS
# MA20 AND MA50
# ============================================================

fig, axes = plt.subplots(
    5, 1,
    figsize=(14, 20),
    sharex=True
)

for i, stock in enumerate(STOCKS):

    ma20 = df[stock].rolling(20).mean()
    ma50 = df[stock].rolling(50).mean()

    axes[i].plot(
        df.index,
        df[stock],
        label='Closing Price'
    )

    axes[i].plot(
        df.index,
        ma20,
        label='MA20'
    )

    axes[i].plot(
        df.index,
        ma50,
        label='MA50'
    )

    axes[i].set_title(
        f'{stock} — Price History with MA20 and MA50'
    )

    axes[i].set_ylabel('Price (₹)')
    axes[i].legend()

axes[-1].set_xlabel('Date')

fig.suptitle(
    'Time-Series Analysis — Price History and Moving Averages',
    fontsize=16,
    y=0.995
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        'SagnikChakraborty_chart10_moving_average.png'
    ),
    dpi=150,
    bbox_inches='tight'
)

plt.show()


# ============================================================
# 4.2 ROLLING VOLATILITY ANALYSIS
# 30-DAY ANNUALISED VOLATILITY
# ============================================================

rolling_volatility = pd.DataFrame(index=returns.index)

for stock in STOCKS:

    rolling_volatility[stock] = (
        returns[stock]
        .rolling(window=30)
        .std()
        * np.sqrt(252)
    )

plt.figure(figsize=(14, 8))

for stock in STOCKS:

    plt.plot(
        rolling_volatility.index,
        rolling_volatility[stock],
        label=stock
    )

plt.title(
    '30-Day Rolling Annualised Volatility — Indian Banking Stocks',
    fontsize=15
)

plt.xlabel('Date')
plt.ylabel('Annualised Volatility (%)')

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        'SagnikChakraborty_chart11_rolling_volatility.png'
    ),
    dpi=150,
    bbox_inches='tight'
)

plt.show()


# ============================================================
# PRINT ROLLING VOLATILITY SUMMARY
# ============================================================

print("\n================ ROLLING VOLATILITY SUMMARY ================")

rolling_vol_summary = pd.DataFrame({
    'Average Rolling Volatility (%)':
        rolling_volatility.mean(),

    'Maximum Rolling Volatility (%)':
        rolling_volatility.max(),

    'Minimum Rolling Volatility (%)':
        rolling_volatility.min()
}).round(2)

print(rolling_vol_summary)


# ============================================================
# 4.3 CUMULATIVE RETURN ANALYSIS
# ₹10,000 INVESTMENT
# ============================================================

initial_investment = 10000

cumulative_value = (
    (1 + returns / 100)
    .cumprod()
    * initial_investment
)

plt.figure(figsize=(14, 8))

for stock in STOCKS:

    plt.plot(
        cumulative_value.index,
        cumulative_value[stock],
        label=stock
    )

plt.axhline(
    y=initial_investment,
    linestyle='--',
    linewidth=1.5,
    label='Initial Investment (₹10,000)'
)

plt.title(
    'Cumulative Return Analysis',
    fontsize=15
)

plt.xlabel('Date')
plt.ylabel('Portfolio Value (₹)')

plt.gca().yaxis.set_major_formatter(
    mticker.StrMethodFormatter('₹{x:,.0f}')
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        'SagnikChakraborty_chart12_cumulative_return.png'
    ),
    dpi=150,
    bbox_inches='tight'
)

plt.show()


# ============================================================
# FINAL CUMULATIVE RETURN SUMMARY
# ============================================================

final_values = cumulative_value.iloc[-1]

cumulative_return_summary = (
    (final_values / initial_investment) - 1
) * 100

print("\n================ CUMULATIVE RETURN SUMMARY ================")

section4_cumulative = pd.DataFrame({
    'Final Value of ₹10,000':
        final_values.round(2),

    'Cumulative Return (%)':
        cumulative_return_summary.round(2)
})

print(section4_cumulative)

