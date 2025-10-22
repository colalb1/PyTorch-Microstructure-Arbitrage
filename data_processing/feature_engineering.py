import numpy as np
import pandas as pd


def simulate_unified_data(num_rows: int = 10000, start_time_ns=None) -> pd.DataFrame:
    """
    Generates a simulated high-resolution, synchronized DataFrame for two exchanges.

    This function creates a realistic-looking dataset with nanosecond timestamps,
    best bid/ask prices, top-level volumes, and sporadic trades.

    Args:
        num_rows (int): The number of data points (timestamps) to generate.
        start_time_ns (int, optional): The starting timestamp in nanoseconds.
                                       Defaults to the current time.

    Returns:
        pd.DataFrame: A DataFrame with simulated market data, indexed by timestamp.
    """
    if start_time_ns is None:
        start_time_ns = pd.Timestamp.now().value

    # Generate a high-resolution, slightly irregular timestamp index
    time_increments = np.random.randint(1_000_000, 100_000_000, size=num_rows)
    timestamps = start_time_ns + np.cumsum(time_increments)
    index = pd.to_datetime(timestamps, unit="ns")

    # Simulate a random walk for the base price
    base_price = 10000 + np.random.randn(num_rows).cumsum() * 0.1

    # Create data for Exchange A
    a_spread = np.random.uniform(0.5, 1.5, size=num_rows)

    # Multi-level data for Exchange A
    data_a = {
        "A_best_ask": base_price + a_spread / 2,
        "A_best_bid": base_price - a_spread / 2,
    }

    for i in range(1, 6):  # 5 levels of depth
        data_a[f"A_ask_vol_{i}"] = np.random.uniform(
            0.1, 5 - (i - 1) * 0.5, size=num_rows
        )
        data_a[f"A_bid_vol_{i}"] = np.random.uniform(
            0.1, 5 - (i - 1) * 0.5, size=num_rows
        )
    df_a = pd.DataFrame(data_a, index=index)

    # Multi-level data for Exchange Bs
    b_spread = np.random.uniform(0.6, 1.6, size=num_rows)
    price_lag = np.random.normal(0, 0.05, size=num_rows)
    data_b = {
        "B_best_ask": base_price + price_lag + b_spread / 2,
        "B_best_bid": base_price + price_lag - b_spread / 2,
    }

    for i in range(1, 6):  # 5 levels of depth
        data_b[f"B_ask_vol_{i}"] = np.random.uniform(
            0.1, 5 - (i - 1) * 0.5, size=num_rows
        )
        data_b[f"B_bid_vol_{i}"] = np.random.uniform(
            0.1, 5 - (i - 1) * 0.5, size=num_rows
        )
    df_b = pd.DataFrame(data_b, index=index)

    # Simulate trades (less frequent than book updates)
    trade_mask = np.random.random(num_rows) < 0.05
    trade_prices = base_price + np.random.normal(0, 0.1, size=num_rows)
    trade_volumes = np.random.uniform(0.01, 1, size=num_rows)

    trades = pd.DataFrame(
        {
            "trade_price": np.where(trade_mask, trade_prices, np.nan),
            "trade_volume": np.where(trade_mask, trade_volumes, np.nan),
        },
        index=index,
    )

    # Combine all data and forward-fill to simulate the stateful nature of market data
    df = pd.concat([df_a, df_b, trades], axis=1)
    df.ffill(inplace=True)

    return df


def engineer_book_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates intra-exchange (order book) features.

    Args:
        df (pd.DataFrame): The input DataFrame with synchronized book data.

    Returns:
        pd.DataFrame: The DataFrame with added book-based features.
    """
    for ex in ["A", "B"]:
        # Mid-Price
        df[f"{ex}_mid_price"] = (df[f"{ex}_best_bid"] + df[f"{ex}_best_ask"]) / 2

        # Weighted Mid-Price (WMP)
        df[f"{ex}_wmp"] = (
            df[f"{ex}_best_bid"] * df[f"{ex}_ask_vol_1"]
            + df[f"{ex}_best_ask"] * df[f"{ex}_bid_vol_1"]
        ) / (df[f"{ex}_bid_vol_1"] + df[f"{ex}_ask_vol_1"])

        # Spread
        df[f"{ex}_spread"] = df[f"{ex}_best_ask"] - df[f"{ex}_best_bid"]

        # Order Book Imbalance (OBI)
        df[f"{ex}_obi"] = (df[f"{ex}_bid_vol_1"] - df[f"{ex}_ask_vol_1"]) / (
            df[f"{ex}_bid_vol_1"] + df[f"{ex}_ask_vol_1"]
        )

        # Depth Ratio (5 levels of the simulated book)
        bid_vols = [df[f"{ex}_bid_vol_{i}"] for i in range(1, 6)]
        ask_vols = [df[f"{ex}_ask_vol_{i}"] for i in range(1, 6)]

        df[f"{ex}_depth_ratio"] = sum(bid_vols) / sum(ask_vols)

    return df


def engineer_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates intra-exchange (trade) features.

    Args:
        df (pd.DataFrame): The input DataFrame with book and trade data.

    Returns:
        pd.DataFrame: The DataFrame with added trade-based features.
    """
    # Use Exchange A's mid-price as the reference for trade signing
    mid_price = df["A_mid_price"]

    # Determine trade sign: +1 for buys (trades above mid), -1 for sells (trades below mid).
    trade_sign = np.sign(df["trade_price"] - mid_price)

    # Apply the sign to the trade volume. NaNs will propagate correctly.
    df["signed_trade_volume"] = trade_sign * df["trade_volume"]

    # Realized Volatility (5-second rolling window)
    mid_price_returns = df["A_mid_price"].pct_change().dropna()
    df["realized_vol_5s"] = mid_price_returns.rolling("5s").std() * np.sqrt(
        252 * 24 * 60 * 12
    )  # Annualized

    # Order Flow Imbalance (OFI) Proxy (1s and 5s rolling sums)
    df["ofi_1s"] = df["signed_trade_volume"].rolling("1s").sum()
    df["ofi_5s"] = df["signed_trade_volume"].rolling("5s").sum()

    # Forward fill the trade-based features as they only change on new trades
    trade_cols = ["signed_trade_volume", "realized_vol_5s", "ofi_1s", "ofi_5s"]
    df[trade_cols] = df[trade_cols].ffill()

    return df


def engineer_cross_exchange_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates cross-exchange features.

    Args:
        df (pd.DataFrame): The input DataFrame with features for both exchanges.

    Returns:
        pd.DataFrame: The DataFrame with added cross-exchange features.
    """
    # Basis/Spread
    df["cross_basis"] = df["A_mid_price"] - df["B_mid_price"]

    # Ratio
    df["cross_ratio"] = df["A_mid_price"] / df["B_mid_price"]

    # Lead/Lag Indicator (A's price vs. B's lagged price)
    b_mid_price_lagged = df["B_mid_price"].shift(freq="100ms")
    # Reindex to align timestamps before calculating the difference
    b_mid_price_lagged = b_mid_price_lagged.reindex(df.index, method="ffill")
    df["lead_lag_100ms"] = df["A_mid_price"] - b_mid_price_lagged

    return df


def main():
    """
    Main function to run the feature engineering pipeline.
    """
    print("--- Starting Feature Engineering Pipeline ---")

    # 1. Simulate synchronized data
    unified_df = simulate_unified_data(num_rows=50000)
    print(f"Simulated data created with shape: {unified_df.shape}")
    initial_cols = unified_df.shape[1]

    # 2. Engineer features
    features_df = engineer_book_features(unified_df)
    features_df = engineer_trade_features(features_df)
    features_df = engineer_cross_exchange_features(features_df)

    # Clean up any NaNs created by rolling windows or lags
    features_df.dropna(inplace=True)

    print(f"\nOriginal number of columns: {initial_cols}")
    print(f"Final number of features: {features_df.shape[1]}")
    print("-" * 43)

    # 3. Output and Verification
    print("\n--- Final Feature DataFrame (Head) ---")
    print(features_df.head())
    print("-" * 38)


if __name__ == "__main__":
    main()
