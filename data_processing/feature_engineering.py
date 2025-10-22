import logging
import sys

import numpy as np
import pandas as pd
from data_alignment import align_data

logger = logging.getLogger(__name__)


def add_simulated_depth_and_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments a real aligned DataFrame with simulated multi-level depth and trades.

    Since the L2 data feeds don't contain trade data and we only extracted
    top-level volume, this function adds the missing columns needed for
    advanced feature engineering.

    Args:
        df (pd.DataFrame): The real aligned DataFrame from align_data().

    Returns:
        pd.DataFrame: The DataFrame augmented with simulated data.
    """
    num_rows = len(df)

    # Simulate and add multi-level depth
    for ex in ["coinbase", "kraken"]:
        for i in range(2, 6):  # Add levels 2 through 5
            # Simulate volumes that decrease as we go deeper into the book
            df[f"{ex}_ask_vol_{i}"] = np.random.uniform(
                0.1, 4 - (i - 1) * 0.5, size=num_rows
            )
            df[f"{ex}_bid_vol_{i}"] = np.random.uniform(
                0.1, 4 - (i - 1) * 0.5, size=num_rows
            )

    # Simulate and add trades
    trade_mask = np.random.random(num_rows) < 0.05

    # Trades should hover around the mid-price of one of the exchanges
    base_price = (df["coinbase_best_bid"] + df["coinbase_best_ask"]) / 2
    trade_prices = base_price + np.random.normal(0, 0.1, size=num_rows)
    trade_volumes = np.random.uniform(0.01, 1, size=num_rows)

    df["trade_price"] = np.where(trade_mask, trade_prices, np.nan)
    df["trade_volume"] = np.where(trade_mask, trade_volumes, np.nan)

    # Forward-fill the trade data
    df[["trade_price", "trade_volume"]] = df[["trade_price", "trade_volume"]].ffill()

    return df


def engineer_book_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates intra-exchange (order book) features from real data.
    """

    for ex in ["coinbase", "kraken"]:
        df[f"{ex}_mid_price"] = (df[f"{ex}_best_bid"] + df[f"{ex}_best_ask"]) / 2

        # Weighted mid-price
        df[f"{ex}_wmp"] = (
            df[f"{ex}_best_bid"] * df[f"{ex}_ask_vol_1"]
            + df[f"{ex}_best_ask"] * df[f"{ex}_bid_vol_1"]
        ) / (df[f"{ex}_bid_vol_1"] + df[f"{ex}_ask_vol_1"])

        df[f"{ex}_spread"] = df[f"{ex}_best_ask"] - df[f"{ex}_best_bid"]

        # Order book imbalance (OBI)
        df[f"{ex}_obi"] = (df[f"{ex}_bid_vol_1"] - df[f"{ex}_ask_vol_1"]) / (
            df[f"{ex}_bid_vol_1"] + df[f"{ex}_ask_vol_1"]
        )

        # Depth Ratio
        bid_vols = [df[f"{ex}_bid_vol_{i}"] for i in range(1, 6)]
        ask_vols = [df[f"{ex}_ask_vol_{i}"] for i in range(1, 6)]
        df[f"{ex}_depth_ratio"] = sum(bid_vols) / sum(ask_vols)

    return df


def engineer_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates intra-exchange (trade) features from augmented data.
    """
    mid_price = df["coinbase_mid_price"]

    # 1 for buy, -1 for sell
    trade_sign = np.sign(df["trade_price"] - mid_price)

    # Apply the sign to the trade volume. Will propagate None correctly
    df["signed_trade_volume"] = trade_sign * df["trade_volume"]

    mid_price_returns = df["coinbase_mid_price"].pct_change().dropna()
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
    Generates cross-exchange features from real data.
    """
    df["cross_basis"] = df["coinbase_mid_price"] - df["kraken_mid_price"]
    df["cross_ratio"] = df["coinbase_mid_price"] / df["kraken_mid_price"]

    kraken_mid_price_lagged = df["kraken_mid_price"].shift(freq="100ms")
    kraken_mid_price_lagged = kraken_mid_price_lagged.reindex(df.index, method="ffill")
    df["lead_lag_100ms"] = df["coinbase_mid_price"] - kraken_mid_price_lagged

    return df


def main():
    """
    Main function to run the full data processing and feature engineering pipeline.
    """
    logger.info("--- Starting Data Alignment and Feature Engineering Pipeline ---")

    # Load and align real data
    aligned_df = align_data()

    if aligned_df is None:
        logger.error("Data alignment failed. Exiting.")
        return

    logger.info(f"Real aligned data loaded with shape: {aligned_df.shape}")

    # Augment with simulated data for advanced features
    augmented_df = add_simulated_depth_and_trades(aligned_df.copy())
    logger.info(f"Augmented data created with shape: {augmented_df.shape}")
    initial_cols = augmented_df.shape[1]

    # Engineer features
    features_df = engineer_book_features(augmented_df)
    features_df = engineer_trade_features(features_df)
    features_df = engineer_cross_exchange_features(features_df)

    # Clean Nones
    features_df.dropna(inplace=True)

    logger.info(f"Original number of columns (post-augmentation): {initial_cols}")
    logger.info(f"Final number of features: {features_df.shape[1]}")

    # Output and Verification
    logger.info("--- Final Feature DataFrame (Head) ---")
    logger.info(f"\n{features_df.head()}")


if __name__ == "__main__":
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    main()
