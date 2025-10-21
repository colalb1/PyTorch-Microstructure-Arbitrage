import json

import pandas as pd
from sortedcontainers import SortedList


class OrderBook:
    """
    A class to maintain the state of an order book using sorted lists for efficiency.
    """

    def __init__(self):
        # Store bids as (price, quantity) tuples. Prices are negative for max-heap behavior.
        self.bids = SortedList()
        # Store asks as (price, quantity) tuples.
        self.asks = SortedList()

    def update(self, side: str, price: float, quantity: float) -> None:
        """
        Updates the order book with a new price level.

        If quantity is 0, the price level is removed.

        Args:
            side (str): The side of the order book to update ('bid' or 'ask').
            price (float): The price level to update.
            quantity (float): The new quantity at the price level.
        """
        price = float(price)
        quantity = float(quantity)
        book_side = self.bids if side == "bid" else self.asks
        # For bids, we store prices as negative to easily find the max price (min neg price)
        price_key = -price if side == "bid" else price

        # Create a representative entry to search for the price level
        entry = (price_key, 0)
        idx = book_side.bisect_left(entry)

        # Remove existing entry if it exists
        if idx < len(book_side) and book_side[idx][0] == price_key:
            book_side.pop(idx)

        # Add new entry if quantity is not zero
        if quantity > 0:
            book_side.add((price_key, quantity))

    def get_bbo(self) -> tuple[float | None, float | None]:
        """
        Returns the best bid and best ask from the order book.

        Returns:
            tuple[float | None, float | None]: A tuple containing the best bid and best ask.
                                               Returns None for a side if the book is empty.
        """
        best_bid = -self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return best_bid, best_ask


def process_coinbase_l2_data(file_path: str) -> list[dict]:
    """
    Processes Coinbase L2 data from a JSONL file to extract BBO at each timestamp.

    Args:
        file_path (str): The path to the Coinbase JSONL data file.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a BBO update
                    with 'system_ts_ns', 'best_bid', and 'best_ask'.
    """
    book = OrderBook()
    bbo_data = []

    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)

                system_ts = data["system_ts_ns"]
                payload = data.get("payload", {})
                events = payload.get("events", [])

                for event in events:
                    updates = event.get("updates", [])

                    for update in updates:
                        side = update["side"]
                        price = update["price_level"]
                        quantity = update["new_quantity"]

                        book.update(side, price, quantity)

                best_bid, best_ask = book.get_bbo()

                if best_bid is not None and best_ask is not None:
                    bbo_data.append(
                        {
                            "system_ts_ns": system_ts,
                            "best_bid": best_bid,
                            "best_ask": best_ask,
                        }
                    )
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line: {line.strip()}")
                continue
    return bbo_data


def process_kraken_l2_data(file_path: str) -> list[dict]:
    """
    Processes Kraken L2 data from a JSONL file to extract BBO at each timestamp.

    Handles different payload structures (initial snapshot vs. updates).

    Args:
        file_path (str): The path to the Kraken JSONL data file.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a BBO update
                    with 'system_ts_ns', 'best_bid', and 'best_ask'.
    """
    book = OrderBook()
    bbo_data = []

    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)

                system_ts = data["system_ts_ns"]
                payload = data.get("payload")

                if not isinstance(payload, list) or len(payload) < 2:
                    continue

                updates = payload[1]

                # Initial update
                if "as" in updates and "bs" in updates:
                    for price, quantity, _ in updates["bs"]:
                        book.update("bid", price, quantity)
                    for price, quantity, _ in updates["as"]:
                        book.update("ask", price, quantity)
                # Subsequent updates
                else:
                    if "b" in updates:
                        for price, quantity, *_ in updates["b"]:
                            book.update("bid", price, quantity)
                    if "a" in updates:
                        for price, quantity, *_ in updates["a"]:
                            book.update("ask", price, quantity)

                best_bid, best_ask = book.get_bbo()

                if best_bid is not None and best_ask is not None:
                    bbo_data.append(
                        {
                            "system_ts_ns": system_ts,
                            "best_bid": best_bid,
                            "best_ask": best_ask,
                        }
                    )
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Warning: Could not process line: {line.strip()} due to {e}")
                continue
    return bbo_data


def main():
    """
    Main function to implement the Data Alignment & Synchronization process
    using real Coinbase and Kraken data.
    """
    # Define paths to the raw files
    coinbase_file = "data/raw/coinbase_l2.jsonl"
    kraken_file = "data/raw/kraken_l2.jsonl"

    print("--- Processing Raw Data ---")

    # Parsing and Order Book Reconstruction
    data_a = process_coinbase_l2_data(coinbase_file)
    data_b = process_kraken_l2_data(kraken_file)

    print(f"Processed {len(data_a)} BBO updates from Coinbase.")
    print(f"Processed {len(data_b)} BBO updates from Kraken.")
    print("-" * 28)

    if not data_a or not data_b:
        print("Error: No data processed. Exiting.")
        return

    # Indexing
    df_a = pd.DataFrame(data_a)
    df_b = pd.DataFrame(data_b)

    # Convert timestamp to a proper nanosecond-resolution datetime index
    df_a.set_index(pd.to_datetime(df_a["system_ts_ns"], unit="ns"), inplace=True)
    df_b.set_index(pd.to_datetime(df_b["system_ts_ns"], unit="ns"), inplace=True)

    # Drop original timestamp column
    df_a = df_a.drop(columns=["system_ts_ns"])
    df_b = df_b.drop(columns=["system_ts_ns"])

    # Add exchange identifier to column names for clarity after merge
    df_a = df_a.add_prefix("coinbase_")
    df_b = df_b.add_prefix("kraken_")

    print("\n--- Indexed DataFrames (Pre-Merge) ---")
    print("Coinbase DataFrame Head:\n", df_a.head(3))
    print("\nKraken DataFrame Head:\n", df_b.head(3))
    print("-" * 36)

    # Master Alignment
    # Merge using an outer join to create a master index with all unique timestamps
    merged_df = pd.concat([df_a, df_b], axis=1, sort=True)

    # Forward-filling to propagate the last known state
    aligned_df = merged_df.ffill()

    # Find the time when both feeds are active
    first_valid_a = df_a.index.min()
    first_valid_b = df_b.index.min()
    start_time = max(first_valid_a, first_valid_b)

    # Trim the data to the period where both feeds are active
    aligned_df = aligned_df.loc[start_time:]

    # Drop any remaining rows with NaNs to ensure data integrity
    aligned_df.dropna(inplace=True)

    # Output and Verification
    print("\n--- Final Aligned & Synchronized DataFrame ---")
    print(f"Shape of the final DataFrame: {aligned_df.shape}")

    # Verify the time resolution of the index
    if isinstance(aligned_df.index, pd.DatetimeIndex):
        print("Index type: pd.DatetimeIndex")
        print("Time resolution of index: Nanosecond (inferred from data)")
    else:
        print("Index is not a DatetimeIndex.")

    print("\nFirst 5 rows of the final DataFrame:")
    print(aligned_df.head())

    print("\nLast 5 rows of the final DataFrame:")
    print(aligned_df.tail())

    print(
        f"\nVerification: Any remaining NaN values? {aligned_df.isnull().values.any()}"
    )
    print("-" * 44)


if __name__ == "__main__":
    main()
