import asyncio
import json
import time
from pathlib import Path

import aiofiles
import websockets

"""
Robust, async data collector.
Uses the asyncio library to manage simultaneous,
persistent WebSocket connections to multiple crypto
exchanges (Coinbase and Kraken).

To Run:
python data_collection/collector.py
"""

COLLECTION_DURATION_SECONDS = 60

# Define WebSocket endpoints and subscriptions messages for each exchange
EXCHANGES = {
    "coinbase": {
        "uri": "wss://advanced-trade-ws.coinbase.com",
        "subscription": {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["level2"],
        },
        "output_file": "data/raw/coinbase_l2.jsonl",
    },
    "kraken": {
        "uri": "wss://ws.kraken.com",
        "subscription": {
            "event": "subscribe",
            "pair": ["XBT/USD"],
            "subscription": {"name": "book"},
        },
        "output_file": "data/raw/kraken_l2.jsonl",
    },
}


async def websocket_handler(
    exchange_name: str, config: dict, writer_queue: asyncio.Queue
) -> None:
    """
    Handles the WebSocket connection for a single exchange.

    Connects, subscribes, listens for messages, and pushes them to the writer queue.
    Includes automatic reconnection with exponential backoff.

    Args:
        exchange_name (str): The name of the exchange (e.g., 'coinbase').
        config (dict): The configuration dictionary for the exchange.
        writer_queue (asyncio.Queue): The queue to which timestamped messages are pushed.

    Returns:
        None
    """
    uri = config["uri"]
    subscription = config["subscription"]
    backoff_delay = 1  # Init delay in seconds for reconnection

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print(f"[{exchange_name.capitalize()}] Connection successful.")
                backoff_delay = 1  # Reset backoff upon successful connection

                # Send subscription message
                await websocket.send(json.dumps(subscription))

                print(f"[{exchange_name.capitalize()}] Subscribed with: {subscription}")

                # Message listener
                async for message in websocket:
                    try:
                        # High-res timestamp
                        data = json.loads(message)
                        data["system_ts_ns"] = time.time_ns()

                        await writer_queue.put((exchange_name, data))
                    except json.JSONDecodeError:
                        print(
                            f"[{exchange_name.capitalize()}] Error decoding JSON: {message}"
                        )
                        continue

        except (
            websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.ConnectionClosedOK,
        ) as e:
            print(
                f"[{exchange_name.capitalize()}] Connection closed: {e}. Reconnecting in {backoff_delay}s..."
            )
        except Exception as e:
            print(
                f"[{exchange_name.capitalize()}] An unexpected error occurred: {e}. Reconnecting in {backoff_delay}s..."
            )

        await asyncio.sleep(backoff_delay)
        backoff_delay = min(backoff_delay * 2, 60)  # Backoff capped at 60s


async def file_writer(writer_queue: asyncio.Queue) -> None:
    """
    Asynchronously writes data from the queue to the appropriate .jsonl file.

    This runs as a separate, dedicated task to prevent I/O blocking.

    Args:
        writer_queue (asyncio.Queue): The queue from which messages are read to be written to files.

    Returns:
        None
    """
    # Open files in append mode asynchronously
    file_handlers = {
        name: await aiofiles.open(config["output_file"], mode="a")
        for name, config in EXCHANGES.items()
    }
    print("File writer started. Ready to persist data.")

    try:
        while True:
            exchange_name, data = await writer_queue.get()
            handler = file_handlers[exchange_name]

            await handler.write(json.dumps(data) + "\n")
            await handler.flush()

            writer_queue.task_done()
    except asyncio.CancelledError:
        print("File writer received cancellation request.")
    finally:
        # Close file handlers
        for name, handler in file_handlers.items():
            await handler.close()
            print(f"Closed file: {EXCHANGES[name]['output_file']}")

        print("File writer has shut down.")


async def main():
    """
    Main function to set up and run the data collection pipeline.
    """
    # Ensure the data/raw directory exists
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    # A queue to decouple WebSocket handlers from the file writer
    writer_queue = asyncio.Queue()

    # Create the dedicated file writer task
    writer_task = asyncio.create_task(file_writer(writer_queue))

    # Create a handler task for each exchange
    handler_tasks = [
        asyncio.create_task(websocket_handler(name, config, writer_queue))
        for name, config in EXCHANGES.items()
    ]

    print(f"Starting data collection for {COLLECTION_DURATION_SECONDS} seconds...")
    start_time = time.time()

    # This loop keeps the main function alive while the tasks run.
    # It also enforces the total collection duration.
    while time.time() - start_time < COLLECTION_DURATION_SECONDS:
        await asyncio.sleep(1)

    print("Collection duration finished. Shutting down tasks...")

    # Gracefully cancel all running tasks
    for task in handler_tasks:
        task.cancel()

    # Wait for the writer queue to be fully processed before shutting down
    await writer_queue.join()
    writer_task.cancel()

    # Wait for all tasks to complete their cancellation
    await asyncio.gather(*handler_tasks, writer_task, return_exceptions=True)

    print("All tasks have been shut down. Script finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting.")
