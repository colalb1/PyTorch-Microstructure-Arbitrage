import json
import threading
import time
from pathlib import Path
from queue import Queue

import websocket

"""
Robust, async data collector.
Uses the asyncio library to manage simultaneous,
persistent WebSocket connections to multiple crypto
exchanges (Coinbase and Kraken).

To Run:
python data_collection/collector.py
"""


COLLECTION_DURATION_SECONDS = 5

# Define WebSocket endpoints and subscriptions messages for each exchange
EXCHANGES = {
    "coinbase": {
        "uri": "wss://advanced-trade-ws.coinbase.com",
        "subscription": {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channel": "level2",
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


def file_writer(writer_queue: Queue) -> None:
    """
    Pulls data from a thread-safe queue and writes it to the appropriate file.
    This runs in its own dedicated thread.

    Args:
        writer_queue (Queue): The thread-safe queue to pull messages from.
    """
    file_handlers = {
        name: open(config["output_file"], "a") for name, config in EXCHANGES.items()
    }
    print("File writer started.")

    while True:
        message = writer_queue.get()

        if message is None:  # Sentinel for stopping
            break

        exchange_name, data = message
        handler = file_handlers[exchange_name]
        handler.write(json.dumps(data) + "\n")
        handler.flush()
        writer_queue.task_done()

    for handler in file_handlers.values():
        handler.close()
    print("File writer has shut down.")


def create_websocket_handler(
    exchange_name: str, subscription: dict, writer_queue: Queue
) -> websocket.WebSocketApp:
    """
    Factory function to create the websocket.WebSocketApp with the correct callbacks.

    Args:
        exchange_name (str): The name of the exchange.
        subscription (dict): The subscription message for the exchange.
        writer_queue (Queue): The queue to which messages will be pushed.

    Returns:
        websocket.WebSocketApp: The configured WebSocketApp instance.
    """

    def on_open(ws):
        print(f"[{exchange_name.capitalize()}] Connection opened. Subscribing...")
        ws.send(json.dumps(subscription))
        print(f"[{exchange_name.capitalize()}] Subscribed with: {subscription}")

    def on_message(ws, message):
        try:
            payload = json.loads(message)
            # Standardize the message format and add timestamp
            standardized_message = {
                "system_ts_ns": time.time_ns(),
                "payload": payload,
            }
            writer_queue.put((exchange_name, standardized_message))
        except json.JSONDecodeError:
            print(f"[{exchange_name.capitalize()}] Error decoding JSON: {message}")

    def on_error(ws, error):
        print(f"[{exchange_name.capitalize()}] Error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print(f"[{exchange_name.capitalize()}] Connection closed.")

    return websocket.WebSocketApp(
        EXCHANGES[exchange_name]["uri"],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )


def main() -> None:
    """
    Main function to set up and run the data collection pipeline.

    Initializes the writer thread and a connection thread for each exchange,
    runs for a configured duration, and then gracefully shuts everything down.
    """
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    writer_queue = Queue()

    # Start the file writer thread
    writer_thread = threading.Thread(target=file_writer, args=(writer_queue,))
    writer_thread.start()

    # Create and start a thread for each exchange
    ws_threads = []
    ws_apps = []

    for name, config in EXCHANGES.items():
        ws_app = create_websocket_handler(name, config["subscription"], writer_queue)
        ws_apps.append(ws_app)

        thread = threading.Thread(target=ws_app.run_forever)
        thread.daemon = True
        ws_threads.append(thread)

        thread.start()

    print(f"Starting data collection for {COLLECTION_DURATION_SECONDS} seconds...")
    time.sleep(COLLECTION_DURATION_SECONDS)

    print("Collection duration finished. Shutting down...")

    # Stop WebSocket connections
    for ws_app in ws_apps:
        ws_app.close()

    # Stop the writer thread
    writer_queue.put(None)
    writer_thread.join()

    print("All threads have been shut down. Script finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting.")
