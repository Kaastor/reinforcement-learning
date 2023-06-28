from binance.client import Client
import pandas as pd
import datetime

api_key = 'L9jWMnd4iU6OMr2umUulIxYScV5lk0lbyfoKQZxGkC6yTJxLmPytlzCLJAmAftuC'
api_secret = '3B3ZOte0WjjHBqUUJ2QgTcUD7QKdKWIWs6Bpo9dlG89fq7WodCdwkF9HUvUGz2CY'

# Initialize the client
client = Client(api_key, api_secret)


def get_bitcoin_data():
    # Define start and end times in milliseconds since Unix epoch
    start_time = int(datetime.datetime(2022, 1, 1).timestamp()) * 1000
    end_time = int(datetime.datetime(2023, 1, 1).timestamp()) * 1000

    # Get klines
    candles = client.get_klines(symbol='BTCUSDT',
                                interval=Client.KLINE_INTERVAL_1HOUR,
                                startTime=start_time,
                                endTime=end_time)

    while True:
        # Get next batch of candles
        new_candles = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, startTime=candles[-1][0] + 1, endTime=end_time)
        # If there are no more candles, stop the loop
        if not new_candles:
            break
        # Add new candles to the list
        candles += new_candles

    # Convert klines to DataFrame
    df = pd.DataFrame(candles, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

    # Convert timestamps to datetime
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')

    # Select the OHLCV columns
    df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df