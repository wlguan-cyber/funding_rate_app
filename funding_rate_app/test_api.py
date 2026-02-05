import asyncio
import pandas as pd
from datetime import datetime, timedelta
from exchange_utils import Exchange, get_exchange_client

async def test_exchange_funding(exchange: Exchange, symbol: str):
    print(f"\n--- Testing {exchange.value} ({symbol}) ---")
    client = get_exchange_client(exchange)
    
    # Test 1: Get Instruments
    print(f"Fetching instruments...")
    try:
        instruments = await client.get_instruments_info()
        print(f"Found {len(instruments)} instruments.")
        if exchange == Exchange.PACIFICA:
            print(f"First 5 Pacifica symbols: {[i['symbol'] for i in instruments[:5]]}")
        if not any(i['symbol'] == symbol for i in instruments):
            print(f"Warning: {symbol} not found in instrument list.")
    except Exception as e:
        print(f"Error fetching instruments: {e}")

    # Test 2: Get Funding Rates (Last 45 days to ensure we cross Jan/Feb boundary)
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=60)).timestamp() * 1000)
    
    print(f"Fetching funding rates from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}...")
    try:
        df = await client.get_funding_rates(symbol, start_time, end_time)
        if df.empty:
            print("Result: Empty DataFrame returned.")
        else:
            print(f"Result: {len(df)} records returned.")
            print(f"Earliest record: {df.index.min()}")
            print(f"Latest record:   {df.index.max()}")
            
            # Print first and last 5 to see the range
            print("First 5 records:")
            print(df.head(5))
            print("Last 5 records:")
            print(df.tail(5))
            
            # Check for gaps or truncation
            if df.index.max() < datetime.now() - timedelta(days=1):
                print(f"CRITICAL: Data appears truncated! Latest date is {df.index.max()}")
            else:
                print("Success: Data reaches recent dates.")
    except Exception as e:
        import traceback
        print(f"Error fetching funding rates: {e}")
        traceback.print_exc()

async def main():
    # Test Hyperliquid specifically first
    await test_exchange_funding(Exchange.HYPERLIQUID, "BTC")
    
    # Test Hyperliquid DEX symbol
    await test_exchange_funding(Exchange.HYPERLIQUID, "xyz:AAPL")
    
    # Test Binance as a control
    await test_exchange_funding(Exchange.BINANCE, "BTCUSDT")
    
    # Test Aster
    await test_exchange_funding(Exchange.ASTER, "BTCUSDT")

    # Test Pacifica
    await test_exchange_funding(Exchange.PACIFICA, "BTC")

if __name__ == "__main__":
    asyncio.run(main())
