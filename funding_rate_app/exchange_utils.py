"""
Exchange Utilities for Funding Rate App

This module provides simplified REST clients for fetching funding rates and
instrument metadata from various cryptocurrency exchanges.

Supported Exchanges:
- Aster (Binance-compatible API)
- Binance Futures
- Hyperliquid
- Lighter
- Pacifica
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import pandas as pd
import aiohttp
import requests


# =============================================================================
# Exchange Enum
# =============================================================================

class Exchange(Enum):
    """Supported exchanges for funding rate queries."""
    ASTER = "Aster"
    BINANCE = "Binance"
    HYPERLIQUID = "Hyperliquid"
    LIGHTER = "Lighter"
    PACIFICA = "Pacifica"


# =============================================================================
# HTTP Client
# =============================================================================

class HTTPClient:
    """
    Simple async/sync HTTP client for REST API requests.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.default_headers = {"Accept": "application/json"}

    async def get(self, url_path: str, params: Dict = None, headers: Dict = None) -> Any:
        """Async GET request."""
        params = params or {}
        url = self.base_url + url_path
        request_headers = {**self.default_headers, **(headers or {})}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=request_headers) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise Exception(f"HTTP {response.status}: {text}")
                return await response.json()

    async def post(self, url_path: str, payload: Any = None, headers: Dict = None) -> Any:
        """Async POST request."""
        payload = payload or {}
        url = self.base_url + url_path
        request_headers = {**self.default_headers, **(headers or {})}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=request_headers) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise Exception(f"HTTP {response.status}: {text}")
                return await response.json()

    def get_sync(self, url_path: str, params: Dict = None, headers: Dict = None) -> Any:
        """Synchronous GET request."""
        params = params or {}
        url = self.base_url + url_path
        request_headers = {**self.default_headers, **(headers or {})}
        response = requests.get(url, params=params, headers=request_headers)
        response.raise_for_status()
        return response.json()

    def post_sync(self, url_path: str, payload: Any = None, headers: Dict = None) -> Any:
        """Synchronous POST request."""
        payload = payload or {}
        url = self.base_url + url_path
        request_headers = {**self.default_headers, **(headers or {})}
        response = requests.post(url, json=payload, headers=request_headers)
        response.raise_for_status()
        return response.json()


# =============================================================================
# Base Client
# =============================================================================

class BaseExchangeClient:
    """Base class for exchange clients."""

    def __init__(self, exchange: Exchange, rest_url: str):
        self.exchange = exchange
        self.rest_url = rest_url
        self.client = HTTPClient(base_url=rest_url)

    async def get_instruments_info(self) -> List[Dict]:
        """Get available instruments/symbols. Override in subclasses."""
        raise NotImplementedError

    async def get_funding_rates(
        self,
        symbol: str,
        start_time: int,
        end_time: int
    ) -> pd.DataFrame:
        """Get historical funding rates. Override in subclasses."""
        raise NotImplementedError


# =============================================================================
# Aster Client (Binance-compatible API)
# =============================================================================

class AsterClient(BaseExchangeClient):
    """
    Aster exchange client (uses Binance-compatible API).
    REST URL: https://fapi.asterdex.com
    """

    def __init__(self):
        super().__init__(
            exchange=Exchange.ASTER,
            rest_url="https://fapi.asterdex.com"
        )

    async def get_instruments_info(self) -> List[Dict]:
        """Get available perpetual instruments from Aster."""
        response = await self.client.get("/fapi/v1/exchangeInfo")
        symbols = response.get("symbols", [])
        return [
            {
                "symbol": s["symbol"],
                "status": s.get("status", "TRADING"),
                "base_asset": s.get("baseAsset", ""),
                "quote_asset": s.get("quoteAsset", ""),
            }
            for s in symbols
            if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"
        ]

    async def get_funding_rates(
        self,
        symbol: str,
        start_time: int,
        end_time: int
    ) -> pd.DataFrame:
        """
        Get historical funding rates with automatic pagination.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            
        Returns:
            DataFrame with columns: funding_rate, symbol. Index is datetime.
        """
        all_funding = []
        current_start = start_time
        max_records = 1000

        while current_start < end_time:
            params = {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": end_time,
                "limit": max_records,
            }

            response = await self.client.get("/fapi/v1/fundingRate", params=params)

            if not response or len(response) == 0:
                break

            if isinstance(response, dict) and "code" in response:
                break

            all_funding.extend(response)

            last_funding_time = response[-1].get('fundingTime', 0)
            if last_funding_time <= current_start:
                break

            current_start = last_funding_time + 1

            if len(response) < max_records:
                break

        if not all_funding:
            return pd.DataFrame(columns=['symbol', 'funding_rate'])

        df = pd.DataFrame(all_funding)

        # Rename columns to standard format
        column_map = {
            'fundingTime': 'timestamp',
            'fundingRate': 'funding_rate',
        }
        df = df.rename(columns=column_map)

        df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df.index = df.index.round("1h")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]

        return df[['symbol', 'funding_rate']]


# =============================================================================
# Binance Client
# =============================================================================

class BinanceClient(BaseExchangeClient):
    """
    Binance USD-M Perpetual Futures client.
    REST URL: https://fapi.binance.com
    """

    def __init__(self):
        super().__init__(
            exchange=Exchange.BINANCE,
            rest_url="https://fapi.binance.com"
        )

    async def get_instruments_info(self) -> List[Dict]:
        """Get available perpetual instruments from Binance."""
        response = await self.client.get("/fapi/v1/exchangeInfo")
        symbols = response.get("symbols", [])
        return [
            {
                "symbol": s["symbol"],
                "status": s.get("status", "TRADING"),
                "base_asset": s.get("baseAsset", ""),
                "quote_asset": s.get("quoteAsset", ""),
            }
            for s in symbols
            if s.get("contractType") in ["PERPETUAL", "TRADIFI_PERPETUAL"] and s.get("status") == "TRADING"
        ]

    async def get_funding_rates(
        self,
        symbol: str,
        start_time: int,
        end_time: int
    ) -> pd.DataFrame:
        """
        Get historical funding rates with automatic pagination.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            
        Returns:
            DataFrame with columns: funding_rate, symbol. Index is datetime.
        """
        all_funding = []
        current_start = start_time
        limit = 1000

        while current_start < end_time:
            params = {
                'symbol': symbol,
                'startTime': current_start,
                'endTime': end_time,
                'limit': limit
            }

            try:
                response = await self.client.get("/fapi/v1/fundingRate", params=params)

                if not response or len(response) == 0:
                    break

                all_funding.extend(response)

                last_funding_time = response[-1]['fundingTime']
                current_start = last_funding_time + 1

                if len(response) < limit:
                    break

            except Exception as e:
                print(f"[BinanceClient] Error fetching funding rates: {e}")
                break

        if not all_funding:
            return pd.DataFrame(columns=['symbol', 'funding_rate'])

        df = pd.DataFrame(all_funding)

        if 'fundingRate' in df.columns:
            df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True)

        if 'funding_rate' in df.columns:
            df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')

        if 'fundingTime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df.set_index('timestamp', inplace=True)

        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]

        return df[['symbol', 'funding_rate']]


# =============================================================================
# Hyperliquid Client
# =============================================================================

class HyperliquidClient(BaseExchangeClient):
    """
    Hyperliquid perpetual futures client.
    REST URL: https://api.hyperliquid.xyz
    """

    def __init__(self):
        super().__init__(
            exchange=Exchange.HYPERLIQUID,
            rest_url="https://api.hyperliquid.xyz"
        )

    async def get_instruments_info(self) -> List[Dict]:
        """Get available perpetual instruments from Hyperliquid (Main + DEXs)."""
        all_instruments = []
        print(f"[HyperliquidClient] Fetching instruments...")
        
        # 1. Get Main Exchange instruments
        try:
            response = await self.client.post("/info", {"type": "metaAndAssetCtxs"})
            if response and len(response) >= 1:
                universe = response[0].get('universe', [])
                for asset in universe:
                    all_instruments.append({
                        "symbol": asset.get("name", ""),
                        "status": "TRADING",
                        "sz_decimals": asset.get("szDecimals", 0),
                        "dex": None
                    })
                print(f"[HyperliquidClient] Found {len(universe)} main instruments.")
        except Exception as e:
            print(f"[HyperliquidClient] Error fetching main instruments: {e}")
        
        # 2. Get DEX (HIP-3) instruments
        try:
            dexs_response = await self.client.post("/info", {"type": "perpDexs"})
            # perpDexs returns [None, {dex1}, {dex2}, ...]
            if dexs_response and isinstance(dexs_response, list):
                # Use asyncio.gather to fetch all DEX meta in parallel
                dex_tasks = []
                dex_names = []
                for dex_info in dexs_response:
                    if dex_info and isinstance(dex_info, dict):
                        dex_name = dex_info.get('name')
                        if dex_name:
                            dex_names.append(dex_name)
                            dex_tasks.append(self.client.post("/info", {
                                "type": "metaAndAssetCtxs",
                                "dex": dex_name
                            }))
                
                if dex_tasks:
                    print(f"[HyperliquidClient] Fetching meta for {len(dex_tasks)} DEXs...")
                    dex_metas = await asyncio.gather(*dex_tasks, return_exceptions=True)
                    for dex_name, dex_meta in zip(dex_names, dex_metas):
                        if isinstance(dex_meta, Exception):
                            print(f"[HyperliquidClient] Error fetching meta for DEX {dex_name}: {dex_meta}")
                            continue
                            
                        if dex_meta and len(dex_meta) >= 1:
                            dex_universe = dex_meta[0].get('universe', [])
                            for asset in dex_universe:
                                asset_name = asset.get("name", "")
                                all_instruments.append({
                                    "symbol": asset_name,
                                    "status": "TRADING",
                                    "sz_decimals": asset.get("szDecimals", 0),
                                    "dex": dex_name
                                })
                    print(f"[HyperliquidClient] Total instruments after DEXs: {len(all_instruments)}")
        except Exception as e:
            print(f"[HyperliquidClient] Error fetching DEX instruments: {e}")
            
        return all_instruments

    async def get_funding_rates(
        self,
        symbol: str,
        start_time: int,
        end_time: int
    ) -> pd.DataFrame:
        """
        Get historical funding rates with automatic pagination.
        
        Args:
            symbol: The coin/asset symbol (e.g., "BTC", "xyz:AAPL")
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            
        Returns:
            DataFrame with columns: symbol, funding_rate, premium. Index is datetime.
        """
        all_funding = []
        current_start = start_time
        
        # Determine if this is a DEX symbol (contains a colon like "xyz:AAPL")
        dex_name = None
        if ":" in symbol:
            dex_name = symbol.split(":")[0]

        # Hyperliquid returns up to 1000 records per request.
        # It returns them in CHRONOLOGICAL order (earliest first) when startTime is provided.
        # We paginate forward from start_time.
        while current_start < end_time:
            payload = {
                "type": "fundingHistory",
                "coin": symbol,
                "startTime": current_start,
                "endTime": end_time,
            }
            if dex_name:
                payload["dex"] = dex_name

            response = await self.client.post("/info", payload)

            if not response or len(response) == 0:
                break
            
            all_funding.extend(response)
            
            # Since Hyperliquid returns earliest first, response[-1] is the LATEST in the batch.
            latest_in_batch = response[-1]
            latest_ts = latest_in_batch.get('time', 0)
            
            if latest_ts <= current_start:
                # No progress made
                break
                
            current_start = latest_ts + 1
            
            # If we got 1000 records, there's likely more data after this batch.
            # (Note: Hyperliquid limit is actually 500 or 1000 depending on the endpoint, 
            # but checking for >= 500 is safer if we want to be sure about pagination).
            if len(response) < 500:
                break

        if not all_funding:
            return pd.DataFrame(columns=['symbol', 'funding_rate'])

        df = pd.DataFrame(all_funding)
        
        # Sort by time and remove duplicates
        df = df.sort_values('time').drop_duplicates('time')
        
        # Convert numeric columns
        df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
        if 'premium' in df.columns:
            df['premium'] = pd.to_numeric(df['premium'], errors='coerce')
        
        # Rename coin to symbol
        df = df.rename(columns={'coin': 'symbol'})
        
        # Set timestamp as index
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        df = df.set_index('timestamp')
        df.index = df.index.round("1h")
        df = df.sort_index()

        cols = ['symbol', 'funding_rate']
        if 'premium' in df.columns:
            cols.append('premium')
        return df[cols]


# =============================================================================
# Lighter Client
# =============================================================================

class LighterClient(BaseExchangeClient):
    """
    Lighter perpetual futures client (zkLighter on Arbitrum).
    REST URL: https://mainnet.zklighter.elliot.ai
    """

    def __init__(self):
        super().__init__(
            exchange=Exchange.LIGHTER,
            rest_url="https://mainnet.zklighter.elliot.ai"
        )
        self.symbol_to_market_id: Dict[str, int] = {}
        self._load_instruments_sync()

    def _load_instruments_sync(self):
        """Load instrument metadata synchronously on init."""
        try:
            response = self.client.get_sync("/api/v1/orderBookDetails")
            details = response.get('order_book_details', [])
            for item in details:
                symbol = item.get('symbol', '')
                market_id = item.get('market_id')
                if symbol and market_id is not None:
                    self.symbol_to_market_id[symbol] = market_id
        except Exception as e:
            print(f"[LighterClient] Error loading instruments: {e}")

    async def get_instruments_info(self) -> List[Dict]:
        """Get available perpetual instruments from Lighter."""
        response = await self.client.get("/api/v1/orderBookDetails")
        details = response.get('order_book_details', [])
        
        # Update cache
        for item in details:
            symbol = item.get('symbol', '')
            market_id = item.get('market_id')
            if symbol and market_id is not None:
                self.symbol_to_market_id[symbol] = market_id
        
        return [
            {
                "symbol": item.get("symbol", ""),
                "market_id": item.get("market_id"),
                "status": "TRADING",
            }
            for item in details
        ]

    async def get_funding_rates(
        self,
        symbol: str,
        start_time: int,
        end_time: int
    ) -> pd.DataFrame:
        """
        Get historical funding rates with automatic pagination.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC", "ETH")
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            
        Returns:
            DataFrame with columns: symbol, funding_rate. Index is datetime.
        """
        market_id = self.symbol_to_market_id.get(symbol)
        if market_id is None:
            print(f"[LighterClient] Unknown symbol: {symbol}")
            return pd.DataFrame(columns=['symbol', 'funding_rate'])

        all_funding = []
        current_end = end_time
        limit = 500

        while current_end > start_time:
            params = {
                'market_id': market_id,
                'resolution': "1h",
                'start_timestamp': start_time,
                'end_timestamp': current_end,
                'count_back': limit
            }

            try:
                response = await self.client.get("/api/v1/fundings", params=params)
                data = response.get('fundings', response) if isinstance(response, dict) else response

                if not data or len(data) == 0:
                    break

                all_funding.extend(data)

                first_entry = data[0]
                earliest_ts = first_entry.get('timestamp', first_entry.get('funding_time', 0))
                if isinstance(earliest_ts, str):
                    earliest_ts = int(earliest_ts)

                # Check if milliseconds or seconds
                comp_ts = earliest_ts if earliest_ts > 1e11 else earliest_ts * 1000

                if comp_ts <= start_time:
                    break

                current_end = comp_ts - 1

                if len(data) < limit:
                    break

            except Exception as e:
                print(f"[LighterClient] Error fetching funding rates: {e}")
                break

        if not all_funding:
            return pd.DataFrame(columns=['symbol', 'funding_rate'])

        df = pd.DataFrame(all_funding)

        # Standardize column names
        column_mapping = {
            'rate': 'funding_rate',
            'fundingRate': 'funding_rate',
        }
        df.rename(columns=column_mapping, inplace=True)

        if 'funding_rate' in df.columns:
            df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
            df['funding_rate'] = df['funding_rate'] / 100  # Convert from percentage

        # Find timestamp column
        timestamp_col = None
        for col in ['timestamp', 'funding_time', 'time']:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col:
            example_ts = df[timestamp_col].iloc[0]
            if isinstance(example_ts, str):
                example_ts = int(example_ts)
            unit = 'ms' if example_ts > 1e11 else 's'
            df['timestamp'] = pd.to_datetime(df[timestamp_col], unit=unit)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='first')]

        # Add symbol column
        df['symbol'] = symbol

        return df[['symbol', 'funding_rate']]


# =============================================================================
# Pacifica Client
# =============================================================================

class PacificaClient(BaseExchangeClient):
    """
    Pacifica perpetual futures client (Solana-based).
    REST URL: https://api.pacifica.fi/api/v1
    """

    def __init__(self):
        super().__init__(
            exchange=Exchange.PACIFICA,
            rest_url="https://api.pacifica.fi/api/v1"
        )

    async def get_instruments_info(self) -> List[Dict]:
        """Get available perpetual instruments from Pacifica."""
        response = await self.client.get("/info")
        if not response.get("success"):
            return []
        
        data = response.get("data", [])
        return [
            {
                "symbol": item.get("symbol", ""),
                "status": "TRADING",
            }
            for item in data
        ]

    async def get_funding_rates(
        self,
        symbol: str,
        start_time: int,
        end_time: int
    ) -> pd.DataFrame:
        """
        Get historical funding rates.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC", "ETH")
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            
        Returns:
            DataFrame with columns: symbol, funding_rate. Index is datetime.
        """
        # Pacifica API returns data in reverse chronological order (latest first).
        # It seems to ignore start_time/end_time parameters on the funding history endpoint.
        # However, it supports a large 'limit' (up to 4000).
        # 4000 records can cover several months of data (e.g., back to August 2025).
        # Since we can't filter on the server, we fetch a large batch and filter locally.
        
        limit = 4000
        params = {
            'symbol': symbol,
            'limit': limit
        }

        try:
            response = await self.client.get("/funding_rate/history", params=params)

            if not response.get('success'):
                print(f"[PacificaClient] Error: {response.get('error')}")
                return pd.DataFrame(columns=['symbol', 'funding_rate'])

            data = response.get('data', [])
            if not data:
                return pd.DataFrame(columns=['symbol', 'funding_rate'])

            df = pd.DataFrame(data)

            # Standardize column names
            if 'rate' in df.columns and 'funding_rate' not in df.columns:
                df.rename(columns={'rate': 'funding_rate'}, inplace=True)

            if 'funding_rate' in df.columns:
                df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')

            column_mapping = {"created_at": "timestamp"}
            df.rename(columns=column_mapping, inplace=True)

            timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'funding_time'
            if timestamp_col in df.columns:
                df['timestamp'] = pd.to_datetime(df[timestamp_col], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.index = df.index.round("1h")

            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]

            # Post-filter by time range since Pacifica API ignores time parameters
            start_dt = pd.to_datetime(start_time, unit='ms').round("1h")
            end_dt = pd.to_datetime(end_time, unit='ms').round("1h")
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            if 'symbol' not in df.columns:
                df['symbol'] = symbol

            return df[['symbol', 'funding_rate']]

        except Exception as e:
            print(f"[PacificaClient] Error fetching funding rates: {e}")
            return pd.DataFrame(columns=['symbol', 'funding_rate'])


# =============================================================================
# Factory Function
# =============================================================================

def get_exchange_client(exchange: Exchange) -> BaseExchangeClient:
    """
    Factory function to get the appropriate exchange client.
    
    Args:
        exchange: The exchange enum value
        
    Returns:
        An instance of the exchange client
    """
    clients = {
        Exchange.ASTER: AsterClient,
        Exchange.BINANCE: BinanceClient,
        Exchange.HYPERLIQUID: HyperliquidClient,
        Exchange.LIGHTER: LighterClient,
        Exchange.PACIFICA: PacificaClient,
    }
    
    client_class = clients.get(exchange)
    if client_class is None:
        raise ValueError(f"Unsupported exchange: {exchange}")
    
    return client_class()


# =============================================================================
# Convenience Functions
# =============================================================================

async def fetch_all_funding_rates(
    exchanges: List[Exchange],
    symbols: Dict[Exchange, List[str]],
    start_time: int,
    end_time: int
) -> Dict[str, pd.DataFrame]:
    """
    Fetch funding rates from multiple exchanges for multiple symbols.
    
    Args:
        exchanges: List of exchanges to query
        symbols: Dictionary mapping exchange to list of symbols
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
        
    Returns:
        Dictionary mapping "exchange:symbol" to DataFrame of funding rates
    """
    results = {}
    
    for exchange in exchanges:
        client = get_exchange_client(exchange)
        exchange_symbols = symbols.get(exchange, [])
        
        for symbol in exchange_symbols:
            key = f"{exchange.value}:{symbol}"
            try:
                df = await client.get_funding_rates(symbol, start_time, end_time)
                results[key] = df
            except Exception as e:
                print(f"Error fetching {key}: {e}")
                results[key] = pd.DataFrame(columns=['symbol', 'funding_rate'])
    
    return results
