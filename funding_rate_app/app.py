"""
Funding Rate Explorer

A Streamlit application for querying and visualizing historical funding rates
from multiple cryptocurrency perpetual futures exchanges.

Supported Exchanges:
- Aster
- Binance Futures
- Hyperliquid
- Lighter
- Pacifica
"""

import streamlit as st
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

from exchange_utils import (
    Exchange,
    get_exchange_client,
    AsterClient,
    BinanceClient,
    HyperliquidClient,
    LighterClient,
    PacificaClient,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Funding Rate Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Caching Functions
# =============================================================================

@st.cache_data(ttl=3600)
def get_instruments_cached(exchange_name: str) -> List[Dict]:
    """
    Fetch and cache instruments for an exchange.
    Cache expires after 1 hour.
    """
    exchange = Exchange(exchange_name)
    client = get_exchange_client(exchange)
    
    # Run async function in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        instruments = loop.run_until_complete(client.get_instruments_info())
    finally:
        loop.close()
    
    return instruments


def get_symbol_list(exchange_name: str) -> List[str]:
    """Get list of available symbols for an exchange."""
    try:
        instruments = get_instruments_cached(exchange_name)
        return sorted([inst.get("symbol", "") for inst in instruments if inst.get("symbol")])
    except Exception as e:
        st.error(f"Error fetching instruments for {exchange_name}: {e}")
        return []


async def fetch_funding_rates_async(
    exchange: Exchange,
    symbol: str,
    start_time: int,
    end_time: int
) -> pd.DataFrame:
    """Fetch funding rates for a single exchange/symbol pair."""
    client = get_exchange_client(exchange)
    return await client.get_funding_rates(symbol, start_time, end_time)


def fetch_funding_rates(
    exchange: Exchange,
    symbol: str,
    start_time: int,
    end_time: int
) -> pd.DataFrame:
    """Sync wrapper for fetching funding rates."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            fetch_funding_rates_async(exchange, symbol, start_time, end_time)
        )
    finally:
        loop.close()


# =============================================================================
# Main Application
# =============================================================================

def main():
    st.title("📈 Funding Rate Explorer")
    st.markdown(
        "Query and visualize historical funding rates from multiple cryptocurrency exchanges."
    )

    # -------------------------------------------------------------------------
    # Sidebar - Exchange Selection
    # -------------------------------------------------------------------------
    st.sidebar.header("Configuration")

    if st.sidebar.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

    available_exchanges = [e.value for e in Exchange]
    selected_exchanges = st.sidebar.multiselect(
        "Select Exchanges",
        options=available_exchanges,
        default=["Binance"],
        help="Select one or more exchanges to query funding rates from.",
    )

    if not selected_exchanges:
        st.warning("Please select at least one exchange.")
        return

    # -------------------------------------------------------------------------
    # Sidebar - Symbol Selection (per exchange)
    # -------------------------------------------------------------------------
    st.sidebar.subheader("Symbol Selection")

    exchange_symbols: Dict[str, List[str]] = {}

    for exchange_name in selected_exchanges:
        with st.sidebar.expander(f"{exchange_name} Symbols", expanded=True):
            symbols = get_symbol_list(exchange_name)
            
            if not symbols:
                st.warning(f"No symbols available for {exchange_name}")
                continue

            # Default to BTC-related symbols if available
            default_symbols = []
            for sym in symbols:
                if "BTC" in sym.upper():
                    default_symbols.append(sym)
                    break
            
            selected = st.multiselect(
                f"Select {exchange_name} symbols",
                options=symbols,
                default=default_symbols[:1] if default_symbols else symbols[:1],
                key=f"symbols_{exchange_name}",
            )
            
            if selected:
                exchange_symbols[exchange_name] = selected

    if not any(exchange_symbols.values()):
        st.warning("Please select at least one symbol from any exchange.")
        return

    # -------------------------------------------------------------------------
    # Sidebar - Date Range
    # -------------------------------------------------------------------------
    st.sidebar.subheader("Date Range")

    # Default to last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start = st.date_input(
            "Start Date",
            value=start_date,
            max_value=end_date,
        )
    with col2:
        end = st.date_input(
            "End Date",
            value=end_date,
            min_value=start,
        )

    # Convert to milliseconds
    start_ms = int(datetime.combine(start, datetime.min.time()).timestamp() * 1000)
    end_ms = int(datetime.combine(end, datetime.max.time()).timestamp() * 1000)

    # -------------------------------------------------------------------------
    # Sidebar - Table Options
    # -------------------------------------------------------------------------
    st.sidebar.subheader("Table Options")
    table_daily_annualized = st.sidebar.checkbox(
        "Show Daily Annualized (%)",
        value=False,
        help="If checked, the data table will show daily mean annualized funding rates instead of raw hourly rates."
    )

    # -------------------------------------------------------------------------
    # Fetch Data Button
    # -------------------------------------------------------------------------
    st.sidebar.divider()

    if st.sidebar.button("🔍 Get Funding Rates", type="primary", use_container_width=True):
        all_data: Dict[str, pd.DataFrame] = {}
        
        # Progress bar
        total_queries = sum(len(syms) for syms in exchange_symbols.values())
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        query_count = 0
        
        for exchange_name, symbols in exchange_symbols.items():
            exchange = Exchange(exchange_name)
            
            for symbol in symbols:
                query_count += 1
                status_text.text(f"Fetching {exchange_name} - {symbol}...")
                progress_bar.progress(query_count / total_queries)
                
                try:
                    df = fetch_funding_rates(exchange, symbol, start_ms, end_ms)
                    if not df.empty:
                        # Add exchange column for identification
                        df['exchange'] = exchange_name
                        all_data[f"{exchange_name}:{symbol}"] = df
                except Exception as e:
                    st.error(f"Error fetching {exchange_name} - {symbol}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Store in session state
        st.session_state['funding_data'] = all_data
        st.session_state['query_params'] = {
            'start': start,
            'end': end,
            'exchanges': exchange_symbols,
        }

    # -------------------------------------------------------------------------
    # Display Results
    # -------------------------------------------------------------------------
    if 'funding_data' in st.session_state and st.session_state['funding_data']:
        all_data = st.session_state['funding_data']
        query_params = st.session_state.get('query_params', {})
        
        st.success(
            f"Loaded funding rates for {len(all_data)} symbol(s) "
            f"from {query_params.get('start')} to {query_params.get('end')}"
        )

        # ---------------------------------------------------------------------
        # Visualization Tabs
        # ---------------------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["📊 Chart", "📋 Data Table", "📈 Statistics"])

        with tab1:
            st.subheader("Funding Rate Chart")
            
            # Combine all data for charting
            chart_data = pd.DataFrame()
            
            for key, df in all_data.items():
                if not df.empty and 'funding_rate' in df.columns:
                    # Resample to daily (sum) to get total daily funding
                    # Annualize: sum_daily_rate * 365 days * 100 (%)
                    daily_annualized = df['funding_rate'].resample('1D').sum() * 365 * 100
                    daily_annualized.name = key.replace(":", " - ")
                    chart_data = pd.concat([chart_data, daily_annualized], axis=1)
            
            if not chart_data.empty:
                st.markdown("**Daily Annualized Funding Rate (%)**")
                st.line_chart(chart_data)
                
                with st.expander("Chart Details"):
                    st.info("Funding rates are aggregated to daily sum and annualized (Daily Sum * 365 * 100%).")
            else:
                st.info("No data to display.")

        with tab2:
            st.subheader("Data Table")
            
            # Create a wide-format DataFrame with MultiIndex columns
            wide_df = pd.DataFrame()
            
            for key, df in all_data.items():
                if not df.empty:
                    # Split only on the first colon to handle symbols that contain colons (e.g., Hyperliquid DEXs)
                    parts = key.split(":", 1)
                    if len(parts) == 2:
                        exchange_name, symbol = parts
                    else:
                        exchange_name = "Unknown"
                        symbol = key
                    
                    if table_daily_annualized:
                        # Resample to daily sum and annualize
                        series = df['funding_rate'].resample('1D').sum() * 365 * 100
                    else:
                        # Use raw funding rates and round index to nearest hour for alignment
                        series = df['funding_rate'].copy()
                        series.index = series.index.round("1h")
                    
                    # Create a MultiIndex for this column
                    series.name = (exchange_name, symbol)
                    wide_df = pd.concat([wide_df, series], axis=1)
            
            if not wide_df.empty:
                # Sort index (time) descending
                wide_df = wide_df.sort_index(ascending=False)
                
                # Format for display
                st.dataframe(
                    wide_df,
                    use_container_width=True,
                )
                
                # Download button
                csv = wide_df.to_csv()
                suffix = "daily_annualized" if table_daily_annualized else "hourly"
                st.download_button(
                    label=f"📥 Download CSV ({suffix.replace('_', ' ').title()})",
                    data=csv,
                    file_name=f"funding_rates_{suffix}_{query_params.get('start')}_{query_params.get('end')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No data to display.")

        with tab3:
            st.subheader("Statistics")
            
            stats_data = []
            for key, df in all_data.items():
                if not df.empty and 'funding_rate' in df.columns:
                    funding = df['funding_rate']
                    stats_data.append({
                        'Source': key,
                        'Count': len(funding),
                        'Mean (%)': f"{funding.mean() * 100:.4f}",
                        'Std (%)': f"{funding.std() * 100:.4f}",
                        'Min (%)': f"{funding.min() * 100:.4f}",
                        'Max (%)': f"{funding.max() * 100:.4f}",
                        'Annualized Return (%)': f"{(funding.sum() / (len(funding.resample('1D')) if len(funding.resample('1D')) > 0 else 1)) * 365 * 100:.2f}",
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            else:
                st.info("No statistics to display.")

    else:
        # Initial state - show instructions
        st.info(
            "👈 Select exchanges and symbols from the sidebar, then click "
            "'Get Funding Rates' to fetch data."
        )

        # Show exchange info
        st.subheader("Available Exchanges")
        
        exchange_info = {
            "Aster": "Binance-compatible perpetual futures (https://asterdex.com)",
            "Binance": "Binance USD-M Perpetual Futures (https://binance.com)",
            "Hyperliquid": "Decentralized perpetual futures (https://hyperliquid.xyz)",
            "Lighter": "zkLighter perpetual futures on Arbitrum (https://lighter.xyz)",
            "Pacifica": "Solana-based perpetual futures (https://pacifica.fi)",
        }
        
        for name, description in exchange_info.items():
            st.markdown(f"- **{name}**: {description}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
