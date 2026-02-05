# Funding Rate Explorer

A Streamlit application for querying and visualizing historical funding rates from multiple cryptocurrency perpetual futures exchanges.

## Supported Exchanges

- **Aster** - Binance-compatible perpetual futures
- **Binance** - USD-M Perpetual Futures
- **Hyperliquid** - Decentralized perpetual futures
- **Lighter** - zkLighter perpetual futures on Arbitrum
- **Pacifica** - Solana-based perpetual futures

## Features

- Multi-exchange support with unified interface
- Dynamic symbol selection based on exchange metadata
- Configurable date range (default: last 30 days)
- Interactive charts with annualized funding rates
- Data export to CSV
- Statistical summary (mean, std, min, max, annualized)

## Installation

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

### Streamlit Community Cloud Deployment

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository and branch
5. Set the main file path to `app.py`
6. Click "Deploy"

## File Structure

```
funding_rate_app/
├── app.py              # Streamlit application entry point
├── exchange_utils.py   # Exchange clients and utilities
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Usage

1. **Select Exchanges**: Use the sidebar to choose which exchanges to query
2. **Select Symbols**: For each exchange, select the trading pairs you're interested in
3. **Set Date Range**: Choose the start and end dates (default: last 30 days)
4. **Get Data**: Click the "Get Funding Rates" button to fetch data
5. **Explore**: View charts, data tables, and statistics in the main panel

## API Endpoints Used

| Exchange | Instruments | Funding Rates |
|----------|-------------|---------------|
| Aster | `/fapi/v1/exchangeInfo` | `/fapi/v1/fundingRate` |
| Binance | `/fapi/v1/exchangeInfo` | `/fapi/v1/fundingRate` |
| Hyperliquid | POST `/info` (metaAndAssetCtxs) | POST `/info` (fundingHistory) |
| Lighter | `/api/v1/orderBookDetails` | `/api/v1/fundings` |
| Pacifica | `/info` | `/funding_rate/history` |

## Notes

- Funding rates are typically paid every 8 hours
- Annualized rates are calculated as: `rate * 3 * 365 * 100%`
- Data is cached for 1 hour to reduce API calls
- No API keys are required (public endpoints only)

## License

MIT
