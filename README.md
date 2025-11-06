# Solana Token Sniper Bot 🚀

**Ultra-fast Solana token buyer** that monitors Telegram channels for new token listings and executes swaps using Jupiter Aggregator for best routing. Built for low latency, reliability, and safe testing.

## Features
- Real-time Telegram monitoring (Telethon)
- Jupiter v6 swap integration
- Multi-RPC broadcasting for faster inclusion
- Priority fee tuning and test mode
- Robust logging and retry/error handling

## Quick start
1. Copy `.env.sniper.example` → `.env` and fill required keys (TG_API_ID, TG_API_HASH, WALLET_SECRET_KEY, RPC_ENDPOINTS).
2. Install deps:
```bash
pip install -r requirements.txt
# solana-sniper-bot
Ultra-fast Solana token buyer that monitors Telegram channels and executes low-latency swaps via Jupiter, with multi-RPC broadcasting, priority fees, and safe test mode.
