import asyncio
import base64
import json
import re
import time
import httpx
from typing import Optional, List, Dict, Any, Set
from datetime import datetime

# Telegram
from telethon import TelegramClient, events
import aiohttp

# Solana
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TxOpts
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Bot configuration with hardcoded values"""
    
    # Safety Mechanisms - AUTO SELL
    ENABLE_AUTO_SELL = True
    SELL_50_PERCENT_AT_ROI = 100  # Sell 50% at 100% profit (2X)
    SELL_100_PERCENT_AT_ROI = 200  # Sell 100% at 200% profit (3X)

    # Telegram Monitoring
    TG_API_ID = 20220319
    TG_API_HASH = "f18fed313119349048b31edd3dc5241d"
    TG_CHANNEL = "@test_mode4"
    
    # Telegram Bot (for notifications)
    BOT_TOKEN = 'add bot token'
    CHAT_ID = 'add chat id'
    
    # Solana Wallet
    WALLET_SECRET_KEY = "[add wallet secret key]"
    
    # RPC Endpoints - Premium RPCs for maximum speed
    RPC_ENDPOINTS = [
        "https://mainnet.helius-rpc.com/?api-key=YOUR_HELIUS_KEY",  # Replace with your key
        "https://rpc.ankr.com/solana",
        "https://api.mainnet-beta.solana.com",
        "https://solana-api.projectserum.com"
    ]
    
    # Jupiter API
    JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
    JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"
    
    # Trading Parameters
    INPUT_TOKEN = "So11111111111111111111111111111111111111112"  # SOL (wrapped)
    BUY_AMOUNT_SOL = 0.01
    SLIPPAGE_BPS = 300  # 3%
    
    # Performance - ULTRA MAXIMUM SPEED OPTIMIZATION
    PRIORITY_FEE_MICRO_LAMPORTS = 1000000  # 1M micro-lamports for ULTRA speed (Top 1%)
    SKIP_PREFLIGHT = True
    COMPUTE_UNIT_LIMIT = 200000
    
    # Safety
    TEST_MODE = True
    MAX_RETRIES = 1
    REQUEST_TIMEOUT = 2
    
    # Advanced RPC optimization
    USE_SINGLE_FASTEST_RPC = True
    
    # Price Tracking - JUPITER ONLY
    TRACKING_POLL_INTERVAL = 1  # 1 second polling interval

# ============================================================================
# UTILITIES
# ============================================================================

class Logger:
    """Enhanced logging with timestamps"""
    
    @staticmethod
    def info(msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [INFO] {msg}")
    
    @staticmethod
    def success(msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [SUCCESS] ✓ {msg}")
    
    @staticmethod
    def error(msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [ERROR] ✗ {msg}")
    
    @staticmethod
    def warning(msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [WARNING] ⚠ {msg}")

log = Logger()

# Mint address patterns
CA_PATTERN = re.compile(r"(?i)\bca\s*[:=]?\s*([1-9A-HJ-NP-Za-km-z]{43,44})\b")
BARE_MINT_PATTERN = re.compile(r"\b([1-9A-HJ-NP-Za-km-z]{43,44})\b")
PUMP_PATTERN = re.compile(
    r"\b([1-9A-HJ-NP-Za-km-z]{43,44})(?=\s*pump\b)", re.IGNORECASE
)
print("ACTIVE PUMP PATTERN:", PUMP_PATTERN.pattern)


def parse_mint_from_text(text: str) -> Optional[str]:
    """Extract Solana mint address from text correctly."""
    if not text:
        return None

    text = str(text).strip()

    # 1) Check CA: ... patterns
    m = CA_PATTERN.search(text)
    if m:
        return m.group(1)

    # 2) Check pump pattern (strip the word pump)
    m = PUMP_PATTERN.search(text)
    if m:
        return m.group(1)  # << important: only group(1)

    # 3) Fallback bare mint
    m = BARE_MINT_PATTERN.search(text)
    if m:
        return m.group(1)

    return None

def load_keypair_from_env(secret: str) -> Keypair:
    """Load Solana keypair from JSON array"""
    try:
        secret = secret.strip()
        arr = json.loads(secret)
        if not isinstance(arr, list) or len(arr) != 64:
            raise ValueError(f"Secret key must be 64 integers, got {len(arr) if isinstance(arr, list) else 'non-list'}")
        
        sk_bytes = bytes(arr)
        return Keypair.from_bytes(sk_bytes)
    except Exception as e:
        raise RuntimeError(f"Invalid WALLET_SECRET_KEY: {e}") from e

# ============================================================================
# TELEGRAM NOTIFICATION BOT
# ============================================================================

class TelegramNotifier:
    """Send notifications to personal Telegram bot"""
    
    @staticmethod
    async def send_message(message: str):
        """Send message to personal Telegram (non-blocking)"""
        url = f"https://api.telegram.org/bot{Config.BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": Config.CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
            log.info(f"Sending notification to Telegram...")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    response_text = await response.text()
                    if response.status == 200:
                        log.success(f"Notification sent successfully!")
                    else:
                        log.error(f"Notification failed! Status: {response.status}")
                        log.error(f"Response: {response_text}")
                        
                        if "chat not found" in response_text.lower():
                            log.error("Chat not found! Did you start the bot in Telegram?")
                        elif "blocked" in response_text.lower():
                            log.error("Bot is blocked! Unblock it in Telegram.")
                        elif "unauthorized" in response_text.lower():
                            log.error("Invalid bot token!")
        except asyncio.TimeoutError:
            log.error(f"Notification timeout after 5s")
        except Exception as e:
            log.error(f"Notification error: {type(e).__name__}: {e}")
    
    @staticmethod
    async def send_startup_notification(wallet_pubkey: str, balance_sol: float, rpc_url: str):
        """Send detailed startup notification"""
        startup_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        balance_status = "✅" if balance_sol >= 0.01 else "⚠️"
        mode_emoji = "🧪" if Config.TEST_MODE else "🔴"
        
        message = (
            f"🤖 <b>SOLANA SNIPER BOT STARTED</b>\n\n"
            f"⏰ <b>Time:</b> {startup_time}\n"
            f"{mode_emoji} <b>Mode:</b> {'TEST MODE' if Config.TEST_MODE else 'LIVE TRADING'}\n\n"
            f"💰 <b>WALLET INFO</b>\n"
            f"├ Address: <code>{wallet_pubkey}</code>\n"
            f"├ Balance: {balance_sol:.6f} SOL {balance_status}\n"
            f"└ RPC: {rpc_url.split('//')[1] if '//' in rpc_url else rpc_url}\n\n"
            f"⚙️ <b>TRADING CONFIG</b>\n"
            f"├ Buy Amount: {Config.BUY_AMOUNT_SOL} SOL per trade\n"
            f"├ Slippage: {Config.SLIPPAGE_BPS / 100}%\n"
            f"├ Priority Fee: {Config.PRIORITY_FEE_MICRO_LAMPORTS / 1000}K micro-lamports\n"
            f"└ Channel: {Config.TG_CHANNEL}\n\n"
            f"⚡ <b>SPEED OPTIMIZATION</b>\n"
            f"├ Instant Buy: Enabled (no pre-checks)\n"
            f"├ Skip Preflight: {Config.SKIP_PREFLIGHT}\n"
            f"├ API Timeout: {Config.REQUEST_TIMEOUT}s\n"
            f"└ Price Tracking: Jupiter polling ({Config.TRACKING_POLL_INTERVAL}s intervals)\n\n"
        )
        
        if balance_sol < Config.BUY_AMOUNT_SOL:
            message += (
                f"⚠️ <b>WARNING</b>\n"
                f"Balance ({balance_sol:.6f} SOL) is less than buy amount ({Config.BUY_AMOUNT_SOL} SOL).\n\n"
            )
        
        message += "✅ Bot is now monitoring for new tokens...\n🎯 Ready to snipe!"
        
        await TelegramNotifier.send_message(message)

# ============================================================================
# TOKEN DATA FETCHER (Background only - not blocking buys)
# ============================================================================

class TokenDataFetcher:
    """Fetch token information in background"""
    
    @staticmethod
    async def get_token_info(mint: str) -> Optional[Dict[str, Any]]:
        """Fetch token metadata and market cap (background task)"""
        result = {
            "name": "Unknown Token",
            "symbol": "???",
            "chain": "Solana",
            "mint": mint,
            "market_cap": None,
            "price_usd": None
        }
        
        try:
            url = f"https://api.mainnet-beta.solana.com"
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getAsset",
                    "params": {"id": mint}
                }
                
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "result" in data:
                            res = data["result"]
                            result["name"] = res.get("content", {}).get("metadata", {}).get("name", "Unknown")
                            result["symbol"] = res.get("content", {}).get("metadata", {}).get("symbol", "???")
        except Exception as e:
            log.warning(f"Token metadata fetch error: {e}")
        
        # Fetch market cap from DexScreener API
        try:
            async with aiohttp.ClientSession() as session:
                dex_url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
                async with session.get(dex_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("pairs") and len(data["pairs"]) > 0:
                            pairs = data["pairs"]
                            pairs.sort(key=lambda x: float(x.get("liquidity", {}).get("usd", 0) or 0), reverse=True)
                            top_pair = pairs[0]
                            
                            result["market_cap"] = top_pair.get("marketCap")
                            result["price_usd"] = top_pair.get("priceUsd")
                            result["liquidity_usd"] = top_pair.get("liquidity", {}).get("usd")
                            result["fdv"] = top_pair.get("fdv")
                            
                            log.success(f"Market data: MC=${result['market_cap']:,.0f}" if result['market_cap'] else "Market data: MC=N/A")
        except Exception as e:
            log.warning(f"Market cap fetch error: {e}")
        
        return result

# ============================================================================
# JUPITER PRICE TRACKER - USD TRACKING WITH 1S POLLING
# ============================================================================

class JupiterPriceTracker:
    """Real-time price tracking using Jupiter API polling with USD prices"""
    
    def __init__(self, mint_address: str, entry_price: float, trade_info: Dict[str, Any], bot_instance):
        self.mint_address = mint_address
        self.entry_price = entry_price  # Entry price in SOL per token
        self.trade_info = trade_info
        self.bot = bot_instance
        self.notified_multiples: Set[float] = set()  # Changed to float for 1.5X, 2.5X etc
        self.current_price = 0
        self.is_running = True
        self.start_time = time.time()
        self.last_notification_time = 0
        self.notification_cooldown = 5  # 5 seconds between notifications
        self.sol_price_usd = None  # Cache SOL price
    
    async def get_sol_price_usd(self) -> Optional[float]:
        """Get current SOL price in USD using Jupiter (real-time, same source as trades)"""
        try:
            # Quote 1 SOL -> USDC to get SOL price
            # USDC mint: EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v
            usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            one_sol_lamports = int(1e9)  # 1 SOL
            
            quote_data = await JupiterAPI.get_quote(
                Config.INPUT_TOKEN,  # SOL
                usdc_mint,           # USDC
                one_sol_lamports,
                50  # Low slippage for stable pair
            )
            
            if quote_data:
                out_amount = int(quote_data.get("outAmount", 0))
                # USDC has 6 decimals
                sol_price = out_amount / 1e6
                return sol_price if sol_price > 0 else None
        except Exception as e:
            log.warning(f"Failed to get SOL price from Jupiter: {e}")
        return None
    
    async def start_tracking(self):
        """Main entry point for starting tracking"""
        token_symbol = self.trade_info.get('token_symbol', '???')
        log.info(f"📊 Starting Jupiter polling for {token_symbol}")
        log.info(f"   Entry price: {self.entry_price:.15f} SOL/token")
        
        # Get SOL price once at start
        self.sol_price_usd = await self.get_sol_price_usd()
        if self.sol_price_usd:
            entry_price_usd = self.entry_price * self.sol_price_usd
            log.info(f"   Entry price: ${entry_price_usd:.12f} USD/token (SOL @ ${self.sol_price_usd:.2f})")
        
        try:
            await self._track_jupiter_polling()
        except Exception as e:
            log.error(f"Tracking error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
    
    async def _track_jupiter_polling(self):
        """Jupiter API polling at 1 second intervals"""
        max_duration = 86400  # 24 hours
        poll_interval = 1  # 1 second polling for fast updates
        
        log.info(f"   Using {poll_interval}s poll interval (fast mode)")
        
        # Get the amount we used for entry
        entry_sol_amount = self.trade_info.get("buy_amount", 0.01)
        entry_sol_lamports = int(entry_sol_amount * 1e9)
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running and (time.time() - self.start_time) < max_duration:
            try:
                # Use EXACT SAME quote as entry for consistent pricing
                quote_data = await JupiterAPI.get_quote(
                    Config.INPUT_TOKEN,  # From SOL
                    self.mint_address,   # To Token
                    entry_sol_lamports,  # Same amount as entry
                    Config.SLIPPAGE_BPS
                )
                
                if quote_data:
                    consecutive_errors = 0  # Reset error counter
                    
                    in_amount_lamports = int(quote_data.get("inAmount", 0))
                    out_amount_tokens = int(quote_data.get("outAmount", 0))
                    
                    # Calculate current price EXACTLY like entry price
                    if out_amount_tokens > 0 and in_amount_lamports > 0:
                        in_amount_sol = in_amount_lamports / 1e9
                        current_price_per_token_sol = in_amount_sol / out_amount_tokens
                        
                        # Calculate multiple
                        multiple = current_price_per_token_sol / self.entry_price if self.entry_price > 0 else 0
                        
                        # Convert to USD if we have SOL price
                        if self.sol_price_usd:
                            current_price_usd = current_price_per_token_sol * self.sol_price_usd
                            entry_price_usd = self.entry_price * self.sol_price_usd
                            log.info(f"   💲 Price: ${current_price_usd:.12f} | Entry: ${entry_price_usd:.12f} | {multiple:.2f}X")
                        else:
                            log.info(f"   💎 Price: {current_price_per_token_sol:.15f} SOL | {multiple:.2f}X")
                        
                        if multiple > 100000:  # Sanity check
                            log.warning(f"⚠️ Suspicious price: {multiple:.2f}X - might be data error")
                        elif multiple > 0.01:  # Only track if price makes sense
                            await self._process_price_update(current_price_per_token_sol, multiple)
                        else:
                            log.warning(f"⚠️ Token price crashed: {multiple:.4f}X")
                    else:
                        log.warning("Got zero amount from quote")
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        log.error(f"Too many consecutive errors ({consecutive_errors}), backing off...")
                        await asyncio.sleep(poll_interval * 5)
                        consecutive_errors = 0
                    else:
                        log.warning(f"Quote returned None (error {consecutive_errors}/{max_consecutive_errors})")
                
                await asyncio.sleep(poll_interval)
            
            except Exception as e:
                consecutive_errors += 1
                log.warning(f"Poll error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                if consecutive_errors >= max_consecutive_errors:
                    await asyncio.sleep(poll_interval * 10)  # Long backoff
                    consecutive_errors = 0
                else:
                    await asyncio.sleep(poll_interval * 2)
    
    async def _process_price_update(self, current_price_sol: float, multiple: float):
        """Process price update and check for profit milestones"""
        self.current_price = current_price_sol
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_notification_time < self.notification_cooldown:
            return
        
        # Define milestones: 1.5X, 2X, 2.5X, 3X, 3.5X, 4X, 4.5X, 5X, 6X, 7X, 8X, 9X, 10X, 15X, 20X, 25X, 50X, 100X, 500X, 1000X
        milestones = [
            1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            15.0, 20.0, 25.0, 30.0, 40.0, 50.0,
            75.0, 100.0, 150.0, 200.0, 500.0, 1000.0
        ]
        
        # Find the highest milestone reached that we haven't notified yet
        highest_reached = None
        for milestone in milestones:
            if multiple >= milestone and milestone not in self.notified_multiples:
                highest_reached = milestone
        
        # Notify for the highest milestone
        if highest_reached is not None:
            self.notified_multiples.add(highest_reached)
            self.last_notification_time = current_time
            await self._send_profit_notification(highest_reached, multiple, current_price_sol)
    
    async def _send_profit_notification(self, target_multiple: float, actual_multiple: float, current_price_sol: float):
        """Send profit milestone notification with USD prices"""
        token_name = self.trade_info.get("token_name", "Unknown")
        token_symbol = self.trade_info.get("token_symbol", "???")
        buy_amount_sol = self.trade_info.get("buy_amount", 0.01)
        tokens_owned = self.trade_info.get("tokens_bought", 0)
        
        current_value_sol = tokens_owned * current_price_sol if tokens_owned > 0 else 0
        profit_sol = current_value_sol - buy_amount_sol
        profit_pct = (actual_multiple - 1) * 100
        
        elapsed_time = self.bot._format_time_elapsed(self.start_time)
        
        # Calculate USD values if we have SOL price
        entry_price_usd = self.entry_price * self.sol_price_usd if self.sol_price_usd else None
        current_price_usd = current_price_sol * self.sol_price_usd if self.sol_price_usd else None
        buy_amount_usd = buy_amount_sol * self.sol_price_usd if self.sol_price_usd else None
        current_value_usd = current_value_sol * self.sol_price_usd if self.sol_price_usd else None
        profit_usd = profit_sol * self.sol_price_usd if self.sol_price_usd else None
        
        # Format prices
        if entry_price_usd and current_price_usd:
            if entry_price_usd < 0.000001:
                entry_str = f"${entry_price_usd:.12f}"
            elif entry_price_usd < 0.001:
                entry_str = f"${entry_price_usd:.8f}"
            else:
                entry_str = f"${entry_price_usd:.6f}"
            
            if current_price_usd < 0.000001:
                current_str = f"${current_price_usd:.12f}"
            elif current_price_usd < 0.001:
                current_str = f"${current_price_usd:.8f}"
            else:
                current_str = f"${current_price_usd:.6f}"
            
            price_info = (
                f"<b>Entry Price:</b> {entry_str}\n"
                f"<b>Current Price:</b> {current_str}\n"
            )
        else:
            # Fallback to SOL prices
            price_info = (
                f"<b>Entry Price:</b> {self.entry_price:.15f} SOL\n"
                f"<b>Current Price:</b> {current_price_sol:.15f} SOL\n"
            )
        
        # Format value section
        if buy_amount_usd and current_value_usd and profit_usd:
            value_section = (
                f"💰 <b>VALUE (USD)</b>\n"
                f"├ Entry: ${buy_amount_usd:.2f} ({buy_amount_sol:.4f} SOL)\n"
                f"├ Current: ${current_value_usd:.2f} ({current_value_sol:.4f} SOL)\n"
                f"└ Profit: ${profit_usd:+.2f} ({profit_sol:+.4f} SOL)\n\n"
            )
        else:
            value_section = (
                f"💰 <b>VALUE (SOL)</b>\n"
                f"├ Entry: {buy_amount_sol:.4f} SOL\n"
                f"├ Current: {current_value_sol:.4f} SOL\n"
                f"└ Profit: {profit_sol:+.4f} SOL\n\n"
            )
        
        # Format milestone display
        if target_multiple == int(target_multiple):
            milestone_str = f"{int(target_multiple)}X"
        else:
            milestone_str = f"{target_multiple:.1f}X"
        
        message = (
            f"🚀 <b>{milestone_str} PROFIT MILESTONE!</b>\n\n"
            f"<b>Token:</b> {token_name} ({token_symbol})\n"
            f"{price_info}"
            f"<b>Actual Multiple:</b> <b>{actual_multiple:.2f}X</b>\n"
            f"<b>Profit:</b> +{profit_pct:.2f}%\n\n"
            f"{value_section}"
            f"📊 Tokens: {tokens_owned:,.0f}\n"
            f"⏱️ Time Held: {elapsed_time}\n"
            f"📈 Polling every 1 second"
        )
        
        log.success(f"🚀 {token_symbol}: {milestone_str} milestone! (actual: {actual_multiple:.2f}X)")
        await TelegramNotifier.send_message(message)
    
    async def stop(self):
        """Stop tracking"""
        self.is_running = False

# ============================================================================
# JUPITER API INTEGRATION
# ============================================================================

class JupiterAPI:
    """Jupiter Aggregator API client - Ultra-fast mode"""
    
    _session: Optional[aiohttp.ClientSession] = None
    
    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        """Get or create persistent session for connection reuse"""
        if cls._session is None or cls._session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,
                ttl_dns_cache=300,
                enable_cleanup_closed=True
            )
            cls._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)
            )
        return cls._session
    
    @classmethod
    async def close_session(cls):
        """Close persistent session"""
        if cls._session and not cls._session.closed:
            await cls._session.close()
    
    @classmethod
    async def get_quote(
        cls,
        input_mint: str,
        output_mint: str,
        amount_lamports: int,
        slippage_bps: int
    ) -> Optional[Dict[str, Any]]:
        """Get swap quote from Jupiter API - ULTRA FAST"""
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount_lamports),
            "slippageBps": str(slippage_bps),
            "onlyDirectRoutes": "true",
            "asLegacyTransaction": "false"
        }
        
        try:
            session = await cls.get_session()
            async with session.get(Config.JUPITER_QUOTE_URL, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    log.error(f"Quote failed: {response.status}")
                    return None
        except asyncio.TimeoutError:
            log.error("Quote timeout")
            return None
        except Exception as e:
            log.error(f"Quote error: {e}")
            return None
    
    @classmethod
    async def get_swap_transaction(
        cls,
        route: Dict[str, Any],
        user_public_key: str
    ) -> Optional[str]:
        """Get swap transaction from Jupiter API - ULTRA FAST"""
        body = {
            "quoteResponse": route,
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": True,
            "computeUnitPriceMicroLamports": Config.PRIORITY_FEE_MICRO_LAMPORTS,
            "dynamicComputeUnitLimit": True
        }
        
        try:
            session = await cls.get_session()
            async with session.post(Config.JUPITER_SWAP_URL, json=body) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("swapTransaction")
                else:
                    error_text = await response.text()
                    log.error(f"Swap tx failed: {response.status}")
                    log.error(f"Response: {error_text[:200]}")
                    return None
        except asyncio.TimeoutError:
            log.error("Swap tx timeout")
            return None
        except Exception as e:
            log.error(f"Swap tx error: {e}")
            return None
            
# ============================================================================
# SOLANA TRANSACTION HANDLING
# ============================================================================

class SolanaTransactionHandler:
    """Handle Solana transaction signing and broadcasting - ULTRA FAST"""
    
    def __init__(self, wallet: Keypair):
        self.wallet = wallet
        self.rpc_clients: Dict[str, AsyncClient] = {}
        self.fastest_rpc: Optional[str] = None
    
    async def initialize_rpc_clients(self, endpoints: List[str]):
        """Pre-initialize RPC clients for faster execution"""
        for endpoint in endpoints:
            try:
                client = AsyncClient(endpoint)
                self.rpc_clients[endpoint] = client
                log.success(f"Pre-initialized RPC: {endpoint}")
            except Exception as e:
                log.warning(f"Failed to pre-init RPC {endpoint}: {e}")
    
    async def find_fastest_rpc(self) -> Optional[str]:
        """Benchmark RPCs and find the fastest one"""
        if self.fastest_rpc:
            return self.fastest_rpc
        
        log.info("Benchmarking RPCs for speed...")
        
        async def test_rpc_speed(endpoint: str) -> tuple:
            try:
                start = time.time()
                async with AsyncClient(endpoint) as client:
                    await asyncio.wait_for(
                        client.get_slot(),
                        timeout=3.0
                    )
                latency = (time.time() - start) * 1000
                return (endpoint, latency)
            except Exception:
                return (endpoint, float('inf'))
        
        results = await asyncio.gather(*[test_rpc_speed(ep) for ep in Config.RPC_ENDPOINTS])
        results.sort(key=lambda x: x[1])
        
        if results[0][1] != float('inf'):
            self.fastest_rpc = results[0][0]
            log.success(f"Fastest RPC: {self.fastest_rpc} ({results[0][1]:.0f}ms)")
            return self.fastest_rpc
        
        return None
    
    def sign_transaction(self, tx_bytes: bytes) -> Optional[bytes]:
        """Sign transaction using wallet - FIXED for VersionedTransaction"""
        try:
            # Deserialize the transaction
            vtx = VersionedTransaction.from_bytes(tx_bytes)
            
            # Sign it (VersionedTransaction uses a different signing method)
            # Create a new signed transaction with the wallet signature
            signed_tx = VersionedTransaction.populate(
                vtx.message,
                [self.wallet.sign_message(bytes(vtx.message))]
            )
            
            return bytes(signed_tx)
        except Exception as e:
            log.error(f"Signing failed: {e}")
            # Return original bytes as fallback
            return tx_bytes
    
    async def broadcast_ultra_fast(self, tx_bytes: bytes) -> bool:
        """Broadcast to fastest RPC only - MAXIMUM SPEED"""
        if not self.fastest_rpc:
            await self.find_fastest_rpc()
        
        if not self.fastest_rpc:
            log.error("No working RPC found")
            return False
        
        try:
            async with AsyncClient(self.fastest_rpc) as client:
                resp = await asyncio.wait_for(
                    client.send_raw_transaction(
                        tx_bytes,
                        opts=TxOpts(
                            skip_preflight=Config.SKIP_PREFLIGHT,
                            max_retries=0
                        )
                    ),
                    timeout=Config.REQUEST_TIMEOUT
                )
                log.success(f"TX broadcast: {resp.value if hasattr(resp, 'value') else resp}")
                return True
        except asyncio.TimeoutError:
            log.error("Broadcast timeout")
            return False
        except Exception as e:
            log.error(f"Broadcast error: {e}")
            return False
    
    async def broadcast_to_rpcs(self, tx_bytes: bytes, endpoints: List[str]) -> List[Dict[str, Any]]:
        """Broadcast transaction to multiple RPC endpoints in parallel"""
        async def send_to_rpc(endpoint: str) -> Dict[str, Any]:
            try:
                async with AsyncClient(endpoint) as client:
                    resp = await asyncio.wait_for(
                        client.send_raw_transaction(
                            tx_bytes,
                            opts=TxOpts(
                                skip_preflight=Config.SKIP_PREFLIGHT,
                                max_retries=0
                            )
                        ),
                        timeout=Config.REQUEST_TIMEOUT
                    )
                    return {"endpoint": endpoint, "success": True, "response": resp}
            except Exception as e:
                return {"endpoint": endpoint, "success": False, "error": str(e)}
        
        tasks = [send_to_rpc(ep) for ep in endpoints]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def cleanup(self):
        """Cleanup RPC clients"""
        for client in self.rpc_clients.values():
            try:
                await client.close()
            except:
                pass

# ============================================================================
# CORE BOT LOGIC - OPTIMIZED FOR SPEED
# ============================================================================


class SolanaSnipeBot:
    """Main bot class with ultra-fast execution"""
    
    def __init__(self):
        self.wallet = load_keypair_from_env(Config.WALLET_SECRET_KEY)
        self.tx_handler = SolanaTransactionHandler(self.wallet)
        self.trade_count = 0
        self.active_trades: Dict[str, Dict[str, Any]] = {}
        self.profit_tracking_tasks: Dict[str, asyncio.Task] = {}
        self.balance_sol = 0
        self.working_rpc = None
        
        log.info("=" * 70)
        log.info("Solana Token Sniper Bot - MAXIMUM SPEED MODE")
        log.info("=" * 70)
        log.info(f"Wallet: {self.wallet.pubkey()}")
        log.info(f"Channel: {Config.TG_CHANNEL}")
        log.info(f"Buy Amount: {Config.BUY_AMOUNT_SOL} SOL")
        log.info(f"Test Mode: {Config.TEST_MODE}")
        log.info(f"Priority Fee: {Config.PRIORITY_FEE_MICRO_LAMPORTS / 1000}K micro-lamports")
        log.info(f"Request Timeout: {Config.REQUEST_TIMEOUT}s")
        log.info(f"Single RPC Mode: {Config.USE_SINGLE_FASTEST_RPC}")
        log.info(f"Price Tracking: Jupiter polling ({max(Config.TRACKING_POLL_INTERVAL, 3)}s intervals)")
        log.info("=" * 70)        

    
    async def check_balance_and_rpc(self) -> bool:
        """Check wallet balance and find working RPC - OPTIMIZED"""
        log.info("Initializing ultra-fast RPC connections...") 
        
        await self.tx_handler.find_fastest_rpc()
        
        if not self.tx_handler.fastest_rpc:
            log.error("No working RPC found!")
            return False
        
        try:
            async with AsyncClient(self.tx_handler.fastest_rpc) as client:
                response = await asyncio.wait_for(
                    client.get_balance(self.wallet.pubkey()),
                    timeout=5.0
                )
                
                if response.value is not None:
                    self.balance_sol = response.value / 1e9
                    self.working_rpc = self.tx_handler.fastest_rpc
                    
                    log.success(f"Fastest RPC: {self.working_rpc}")
                    log.success(f"Balance: {self.balance_sol:.6f} SOL")
                    
                    if self.balance_sol < Config.BUY_AMOUNT_SOL:
                        log.warning(f"Balance ({self.balance_sol:.6f} SOL) < Buy amount ({Config.BUY_AMOUNT_SOL} SOL)")
                    
                    return True
        except Exception as e:
            log.error(f"RPC check failed: {str(e)[:50]}")
        
        return False
    
    async def attempt_swap_fast(self, mint_address: str):
        """ULTRA-FAST swap - buy immediately, notify later"""
        start_time = time.time()
        self.trade_count += 1
        trade_id = f"TRADE_{self.trade_count}"
        
        log.info(f"\n{'='*70}")
        log.info(f"[{trade_id}] FAST SNIPE: {mint_address}")
        log.info(f"{'='*70}")
        
        # Send immediate detection notification (BEFORE buy)
        detection_msg = (
            f"🎯 <b>TOKEN DETECTED!</b>\n\n"
            f"<b>Mint:</b> <code>{mint_address}</code>\n"
            f"<b>Trade ID:</b> {trade_id}\n"
            f"<b>Status:</b> Attempting buy...\n\n"
            f"⚡ Executing at TOP 1% speed..."
        )
        await TelegramNotifier.send_message(detection_msg)
        
        # Store basic trade info
        self.active_trades[mint_address] = {
            "trade_id": trade_id,
            "mint": mint_address,
            "detected_at": datetime.now(),
            "entry_time": None,
            "entry_price": None,
            "buy_amount": Config.BUY_AMOUNT_SOL
        }
        
        try:
            # STEP 1: Get quote (FAST - no other checks)
            log.info("[1/4] ⚡ Getting quote...")
            amount_lamports = int(Config.BUY_AMOUNT_SOL * 1e9)
            
            quote_data = await JupiterAPI.get_quote(
                Config.INPUT_TOKEN,
                mint_address,
                amount_lamports,
                Config.SLIPPAGE_BPS
            )
            
            if not quote_data:
                log.error("No quote available")
                await TelegramNotifier.send_message(
                    f"❌ <b>Quote Failed</b>\n\n"
                    f"<b>Mint:</b> <code>{mint_address}</code>\n"
                    f"<b>Reason:</b> No liquidity or invalid token"
                )
                return
            
            route = quote_data
            in_amount = int(route.get("inAmount", 0)) / 1e9
            out_amount_str = route.get("outAmount", "0")
            out_amount = int(out_amount_str) if out_amount_str else 0
            
            log.success(f"Quote: {in_amount} SOL → {out_amount} tokens")
            
            # STEP 2: Get swap transaction
            log.info("[2/4] ⚡ Getting swap tx...")
            swap_tx_b64 = await JupiterAPI.get_swap_transaction(route, str(self.wallet.pubkey()))
            
            if not swap_tx_b64:
                log.error("Failed to get swap transaction")
                await TelegramNotifier.send_message(
                    f"❌ <b>Swap TX Failed</b>\n\n"
                    f"<b>Mint:</b> <code>{mint_address}</code>\n"
                    f"<b>Reason:</b> Could not generate transaction"
                )
                return
            
            log.success("Swap tx received")
            
            # STEP 3: Sign transaction
            log.info("[3/4] ⚡ Signing...")
            tx_bytes = base64.b64decode(swap_tx_b64)
            signed_tx = self.tx_handler.sign_transaction(tx_bytes)
            
            if not signed_tx:
                log.error("Failed to sign transaction")
                return
            
            log.success("Signed")
            
            # ============================================================
            # CRITICAL: Calculate and VERIFY entry price
            # ============================================================
            if out_amount > 0 and in_amount > 0:
                entry_price_per_token = in_amount / out_amount
                log.info(f"💎 ENTRY PRICE CALCULATION:")
                log.info(f"   IN:  {in_amount} SOL")
                log.info(f"   OUT: {out_amount:,} tokens")
                log.info(f"   PRICE: {entry_price_per_token:.15f} SOL/token")
            else:
                log.error(f"❌ Invalid amounts! in={in_amount}, out={out_amount}")
                entry_price_per_token = 0
            
            # Store entry info IMMEDIATELY
            self.active_trades[mint_address]["entry_time"] = datetime.now()
            self.active_trades[mint_address]["entry_price"] = entry_price_per_token
            self.active_trades[mint_address]["tokens_bought"] = out_amount
            
            log.info(f"✅ Stored entry_price: {self.active_trades[mint_address]['entry_price']:.15f}")
            
            # STEP 4: Broadcast (ULTRA FAST)
            if Config.TEST_MODE:
                log.warning("[4/4] ⚡ TEST MODE - Simulating")
                log.info(f"Would broadcast {len(signed_tx)} bytes")
                
                elapsed = (time.time() - start_time) * 1000
                log.success(f"⚡⚡⚡ TEST SNIPE COMPLETE: {elapsed:.0f}ms")
                
                # Send notification immediately (don't use create_task in test mode)
                await self._send_buy_notification(
                    mint_address, in_amount, out_amount, elapsed, test_mode=True
                )
                
            else:
                # Use single fastest RPC or parallel broadcast
                if Config.USE_SINGLE_FASTEST_RPC:
                    log.info(f"[4/4] ⚡⚡⚡ Broadcasting to FASTEST RPC...")
                    success = await self.tx_handler.broadcast_ultra_fast(signed_tx)
                    
                    elapsed = (time.time() - start_time) * 1000
                    
                    if success:
                        log.success(f"⚡⚡⚡ LIVE SNIPE COMPLETE: {elapsed:.0f}ms")
                        
                        # Send notification (background for live mode to not slow down)
                        asyncio.create_task(self._send_buy_notification(
                            mint_address, in_amount, out_amount, elapsed, test_mode=False
                        ))
                    else:
                        log.error("Fastest RPC broadcast failed")
                        await TelegramNotifier.send_message(
                            f"❌ <b>Broadcast Failed</b>\n\n"
                            f"<b>Mint:</b> <code>{mint_address}</code>\n"
                            f"<b>RPC:</b> {self.tx_handler.fastest_rpc}"
                        )
                else:
                    log.info(f"[4/4] ⚡ Broadcasting to {len(Config.RPC_ENDPOINTS)} RPCs...")
                    results = await self.tx_handler.broadcast_to_rpcs(signed_tx, Config.RPC_ENDPOINTS)
                    
                    success_count = sum(1 for r in results if r["success"])
                    elapsed = (time.time() - start_time) * 1000
                    
                    if success_count > 0:
                        log.success(f"⚡⚡ LIVE SNIPE COMPLETE: {elapsed:.0f}ms ({success_count}/{len(results)} RPCs)")
                        
                        # Send notification (background for live mode to not slow down)
                        asyncio.create_task(self._send_buy_notification(
                            mint_address, in_amount, out_amount, elapsed, test_mode=False
                        ))
                    else:
                        log.error("All RPC broadcasts failed")
                        await TelegramNotifier.send_message(
                            f"❌ <b>Broadcast Failed</b>\n\n"
                            f"<b>Mint:</b> <code>{mint_address}</code>\n"
                            f"<b>Reason:</b> All RPCs rejected transaction"
                        )
            
            # ============================================================
            # CRITICAL: Start profit tracking ONCE with duplicate prevention
            # ============================================================
            if entry_price_per_token > 0:
                # ONLY start tracking if not already tracking
                if mint_address not in self.profit_tracking_tasks:
                    log.info(f"🚀 Starting profit tracking for {mint_address}")
                    tracking_task = asyncio.create_task(self.track_profit_jupiter(mint_address))
                    self.profit_tracking_tasks[mint_address] = tracking_task
                else:
                    log.warning(f"⚠️ Already tracking {mint_address} - skipping duplicate")
            else:
                log.error(f"⚠️ NOT starting tracking - entry_price={entry_price_per_token}") 
                           
        except Exception as e:
            log.error(f"Swap failed: {e}")
            import traceback
            traceback.print_exc()
            await TelegramNotifier.send_message(
                f"❌ <b>Error</b>\n\n"
                f"<b>Mint:</b> <code>{mint_address}</code>\n"
                f"<b>Error:</b> {str(e)[:200]}"
            )
    
    async def _send_buy_notification(self, mint_address: str, in_amount: float, out_amount: int, elapsed_ms: float, test_mode: bool):
        """Send buy notification with token info including MC and price (background task)"""
        # Fetch token info in background (doesn't block buy)
        token_info = await TokenDataFetcher.get_token_info(mint_address)
        token_name = token_info.get("name", "Unknown")
        token_symbol = token_info.get("symbol", "???")

        # Update stored trade info
        if mint_address in self.active_trades:
            self.active_trades[mint_address]["token_name"] = token_name
            self.active_trades[mint_address]["token_symbol"] = token_symbol
            self.active_trades[mint_address]["market_cap"] = token_info.get("market_cap")
            self.active_trades[mint_address]["price_usd"] = token_info.get("price_usd")

        mode_text = "TEST MODE - BUY SIMULATED" if test_mode else "BUY EXECUTED - LIVE"
        mode_emoji = "🧪" if test_mode else "✅"

        # Get entry price from stored trade data
        entry_price_sol_per_token = self.active_trades[mint_address].get("entry_price", 0)
        
        # If not stored yet, calculate it
        if entry_price_sol_per_token == 0 and out_amount > 0:
            entry_price_sol_per_token = in_amount / out_amount

        # Format market data
        mc_text = f"${token_info.get('market_cap', 0):,.0f}" if token_info.get('market_cap') else "N/A"
        
        # Get token price in USD and calculate entry price in USD
        price_usd = token_info.get('price_usd')
        entry_price_usd_per_token = None
        
        if price_usd:
            try:
                price_float = float(price_usd)
                if price_float < 0.000001:
                    price_text = f"${price_float:.12f}"
                elif price_float < 0.001:
                    price_text = f"${price_float:.8f}"
                elif price_float < 1:
                    price_text = f"${price_float:.6f}"
                else:
                    price_text = f"${price_float:.4f}"
                
                # Calculate entry price in USD (this is what we paid per token in USD)
                # We need SOL price in USD for this
                entry_price_usd_per_token = price_float
                
            except (ValueError, TypeError):
                price_text = "N/A"
        else:
            price_text = "N/A"

        # Format entry price in USD
        if entry_price_usd_per_token is not None:
            if entry_price_usd_per_token < 0.000001:
                entry_price_text = f"${entry_price_usd_per_token:.12f}/token"
            elif entry_price_usd_per_token < 0.001:
                entry_price_text = f"${entry_price_usd_per_token:.8f}/token"
            elif entry_price_usd_per_token < 1:
                entry_price_text = f"${entry_price_usd_per_token:.6f}/token"
            else:
                entry_price_text = f"${entry_price_usd_per_token:.4f}/token"
        else:
            # Fallback to SOL price if USD not available
            if entry_price_sol_per_token < 0.00000001:
                entry_price_text = f"{entry_price_sol_per_token:.6e} SOL/token"
            elif entry_price_sol_per_token < 0.0001:
                entry_price_text = f"{entry_price_sol_per_token:.15f} SOL/token"
            else:
                entry_price_text = f"{entry_price_sol_per_token:.10f} SOL/token"

        liquidity_usd = token_info.get('liquidity_usd')
        liquidity_text = f"${liquidity_usd:,.0f}" if liquidity_usd else "N/A"

        fdv = token_info.get('fdv')
        fdv_text = f"${fdv:,.0f}" if fdv else "N/A"

        buy_msg = (
            f"{mode_emoji} <b>{mode_text}</b>\n\n"
            f"<b>Token:</b> {token_name} ({token_symbol})\n"
            f"<b>Mint:</b> <code>{mint_address}</code>\n\n"
            f"💰 <b>MARKET DATA</b>\n"
            f"├ Market Cap: {mc_text}\n"
            f"├ Current Price: {price_text}\n"
            f"├ Liquidity: {liquidity_text}\n"
            f"└ FDV: {fdv_text}\n\n"
            f"📊 <b>TRADE INFO</b>\n"
            f"├ Entry Amount: {in_amount:.4f} SOL\n"
            f"├ Tokens Bought: {out_amount:,.0f}\n"
            f"├ Entry Price: {entry_price_text}\n"
            f"└ Execution Time: ⚡ {elapsed_ms:.0f}ms\n\n"
            f"📈 Jupiter polling active ({max(Config.TRACKING_POLL_INTERVAL, 3)}s intervals)..."
        )

        await TelegramNotifier.send_message(buy_msg)

    async def track_profit_jupiter(self, mint_address: str):
        """Track profit using Jupiter API polling only"""
        if mint_address not in self.active_trades:
            log.error(f"❌ Cannot track {mint_address} - not in active trades")
            return

        trade = self.active_trades[mint_address]
        entry_price = trade.get("entry_price")
        token_symbol = trade.get("token_symbol", "???")
        token_name = trade.get("token_name", "Unknown")

        # CRITICAL: Validate entry price
        if not entry_price or entry_price == 0:
            log.error(f"❌ STOPPING TRACKING for {token_name}")
            log.error(f"   Reason: Invalid entry price = {entry_price}")
            log.error(f"   Trade data: {trade}")
            return

        # Create tracker instance
        tracker = JupiterPriceTracker(mint_address, entry_price, trade, self)
        
        # Store tracker instance to prevent duplicates
        self.active_trades[mint_address]["tracker"] = tracker
        
        try:
            # Start tracking using the correct method
            await tracker.start_tracking()
        except Exception as e:
            log.error(f"Tracking failed for {token_symbol}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await tracker.stop()
            # Remove from tracking tasks when done
            if mint_address in self.profit_tracking_tasks:
                del self.profit_tracking_tasks[mint_address]

    @staticmethod
    def _format_time_elapsed(start_time: float) -> str:
        """Format elapsed time in human readable format"""
        elapsed = time.time() - start_time

        if elapsed < 60:
            return f"{int(elapsed)}s"
        elif elapsed < 3600:
            return f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        elif elapsed < 86400:
            return f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"
        else:
            return f"{int(elapsed // 86400)}d {int((elapsed % 86400) // 3600)}h"

    async def handle_new_message(self, event):
        """Handle new messages from monitored channel - AGGRESSIVE CA EXTRACTION"""
        try:
            # Get message text
            message_text = event.message.message or ""
            full_text = message_text

            # Extract from media caption
            if hasattr(event.message, 'media') and event.message.media:
                if hasattr(event.message.media, 'caption'):
                    caption = event.message.media.caption or ""
                    full_text = f"{message_text} {caption}"

                # Extract from webpage preview (embedded cards)
                if hasattr(event.message.media, 'webpage'):
                    webpage = event.message.media.webpage

                    # Get URL
                    if hasattr(webpage, 'url'):
                        full_text = f"{full_text} {webpage.url}"
                        log.info(f"🔗 Webpage URL: {webpage.url}")

                    # Get description (often contains CA)
                    if hasattr(webpage, 'description'):
                        desc = webpage.description or ""
                        full_text = f"{full_text} {desc}"
                        log.info(f"📝 Webpage description: {desc[:100]}")

                    # Get title (might contain CA)
                    if hasattr(webpage, 'title'):
                        title = webpage.title or ""
                        full_text = f"{full_text} {title}"

                    # Get display URL (sometimes has CA)
                    if hasattr(webpage, 'display_url'):
                        display = webpage.display_url or ""
                        full_text = f"{full_text} {display}"

                    # Get site name
                    if hasattr(webpage, 'site_name'):
                        site = webpage.site_name or ""
                        full_text = f"{full_text} {site}"

            # Extract from button URLs
            if hasattr(event.message, 'buttons') and event.message.buttons:
                for button_row in event.message.buttons:
                    for button in button_row:
                        if hasattr(button, 'url') and button.url:
                            full_text = f"{full_text} {button.url}"
                        if hasattr(button, 'text') and button.text:
                            full_text = f"{full_text} {button.text}"

            # Extract from message entities (links, mentions, etc)
            if hasattr(event.message, 'entities') and event.message.entities:
                for entity in event.message.entities:
                    if hasattr(entity, 'url') and entity.url:
                        full_text = f"{full_text} {entity.url}"

            # Get raw message object dump for deep inspection
            if hasattr(event.message, 'to_dict'):
                try:
                    msg_dict = event.message.to_dict()
                    # Convert to string with a custom JSON encoder that handles datetime
                    import datetime as dt
                    
                    def json_serial(obj):
                        """JSON serializer for objects not serializable by default json code"""
                        if isinstance(obj, (dt.datetime, dt.date)):
                            return obj.isoformat()
                        raise TypeError(f"Type {type(obj)} not serializable")
                    
                    msg_str = json.dumps(msg_dict, default=json_serial)
                    # Search for CA pattern in entire message structure
                    ca_in_json = parse_mint_from_text(msg_str)
                    if ca_in_json:
                        full_text = f"{full_text} {ca_in_json}"
                        log.success(f"🔍 Found CA in message JSON: {ca_in_json}")
                except Exception as e:
                    log.warning(f"Could not parse message dict: {e}")
                    
            if not full_text.strip():
                log.warning("📨 Empty message - dumping full message object for debug")
                try:
                    log.info(f"DEBUG: Message raw: {event.message}")
                except:
                    pass
                return

            log.info(f"📨 New message: {message_text[:100] if message_text else '(no text)'}...")
            log.info(f"🔍 Full extracted text: {full_text[:200]}...")

            # Try to find mint address
            mint_address = parse_mint_from_text(full_text)

            if mint_address:
                log.success(f"🎯 Mint detected: {mint_address}")

                if mint_address in self.active_trades:
                    log.warning(f"Already tracking {mint_address}")
                    return

                # IMMEDIATE FAST SNIPE
                await self.attempt_swap_fast(mint_address)
            else:
                log.warning("❌ No valid mint found - Full message dump:")
                log.info(f"Message text: {message_text}")
                log.info(f"Full extracted: {full_text}")

                # Emergency: try to extract from raw message string representation
                raw_str = str(event.message)
                emergency_ca = parse_mint_from_text(raw_str)
                if emergency_ca:
                    log.success(f"🚨 EMERGENCY: Found CA in raw string: {emergency_ca}")
                    await self.attempt_swap_fast(emergency_ca)

        except Exception as e:
            log.error(f"Message handler error: {e}")
            import traceback
            traceback.print_exc()

    async def start(self):
        """Start the bot"""
        log.info("=" * 70)
        log.info("STARTING BOT...")
        log.info("=" * 70)

        # Step 1: Send initial startup notification
        try:
            await TelegramNotifier.send_message(
                "🚀 <b>BOT STARTING...</b>\n\n"
                "⏳ Checking balance and RPC connections..."
            )
            log.success("Initial notification sent")
        except Exception as e:
            log.error(f"Failed to send initial notification: {e}")
            log.warning("Continuing without notifications - check your bot setup!")

        # Step 2: Check balance and RPC
        if not await self.check_balance_and_rpc():
            error_msg = "❌ <b>STARTUP FAILED</b>\n\nCould not connect to any RPC endpoint."
            try:
                await TelegramNotifier.send_message(error_msg)
            except:
                pass
            log.error("Failed to initialize bot. Exiting.")
            return

        # Step 3: Send detailed startup notification
        try:
            await TelegramNotifier.send_startup_notification(
                wallet_pubkey=str(self.wallet.pubkey()),
                balance_sol=self.balance_sol,
                rpc_url=self.working_rpc
            )
            log.success("Detailed startup notification sent")
        except Exception as e:
            log.error(f"Failed to send startup notification: {e}")
            log.warning("Bot will continue but notifications may not work!")

        # Step 4: Connect to Telegram for monitoring
        log.info(f"Connecting to Telegram for channel monitoring...")
        client = TelegramClient(
            'sniper_session',
            Config.TG_API_ID,
            Config.TG_API_HASH
        )

        try:
            await client.start()
            log.success("Telegram client connected")

            # Send ready notification
            try:
                await TelegramNotifier.send_message(
                    "✅ <b>BOT READY!</b>\n\n"
                    f"📡 Monitoring: {Config.TG_CHANNEL}\n"
                    f"⚡ Speed Mode: TOP 1%\n"
                    f"💰 Balance: {self.balance_sol:.4f} SOL\n"
                    f"📊 Tracking: Jupiter polling ({max(Config.TRACKING_POLL_INTERVAL, 3)}s)\n\n"
                    "🎯 Ready to snipe tokens!"
                )
                log.success("Ready notification sent")
            except Exception as e:
                log.error(f"Failed to send ready notification: {e}")

            @client.on(events.NewMessage(chats=Config.TG_CHANNEL))
            async def message_handler(event):
                await self.handle_new_message(event)

            log.success(f"⚡ SPEED MODE: Monitoring {Config.TG_CHANNEL}")
            log.info("Bot is now running. Press Ctrl+C to stop.")

            await client.run_until_disconnected()

        except KeyboardInterrupt:
            log.info("Shutdown signal received...")

            for task in self.profit_tracking_tasks.values():
                task.cancel()

            log.info("Bot stopped gracefully")

        except Exception as e:
            log.error(f"Fatal error: {e}")
            await TelegramNotifier.send_message(f"❌ <b>Bot Crashed</b>\n\n{str(e)}")

        finally:
            await client.disconnect()
            await JupiterAPI.close_session()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    try:
        bot = SolanaSnipeBot()
        await bot.start()
    except Exception as e:
        log.error(f"Startup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("\nBot terminated by user")
    except Exception as e:
        log.error(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
