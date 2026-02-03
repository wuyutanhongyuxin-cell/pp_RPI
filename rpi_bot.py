#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPI Bot - 100% RPI 触发交易机器人

核心原理:
    RPI (Rebate Point Index) 只出现在 TAKER 订单上
    使用市价单进行快速买入/卖出，每笔交易都是 TAKER
    配合 interactive token 实现 0 手续费 + RPI 积分

核心策略:
    1. 只使用市价单 (TAKER) - 不使用限价单 (MAKER)
    2. 固定交易大小 0.003 BTC (分析显示此大小 RPI 概率最高)
    3. 快速开平仓循环，最大化 RPI 获取

基于 pp2 项目改进: https://github.com/wuyutanhongyuxin-cell/pp2
"""

import os
import sys
import json
import time
import asyncio
import logging
import signal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from decimal import Decimal, ROUND_DOWN

from dotenv import load_dotenv

# 全局退出标志
_shutdown_requested = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('RPI-BOT')


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class RPIConfig:
    """RPI 交易配置"""

    # 核心参数: 固定交易大小 (BTC)
    # 分析显示 0.003 BTC 的 RPI 触发率最高
    trade_size: str = "0.003"

    # 交易间隔 (秒)
    # 设置适当间隔避免频繁交易被限速
    trade_interval: float = 5.0

    # 市场
    market: str = "BTC-USD-PERP"

    # 限速设置
    limits_per_second: int = 2
    limits_per_minute: int = 30
    limits_per_hour: int = 300
    limits_per_day: int = 1000

    # Spread 优化配置
    max_spread_pct: float = 0.03  # 最大允许价差百分比

    # 止盈止损配置
    stop_loss_pct: float = 0.015  # 止损百分比 (价格下跌此比例则立即平仓)

    # 止盈配置
    min_profit_pct: float = 0.005  # 最小止盈百分比
    max_wait_seconds: float = 8.0  # 等待止盈最长时间
    check_interval: float = 0.5  # 检查价格间隔

    # 运行状态
    enabled: bool = True


@dataclass
class RateLimitState:
    """限速状态"""
    day: str = ""
    trades: List[int] = field(default_factory=list)


@dataclass
class AccountInfo:
    """账号信息"""
    l2_private_key: str
    l2_address: str
    name: str = ""


# =============================================================================
# Paradex API 客户端 (使用 Interactive Token)
# =============================================================================

class ParadexInteractiveClient:
    """
    Paradex API 客户端
    关键：使用 ?token_usage=interactive 认证获取 0 手续费 token
    """

    def __init__(self, l2_private_key: str, l2_address: str, environment: str = "prod"):
        self.l2_private_key = l2_private_key
        self.l2_address = l2_address
        self.environment = environment

        self.base_url = f"https://api.{'prod' if environment == 'prod' else 'testnet'}.paradex.trade/v1"
        self.jwt_token: Optional[str] = None
        self.jwt_expires_at: int = 0
        self.market_info: Dict[str, Any] = {}

        # 导入 paradex-py SDK
        try:
            from paradex_py import ParadexSubkey
            from paradex_py.environment import PROD, TESTNET

            env = PROD if environment == "prod" else TESTNET
            self.paradex = ParadexSubkey(
                env=env,
                l2_private_key=l2_private_key,
                l2_address=l2_address,
            )
            log.info(f"Paradex SDK 初始化成功 (环境: {environment})")
        except ImportError:
            log.error("请先安装 paradex-py: pip install paradex-py")
            raise

    async def authenticate_interactive(self) -> bool:
        """
        使用 interactive 模式认证
        关键：POST /v1/auth?token_usage=interactive
        这是实现 0 手续费的核心
        """
        try:
            import aiohttp

            auth_headers = self.paradex.account.auth_headers()

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                url = f"{self.base_url}/auth?token_usage=interactive"

                headers = {
                    "Content-Type": "application/json",
                    **auth_headers
                }

                async with session.post(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.jwt_token = data.get("jwt_token")

                        # 解析 token 获取过期时间
                        import base64
                        payload = self.jwt_token.split('.')[1]
                        payload += '=' * (4 - len(payload) % 4)
                        decoded = json.loads(base64.b64decode(payload))

                        self.jwt_expires_at = decoded.get("exp", 0)
                        token_usage = decoded.get("token_usage", "unknown")

                        log.info(f"认证成功! token_usage={token_usage}")

                        if token_usage != "interactive":
                            log.warning("警告: token_usage 不是 interactive，手续费可能不是 0!")

                        return True
                    else:
                        error = await resp.text()
                        log.error(f"认证失败: {resp.status} - {error}")
                        return False

        except Exception as e:
            log.error(f"认证异常: {e}")
            return False

    async def ensure_authenticated(self) -> bool:
        """确保已认证且 token 未过期"""
        now = int(time.time())

        if self.jwt_token and self.jwt_expires_at > now + 60:
            return True

        log.info("Token 已过期或不存在，重新认证...")
        return await self.authenticate_interactive()

    def _get_auth_headers(self) -> Dict[str, str]:
        """获取带认证的请求头"""
        return {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json"
        }

    async def get_balance(self) -> Optional[float]:
        """获取 USDC 余额"""
        try:
            if not await self.ensure_authenticated():
                return None

            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                url = f"{self.base_url}/balance"
                async with session.get(url, headers=self._get_auth_headers()) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data.get("results", []):
                            if item.get("token") == "USDC":
                                return float(item.get("size", 0))
            return 0
        except Exception as e:
            log.error(f"获取余额失败: {e}")
            return None

    async def get_positions(self, market: str = None) -> List[Dict]:
        """获取持仓"""
        try:
            if not await self.ensure_authenticated():
                return []

            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                url = f"{self.base_url}/positions"
                async with session.get(url, headers=self._get_auth_headers()) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        positions = data.get("results", [])

                        if market:
                            positions = [p for p in positions if p.get("market") == market]

                        return [p for p in positions if p.get("status") != "CLOSED" and float(p.get("size", 0)) > 0]
            return []
        except Exception as e:
            log.error(f"获取持仓失败: {e}")
            return []

    async def get_market_info(self, market: str) -> Optional[Dict]:
        """获取市场信息"""
        if market in self.market_info:
            return self.market_info[market]

        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                url = f"{self.base_url}/markets"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for m in data.get("results", []):
                            self.market_info[m.get("symbol")] = m
                        return self.market_info.get(market)
            return None
        except Exception as e:
            log.error(f"获取市场信息失败: {e}")
            return None

    async def get_bbo(self, market: str) -> Optional[Dict]:
        """获取最优买卖价 (Best Bid/Offer)"""
        try:
            if not await self.ensure_authenticated():
                return None

            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                url = f"{self.base_url}/orderbook/{market}?depth=1"

                async with session.get(url, headers=self._get_auth_headers()) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        bids = data.get("bids", [])
                        asks = data.get("asks", [])

                        best_bid = data.get("best_bid_api") or (bids[0] if bids else None)
                        best_ask = data.get("best_ask_api") or (asks[0] if asks else None)

                        if best_bid and best_ask:
                            return {
                                "bid": float(best_bid[0]),
                                "ask": float(best_ask[0]),
                                "bid_size": float(best_bid[1]),
                                "ask_size": float(best_ask[1]),
                            }
            return None
        except Exception as e:
            log.error(f"获取 BBO 失败: {e}")
            return None

    async def place_market_order(
        self,
        market: str,
        side: str,
        size: str,
        reduce_only: bool = False
    ) -> Optional[Dict]:
        """
        下市价单 (TAKER 订单 - 这是获取 RPI 的关键!)
        """
        try:
            if not await self.ensure_authenticated():
                return None

            from paradex_py.common.order import Order, OrderSide, OrderType
            from decimal import Decimal

            order_side = OrderSide.Buy if side.upper() == "BUY" else OrderSide.Sell

            order = Order(
                market=market,
                order_type=OrderType.Market,
                order_side=order_side,
                size=Decimal(size),
                client_id=f"rpi_{int(time.time()*1000)}",
                reduce_only=reduce_only,
                signature_timestamp=int(time.time() * 1000),
            )

            order.signature = self.paradex.account.sign_order(order)

            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                url = f"{self.base_url}/orders"
                payload = order.dump_to_dict()

                async with session.post(url, headers=self._get_auth_headers(), json=payload) as resp:
                    if resp.status == 201:
                        result = await resp.json()
                        log.info(f"市价单成功: {side} {size} BTC, order_id={result.get('id')}")
                        return result
                    else:
                        error = await resp.text()
                        log.error(f"市价单失败: {resp.status} - {error}")
                        return None

        except Exception as e:
            log.error(f"市价单失败: {e}")
            return None

    async def cancel_all_orders(self, market: str = None) -> int:
        """取消所有挂单"""
        try:
            if not await self.ensure_authenticated():
                return 0

            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                url = f"{self.base_url}/orders"
                params = {"status": "OPEN"}
                if market:
                    params["market"] = market

                async with session.get(url, headers=self._get_auth_headers(), params=params) as resp:
                    if resp.status != 200:
                        return 0
                    data = await resp.json()
                    orders = data.get("results", [])

                if not orders:
                    return 0

                cancelled = 0
                for order in orders:
                    order_id = order.get("id")
                    if order_id:
                        cancel_url = f"{self.base_url}/orders/{order_id}"
                        async with session.delete(cancel_url, headers=self._get_auth_headers()) as cancel_resp:
                            if cancel_resp.status in [200, 204]:
                                cancelled += 1

                log.info(f"已取消 {cancelled}/{len(orders)} 个挂单")
                return cancelled

        except Exception as e:
            log.error(f"取消订单失败: {e}")
            return 0

    async def close_all_positions(self, market: str = None) -> int:
        """平掉所有仓位"""
        try:
            if not await self.ensure_authenticated():
                return 0

            positions = await self.get_positions()
            if not positions:
                return 0

            closed = 0
            for pos in positions:
                pos_market = pos.get("market")
                if market and pos_market != market:
                    continue

                size = pos.get("size", "0")
                side = pos.get("side", "")

                if float(size) == 0:
                    continue

                close_side = "SELL" if side == "LONG" else "BUY"

                result = await self.place_market_order(
                    market=pos_market,
                    side=close_side,
                    size=size,
                    reduce_only=True
                )

                if result:
                    closed += 1
                    log.info(f"已平仓 {pos_market}: {close_side} {size}")

            return closed

        except Exception as e:
            log.error(f"平仓失败: {e}")
            return 0


# =============================================================================
# 多账号管理器
# =============================================================================

class AccountManager:
    """多账号管理器 - 当一个账号达到限制时自动切换"""

    def __init__(self, accounts: List[AccountInfo], environment: str = "prod"):
        if not accounts:
            raise ValueError("至少需要配置一个账号")

        self.accounts = accounts
        self.environment = environment
        self.current_index = 0
        self.clients: Dict[int, ParadexInteractiveClient] = {}
        self.rate_states: Dict[int, RateLimitState] = {}
        self.daily_limits = 1000

        for i in range(len(accounts)):
            self.rate_states[i] = RateLimitState()

        log.info(f"账号管理器: 共 {len(accounts)} 个账号")

    def get_current_client(self) -> Optional[ParadexInteractiveClient]:
        """获取当前活跃的客户端"""
        if self.current_index >= len(self.accounts):
            return None

        if self.current_index not in self.clients:
            account = self.accounts[self.current_index]
            try:
                client = ParadexInteractiveClient(
                    l2_private_key=account.l2_private_key,
                    l2_address=account.l2_address,
                    environment=self.environment
                )
                self.clients[self.current_index] = client
                log.info(f"已加载账号 #{self.current_index + 1}: {account.name or account.l2_address[:10]}...")
            except Exception as e:
                log.error(f"加载账号 #{self.current_index + 1} 失败: {e}")
                return None

        return self.clients[self.current_index]

    def get_current_rate_state(self) -> RateLimitState:
        """获取当前账号的限速状态"""
        return self.rate_states[self.current_index]

    def get_current_account_name(self) -> str:
        """获取当前账号名称"""
        if self.current_index >= len(self.accounts):
            return "无可用账号"
        account = self.accounts[self.current_index]
        return account.name or f"账号#{self.current_index + 1}"

    def _count_hour_trades(self, account_index: int) -> int:
        """统计某账号过去1小时的交易数"""
        state = self.rate_states[account_index]
        cutoff = int(time.time() * 1000) - 3600000
        return len([t for t in state.trades if t > cutoff])

    def is_account_hour_limited(self, account_index: int) -> bool:
        """检查某账号是否达到小时限制"""
        return self._count_hour_trades(account_index) >= 300

    def switch_to_next_available_account(self) -> str:
        """切换到下一个可用账号"""
        today = datetime.now().strftime("%Y-%m-%d")

        for _ in range(len(self.accounts)):
            self.current_index = (self.current_index + 1) % len(self.accounts)

            state = self.rate_states[self.current_index]
            if state.day == today and len(state.trades) >= self.daily_limits:
                continue

            if not self.is_account_hour_limited(self.current_index):
                log.info(f"切换到 {self.get_current_account_name()}")
                return "switched"

        has_day_available = any(
            self.rate_states[i].day != today or len(self.rate_states[i].trades) < self.daily_limits
            for i in range(len(self.accounts))
        )

        if has_day_available:
            return "all_hour_limited"

        return "all_day_limited"

    def all_accounts_exhausted(self) -> bool:
        """检查是否所有账号都已用完今日额度"""
        today = datetime.now().strftime("%Y-%m-%d")
        return all(
            self.rate_states[i].day == today and len(self.rate_states[i].trades) >= self.daily_limits
            for i in range(len(self.accounts))
        )

    def save_state(self, filepath: str = "rpi_account_states.json"):
        """保存状态"""
        data = {
            "current_index": self.current_index,
            "rate_states": {
                str(i): {"day": state.day, "trades": state.trades[-1000:]}
                for i, state in self.rate_states.items()
            }
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            log.error(f"保存状态失败: {e}")

    def load_state(self, filepath: str = "rpi_account_states.json"):
        """加载状态"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.current_index = data.get("current_index", 0)
                    for i_str, state_data in data.get("rate_states", {}).items():
                        i = int(i_str)
                        if i < len(self.accounts):
                            self.rate_states[i] = RateLimitState(
                                day=state_data.get("day", ""),
                                trades=state_data.get("trades", [])
                            )
                log.info(f"已加载状态，当前账号: {self.get_current_account_name()}")
        except Exception as e:
            log.warning(f"加载状态失败: {e}")


# =============================================================================
# RPI 交易机器人
# =============================================================================

class RPIBot:
    """
    RPI 机器人 - 专注于获取 100% RPI

    核心逻辑:
    1. 使用市价单开仓 (TAKER -> 获得 RPI)
    2. 立即使用市价单平仓 (TAKER -> 再次获得 RPI)
    3. 循环执行，每次交易都是 TAKER，每次都获得 RPI
    """

    def __init__(
        self,
        client: ParadexInteractiveClient,
        config: RPIConfig,
        account_manager: Optional[AccountManager] = None
    ):
        self.client = client
        self.config = config
        self.account_manager = account_manager
        self.rate_state = RateLimitState()

        # 统计
        self.total_trades = 0
        self.rpi_trades = 0
        self.start_time = None

        if self.account_manager:
            self.rate_state = self.account_manager.get_current_rate_state()

    def _day_key(self) -> str:
        """获取当天日期键"""
        return datetime.now().strftime("%Y-%m-%d")

    def _prune_trades(self):
        """清理过期的交易记录"""
        cutoff = int(time.time() * 1000) - 86400000
        self.rate_state.trades = [t for t in self.rate_state.trades if t > cutoff]

    def _count_trades_in_window(self, window_ms: int) -> int:
        """统计时间窗口内的交易数"""
        cutoff = int(time.time() * 1000) - window_ms
        return len([t for t in self.rate_state.trades if t > cutoff])

    def _can_trade(self) -> tuple[bool, Optional[str], Dict]:
        """检查是否可以交易"""
        if self.account_manager:
            self.rate_state = self.account_manager.get_current_rate_state()

        if self.rate_state.day != self._day_key():
            self.rate_state.day = self._day_key()
            self.rate_state.trades = []

        self._prune_trades()

        usage = {
            "sec": self._count_trades_in_window(1000),
            "min": self._count_trades_in_window(60000),
            "hour": self._count_trades_in_window(3600000),
            "day": len(self.rate_state.trades),
        }

        if self.account_manager:
            usage["account"] = self.account_manager.get_current_account_name()

            if usage["day"] >= self.config.limits_per_day:
                if self.account_manager.all_accounts_exhausted():
                    return False, "all_accounts_exhausted", usage
                return False, "day_limit_switch", usage

            if usage["hour"] >= self.config.limits_per_hour:
                return False, "hour_switch", usage

        else:
            if usage["day"] >= self.config.limits_per_day:
                return False, "day", usage
            if usage["hour"] >= self.config.limits_per_hour:
                return False, "hour", usage

        if usage["min"] >= self.config.limits_per_minute:
            return False, "min", usage
        if usage["sec"] >= self.config.limits_per_second:
            return False, "sec", usage

        return True, None, usage

    def _record_trade(self):
        """记录一次交易"""
        self.rate_state.trades.append(int(time.time() * 1000))
        self.total_trades += 1

        if self.account_manager:
            self.account_manager.save_state()

    async def run_rpi_cycle(self) -> tuple[bool, str]:
        """
        执行一个 RPI 交易周期 (优化版):
        1. 检查 Spread 是否在允许范围内
        2. 市价买入 (TAKER -> RPI)
        3. 等待价格上涨或超时
        4. 市价卖出 (TAKER -> RPI)
        """
        market = self.config.market
        size = self.config.trade_size

        # 1. 检查限速
        can_trade, reason, usage = self._can_trade()
        if not can_trade:
            return False, f"限速: {reason}"

        # 2. 检查余额
        log.info("[检查] 获取账户余额...")
        balance = await self.client.get_balance()
        if not balance:
            return False, "无法获取余额"

        log.info(f"[检查] 余额: {balance:.2f} USDC")

        # 获取当前价格估算所需金额
        log.info("[检查] 获取市场价格...")
        bbo = await self.client.get_bbo(market)
        if not bbo:
            return False, "无法获取市场价格"

        # 3. [优化] 检查 Spread 是否在允许范围内
        bid = bbo["bid"]
        ask = bbo["ask"]
        spread_pct = (ask - bid) / bid * 100
        log.info(f"[检查] Spread: {spread_pct:.4f}% (限制: {self.config.max_spread_pct}%)")

        if spread_pct > self.config.max_spread_pct:
            return False, f"Spread 过大: {spread_pct:.4f}% > {self.config.max_spread_pct}%"

        # 考虑杠杆 (Paradex 默认最高 50x)，只需要 2% 保证金 + 10% 缓冲
        leverage = 50
        required = float(size) * bbo["ask"] / leverage * 1.5  # 保证金 + 50% 安全缓冲
        if balance < required:
            return False, f"余额不足: {balance:.2f} < {required:.2f} USD"

        # 4. 市价买入 (TAKER -> 获得 RPI)
        log.info(f"[开仓] 市价买入 {size} BTC @ ~${ask:.1f}...")
        buy_result = await self.client.place_market_order(
            market=market,
            side="BUY",
            size=size,
            reduce_only=False
        )

        if not buy_result:
            return False, "买入失败"

        self._record_trade()
        entry_price = ask  # 记录入场价格

        # 检查是否获得 RPI
        buy_flags = buy_result.get("flags", [])
        if "rpi" in [f.lower() for f in buy_flags]:
            self.rpi_trades += 1
            log.info(f"  -> RPI 获得! flags={buy_flags}")
        else:
            log.info(f"  -> flags={buy_flags}")

        # 5. [优化] 等待价格上涨或止损 (或立即平仓)
        best_bid = bid
        exit_reason = "instant"

        # 如果 MAX_WAIT_SECONDS <= 0，立即平仓模式
        if self.config.max_wait_seconds <= 0:
            log.info(f"[立即平仓] 极速模式，不等待")
        else:
            target_price = entry_price * (1 + self.config.min_profit_pct / 100)
            stop_price = entry_price * (1 - self.config.stop_loss_pct / 100)
            log.info(f"[等待] 止盈: ${target_price:.1f} (+{self.config.min_profit_pct}%) | 止损: ${stop_price:.1f} (-{self.config.stop_loss_pct}%)")

            wait_start = time.time()
            exit_reason = "timeout"

            while time.time() - wait_start < self.config.max_wait_seconds:
                # 检查是否收到退出信号
                if _shutdown_requested:
                    log.info("[中断] 收到退出信号，立即平仓")
                    exit_reason = "shutdown"
                    break

                # 获取最新价格
                new_bbo = await self.client.get_bbo(market)
                if new_bbo:
                    best_bid = new_bbo["bid"]

                    # 检查止盈
                    if best_bid >= target_price:
                        waited = time.time() - wait_start
                        log.info(f"[止盈] Bid: ${best_bid:.1f} >= 目标: ${target_price:.1f} (等待 {waited:.1f}s)")
                        exit_reason = "take_profit"
                        break

                    # 检查止损
                    if best_bid <= stop_price:
                        waited = time.time() - wait_start
                        log.info(f"[止损] Bid: ${best_bid:.1f} <= 止损: ${stop_price:.1f} (等待 {waited:.1f}s)")
                        exit_reason = "stop_loss"
                        break

                await asyncio.sleep(self.config.check_interval)
            else:
                waited = time.time() - wait_start
                log.info(f"[超时] 等待 {waited:.1f}s，Bid: ${best_bid:.1f}，执行平仓")

        # 6. 市价卖出 (TAKER -> 再次获得 RPI)
        log.info(f"[平仓] 市价卖出 {size} BTC @ ~${best_bid:.1f}...")
        sell_result = await self.client.place_market_order(
            market=market,
            side="SELL",
            size=size,
            reduce_only=True
        )

        if not sell_result:
            # 尝试获取实际仓位再平
            positions = await self.client.get_positions(market)
            if positions:
                pos = positions[0]
                actual_size = pos.get("size", size)
                log.info(f"  重试平仓，实际仓位: {actual_size}")
                sell_result = await self.client.place_market_order(
                    market=market,
                    side="SELL",
                    size=str(actual_size),
                    reduce_only=True
                )

        if sell_result:
            self._record_trade()

            # 检查是否获得 RPI
            sell_flags = sell_result.get("flags", [])
            if "rpi" in [f.lower() for f in sell_flags]:
                self.rpi_trades += 1
                log.info(f"  -> RPI 获得! flags={sell_flags}")
            else:
                log.info(f"  -> flags={sell_flags}")

            # 计算本次收益
            pnl = (best_bid - entry_price) * float(size)
            log.info(f"  -> 预估 PnL: ${pnl:.4f}")
        else:
            log.warning("  -> 平仓失败，稍后重试")

        # 统计
        rpi_rate = (self.rpi_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        log.info(f"[统计] 总交易: {self.total_trades}, RPI: {self.rpi_trades} ({rpi_rate:.1f}%)")

        return True, "RPI 周期完成"


    async def _cleanup_on_exit(self):
        """退出时清理"""
        log.info("=" * 50)
        log.info("执行退出清理...")

        if self.account_manager:
            for idx, client in self.account_manager.clients.items():
                account_name = self.account_manager.accounts[idx].name or f"账号#{idx + 1}"
                log.info(f"[{account_name}] 清理中...")

                try:
                    if await client.ensure_authenticated():
                        await client.cancel_all_orders(self.config.market)
                        await client.close_all_positions(self.config.market)
                except Exception as e:
                    log.error(f"[{account_name}] 清理异常: {e}")
        else:
            try:
                await self.client.cancel_all_orders(self.config.market)
                await self.client.close_all_positions(self.config.market)
            except Exception as e:
                log.error(f"清理异常: {e}")

        log.info("清理完成")
        log.info("=" * 50)

    async def _switch_account(self) -> str:
        """切换账号"""
        if not self.account_manager:
            return "switched"

        current = self.account_manager.get_current_account_name()
        log.info(f"[{current}] 切换账号...")

        # 清理当前账号
        await self.client.cancel_all_orders(self.config.market)
        await self.client.close_all_positions(self.config.market)
        self.account_manager.save_state()

        # 切换
        result = self.account_manager.switch_to_next_available_account()

        if result == "switched":
            new_client = self.account_manager.get_current_client()
            if new_client:
                self.client = new_client
                self.rate_state = self.account_manager.get_current_rate_state()

                if await self.client.authenticate_interactive():
                    log.info(f"已切换到 {self.account_manager.get_current_account_name()}")
                else:
                    log.error("新账号认证失败!")

        return result

    async def run(self):
        """主运行循环"""
        global _shutdown_requested

        self.start_time = time.time()

        log.info("=" * 60)
        log.info("RPI Bot - 100% RPI 触发交易机器人")
        log.info("=" * 60)
        log.info(f"市场: {self.config.market}")
        log.info(f"交易大小: {self.config.trade_size} BTC")
        log.info(f"交易间隔: {self.config.trade_interval} 秒")
        log.info("")
        log.info("核心原理:")
        log.info("  - 只使用市价单 (TAKER) = 100% RPI")
        log.info("  - Interactive Token = 0% 手续费")
        log.info("  - 快速开平仓循环")
        log.info("=" * 60)

        if self.account_manager:
            log.info(f"多账号模式: {len(self.account_manager.accounts)} 个账号")
            log.info(f"当前: {self.account_manager.get_current_account_name()}")
        else:
            log.info("单账号模式")

        log.info("=" * 60)

        # 初始认证
        if not await self.client.authenticate_interactive():
            log.error("初始认证失败!")
            return

        log.info("认证成功，开始 RPI 交易...")
        log.info("")

        while not _shutdown_requested:
            try:
                log.info(f"[周期] 开始 RPI 交易周期...")
                success, msg = await self.run_rpi_cycle()

                if "all_accounts_exhausted" in msg:
                    log.warning("所有账号今日额度已用完，等待明天...")
                    await self._cleanup_on_exit()
                    await self._wait_until_tomorrow()
                    continue

                if "switch" in msg and self.account_manager:
                    result = await self._switch_account()
                    if result == "all_hour_limited":
                        log.info("所有账号小时限制已满，等待 10 分钟...")
                        await asyncio.sleep(600)
                    elif result == "all_day_limited":
                        await self._wait_until_tomorrow()
                    continue

                if not success:
                    log.info(f"[周期] 失败: {msg}")

                if success:
                    # 成功后等待配置的间隔
                    await asyncio.sleep(self.config.trade_interval)
                else:
                    # 失败时短暂等待
                    await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"异常: {e}")
                await asyncio.sleep(1)

        # 退出清理
        if _shutdown_requested:
            await self._cleanup_on_exit()

        # 最终统计
        elapsed = time.time() - self.start_time
        log.info("")
        log.info("=" * 60)
        log.info("最终统计")
        log.info("=" * 60)
        log.info(f"运行时间: {elapsed/60:.1f} 分钟")
        log.info(f"总交易数: {self.total_trades}")
        log.info(f"RPI 交易数: {self.rpi_trades}")
        if self.total_trades > 0:
            log.info(f"RPI 比例: {self.rpi_trades/self.total_trades*100:.1f}%")
        log.info("=" * 60)

        if self.account_manager:
            self.account_manager.save_state()

    async def _wait_until_tomorrow(self):
        """等待到明天凌晨"""
        now = datetime.now()
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        wait_seconds = (tomorrow - now).total_seconds()
        log.info(f"等待 {wait_seconds/3600:.1f} 小时后重新开始...")
        await asyncio.sleep(wait_seconds + 60)


# =============================================================================
# 主入口
# =============================================================================

def parse_accounts(accounts_str: str) -> List[AccountInfo]:
    """解析多账号配置"""
    accounts = []
    if not accounts_str:
        return accounts

    pairs = accounts_str.strip().split(";")
    for i, pair in enumerate(pairs):
        pair = pair.strip()
        if not pair:
            continue

        parts = pair.split(",")
        if len(parts) != 2:
            continue

        private_key = parts[0].strip()
        address = parts[1].strip()

        if not private_key.startswith("0x") or not address.startswith("0x"):
            continue

        accounts.append(AccountInfo(
            l2_private_key=private_key,
            l2_address=address,
            name=f"账号#{i+1}"
        ))

    return accounts


async def main():
    load_dotenv()

    environment = os.getenv("PARADEX_ENVIRONMENT", "prod")
    market = os.getenv("MARKET", "BTC-USD-PERP")

    # 多账号配置
    accounts_str = os.getenv("PARADEX_ACCOUNTS", "")
    accounts = parse_accounts(accounts_str)

    account_manager = None
    client = None

    if accounts:
        log.info(f"检测到多账号配置: {len(accounts)} 个账号")
        account_manager = AccountManager(accounts, environment)
        account_manager.load_state()

        if account_manager.is_account_hour_limited(account_manager.current_index):
            log.info("当前账号已达小时限制，尝试切换...")
            result = account_manager.switch_to_next_available_account()
            if result == "all_day_limited":
                log.error("所有账号今日额度已用完!")
                sys.exit(1)

        client = account_manager.get_current_client()
        if not client:
            log.error("无法初始化账号!")
            sys.exit(1)
    else:
        l2_private_key = os.getenv("PARADEX_L2_PRIVATE_KEY")
        l2_address = os.getenv("PARADEX_L2_ADDRESS")

        if not l2_private_key or not l2_address:
            log.error("请在 .env 文件中配置账号:")
            log.error("  多账号: PARADEX_ACCOUNTS=私钥1,地址1;私钥2,地址2")
            log.error("  单账号: PARADEX_L2_PRIVATE_KEY 和 PARADEX_L2_ADDRESS")
            sys.exit(1)

        client = ParadexInteractiveClient(
            l2_private_key=l2_private_key,
            l2_address=l2_address,
            environment=environment
        )

    # 创建配置
    config = RPIConfig(market=market)

    # 读取交易大小配置
    trade_size = os.getenv("TRADE_SIZE", "0.003").strip()
    config.trade_size = trade_size
    log.info(f"交易大小: {trade_size} BTC")

    # 读取交易间隔配置
    trade_interval = os.getenv("TRADE_INTERVAL", "2.0").strip()
    config.trade_interval = float(trade_interval)

    # 读取优化参数
    config.max_spread_pct = float(os.getenv("MAX_SPREAD_PCT", "0.03"))
    config.min_profit_pct = float(os.getenv("MIN_PROFIT_PCT", "0.005"))
    config.max_wait_seconds = float(os.getenv("MAX_WAIT_SECONDS", "8.0"))
    config.check_interval = float(os.getenv("CHECK_INTERVAL", "0.5"))
    config.stop_loss_pct = float(os.getenv("STOP_LOSS_PCT", "0.015"))
    
    log.info(f"优化参数: spread<{config.max_spread_pct}%, 止盈>{config.min_profit_pct}%, 止损>{config.stop_loss_pct}%, 等待<{config.max_wait_seconds}s")

    # 创建并运行机器人
    bot = RPIBot(client, config, account_manager)

    # 设置信号处理器
    def signal_handler(sig, frame):
        global _shutdown_requested
        log.info("\n收到 Ctrl+C，准备退出...")
        _shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)

    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("程序已退出")
