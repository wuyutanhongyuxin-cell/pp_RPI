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
from collections import deque
import numpy as np

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

    # 趋势过滤
    trend_filter_enabled: bool = True  # 是否启用趋势过滤
    orderbook_imbalance_threshold: float = 0.0  # 订单簿不平衡阈值 (>0 要求买压大于卖压)

    # 波动率过滤
    volatility_filter_enabled: bool = True  # 是否启用波动率过滤
    max_volatility_pct: float = 0.1  # 最大允许波动率 (过去N秒价格变化%)
    volatility_window: int = 10  # 波动率检测窗口 (秒)

    # 入场模式
    entry_mode: str = "trend"  # trend=趋势跟随, mean_reversion=均值回归, hybrid=混合

    # 动态止损
    dynamic_stop_loss: bool = True  # 是否使用动态止损 (基于spread)
    stop_loss_spread_multiplier: float = 2.0  # 止损 = spread * 此倍数

    # ===== 研究优化参数 =====

    # 平仓模式: market=市价单(TAKER), limit=限价单(MAKER)
    exit_order_type: str = "limit"  # 使用限价单平仓可赚取点差

    # 方向信号阈值 (订单簿不平衡度)
    direction_signal_threshold: float = 0.3  # |imbalance| > 0.3 才入场

    # 风险收益比
    risk_reward_ratio: float = 2.0  # 止盈 = 止损 * 此值

    # 交易时段过滤 (UTC时间)
    time_filter_enabled: bool = True  # 是否启用时段过滤
    optimal_hours_start: int = 8  # 最佳时段开始 (UTC)
    optimal_hours_end: int = 21  # 最佳时段结束 (UTC)
    weekend_multiplier: float = 0.5  # 周末仓位倍数

    # Kelly准则仓位管理
    kelly_enabled: bool = True  # 是否启用Kelly仓位管理
    kelly_fraction: float = 0.25  # 使用1/4 Kelly降低风险
    max_position_pct: float = 0.20  # 最大仓位占比

    # 回撤控制
    drawdown_control_enabled: bool = True  # 是否启用回撤控制
    max_daily_loss_pct: float = 0.03  # 日内最大亏损 3%
    max_total_loss_pct: float = 0.10  # 总体最大亏损 10%

    # 尾随止盈 (Trailing Stop)
    trailing_stop_enabled: bool = True  # 是否启用尾随止盈
    trailing_trigger_rr: float = 1.0  # 触发尾随的R/R倍数 (1.0 = 达到1:1时开始尾随)
    trailing_distance_pct: float = 0.5  # 尾随距离 (占止损的百分比, 0.5 = 0.5R)

    # ===== 量化信号增强 (v2.2) =====
    quant_signals_enabled: bool = True  # 是否启用量化信号融合
    rsi_enabled: bool = True  # 是否使用RSI指标
    rsi_period: int = 14  # RSI周期
    rsi_oversold: float = 30.0  # RSI超卖阈值
    rsi_overbought: float = 70.0  # RSI超买阈值
    vwap_enabled: bool = True  # 是否使用VWAP
    order_flow_enabled: bool = True  # 是否使用订单流分析
    signal_consistency_weight: float = 0.8  # 信号一致性权重
    min_signal_strength: float = 0.5  # 最小信号强度阈值

    # 自适应止损
    adaptive_stop_loss: bool = True  # 是否使用自适应止损
    adaptive_stop_base: float = 0.02  # 基础止损百分比
    adaptive_stop_volatility_mult: float = 1.5  # 波动率乘数

    # 连续亏损控制
    max_consecutive_losses: int = 5  # 最大连续亏损次数后暂停

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
# 量化信号模块 (v2.2)
# =============================================================================

class QuantSignalState:
    """量化信号状态 - 存储历史数据用于计算RSI、VWAP等指标"""

    def __init__(self, rsi_period: int = 14, max_history: int = 100):
        self.rsi_period = rsi_period
        # 价格历史
        self.price_history = deque(maxlen=max_history)
        self.spread_history = deque(maxlen=max_history)
        self.imbalance_history = deque(maxlen=max_history)
        # RSI计算
        self.rsi_gains = deque(maxlen=rsi_period)
        self.rsi_losses = deque(maxlen=rsi_period)
        self.last_price = None
        # VWAP计算
        self.vwap_volume = 0.0
        self.vwap_value = 0.0
        self.session_start_time = time.time()
        # 订单流
        self.buy_volume_history = deque(maxlen=50)
        self.sell_volume_history = deque(maxlen=50)
        # 波动率历史
        self.price_changes = deque(maxlen=50)
        # 连续亏损计数
        self.consecutive_losses = 0
        self.session_trades = 0
        self.session_wins = 0

    def update_price(self, price: float, volume: float = 0) -> None:
        """更新价格数据"""
        now = time.time()
        self.price_history.append((now, price))

        # 更新RSI
        if self.last_price is not None:
            change = price - self.last_price
            if change > 0:
                self.rsi_gains.append(change)
                self.rsi_losses.append(0)
            else:
                self.rsi_gains.append(0)
                self.rsi_losses.append(abs(change))
            # 记录价格变化百分比
            self.price_changes.append(abs(change / self.last_price * 100))

        # 更新VWAP
        if volume > 0:
            self.vwap_volume += volume
            self.vwap_value += price * volume

        self.last_price = price

    def update_orderbook(self, imbalance: float, spread_pct: float) -> None:
        """更新订单簿数据"""
        self.imbalance_history.append(imbalance)
        self.spread_history.append(spread_pct)

    def update_trade_flow(self, buy_volume: float, sell_volume: float) -> None:
        """更新交易流数据"""
        self.buy_volume_history.append(buy_volume)
        self.sell_volume_history.append(sell_volume)

    def record_trade_result(self, is_win: bool) -> None:
        """记录交易结果"""
        self.session_trades += 1
        if is_win:
            self.session_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def get_rsi(self) -> float:
        """计算RSI(14)"""
        if len(self.rsi_gains) < self.rsi_period:
            return 50.0  # 数据不足返回中性值

        avg_gain = sum(self.rsi_gains) / self.rsi_period
        avg_loss = sum(self.rsi_losses) / self.rsi_period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_vwap(self) -> Optional[float]:
        """获取VWAP"""
        if self.vwap_volume > 0:
            return self.vwap_value / self.vwap_volume
        return self.last_price

    def get_order_flow_imbalance(self) -> float:
        """获取订单流不平衡度 (-1 到 1)"""
        if len(self.buy_volume_history) < 5:
            return 0.0

        recent_buys = sum(list(self.buy_volume_history)[-10:])
        recent_sells = sum(list(self.sell_volume_history)[-10:])
        total = recent_buys + recent_sells

        if total == 0:
            return 0.0

        return (recent_buys - recent_sells) / total

    def get_price_momentum(self, lookback_seconds: int = 30) -> float:
        """计算价格动量 (过去N秒的价格变化率%)"""
        if len(self.price_history) < 2:
            return 0.0

        now_price = self.price_history[-1][1]
        target_time = time.time() - lookback_seconds

        old_price = now_price
        for ts, price in self.price_history:
            if ts <= target_time:
                old_price = price
                break

        if old_price == 0:
            return 0.0

        return (now_price - old_price) / old_price * 100

    def get_adaptive_stop_loss(self, base_pct: float, volatility_mult: float) -> float:
        """计算自适应止损百分比"""
        if len(self.price_changes) < 10:
            return base_pct

        # 计算近期波动率
        volatility = float(np.std(list(self.price_changes)))
        dynamic_stop = base_pct + volatility * volatility_mult

        # 限制在合理范围
        return min(max(dynamic_stop, 0.01), 0.05)

    def get_spread_zscore(self) -> float:
        """计算spread的Z-score（异常检测）"""
        if len(self.spread_history) < 10:
            return 0.0

        spreads = list(self.spread_history)
        mean = float(np.mean(spreads))
        std = float(np.std(spreads))

        if std == 0:
            return 0.0

        return (spreads[-1] - mean) / std

    def should_pause_trading(self, max_consecutive: int) -> tuple[bool, str]:
        """检查是否应该暂停交易"""
        if self.consecutive_losses >= max_consecutive:
            return True, f"连续亏损 {self.consecutive_losses} 次"
        return False, ""

    def get_session_stats(self) -> dict:
        """获取会话统计"""
        return {
            'trades': self.session_trades,
            'wins': self.session_wins,
            'win_rate': self.session_wins / self.session_trades if self.session_trades > 0 else 0,
            'consecutive_losses': self.consecutive_losses,
            'rsi': self.get_rsi(),
            'vwap': self.get_vwap(),
            'momentum': self.get_price_momentum()
        }

    def reset_session(self) -> None:
        """重置会话数据（新的一天）"""
        self.consecutive_losses = 0
        self.session_trades = 0
        self.session_wins = 0
        self.vwap_volume = 0.0
        self.vwap_value = 0.0
        self.session_start_time = time.time()


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
        self.client_id_format = "rpi"  # 可选: rpi, timestamp, uuid, empty

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

    def _generate_client_id(self) -> str:
        """生成 client_id，支持多种格式"""
        import uuid
        ts = int(time.time() * 1000)
        
        if self.client_id_format == "rpi":
            return f"rpi_{ts}"
        elif self.client_id_format == "timestamp":
            return str(ts)
        elif self.client_id_format == "uuid":
            return str(uuid.uuid4())
        elif self.client_id_format == "empty":
            return ""
        else:
            return f"rpi_{ts}"

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

    async def get_orderbook_imbalance(self, market: str, depth: int = 5) -> Optional[float]:
        """
        获取订单簿不平衡度
        返回值 > 0 表示买压大于卖压（看涨）
        返回值 < 0 表示卖压大于买压（看跌）
        """
        try:
            if not await self.ensure_authenticated():
                return None

            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                url = f"{self.base_url}/orderbook/{market}?depth={depth}"
                async with session.get(url, headers=self._get_auth_headers()) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        bids = data.get("bids", [])
                        asks = data.get("asks", [])

                        # 计算买卖总量
                        total_bid_size = sum(float(b[1]) for b in bids[:depth])
                        total_ask_size = sum(float(a[1]) for a in asks[:depth])

                        if total_bid_size + total_ask_size == 0:
                            return 0

                        # 不平衡度：(买量 - 卖量) / (买量 + 卖量)
                        imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
                        return imbalance
            return None
        except Exception as e:
            log.error(f"获取订单簿不平衡度失败: {e}")
            return None

    async def get_price_volatility(self, market: str, window_seconds: int = 10, samples: int = 5) -> Optional[Dict]:
        """
        获取价格波动率
        返回: {"volatility_pct": 波动率百分比, "trend": 趋势方向, "prices": 价格列表}
        """
        try:
            prices = []
            interval = window_seconds / samples

            for i in range(samples):
                bbo = await self.get_bbo(market)
                if bbo:
                    prices.append(bbo["bid"])
                if i < samples - 1:
                    await asyncio.sleep(interval)

            if len(prices) < 2:
                return None

            # 计算波动率 (最高价 - 最低价) / 平均价
            min_price = min(prices)
            max_price = max(prices)
            avg_price = sum(prices) / len(prices)
            volatility_pct = (max_price - min_price) / avg_price * 100

            # 计算趋势方向
            price_change = prices[-1] - prices[0]
            if price_change > 0:
                trend = "up"
            elif price_change < 0:
                trend = "down"
            else:
                trend = "flat"

            return {
                "volatility_pct": volatility_pct,
                "trend": trend,
                "price_change": price_change,
                "price_change_pct": (price_change / prices[0]) * 100 if prices[0] > 0 else 0,
                "prices": prices,
                "latest_price": prices[-1]
            }
        except Exception as e:
            log.error(f"获取波动率失败: {e}")
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
                client_id=self._generate_client_id(),
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

    async def place_limit_order(
        self,
        market: str,
        side: str,
        size: str,
        price: str,
        post_only: bool = True,
        reduce_only: bool = False,
        time_in_force: str = "GTT",
        ttl_seconds: int = 60
    ) -> Optional[Dict]:
        """
        下限价单 (可选 POST_ONLY 模式)
        POST_ONLY = True 时，订单只能作为 Maker，如果会立即成交则取消
        """
        try:
            if not await self.ensure_authenticated():
                return None

            from paradex_py.common.order import Order, OrderSide, OrderType
            from decimal import Decimal

            order_side = OrderSide.Buy if side.upper() == "BUY" else OrderSide.Sell

            # 构建指令
            instruction = "POST_ONLY" if post_only else "GTC"

            order = Order(
                market=market,
                order_type=OrderType.Limit,
                order_side=order_side,
                size=Decimal(size),
                limit_price=Decimal(price),
                client_id=self._generate_client_id(),
                reduce_only=reduce_only,
                instruction=instruction,
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
                        log.info(f"限价单成功: {side} {size} @ ${price}, order_id={result.get('id')}")
                        return result
                    else:
                        error = await resp.text()
                        log.error(f"限价单失败: {resp.status} - {error}")
                        return None

        except Exception as e:
            log.error(f"限价单失败: {e}")
            return None

    async def wait_order_fill(self, order_id: str, timeout_seconds: float = 5.0) -> Optional[Dict]:
        """
        等待订单成交
        返回: 成交信息 或 None (超时/取消)
        """
        try:
            import aiohttp
            start_time = time.time()

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_seconds + 2)) as session:
                while time.time() - start_time < timeout_seconds:
                    url = f"{self.base_url}/orders/{order_id}"
                    async with session.get(url, headers=self._get_auth_headers()) as resp:
                        if resp.status == 200:
                            order = await resp.json()
                            status = order.get("status", "")

                            if status == "CLOSED":
                                # 完全成交
                                return order
                            elif status in ["CANCELED", "REJECTED"]:
                                # 被取消或拒绝
                                return None
                            # OPEN 或 PARTIAL - 继续等待

                    await asyncio.sleep(0.2)

            # 超时 - 取消订单
            await self.cancel_order(order_id)
            return None

        except Exception as e:
            log.error(f"等待订单成交失败: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """取消单个订单"""
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                url = f"{self.base_url}/orders/{order_id}"
                async with session.delete(url, headers=self._get_auth_headers()) as resp:
                    return resp.status in [200, 204]
        except Exception as e:
            log.error(f"取消订单失败: {e}")
            return False

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

        # 研究优化: 性能追踪
        self.session_start_balance: Optional[float] = None
        self.daily_start_balance: Optional[float] = None
        self.current_day: str = ""
        self.recent_trades: List[Dict] = []  # 最近交易记录用于Kelly计算
        self.total_pnl: float = 0.0

        # v2.2 量化信号状态
        self.quant_state = QuantSignalState(rsi_period=config.rsi_period)

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

    def _record_trade_result(self, pnl: float, win: bool):
        """记录交易结果用于Kelly计算"""
        self.recent_trades.append({
            "time": time.time(),
            "pnl": pnl,
            "win": win
        })
        self.total_pnl += pnl
        # 只保留最近100笔交易
        if len(self.recent_trades) > 100:
            self.recent_trades = self.recent_trades[-100:]

        # v2.2: 更新量化信号状态
        self.quant_state.record_trade_result(win)

    def _check_trading_time(self) -> tuple[bool, float, str]:
        """
        检查当前是否为最佳交易时段
        返回: (是否允许交易, 仓位乘数, 原因)
        """
        if not self.config.time_filter_enabled:
            return True, 1.0, "时段过滤关闭"

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()  # 0=周一, 6=周日

        # 周末检查
        if weekday >= 5:  # 周六或周日
            mult = self.config.weekend_multiplier
            return True, mult, f"周末, 仓位x{mult}"

        # 时段检查
        if self.config.optimal_hours_start <= hour < self.config.optimal_hours_end:
            # 黄金时段 (13:00-17:00 UTC 最佳)
            if 13 <= hour < 17:
                return True, 1.0, "黄金时段"
            return True, 0.8, "良好时段"
        else:
            # 低流动性时段
            return False, 0.3, f"低流动性时段 (UTC {hour}:00)"

    def _calculate_kelly_size(self, balance: float, base_size: float) -> float:
        """
        使用Kelly准则计算最优仓位
        """
        if not self.config.kelly_enabled or len(self.recent_trades) < 10:
            return base_size

        # 统计最近交易
        wins = [t for t in self.recent_trades if t["win"]]
        losses = [t for t in self.recent_trades if not t["win"]]

        if not wins or not losses:
            return base_size

        win_rate = len(wins) / len(self.recent_trades)
        avg_win = sum(t["pnl"] for t in wins) / len(wins)
        avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses))

        if avg_loss == 0:
            return base_size

        # Kelly公式: f* = (bp - q) / b
        # b = avg_win / avg_loss, p = 胜率, q = 1 - p
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly = (b * p - q) / b if b > 0 else 0

        # 使用分数Kelly
        adjusted_kelly = kelly * self.config.kelly_fraction

        # 限制范围
        if adjusted_kelly <= 0:
            log.warning(f"[Kelly] 负期望值! 胜率={win_rate:.1%}, R/R={b:.2f}")
            return base_size * 0.5  # 减半仓位

        # 计算仓位
        max_position = balance * self.config.max_position_pct
        kelly_position = balance * min(adjusted_kelly, self.config.max_position_pct)

        log.info(f"[Kelly] 胜率={win_rate:.1%}, R/R={b:.2f}, Kelly={adjusted_kelly:.2%}")

        # 转换为BTC数量 (假设价格~$100,000)
        # 这里返回一个乘数
        return min(base_size * (1 + adjusted_kelly), base_size * 2)

    def _check_drawdown(self, current_balance: float) -> tuple[bool, str]:
        """
        检查回撤是否超限
        返回: (是否允许交易, 原因)
        """
        if not self.config.drawdown_control_enabled:
            return True, ""

        today = datetime.now().strftime("%Y-%m-%d")

        # 初始化或新的一天
        if self.current_day != today:
            self.current_day = today
            self.daily_start_balance = current_balance

        if self.session_start_balance is None:
            self.session_start_balance = current_balance
            self.daily_start_balance = current_balance

        # 日内回撤检查
        if self.daily_start_balance and self.daily_start_balance > 0:
            daily_loss = (self.daily_start_balance - current_balance) / self.daily_start_balance
            if daily_loss >= self.config.max_daily_loss_pct:
                return False, f"日内回撤 {daily_loss:.1%} >= {self.config.max_daily_loss_pct:.1%}"

        # 总回撤检查
        if self.session_start_balance and self.session_start_balance > 0:
            total_loss = (self.session_start_balance - current_balance) / self.session_start_balance
            if total_loss >= self.config.max_total_loss_pct:
                return False, f"总回撤 {total_loss:.1%} >= {self.config.max_total_loss_pct:.1%}"

        return True, ""

    def _get_direction_signal(self, imbalance: Optional[float], current_price: float = 0) -> tuple[str, float, dict]:
        """
        v2.2 量化信号融合方向判断

        综合以下信号:
        1. 订单簿不平衡 (30%)
        2. 订单流不平衡 (30%)
        3. RSI超买超卖 (20%)
        4. VWAP位置 (20%)

        返回: (方向, 信号强度, 详细信息)
        """
        details = {}

        # 如果禁用量化信号，使用原逻辑
        if not self.config.quant_signals_enabled:
            if imbalance is None:
                return "NEUTRAL", 0.0, {}
            threshold = self.config.direction_signal_threshold
            if imbalance > threshold:
                return "LONG", abs(imbalance), {'imbalance': imbalance}
            elif imbalance < -threshold:
                return "SHORT", abs(imbalance), {'imbalance': imbalance}
            else:
                return "NEUTRAL", abs(imbalance), {'imbalance': imbalance}

        # ========== 量化信号融合 ==========
        direction_score = 0.0

        # 1. 订单簿不平衡 (权重 30%)
        if imbalance is not None:
            direction_score += imbalance * 0.3
            details['imbalance'] = imbalance

        # 2. 订单流不平衡 (权重 30%)
        if self.config.order_flow_enabled:
            ofi = self.quant_state.get_order_flow_imbalance()
            direction_score += ofi * 0.3
            details['order_flow'] = ofi

        # 3. RSI信号 (权重 20%)
        rsi_signal = 0.0
        if self.config.rsi_enabled:
            rsi = self.quant_state.get_rsi()
            details['rsi'] = rsi

            if rsi < self.config.rsi_oversold:
                # 超卖 → 做多信号
                rsi_signal = (self.config.rsi_oversold - rsi) / self.config.rsi_oversold
            elif rsi > self.config.rsi_overbought:
                # 超买 → 做空信号
                rsi_signal = -(rsi - self.config.rsi_overbought) / (100 - self.config.rsi_overbought)

            direction_score += rsi_signal * 0.2
            details['rsi_signal'] = rsi_signal

        # 4. VWAP位置信号 (权重 20%)
        vwap_signal = 0.0
        if self.config.vwap_enabled and current_price > 0:
            vwap = self.quant_state.get_vwap()
            if vwap and vwap > 0:
                vwap_position = (current_price - vwap) / vwap * 100
                details['vwap'] = vwap
                details['vwap_position'] = vwap_position

                momentum = self.quant_state.get_price_momentum(30)
                details['momentum'] = momentum

                # 价格在VWAP下方 + 上涨动量 = 做多
                # 价格在VWAP上方 + 下跌动量 = 做空
                if vwap_position < -0.05 and momentum > 0:
                    vwap_signal = min(abs(vwap_position) * 5, 1.0)
                elif vwap_position > 0.05 and momentum < 0:
                    vwap_signal = -min(abs(vwap_position) * 5, 1.0)

                direction_score += vwap_signal * 0.2
                details['vwap_signal'] = vwap_signal

        # ========== 信号一致性检查 ==========
        signals = [
            imbalance if imbalance else 0,
            self.quant_state.get_order_flow_imbalance() if self.config.order_flow_enabled else 0,
            rsi_signal,
            vwap_signal
        ]

        positive = sum(1 for s in signals if s > 0.1)
        negative = sum(1 for s in signals if s < -0.1)
        consistency = max(positive, negative) / len([s for s in signals if abs(s) > 0.05]) if any(abs(s) > 0.05 for s in signals) else 0

        details['consistency'] = consistency
        details['direction_score'] = direction_score

        # ========== 计算最终信号强度 ==========
        raw_strength = abs(direction_score)

        # 应用一致性权重
        raw_strength *= (consistency * self.config.signal_consistency_weight + (1 - self.config.signal_consistency_weight))

        # Spread异常检测 - 高spread时降低信号强度
        spread_z = self.quant_state.get_spread_zscore()
        if spread_z > 2:
            raw_strength *= 0.5
            details['spread_penalty'] = True

        strength = min(raw_strength, 1.0)
        details['final_strength'] = strength

        # ========== 确定方向 ==========
        threshold = self.config.direction_signal_threshold

        if direction_score > threshold and strength >= self.config.min_signal_strength:
            return "LONG", strength, details
        elif direction_score < -threshold and strength >= self.config.min_signal_strength:
            return "SHORT", strength, details
        else:
            return "NEUTRAL", strength, details

    async def run_rpi_cycle(self) -> tuple[bool, str]:
        """
        执行一个 RPI 交易周期 (研究优化版):
        1. 时段过滤 - 避开低流动性时段
        2. 回撤控制 - 超限暂停交易
        3. 方向信号 - 订单簿不平衡度预测
        4. Kelly仓位 - 动态调整交易大小
        5. 波动率检测 - 高波动时跳过
        6. 限价单平仓 - 赚取点差而非支付
        """
        market = self.config.market
        base_size = self.config.trade_size

        # 1. 检查限速
        can_trade, reason, usage = self._can_trade()
        if not can_trade:
            return False, f"限速: {reason}"

        # 2. [研究优化] 时段过滤
        time_ok, time_mult, time_reason = self._check_trading_time()
        if not time_ok:
            return False, f"时段限制: {time_reason}"
        if time_mult < 1.0:
            log.info(f"[时段] {time_reason}")

        # 3. 检查余额
        log.info("[检查] 获取账户余额...")
        balance = await self.client.get_balance()
        if not balance:
            return False, "无法获取余额"
        log.info(f"[检查] 余额: {balance:.2f} USDC")

        # 4. [研究优化] 回撤控制 - 触发时先平仓再暂停
        drawdown_ok, drawdown_reason = self._check_drawdown(balance)
        if not drawdown_ok:
            log.warning(f"[回撤控制] {drawdown_reason}，平仓后暂停交易...")
            await self.client.cancel_all_orders(market)
            await self.client.close_all_positions(market)
            return False, f"回撤限制: {drawdown_reason}"

        # 4.5 [新增] 检查并清理残留仓位 (支持多空双向)
        positions = await self.client.get_positions(market)
        if positions:
            pos_size = float(positions[0].get("size", 0))
            if pos_size != 0:  # 修复: 做空仓位是负数，需要用 != 0
                pos_type = "多仓" if pos_size > 0 else "空仓"
                log.warning(f"[清理] 检测到残留{pos_type} {abs(pos_size)}，先平仓...")
                await self.client.close_all_positions(market)
                await asyncio.sleep(0.5)  # 等待平仓完成

        # 3. 获取市场价格和Spread
        log.info("[检查] 获取市场价格...")
        bbo = await self.client.get_bbo(market)
        if not bbo:
            return False, "无法获取市场价格"

        bid = bbo["bid"]
        ask = bbo["ask"]
        spread = ask - bid
        spread_pct = spread / bid * 100
        log.info(f"[检查] Spread: ${spread:.2f} ({spread_pct:.4f}%) | 限制: {self.config.max_spread_pct}%")

        if spread_pct > self.config.max_spread_pct:
            return False, f"Spread 过大: {spread_pct:.4f}%"

        # 4. [新增] 波动率检测
        if self.config.volatility_filter_enabled:
            log.info(f"[波动率] 检测中 ({self.config.volatility_window}s)...")
            vol_data = await self.client.get_price_volatility(
                market,
                window_seconds=self.config.volatility_window,
                samples=5
            )
            if vol_data:
                volatility = vol_data["volatility_pct"]
                trend = vol_data["trend"]
                price_change_pct = vol_data["price_change_pct"]

                log.info(f"[波动率] {volatility:.4f}% | 趋势: {trend} | 变化: {price_change_pct:+.4f}%")

                # 波动率过高，跳过 (市场不稳定)
                if volatility > self.config.max_volatility_pct:
                    return False, f"波动率过高: {volatility:.4f}% > {self.config.max_volatility_pct}%"

                # 更新最新价格
                bid = vol_data["latest_price"]
                new_bbo = await self.client.get_bbo(market)
                if new_bbo:
                    ask = new_bbo["ask"]

        # 5. [v2.2] 更新量化信号状态
        mid_price = (bid + ask) / 2
        self.quant_state.update_price(mid_price)
        self.quant_state.update_orderbook(0, spread_pct)  # imbalance稍后更新

        # 5.1 [v2.2] 连续亏损检查
        if self.config.max_consecutive_losses > 0:
            should_pause, pause_reason = self.quant_state.should_pause_trading(
                self.config.max_consecutive_losses
            )
            if should_pause:
                return False, f"[风控] {pause_reason}，暂停交易"

        # 5.2 [研究优化] 订单簿分析 + 量化信号融合
        imbalance = await self.client.get_orderbook_imbalance(market, depth=5)
        if imbalance is not None:
            self.quant_state.update_orderbook(imbalance, spread_pct)

        # v2.2: 使用量化信号融合判断方向
        direction, confidence, signal_details = self._get_direction_signal(imbalance, mid_price)

        if imbalance is not None:
            # 显示量化信号详情
            rsi_info = f"RSI={signal_details.get('rsi', 50):.1f}" if self.config.rsi_enabled else ""
            vwap_info = f"VWAP偏离={signal_details.get('vwap_position', 0):.2f}%" if self.config.vwap_enabled else ""
            ofi_info = f"OFI={signal_details.get('order_flow', 0):.2f}" if self.config.order_flow_enabled else ""

            log.info(f"[量化信号] imb={imbalance:.2f} | {rsi_info} | {vwap_info} | {ofi_info}")
            log.info(f"[方向判断] {direction} | 强度: {confidence:.2f} | 一致性: {signal_details.get('consistency', 0):.2f}")

            # 使用方向信号阈值
            if direction == "NEUTRAL":
                return False, f"无明确方向信号 (强度 {confidence:.2f} < {self.config.min_signal_strength})"

            # 支持双向交易 (LONG做多 / SHORT做空)

        # 6. [简化] 入场模式判断 (支持双向)
        entry_signal = False
        entry_reason = ""
        trade_direction = direction  # LONG 或 SHORT

        if self.config.entry_mode == "trend" or self.config.entry_mode == "hybrid":
            # 趋势跟随：连续上涨做多，连续下跌做空
            if self.config.trend_filter_enabled:
                prices = [bid]
                for i in range(2):
                    await asyncio.sleep(0.2)
                    bbo_check = await self.client.get_bbo(market)
                    if bbo_check:
                        prices.append(bbo_check["bid"])

                if len(prices) >= 3:
                    trend_up = prices[1] >= prices[0] and prices[2] >= prices[1]
                    trend_down = prices[1] <= prices[0] and prices[2] <= prices[1]
                    total_change = prices[2] - prices[0]

                    # LONG: 趋势上涨 + 买压
                    if direction == "LONG" and trend_up and total_change >= 0:
                        entry_signal = True
                        entry_reason = f"[做多] 趋势上涨+买压: +${total_change:.2f}, imb={imbalance:.2f}"
                        bid = prices[2]
                        if bbo_check:
                            ask = bbo_check["ask"]
                    # SHORT: 趋势下跌 + 卖压
                    elif direction == "SHORT" and trend_down and total_change <= 0:
                        entry_signal = True
                        entry_reason = f"[做空] 趋势下跌+卖压: {total_change:.2f}, imb={imbalance:.2f}"
                        bid = prices[2]
                        if bbo_check:
                            ask = bbo_check["ask"]
            else:
                entry_signal = True
                if direction == "LONG":
                    entry_reason = f"[做多] 买压信号: imb={imbalance:.2f}"
                else:
                    entry_reason = f"[做空] 卖压信号: imb={imbalance:.2f}"

        if self.config.entry_mode == "mean_reversion" or (self.config.entry_mode == "hybrid" and not entry_signal):
            # 均值回归：价格下跌+强买压=做多反弹，价格上涨+强卖压=做空回调
            prices = [bid]
            for i in range(2):
                await asyncio.sleep(0.2)
                bbo_check = await self.client.get_bbo(market)
                if bbo_check:
                    prices.append(bbo_check["bid"])

            if len(prices) >= 3:
                price_dropped = prices[2] < prices[0]
                price_rose = prices[2] > prices[0]

                # 做多均值回归: 价格跌 + 强买压
                if price_dropped and imbalance is not None and imbalance > 0.3:
                    entry_signal = True
                    trade_direction = "LONG"
                    entry_reason = f"[做多] 均值回归: 跌${prices[0]-prices[2]:.2f}, 强买压={imbalance:.2f}"
                    bid = prices[2]
                    if bbo_check:
                        ask = bbo_check["ask"]
                # 做空均值回归: 价格涨 + 强卖压
                elif price_rose and imbalance is not None and imbalance < -0.3:
                    entry_signal = True
                    trade_direction = "SHORT"
                    entry_reason = f"[做空] 均值回归: 涨${prices[2]-prices[0]:.2f}, 强卖压={imbalance:.2f}"
                    bid = prices[2]
                    if bbo_check:
                        ask = bbo_check["ask"]

        if not entry_signal:
            return False, f"无入场信号 (模式: {self.config.entry_mode}, 方向: {direction})"

        log.info(f"[入场] {entry_reason}")

        # 7. [研究优化] Kelly仓位计算
        size = base_size
        if self.config.kelly_enabled:
            kelly_size = self._calculate_kelly_size(balance, float(base_size))
            size = str(round(kelly_size, 6))
            if kelly_size != float(base_size):
                log.info(f"[Kelly] 仓位调整: {base_size} -> {size}")

        # 应用时段乘数
        if time_mult < 1.0:
            adjusted_size = float(size) * time_mult
            size = str(round(adjusted_size, 6))
            log.info(f"[时段] 仓位调整: x{time_mult} -> {size}")

        # 8. 检查余额
        leverage = 50
        required = float(size) * ask / leverage * 1.5
        if balance < required:
            return False, f"余额不足: {balance:.2f} < {required:.2f} USD"

        # 9. [优化] 计算止盈止损 (使用风险收益比，并补偿spread)
        # v2.2: 优先使用自适应止损（基于波动率）
        if self.config.adaptive_stop_loss:
            # 自适应止损 = 基础值 + 波动率 * 乘数
            stop_loss_pct = self.quant_state.get_adaptive_stop_loss(
                self.config.adaptive_stop_base,
                self.config.adaptive_stop_volatility_mult
            )
            log.info(f"[自适应止损] {stop_loss_pct:.3f}% (基础{self.config.adaptive_stop_base:.2f}% + 波动率调整)")
        elif self.config.dynamic_stop_loss:
            # 动态止损 = spread * 倍数
            dynamic_stop_pct = spread_pct * self.config.stop_loss_spread_multiplier
            stop_loss_pct = max(0.01, min(dynamic_stop_pct, 0.05))
            log.info(f"[动态止损] {stop_loss_pct:.3f}% (spread {spread_pct:.4f}% x {self.config.stop_loss_spread_multiplier})")
        else:
            stop_loss_pct = self.config.stop_loss_pct

        # 使用风险收益比计算止盈
        # 重要: 需要补偿spread
        # LONG: 在ask买入，检查bid止盈 -> 止盈需要加spread
        # SHORT: 在bid卖出，检查ask止盈 -> 止盈也需要加spread
        raw_take_profit_pct = stop_loss_pct * self.config.risk_reward_ratio
        take_profit_pct = raw_take_profit_pct + spread_pct
        log.info(f"[风险收益] 止盈={take_profit_pct:.3f}% (含spread补偿{spread_pct:.3f}%) : 止损={stop_loss_pct:.3f}% (真实R/R={self.config.risk_reward_ratio})")

        # 9. 开仓 (TAKER -> 获得 RPI)
        # LONG: BUY @ ask, SHORT: SELL @ bid
        if trade_direction == "LONG":
            open_side = "BUY"
            entry_price = ask
            log.info(f"[开仓] 做多 市价买入 {size} BTC @ ~${ask:.1f}...")
        else:  # SHORT
            open_side = "SELL"
            entry_price = bid
            log.info(f"[开仓] 做空 市价卖出 {size} BTC @ ~${bid:.1f}...")

        open_result = await self.client.place_market_order(
            market=market,
            side=open_side,
            size=size,
            reduce_only=False
        )

        if not open_result:
            return False, f"开仓失败 ({trade_direction})"

        self._record_trade()

        # 检查 RPI
        open_flags = open_result.get("flags", [])
        if "rpi" in [f.lower() for f in open_flags]:
            self.rpi_trades += 1
            log.info(f"  -> RPI! flags={open_flags}")
        else:
            log.info(f"  -> flags={open_flags}")

        # 11. 等待出场 (支持尾随止盈，支持双向)
        # LONG: 看bid价格, SHORT: 看ask价格
        best_bid = bid
        best_ask = ask
        check_price = bid if trade_direction == "LONG" else ask
        exit_reason = "instant"

        if self.config.max_wait_seconds <= 0:
            log.info(f"[极速] 立即平仓模式")
        else:
            # 计算止盈止损价格 (方向不同，计算相反)
            if trade_direction == "LONG":
                # LONG: 止盈在上方，止损在下方
                target_price = entry_price * (1 + take_profit_pct / 100)
                stop_price = entry_price * (1 - stop_loss_pct / 100)
                trailing_trigger_price = entry_price * (1 + stop_loss_pct * self.config.trailing_trigger_rr / 100)
            else:
                # SHORT: 止盈在下方，止损在上方
                target_price = entry_price * (1 - take_profit_pct / 100)
                stop_price = entry_price * (1 + stop_loss_pct / 100)
                trailing_trigger_price = entry_price * (1 - stop_loss_pct * self.config.trailing_trigger_rr / 100)

            # 尾随止盈参数
            trailing_active = False
            trailing_stop_price = stop_price
            peak_price = entry_price  # LONG: 最高价, SHORT: 最低价
            trailing_distance = entry_price * (stop_loss_pct * self.config.trailing_distance_pct / 100)

            if trade_direction == "LONG":
                log.info(f"[等待-做多] 止盈: ${target_price:.1f} (+{take_profit_pct:.3f}%) | 止损: ${stop_price:.1f} (-{stop_loss_pct:.3f}%)")
            else:
                log.info(f"[等待-做空] 止盈: ${target_price:.1f} (-{take_profit_pct:.3f}%) | 止损: ${stop_price:.1f} (+{stop_loss_pct:.3f}%)")

            if self.config.trailing_stop_enabled:
                log.info(f"[尾随] 触发价: ${trailing_trigger_price:.1f}")

            wait_start = time.time()
            exit_reason = "timeout"

            while time.time() - wait_start < self.config.max_wait_seconds:
                if _shutdown_requested:
                    exit_reason = "shutdown"
                    break

                new_bbo = await self.client.get_bbo(market)
                if new_bbo:
                    best_bid = new_bbo["bid"]
                    best_ask = new_bbo["ask"]
                    check_price = best_bid if trade_direction == "LONG" else best_ask

                    if trade_direction == "LONG":
                        # LONG: 追踪最高价
                        if check_price > peak_price:
                            peak_price = check_price

                            # 检查是否触发尾随
                            if self.config.trailing_stop_enabled and not trailing_active and check_price >= trailing_trigger_price:
                                trailing_active = True
                                trailing_stop_price = peak_price - trailing_distance
                                log.info(f"[尾随激活] 峰值: ${peak_price:.1f}, 尾随止损: ${trailing_stop_price:.1f}")

                            # 更新尾随止损
                            if trailing_active:
                                new_trailing_stop = peak_price - trailing_distance
                                if new_trailing_stop > trailing_stop_price:
                                    trailing_stop_price = new_trailing_stop
                                    log.info(f"[尾随更新] 峰值: ${peak_price:.1f}, 尾随止损: ${trailing_stop_price:.1f}")

                        # 检查止盈 (价格上涨)
                        if check_price >= target_price:
                            log.info(f"[止盈] ${check_price:.1f} >= ${target_price:.1f}")
                            exit_reason = "take_profit"
                            break

                        # 检查尾随止损
                        if trailing_active and check_price <= trailing_stop_price:
                            log.info(f"[尾随止盈] ${check_price:.1f} <= ${trailing_stop_price:.1f} (峰值: ${peak_price:.1f})")
                            exit_reason = "trailing_stop"
                            break

                        # 检查普通止损 (价格下跌)
                        if check_price <= stop_price:
                            log.info(f"[止损] ${check_price:.1f} <= ${stop_price:.1f}")
                            exit_reason = "stop_loss"
                            break

                    else:  # SHORT
                        # SHORT: 追踪最低价
                        if check_price < peak_price:
                            peak_price = check_price

                            # 检查是否触发尾随 (价格下跌到触发位)
                            if self.config.trailing_stop_enabled and not trailing_active and check_price <= trailing_trigger_price:
                                trailing_active = True
                                trailing_stop_price = peak_price + trailing_distance
                                log.info(f"[尾随激活] 谷值: ${peak_price:.1f}, 尾随止损: ${trailing_stop_price:.1f}")

                            # 更新尾随止损 (跟随价格下跌)
                            if trailing_active:
                                new_trailing_stop = peak_price + trailing_distance
                                if new_trailing_stop < trailing_stop_price:
                                    trailing_stop_price = new_trailing_stop
                                    log.info(f"[尾随更新] 谷值: ${peak_price:.1f}, 尾随止损: ${trailing_stop_price:.1f}")

                        # 检查止盈 (价格下跌)
                        if check_price <= target_price:
                            log.info(f"[止盈] ${check_price:.1f} <= ${target_price:.1f}")
                            exit_reason = "take_profit"
                            break

                        # 检查尾随止损 (价格回升)
                        if trailing_active and check_price >= trailing_stop_price:
                            log.info(f"[尾随止盈] ${check_price:.1f} >= ${trailing_stop_price:.1f} (谷值: ${peak_price:.1f})")
                            exit_reason = "trailing_stop"
                            break

                        # 检查普通止损 (价格上涨)
                        if check_price >= stop_price:
                            log.info(f"[止损] ${check_price:.1f} >= ${stop_price:.1f}")
                            exit_reason = "stop_loss"
                            break

                await asyncio.sleep(self.config.check_interval)
            else:
                log.info(f"[超时] {self.config.max_wait_seconds}s, 检查价: ${check_price:.1f}")

        # 12. 平仓 (强制确保平仓成功，支持双向)
        # LONG: SELL平仓, SHORT: BUY平仓
        close_side = "SELL" if trade_direction == "LONG" else "BUY"
        close_result = None
        # LONG看bid平仓, SHORT看ask平仓
        actual_exit_price = best_bid if trade_direction == "LONG" else best_ask
        max_close_retries = 5  # 最多重试5次

        if self.config.exit_order_type == "limit" and exit_reason != "stop_loss":
            # [研究优化] 使用 POST_ONLY 限价单平仓 (赚取点差)
            close_price = best_bid if trade_direction == "LONG" else best_ask
            log.info(f"[平仓] 限价单{close_side} {size} BTC @ ${close_price:.1f} (POST_ONLY)...")
            limit_result = await self.client.place_limit_order(
                market=market,
                side=close_side,
                size=size,
                price=str(round(close_price, 1)),
                post_only=True,
                reduce_only=True
            )

            if limit_result:
                order_id = limit_result.get("id")
                # 等待成交 (最多3秒，缩短等待时间)
                fill_result = await self.client.wait_order_fill(order_id, timeout_seconds=3.0)
                if fill_result:
                    close_result = fill_result
                    actual_exit_price = float(fill_result.get("avg_fill_price", close_price))
                    log.info(f"  -> 限价单成交 @ ${actual_exit_price:.1f}")
                else:
                    # 限价单未成交，降级为市价单
                    log.info(f"  -> 限价单超时，切换市价单...")

        # 市价单平仓 (作为后备或止损时使用) - 添加重试机制
        for retry in range(max_close_retries):
            if close_result:
                break

            if retry > 0:
                log.warning(f"[平仓重试] 第 {retry + 1}/{max_close_retries} 次尝试...")
                await asyncio.sleep(0.5)  # 重试前等待

            # 获取最新价格
            new_bbo = await self.client.get_bbo(market)
            if new_bbo:
                best_bid = new_bbo["bid"]
                best_ask = new_bbo["ask"]

            close_price = best_bid if trade_direction == "LONG" else best_ask
            log.info(f"[平仓] 市价{close_side} {size} BTC @ ~${close_price:.1f}...")
            close_result = await self.client.place_market_order(
                market=market,
                side=close_side,
                size=size,
                reduce_only=True
            )

            if close_result:
                break

            # 如果市价单失败，检查是否有残留仓位并尝试平掉
            positions = await self.client.get_positions(market)
            if positions:
                actual_size = positions[0].get("size", size)
                log.info(f"[平仓] 检测到残留仓位 {actual_size}，尝试平仓...")
                close_result = await self.client.place_market_order(
                    market=market,
                    side=close_side,
                    size=str(abs(float(actual_size))),
                    reduce_only=True
                )

        # 最终检查：确保没有残留仓位
        if not close_result:
            log.error(f"[严重] 平仓失败 {max_close_retries} 次，执行强制平仓...")
            await self.client.close_all_positions(market)
            # 再次检查
            positions = await self.client.get_positions(market)
            if positions and abs(float(positions[0].get("size", 0))) > 0:
                log.error(f"[严重] 仍有残留仓位: {positions[0].get('size')}")
            else:
                log.info(f"[恢复] 强制平仓成功")
                close_result = {"forced": True}

        if close_result:
            self._record_trade()

            # 检查是否是强制平仓
            if close_result.get("forced"):
                log.warning("  -> 强制平仓完成，PnL 可能有误差")
                # 估算PnL (根据方向不同计算)
                close_price = best_bid if trade_direction == "LONG" else best_ask
                if trade_direction == "LONG":
                    pnl = (close_price - entry_price) * float(size)
                    pnl_pct = (close_price - entry_price) / entry_price * 100
                else:  # SHORT
                    pnl = (entry_price - close_price) * float(size)
                    pnl_pct = (entry_price - close_price) / entry_price * 100
            else:
                close_flags = close_result.get("flags", [])
                if "rpi" in [f.lower() for f in close_flags]:
                    self.rpi_trades += 1
                    log.info(f"  -> RPI! flags={close_flags}")

                # PnL计算 (根据方向不同)
                if trade_direction == "LONG":
                    pnl = (actual_exit_price - entry_price) * float(size)
                    pnl_pct = (actual_exit_price - entry_price) / entry_price * 100
                else:  # SHORT
                    pnl = (entry_price - actual_exit_price) * float(size)
                    pnl_pct = (entry_price - actual_exit_price) / entry_price * 100

            is_win = pnl > 0
            log.info(f"  -> PnL: ${pnl:.4f} ({pnl_pct:+.4f}%) | 出场: {exit_reason}")

            # [研究优化] 记录交易结果用于Kelly计算
            self._record_trade_result(pnl, is_win)
        else:
            log.error("  -> [严重] 平仓完全失败，请检查仓位!")

        rpi_rate = (self.rpi_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        win_rate = sum(1 for t in self.recent_trades if t["win"]) / len(self.recent_trades) * 100 if self.recent_trades else 0
        log.info(f"[统计] 交易: {self.total_trades} | RPI: {self.rpi_trades} ({rpi_rate:.1f}%) | 胜率: {win_rate:.1f}% | 累计PnL: ${self.total_pnl:.4f}")

        return True, f"周期完成 ({exit_reason})"


    async def _cleanup_on_exit(self):
        """退出时清理 - 确保所有仓位都被平掉"""
        log.info("=" * 50)
        log.info("执行退出清理...")
        max_cleanup_retries = 3

        if self.account_manager:
            for idx, client in self.account_manager.clients.items():
                account_name = self.account_manager.accounts[idx].name or f"账号#{idx + 1}"
                log.info(f"[{account_name}] 清理中...")

                try:
                    if await client.ensure_authenticated():
                        await client.cancel_all_orders(self.config.market)

                        # 多次尝试平仓确保成功
                        for retry in range(max_cleanup_retries):
                            await client.close_all_positions(self.config.market)
                            await asyncio.sleep(0.5)

                            positions = await client.get_positions(self.config.market)
                            if not positions or abs(float(positions[0].get("size", 0))) == 0:
                                log.info(f"[{account_name}] 仓位已清空")
                                break
                            else:
                                pos_size = float(positions[0].get("size", 0))
                                pos_type = "多仓" if pos_size > 0 else "空仓"
                                log.warning(f"[{account_name}] 仍有{pos_type} {abs(pos_size)}，重试 {retry + 1}/{max_cleanup_retries}...")
                except Exception as e:
                    log.error(f"[{account_name}] 清理异常: {e}")
        else:
            try:
                await self.client.cancel_all_orders(self.config.market)

                # 多次尝试平仓确保成功
                for retry in range(max_cleanup_retries):
                    await self.client.close_all_positions(self.config.market)
                    await asyncio.sleep(0.5)

                    positions = await self.client.get_positions(self.config.market)
                    if not positions or abs(float(positions[0].get("size", 0))) == 0:
                        log.info("仓位已清空")
                        break
                    else:
                        pos_size = float(positions[0].get("size", 0))
                        pos_type = "多仓" if pos_size > 0 else "空仓"
                        log.warning(f"仍有{pos_type} {abs(pos_size)}，重试 {retry + 1}/{max_cleanup_retries}...")
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

                    # 回撤触发时等待更长时间
                    if "回撤限制" in msg:
                        if "日内" in msg:
                            log.warning("[回撤] 日内回撤超限，暂停交易 30 分钟...")
                            await asyncio.sleep(1800)  # 30分钟
                        else:
                            log.error("[回撤] 总回撤超限，停止交易...")
                            break  # 完全停止

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

    # 设置 client_id 格式
    client_id_format = os.getenv("CLIENT_ID_FORMAT", "rpi")
    client.client_id_format = client_id_format
    log.info(f"Client ID 格式: {client_id_format}")

    # 趋势过滤
    config.trend_filter_enabled = os.getenv("TREND_FILTER", "true").lower() == "true"
    log.info(f"趋势过滤: {'开启' if config.trend_filter_enabled else '关闭'}")

    # 波动率过滤
    config.volatility_filter_enabled = os.getenv("VOLATILITY_FILTER", "true").lower() == "true"
    config.max_volatility_pct = float(os.getenv("MAX_VOLATILITY_PCT", "0.1"))
    config.volatility_window = int(os.getenv("VOLATILITY_WINDOW", "10"))
    log.info(f"波动率过滤: {'开启' if config.volatility_filter_enabled else '关闭'} (最大: {config.max_volatility_pct}%, 窗口: {config.volatility_window}s)")

    # 订单簿阈值
    config.orderbook_imbalance_threshold = float(os.getenv("ORDERBOOK_THRESHOLD", "0.0"))
    log.info(f"订单簿阈值: {config.orderbook_imbalance_threshold}")

    # 入场模式
    config.entry_mode = os.getenv("ENTRY_MODE", "trend")  # trend, mean_reversion, hybrid
    log.info(f"入场模式: {config.entry_mode}")

    # 动态止损
    config.dynamic_stop_loss = os.getenv("DYNAMIC_STOP_LOSS", "true").lower() == "true"
    config.stop_loss_spread_multiplier = float(os.getenv("STOP_LOSS_MULTIPLIER", "2.0"))
    log.info(f"动态止损: {'开启' if config.dynamic_stop_loss else '关闭'} (倍数: {config.stop_loss_spread_multiplier}x)")

    # ===== 研究优化参数 =====
    log.info("")
    log.info("=== 研究优化参数 ===")

    # 平仓模式
    config.exit_order_type = os.getenv("EXIT_ORDER_TYPE", "limit")  # market 或 limit
    log.info(f"平仓模式: {config.exit_order_type} ({'限价单-赚点差' if config.exit_order_type == 'limit' else '市价单-付点差'})")

    # 方向信号阈值
    config.direction_signal_threshold = float(os.getenv("DIRECTION_SIGNAL_THRESHOLD", "0.3"))
    log.info(f"方向信号阈值: {config.direction_signal_threshold}")

    # 风险收益比
    config.risk_reward_ratio = float(os.getenv("RISK_REWARD_RATIO", "2.0"))
    log.info(f"风险收益比: 1:{config.risk_reward_ratio}")

    # 时段过滤
    config.time_filter_enabled = os.getenv("TIME_FILTER", "true").lower() == "true"
    config.optimal_hours_start = int(os.getenv("OPTIMAL_HOURS_START", "8"))
    config.optimal_hours_end = int(os.getenv("OPTIMAL_HOURS_END", "21"))
    config.weekend_multiplier = float(os.getenv("WEEKEND_MULTIPLIER", "0.5"))
    log.info(f"时段过滤: {'开启' if config.time_filter_enabled else '关闭'} (UTC {config.optimal_hours_start}:00-{config.optimal_hours_end}:00, 周末x{config.weekend_multiplier})")

    # Kelly准则
    config.kelly_enabled = os.getenv("KELLY_ENABLED", "true").lower() == "true"
    config.kelly_fraction = float(os.getenv("KELLY_FRACTION", "0.25"))
    config.max_position_pct = float(os.getenv("MAX_POSITION_PCT", "0.20"))
    log.info(f"Kelly准则: {'开启' if config.kelly_enabled else '关闭'} (分数: {config.kelly_fraction}, 最大仓位: {config.max_position_pct:.0%})")

    # 回撤控制
    config.drawdown_control_enabled = os.getenv("DRAWDOWN_CONTROL", "true").lower() == "true"
    config.max_daily_loss_pct = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))
    config.max_total_loss_pct = float(os.getenv("MAX_TOTAL_LOSS_PCT", "0.10"))
    log.info(f"回撤控制: {'开启' if config.drawdown_control_enabled else '关闭'} (日内: {config.max_daily_loss_pct:.0%}, 总体: {config.max_total_loss_pct:.0%})")

    # 尾随止盈
    config.trailing_stop_enabled = os.getenv("TRAILING_STOP", "true").lower() == "true"
    config.trailing_trigger_rr = float(os.getenv("TRAILING_TRIGGER_RR", "1.0"))
    config.trailing_distance_pct = float(os.getenv("TRAILING_DISTANCE_PCT", "0.5"))
    log.info(f"尾随止盈: {'开启' if config.trailing_stop_enabled else '关闭'} (触发: {config.trailing_trigger_rr}R, 距离: {config.trailing_distance_pct}R)")

    # v2.2 量化信号增强
    config.quant_signals_enabled = os.getenv("QUANT_SIGNALS", "true").lower() == "true"
    config.rsi_enabled = os.getenv("RSI_ENABLED", "true").lower() == "true"
    config.rsi_period = int(os.getenv("RSI_PERIOD", "14"))
    config.rsi_oversold = float(os.getenv("RSI_OVERSOLD", "30"))
    config.rsi_overbought = float(os.getenv("RSI_OVERBOUGHT", "70"))
    config.vwap_enabled = os.getenv("VWAP_ENABLED", "true").lower() == "true"
    config.order_flow_enabled = os.getenv("ORDER_FLOW_ENABLED", "true").lower() == "true"
    config.min_signal_strength = float(os.getenv("MIN_SIGNAL_STRENGTH", "0.5"))
    log.info(f"量化信号: {'开启' if config.quant_signals_enabled else '关闭'} (RSI={config.rsi_enabled}, VWAP={config.vwap_enabled}, OFI={config.order_flow_enabled})")

    # 自适应止损
    config.adaptive_stop_loss = os.getenv("ADAPTIVE_STOP_LOSS", "true").lower() == "true"
    config.adaptive_stop_base = float(os.getenv("ADAPTIVE_STOP_BASE", "0.02"))
    config.adaptive_stop_volatility_mult = float(os.getenv("ADAPTIVE_STOP_VOLATILITY_MULT", "1.5"))
    log.info(f"自适应止损: {'开启' if config.adaptive_stop_loss else '关闭'} (基础: {config.adaptive_stop_base:.1%}, 波动乘数: {config.adaptive_stop_volatility_mult}x)")

    # 连续亏损控制
    config.max_consecutive_losses = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))
    log.info(f"连续亏损限制: {config.max_consecutive_losses} 次")

    log.info("=" * 30)

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
