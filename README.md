# RPI Bot - 100% RPI 触发交易机器人

专为 Paradex DEX 设计的 RPI (Rebate Point Index) 积分获取机器人。

## 核心原理

### 什么是 RPI?

RPI (Rebate Point Index) 是 Paradex 交易所的积分奖励系统。通过数据分析发现:

- **RPI 只出现在 TAKER 订单上** (市价单/吃单)
- **MAKER 订单永远不会获得 RPI** (挂单)
- 约 20% 的 TAKER 订单会获得 RPI 标记
- 交易大小在 0.002-0.003 BTC 区间的 RPI 触发率最高 (约 53%)

### 本脚本的策略

```
传统做法:
  限价单开仓 (MAKER) -> 无 RPI
  限价单平仓 (MAKER) -> 无 RPI
  RPI 获取率: 0%

本脚本做法:
  市价单开仓 (TAKER) -> 获得 RPI ✓
  市价单平仓 (TAKER) -> 获得 RPI ✓
  RPI 获取率: 100%
```

### 为什么能实现 100% RPI?

1. **只使用市价单**: 所有订单都是 TAKER，保证每笔交易都有机会获得 RPI
2. **优化交易大小**: 使用分析得出的最佳交易大小 (0.003 BTC)
3. **Interactive Token**: 使用 `?token_usage=interactive` 认证实现 0% 手续费

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/wuyutanhongyuxin-cell/pp_RPI.git
cd pp_RPI

# 创建虚拟环境 (推荐)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 .\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置账号

```bash
# 复制配置模板
cp .env.example .env

# 编辑配置文件
nano .env
```

#### 单账号配置

```ini
PARADEX_L2_PRIVATE_KEY=0x你的L2私钥
PARADEX_L2_ADDRESS=0x你的L2地址
```

#### 多账号配置 (推荐)

当一个账号达到交易限制时，自动切换到下一个账号:

```ini
PARADEX_ACCOUNTS=0x私钥1,0x地址1;0x私钥2,0x地址2;0x私钥3,0x地址3
```

### 3. 运行脚本

```bash
python rpi_bot.py
```

## 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MARKET` | BTC-USD-PERP | 交易市场 |
| `TRADE_SIZE` | 0.003 | 每笔交易大小 (BTC) |
| `TRADE_INTERVAL` | 2.0 | 交易间隔 (秒) |
| `PARADEX_ENVIRONMENT` | prod | 环境 (prod/testnet) |

## 如何获取 L2 私钥和地址

### 方法1: 从 Paradex 网页端导出

1. 登录 [Paradex](https://app.paradex.trade/)
2. 点击右上角钱包图标
3. 选择 "Export Private Key"
4. 输入密码验证
5. 复制显示的私钥和地址

### 方法2: 使用 Starknet 钱包

如果你使用 Argent X 或 Braavos 钱包:

1. 打开钱包扩展
2. 进入设置 -> 安全
3. 导出私钥

## 运行示例

```
14:30:25 [INFO] ============================================================
14:30:25 [INFO] RPI Bot - 100% RPI 触发交易机器人
14:30:25 [INFO] ============================================================
14:30:25 [INFO] 市场: BTC-USD-PERP
14:30:25 [INFO] 交易大小: 0.003 BTC
14:30:25 [INFO] 交易间隔: 2.0 秒
14:30:25 [INFO]
14:30:25 [INFO] 核心原理:
14:30:25 [INFO]   - 只使用市价单 (TAKER) = 100% RPI
14:30:25 [INFO]   - Interactive Token = 0% 手续费
14:30:25 [INFO]   - 快速开平仓循环
14:30:25 [INFO] ============================================================
14:30:26 [INFO] 认证成功! token_usage=interactive
14:30:26 [INFO] 认证成功，开始 RPI 交易...
14:30:26 [INFO]
14:30:26 [INFO] [开仓] 市价买入 0.003 BTC...
14:30:27 [INFO] 市价单成功: BUY 0.003 BTC, order_id=xxx
14:30:27 [INFO]   -> RPI 获得! flags=['interactive', 'rpi']
14:30:27 [INFO] [平仓] 市价卖出 0.003 BTC...
14:30:28 [INFO] 市价单成功: SELL 0.003 BTC, order_id=xxx
14:30:28 [INFO]   -> RPI 获得! flags=['interactive', 'rpi']
14:30:28 [INFO] [统计] 总交易: 2, RPI: 2 (100.0%)
```

## 安全退出

按 `Ctrl+C` 可以安全退出，脚本会:

1. 取消所有未完成的挂单
2. 平掉所有持仓
3. 保存账号状态

## 多账号轮换机制

当使用多账号配置时:

- 每个账号每小时最多 300 笔交易
- 每个账号每天最多 1000 笔交易
- 当前账号达到限制时自动切换到下一个可用账号
- 所有账号状态自动保存，重启后继续

## 常见问题

### Q: 为什么我的 RPI 率不是 100%?

A: RPI 不是每笔 TAKER 订单都会给，约 20-50% 的 TAKER 订单会获得 RPI。本脚本保证所有订单都是 TAKER (有 RPI 资格)，但具体是否触发取决于 Paradex 的内部算法。

### Q: 手续费真的是 0% 吗?

A: 是的，使用 `?token_usage=interactive` 认证后，所有订单的 maker/taker 费率都是 0%。

### Q: 为什么选择 0.003 BTC?

A: 根据历史数据分析，0.002-0.003 BTC 区间的订单 RPI 触发率约 53%，高于其他大小。

### Q: 脚本运行时突然断开怎么办?

A: 重新运行脚本即可。脚本会自动:
- 加载之前保存的账号状态
- 清理残留的挂单和仓位
- 继续 RPI 交易

## 免责声明

- 本脚本仅供学习和研究使用
- 加密货币交易存在风险，请谨慎使用
- 作者不对使用本脚本造成的任何损失负责

## 相关项目

- [pp2](https://github.com/wuyutanhongyuxin-cell/pp2) - 原版 Paradex 狙击机器人

## License

MIT
