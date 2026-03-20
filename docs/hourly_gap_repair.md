# Hourly Gap Repair

## English

### Goal

This repo uses hourly ETH market data, but the Coinbase candle history is not
strictly dense. Some hourly buckets are missing, and the training pipeline in
`datasets/pipeline.py` now hard-fails on non-1-hour timestamp gaps.

To help us compare repair strategies without changing model complexity, the repo
now includes:

- `scripts/check_hourly_intervals.py`
- `scripts/repair_hourly_gaps.py`

### Two repair versions

#### 1. `average`

`average` mode reindexes the CSV onto a strict hourly grid and then applies
linear interpolation across numeric columns.

Why use it:

- Very simple baseline
- Easy to explain
- Produces a dense hourly table quickly

Why be careful:

- It smooths volatility
- It invents synthetic OHLC paths
- It can weaken the realism of trading signals

#### 2. `conservative`

`conservative` mode also reindexes onto a strict hourly grid, but fills price
columns with more market-aware rules:

- `close` / `btc_close`: forward-fill
- `Open` / `btc_open`: previous repaired close
- `high` / `btc_high`: `max(open, close)`
- `low` / `btc_low`: `min(open, close)`
- `volume` / `btc_volume`: `0` for imputed rows
- other numeric columns: forward-fill / backfill
- `market_cap`: recomputed as `close * supply`

Why this is the recommended version:

- Keeps the feature set unchanged
- Avoids adding imputation flags to the model
- Preserves a more defensible OHLCV structure than straight averaging

### Trimming early sparse data

If we want to drop the launch-period region with the worst missingness, we can
trim before a fixed UTC timestamp. A practical choice from the current data is:

```bash
2016-05-26T00:00:00Z
```

This is configurable, not hard-coded.

### Example commands

If you are using the repo-local virtual environment, run the scripts with:

```bash
.venv/bin/python ...
```

Check the original file:

```bash
.venv/bin/python scripts/check_hourly_intervals.py data/metrics/eth_metrics_combined.csv
```

Generate both repaired versions after trimming the early sparse period:

```bash
.venv/bin/python scripts/repair_hourly_gaps.py \
  --input-csv data/metrics/eth_metrics_combined.csv \
  --output-dir data/gap_repair \
  --trim-before 2016-05-26T00:00:00Z
```

Generate only one version:

```bash
.venv/bin/python scripts/repair_hourly_gaps.py \
  --methods conservative \
  --trim-before 2016-05-26T00:00:00Z
```

Validate the outputs:

```bash
.venv/bin/python scripts/check_hourly_intervals.py data/gap_repair/eth_metrics_combined_average_fill.csv
.venv/bin/python scripts/check_hourly_intervals.py data/gap_repair/eth_metrics_combined_conservative_fill.csv
```

### Recommendation

For this repo, `conservative` is the better default.

The reason is not just data quality. It also keeps the rest of the codebase
simple: no new feature schema, no new model input column, and no training logic
changes. `average` is still useful as a comparison baseline.

## 中文

### 目标

这个仓库使用 ETH 的小时级市场数据，但 Coinbase 的历史 candle 数据并不是
严格连续的。有些小时桶缺失，而当前训练流程在
`datasets/pipeline.py` 中会对任何非 1 小时间隔直接报错。

为了在**不增加模型复杂度**的前提下比较不同修复策略，仓库现在提供了：

- `scripts/check_hourly_intervals.py`
- `scripts/repair_hourly_gaps.py`

### 两种修复版本

#### 1. `average`

`average` 模式会先把数据重建到严格的 1 小时网格上，再对数值列做线性插值。

优点：

- 逻辑最简单
- 解释成本低
- 很快就能得到完整的小时级表

缺点：

- 会平滑掉波动
- 会人为制造 OHLC 轨迹
- 对交易信号的真实性不够友好

#### 2. `conservative`

`conservative` 模式同样会先重建到严格的小时网格，但对价格列使用更稳健的
填补规则：

- `close` / `btc_close`：前向填充
- `Open` / `btc_open`：使用上一个修复后的 close
- `high` / `btc_high`：`max(open, close)`
- `low` / `btc_low`：`min(open, close)`
- `volume` / `btc_volume`：对补出来的行设为 `0`
- 其他数值列：前向填充 / 后向填充
- `market_cap`：重新计算为 `close * supply`

为什么更推荐这个版本：

- 不需要修改现有特征集合
- 不需要给模型增加缺失标记列
- 比简单平均更符合交易数据的结构

### 截掉最早期稀疏数据

如果我们想去掉最早那段缺失最严重的启动期数据，可以在修复前先截断。
根据当前数据，一个比较实用的时间点是：

```bash
2016-05-26T00:00:00Z
```

这个时间点是可配置的，不是写死在代码里的。

### 使用示例

如果你使用仓库本地的虚拟环境，可以这样运行：

```bash
.venv/bin/python ...
```

先检查原始文件：

```bash
.venv/bin/python scripts/check_hourly_intervals.py data/metrics/eth_metrics_combined.csv
```

生成两个修复版本，并截掉前面的稀疏区间：

```bash
.venv/bin/python scripts/repair_hourly_gaps.py \
  --input-csv data/metrics/eth_metrics_combined.csv \
  --output-dir data/gap_repair \
  --trim-before 2016-05-26T00:00:00Z
```

只生成一种版本：

```bash
.venv/bin/python scripts/repair_hourly_gaps.py \
  --methods conservative \
  --trim-before 2016-05-26T00:00:00Z
```

验证输出是否已经严格按 1 小时间隔：

```bash
.venv/bin/python scripts/check_hourly_intervals.py data/gap_repair/eth_metrics_combined_average_fill.csv
.venv/bin/python scripts/check_hourly_intervals.py data/gap_repair/eth_metrics_combined_conservative_fill.csv
```

### 建议

对这个仓库来说，`conservative` 更适合作为默认方案。

原因不只是数据更合理，也因为它能保持整个代码库更简单：不需要新增特征
schema，不需要给模型增加额外输入列，也不需要改训练逻辑。`average` 仍然有
价值，但更适合作为对照版本。
