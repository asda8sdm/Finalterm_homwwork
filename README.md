# Household Power Forecasting Course Project

本目录包含 2026 年专硕机器学习课程期末作业的可复现实验代码与报告材料。

## 数据

- 原始用电数据：`individual+household+electric+power+consumption.zip`
- 天气数据：`data/raw/MENSQ_92_previous-1950-2024.csv.gz`
- 处理后的日尺度数据：`data/processed/daily_power_weather.csv`
- 训练/测试划分：`data/processed/train.csv`、`data/processed/test.csv`

用电数据按 PDF 要求处理：

- `global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2`、`sub_metering_3` 按日求和。
- `voltage`、`global_intensity` 按日求均值。
- `sub_metering_remainder = global_active_power * 1000 / 60 - (sub_metering_1 + sub_metering_2 + sub_metering_3)`。
- 天气变量使用 Météo-France 92 省月度气候数据，并按月份映射到每天。

## 运行

推荐使用本机可用 CUDA 的环境：

```bash
/data1/zhinicai/anaconda3/envs/seg_patch/bin/python src/run_experiments.py \
  --models lstm transformer msrt \
  --horizons 90 365 \
  --seeds 2026 2027 2028 2029 2030 \
  --epochs 50 \
  --patience 8 \
  --batch-size 128 \
  --device cuda
```

最终报告中的 MSRT 指标采用了预测长度自适应训练：90 天任务使用默认 `lr=1e-3`，365 天任务使用更稳的 `lr=3e-4`：

```bash
/data1/zhinicai/anaconda3/envs/seg_patch/bin/python src/run_experiments.py \
  --models msrt --horizons 90 --seeds 2026 2027 2028 2029 2030 \
  --epochs 50 --patience 8 --batch-size 128 --lr 1e-3 --device cuda

/data1/zhinicai/anaconda3/envs/seg_patch/bin/python src/run_experiments.py \
  --models msrt --horizons 365 --seeds 2026 2027 2028 2029 2030 \
  --epochs 90 --patience 12 --batch-size 128 --lr 3e-4 --device cuda
```

如果 CUDA 不可用，可把 `--device cuda` 改为 `--device cpu`。

## 输出

- 单轮实验：`outputs/tables/all_runs.csv`
- 均值和标准差：`outputs/tables/summary_metrics.csv`
- 对比曲线：`outputs/figures/comparison_h90.png`、`outputs/figures/comparison_h365.png`
- 模型权重：`outputs/models/`
- 报告：`report/final_report.md`、`report/final_report.html`

## 模型

- `lstm`：两层 LSTM 后接 MLP 多步直接预测头。
- `transformer`：输入投影、位置编码、Transformer Encoder、池化预测头。
- `msrt`：自定义 Horizon-Adaptive MSRT。90 天任务采用 LSTM 锚定的多尺度天气残差 Transformer；365 天任务额外启用 DLinear 趋势/残差分解分支，再用门控与 LSTM 主预测融合。
