# YOLO 自动训练系统迁移指南

本目录是一个可整体迁移的 YOLO 自动训练闭环：自动调参、自动训练、自动评估、断点续跑，支持 LLM 提议超参或 Optuna 贝叶斯搜索两种模式。

## 目录结构

```text
auto_train_deploy/
├── README.md                     # 本迁移指南
├── train_cubit.py                # 单次训练脚本（Ultralytics 封装）
├── make_yaml_from_labels.py      # 扫描标签自动生成 data.yaml / classes.txt
└── auto_train/
    ├── run_auto_train.py         # 自动调参主循环
    ├── trainer.py                # 训练执行器（含 --resume 续跑）
    ├── evaluator.py              # results.csv 指标提取
    ├── validator.py              # 超参范围校验
    ├── ledger.py                 # 实验账本（JSON 持久化）
    ├── tuner.py                  # LLM 调参器
    ├── optuna_tuner.py           # Optuna 调参器
    ├── incremental_train.py      # 增量训练（合并新数据后微调）
    ├── run_auto_train.bat        # Windows 启动脚本
    ├── config.json               # 本机配置模板
    ├── config.server.json        # 服务器配置模板
    ├── config.optuna0.json       # 服务器 Optuna 配置模板
    ├── config.optuna2.json       # 本机/服务器 Optuna 配置模板
    └── config.local.example.json # API 配置模板（复制为 config.local.json）
```

## 安全与隐私

本仓库只包含可公开的自动化训练代码和配置模板，不包含 API 密钥、训练权重、数据集、Optuna 数据库或运行日志。API 配置请复制 `auto_train/config.local.example.json` 为 `auto_train/config.local.json` 并填入自己的密钥，该文件已被 `.gitignore` 排除，不会提交。

## 环境要求

- Python 3.10+，PyTorch CUDA 可用
- ultralytics（8.4.x，训练脚本已验证）
- optuna（使用 `--proposer optuna` 时需要）
- paramiko（可选，用于 SSH 远程执行）

本机推荐使用 conda 环境：

```bash
conda create -n yolo-train python=3.11 -y
conda activate yolo-train
pip install ultralytics optuna paramiko
```

## 第一步：准备数据集

数据使用标准 YOLO 格式：

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

如果数据集没有 `data.yaml`，用标签自动生成：

```bash
python make_yaml_from_labels.py --data /path/to/dataset
python make_yaml_from_labels.py --data /path/to/dataset --names class1,class2,class3
```

类别名来源优先级：`--names` 参数 > 现有 `classes.txt` > 现有 `data.yaml` > `class_0` 占位名。

## 第二步：配置

主配置为 `auto_train/config.json`，常用字段：

```json
{
  "target_metric": "metrics/mAP50(M)",
  "target_value": 0.8,
  "max_trials": 6,
  "training": {
    "data": "/path/to/dataset/data.yaml",
    "weights": "/path/to/yolo11n-seg.pt",
    "project": "/path/to/runs",
    "device": "0",
    "epochs": 200,
    "batch": 16,
    "imgsz": 640
  }
}
```

LLM 调参的 API 信息放在 `config.local.json`（由主循环自动合并，不要提交到仓库）：

```bash
cp auto_train/config.local.example.json auto_train/config.local.json
```

然后把 `api_key` 改成你自己的密钥，`base_url` 和 `model` 按你的 LLM 服务填写。使用 Optuna 模式不需要填 API 信息。

## 第三步：运行

Windows 本机：

```bash
cd auto_train
run_auto_train.bat --proposer optuna --ledger C:/runs/experiments.json --optuna-db C:/runs/optuna.db
run_auto_train.bat --proposer llm --ledger C:/runs/experiments.json
```

Linux 服务器（长任务务必用 tmux）：

```bash
source /path/to/miniconda3/bin/activate yolo-train
cd /path/to/auto_train_deploy/auto_train
tmux new -s auto_train
python run_auto_train.py --proposer optuna --ledger /data/runs/experiments.json --optuna-db /data/runs/optuna.db
```

启动前可以用 mock 模式验证链路：

```bash
python run_auto_train.py --mock --proposer optuna --iterations 2
```

## 断点续跑

关机、断连或手动中断后，使用原命令追加 `--resume`：

```bash
python run_auto_train.py --proposer optuna --resume \
  --ledger /data/runs/experiments.json --optuna-db /data/runs/optuna.db
```

行为：

- 已完成的 trial（目录内有 `args.json`）自动跳过
- 未完成但有 `last.pt` 的 trial 先续训，再继续后续 trial
- 未完成且无 `last.pt` 的 trial 从头重跑
- Optuna 历史与账本保留，trial 编号不重复

## 增量训练

已有新标注数据时，合并进基础数据集并微调：

```bash
python auto_train/incremental_train.py \
  --base-data /path/to/base/data.yaml \
  --new-data /path/to/new_data \
  --out /path/to/merged \
  --weights /path/to/best.pt
```

合并后自动生成新的 `data.yaml` 并微调，结果写入账本。

## 常见问题

- OOM：自动降低 batch 重试，仍失败则该轮作废并继续
- 训练 NaN：改用 `--no-amp` 或服务器上启用 `--tf32`
- Windows 中文路径：数据集和项目路径尽量使用英文，避免 Ultralytics 编码问题
- 长任务：服务器上必须 tmux 托管，nohup 在部分环境下不可靠
- 多 GPU 服务器：训练使用空闲 GPU，避开常驻服务的 GPU
