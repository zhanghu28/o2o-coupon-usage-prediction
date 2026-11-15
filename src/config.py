# -*- coding: utf-8 -*-
"""
config.py —— 全项目唯一的"参数总开关"
==================================================
所有脚本都从这里读路径和参数。想改数据位置、时间切分、模型参数，只改这一个文件。
"""
from pathlib import Path

# ---------- 1. 目录 ----------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_DIR / "data" / "raw"
DATA_CLEAN = PROJECT_DIR / "data" / "clean"
DATA_FEAT = PROJECT_DIR / "data" / "features"
OUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
MOD_DIR = OUT_DIR / "model"

for _d in (DATA_RAW, DATA_CLEAN, DATA_FEAT, FIG_DIR, TAB_DIR, MOD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- 2. 数据文件 ----------
RAW_TRAIN = DATA_RAW / "ccf_offline_stage1_train.csv"
CLEAN_EVENTS = DATA_CLEAN / "events.pkl"        # 清洗后的事件总表
CLEAN_QUALITY = TAB_DIR / "01_数据质量报告.csv"      # 清洗阶段的质量统计
FEAT_TRAIN = DATA_FEAT / "train_features.pkl"   # 训练集（1-4月领券）
FEAT_VALID = DATA_FEAT / "valid_features.pkl"   # 验证集（5-6月领券）
PRED_FILE = TAB_DIR / "07_验证集预测明细.csv"
SUBMIT_FILE = OUT_DIR / "submission.csv"
METRICS_FILE = TAB_DIR / "06_模型评估指标.csv"

# ---------- 3. 业务口径参数 ----------
REDEEM_WINDOW_DAYS = 15        # 官方口径：领券后 15 天内消费 = 核销（label=1）
TRAIN_END = "2016-04-30"       # 训练集：1月1日 ~ 4月30日 领券
VALID_START = "2016-05-01"     # 验证集：5月1日 ~ 6月30日 领券（模拟"预测未来"）
VALID_END = "2016-06-30"

# ---------- 4. 随机种子与模型参数 ----------
SEED = 42
VALID_SIZE_FOR_TUNING = 0.2     # 仅用于训练集内部的小验证，做早停

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    num_threads=2,
    seed=SEED,
    verbose=-1,
)
LGB_ROUNDS = 600
LGB_EARLY_STOP = 40

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    eta=0.05,
    max_depth=6,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    nthread=2,
    seed=SEED,
)
XGB_ROUNDS = 600
XGB_EARLY_STOP = 40

# ---------- 5. 中文绘图 ----------
def setup_matplotlib():
    """让 matplotlib 能正常显示中文（按顺序挑一个本机存在的字体）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    candidates = [
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Source Han Sans CN",
        "SimHei", "Microsoft YaHei", "PingFang SC", "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    return chosen
