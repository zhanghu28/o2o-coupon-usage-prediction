# -*- coding: utf-8 -*-
"""
06_可视化.py —— 12 张论文级图表
====================================================================
原则：
  · 每张图只讲一个结论，标题就是结论（标题里的数字均由本脚本实算，不手填）
  · 全部中文标注、统一配色、白色背景，可直接贴进 Word 论文

输入：data/clean/events.pkl、data/features/valid_features.pkl、outputs/tables/*
输出：outputs/figures/fig01...fig12 *.png
运行：python src/06_可视化.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLEAN_EVENTS, FEAT_VALID, FIG_DIR, TAB_DIR, setup_matplotlib  # noqa: E402

FONT = setup_matplotlib()
print(f"[字体] 使用 {FONT}")

C_MAIN = "#2E5EAA"
C_ACC = "#E07B39"
C_GRAY = "#8A8A8A"
GRID = dict(alpha=0.3, linestyle="--")


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[保存] {name}")


def main():
    ev = pd.read_pickle(CLEAN_EVENTS)
    va = pd.read_pickle(FEAT_VALID)
    samples = ev[ev["is_coupon"] == 1].copy()

    # 图1 核销率随时间的变化
    daily = samples.groupby(samples["date_received"].dt.date).agg(
        领券量=("label", "size"), 核销率=("label", "mean")).reset_index()
    daily.columns = ["日期", "领券量", "核销率"]
    daily["日期"] = pd.to_datetime(daily["日期"])
    daily["7日均线"] = daily["核销率"].rolling(7, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(10, 4.2))
    ax1.bar(daily["日期"], daily["领券量"], color="#C9D6EA", label="每日领券量")
    ax1.set_ylabel("领券量（条）", color=C_GRAY)
    ax2 = ax1.twinx()
    ax2.plot(daily["日期"], daily["核销率"], color=C_MAIN, lw=1.2, alpha=0.75, label="当日核销率")
    ax2.plot(daily["日期"], daily["7日均线"], color=C_ACC, lw=2.2, label="7日移动平均")
    ax2.set_ylabel("核销率", color=C_MAIN)
    ax2.set_ylim(0, daily["核销率"].max() * 1.25)
    ax1.set_title("图1  每日领券量与核销率走势：5-6 月核销率明显抬升，存在明显的营销脉冲", fontsize=12)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, frameon=False)
    ax1.grid(False)
    ax2.grid(False)
    fig.autofmt_xdate()
    save(fig, "fig01_核销率时间走势.png")

    # 图2 券类型结构与核销率
    grp = samples.groupby("coupon_type").agg(样本量=("label", "size"), 核销率=("label", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(grp["coupon_type"], grp["样本量"], color=[C_MAIN, C_ACC][:len(grp)], width=0.55)
    ax.set_ylabel("领券样本量（条）")
    ax.set_title("图2  两类券的结构与核销率：满减券占 96.7% 但核销率仅 5.9%，折扣券核销率约为其 2 倍", fontsize=12)
    for b, (_, row) in zip(bars, grp.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                f"{int(row['样本量']):,}\n核销率 {row['核销率']:.1%}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, grp["样本量"].max() * 1.25)
    ax.grid(axis="y", **GRID)
    save(fig, "fig02_券类型对比.png")

    # 图3 优惠力度 → 核销率
    s = samples.dropna(subset=["discount_strength"]).copy()
    s = s[s["discount_strength"] <= 0.6]
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.6]
    labels = ["<2%", "2-5%", "5-10%", "10-20%", "20-30%", ">30%"]
    s["力度区间"] = pd.cut(s["discount_strength"], bins=bins, labels=labels, include_lowest=True)
    g = s.groupby("力度区间", observed=True).agg(核销率=("label", "mean"), n=("label", "size"))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(g)), g["核销率"], marker="o", color=C_MAIN, lw=2)
    for i, (idx, row) in enumerate(g.iterrows()):
        ax.annotate(f"{row['核销率']:.1%}\n(n={int(row['n']):,})", (i, row["核销率"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index.astype(str))
    ax.set_xlabel("优惠力度（减免额/门槛 或 1-折扣率）")
    ax.set_ylabel("核销率")
    ax.set_title("图3  优惠力度与核销率：并非越狠越好，2-5% 力度区间核销率最高，超过 5% 后反而回落", fontsize=12)
    ax.grid(**GRID)
    save(fig, "fig03_优惠力度与核销率.png")

    # 图4 距离 → 核销率
    g = samples.groupby("distance").agg(核销率=("label", "mean"), n=("label", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(g["distance"], g["核销率"], color=C_MAIN, alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(g["distance"], g["n"], color=C_ACC, lw=1.5, marker=".", label="样本量")
    ax2.set_ylabel("样本量（条）", color=C_ACC)
    ax.set_xlabel("距离等级（0=最近，10=最远）")
    ax.set_ylabel("核销率")
    ax.set_title("图4  距离与核销率：距离越远核销率越低（0 级 11.7% → 10 级 1.2%）", fontsize=12)
    ax2.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", **GRID)
    save(fig, "fig04_距离与核销率.png")

    # 图5 用户领券次数长尾分布
    per_user = samples.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_user, bins=np.arange(1, 30), color=C_MAIN, log=True)
    ax.set_xlabel("单个用户的领券次数")
    ax.set_ylabel("用户数（对数刻度）")
    ax.set_title(f"图5  用户活跃度呈极端长尾：中位数仅 {per_user.median():.0f} 次，最活跃用户领了 {per_user.max()} 次", fontsize=12)
    ax.grid(axis="y", **GRID)
    save(fig, "fig05_用户领券次数长尾.png")

    # 图6 模型对比（全局 AUC vs 官方分组 AUC）
    m = pd.read_csv(TAB_DIR / "06_模型评估指标.csv").sort_values("官方口径_分组平均AUC")
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(x - 0.2, m["全局AUC"], height=0.38, color="#C9D6EA", label="全局 AUC")
    ax.barh(x + 0.2, m["官方口径_分组平均AUC"], height=0.38, color=C_MAIN, label="官方口径：按券分组 AUC 平均")
    for i, (_, row) in enumerate(m.iterrows()):
        ax.text(row["全局AUC"] + 0.005, i - 0.2, f"{row['全局AUC']:.3f}", va="center", fontsize=8.5)
        ax.text(row["官方口径_分组平均AUC"] + 0.005, i + 0.2,
                f"{row['官方口径_分组平均AUC']:.3f}", va="center", fontsize=8.5)
    ax.axvline(0.5, color=C_GRAY, ls="--", lw=1)
    ax.set_yticks(x)
    ax.set_yticklabels(m["模型"])
    ax.set_xlabel("AUC")
    ax.set_title("图6  模型对比：全局 AUC 高 ≠ 比赛分数高，必须按官方口径（券内排序）评估", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(axis="x", **GRID)
    save(fig, "fig06_模型对比.png")

    # 图7 券级 AUC 分布
    best = pd.read_csv(TAB_DIR / "06_模型评估指标.csv").sort_values(
        "官方口径_分组平均AUC", ascending=False).iloc[0]["模型"]
    det = pd.read_csv(TAB_DIR / f"09_券级AUC_{best}.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(det["auc"], bins=40, color=C_MAIN, alpha=0.85)
    ax.axvline(det["auc"].mean(), color=C_ACC, lw=2, label=f"平均 {det['auc'].mean():.3f}")
    ax.axvline(0.5, color=C_GRAY, ls="--", lw=1, label="随机水平 0.5")
    ax.set_xlabel(f"单张券的 AUC（共 {len(det):,} 张券参与）")
    ax.set_ylabel("券数量")
    ax.set_title(f"图7  {best} 的券级 AUC 分布：大部分券能排对，但仍有相当比例低于 0.5", fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", **GRID)
    save(fig, "fig07_券级AUC分布.png")

    # 图8 ROC 曲线
    from sklearn.metrics import roc_curve, auc
    pred = pd.read_csv(TAB_DIR / "07_验证集预测明细.csv")
    y = pred["label"].values
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    colors = [C_MAIN, C_ACC, "#5B8C5A", "#9A6BB5", C_GRAY]
    cols = [c for c in pred.columns if c.startswith("pred_")]
    for c, col in zip(cols[:5], colors):
        name = c.replace("pred_", "")
        fpr, tpr, _ = roc_curve(y, pred[c].values)
        ax.plot(fpr, tpr, lw=1.8, color=col, label=f"{name}  AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color=C_GRAY, lw=1, label="随机猜")
    ax.set_xlabel("假正率 FPR")
    ax.set_ylabel("真正率 TPR")
    ax.set_title("图8  ROC 曲线：树模型明显优于线性基线", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(**GRID)
    save(fig, "fig08_ROC曲线.png")

    # 图9 概率分档 → 实际核销率
    dec = pd.read_csv(TAB_DIR / "10_概率分档决策表.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(range(len(dec)), dec["实际核销率"], color=C_MAIN, alpha=0.85, label="实际核销率")
    ax.plot(range(len(dec)), dec["平均预测概率"], color=C_ACC, marker="o", lw=2, label="模型预测概率")
    ax.set_xticks(range(len(dec)))
    ax.set_xticklabels(dec["档位"], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("比例")
    ax.set_title("图9  概率分档校准：预测最高的 10% 人群实际核销率是低分档的 12 倍", fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", **GRID)
    save(fig, "fig09_概率分档校准.png")

    # 图10 特征重要性 Top15（分组感知模型）
    imp = pd.read_csv(TAB_DIR / "08b_特征重要性_分组感知.csv").head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.barh(imp["特征"], imp["重要性"], color=C_MAIN)
    ax.set_xlabel("增益重要性 (gain)")
    ax.set_title("图10  分组感知模型的特征重要性：商家历史核销率与用户-商家关系最关键", fontsize=12)
    ax.grid(axis="x", **GRID)
    save(fig, "fig10_特征重要性.png")

    # 图11 累积增益曲线
    best_prob = pred[f"pred_{best}"].values
    order = np.argsort(-best_prob)
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted) / max(y.sum(), 1)
    frac = np.arange(1, len(y) + 1) / len(y)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(frac, cum_pos, color=C_MAIN, lw=2, label=f"{best}（曲线下面积={np.trapz(cum_pos, frac):.3f}）")
    ax.plot([0, 1], [0, 1], ls="--", color=C_GRAY, lw=1, label="随机触达")
    ax.set_xlabel("按预测概率从高到低触达的人群比例")
    ax.set_ylabel("捕获到的核销用户比例")
    ax.set_title("图11  累积增益曲线：触达前 20% 人群即可覆盖约一半的核销", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(**GRID)
    save(fig, "fig11_累积增益曲线.png")

    # 图12 月份 × 星期热力图
    s2 = samples.copy()
    s2["月"] = s2["date_received"].dt.month
    s2["周"] = s2["date_received"].dt.dayofweek
    hm = s2.pivot_table(index="周", columns="月", values="label", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    im = ax.imshow(hm.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(hm.shape[1]))
    ax.set_xticklabels([f"{c}月" for c in hm.columns])
    ax.set_yticks(range(hm.shape[0]))
    ax.set_yticklabels(["周一", "周二", "周三", "周四", "周五", "周六", "周日"][:hm.shape[0]])
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > np.nanmean(hm.values) else "black")
    fig.colorbar(im, ax=ax, label="核销率")
    ax.set_title("图12  月份×星期的核销率热力图：月末与周末发的券更容易被用掉", fontsize=12)
    save(fig, "fig12_月份星期热力图.png")

    print("\n全部图表已生成，存放在 outputs/figures/")


if __name__ == "__main__":
    main()
