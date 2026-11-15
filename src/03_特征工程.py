# -*- coding: utf-8 -*-
"""
03_特征工程.py —— 时序特征 / 日期特征 / 训练验证切分
====================================================================
第 2 步的画像特征是"整段训练期的平均"，它有一个隐患：
一个用户在 3 月领的券，画像里却用到了他 4 月的表现 → 时间上是"看了未来"。

这一步做两件事：
  ① 把"看未来"改成"只看过去"：对每一个样本，只统计【这条样本的领券日期之前】
     发生的事情，用"排序 + 累计求和 + 错位（shift）"实现，即 point-in-time 特征。
  ② 加日期类特征，并把数据切成：
        训练集 = 1 月 1 日 ~ 4 月 30 日领的券
        验证集 = 5 月 1 日 ~ 6 月 30 日领的券（模拟"预测未来 15 天"）

输入：data/clean/events.pkl + data/clean/wide_table.pkl
输出：data/features/train_features.pkl、valid_features.pkl
      outputs/tables/03_特征清单.csv
运行：python src/03_特征工程.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    CLEAN_EVENTS, DATA_CLEAN, FEAT_TRAIN, FEAT_VALID, TAB_DIR, TRAIN_END,
)


def main():
    print("=" * 70)
    print("步骤 3 / 特征工程：时序安全特征 + 日期特征 + 数据集切分")
    print("=" * 70)

    ev = pd.read_pickle(CLEAN_EVENTS)
    wide = pd.read_pickle(DATA_CLEAN / "wide_table.pkl")
    print(f"[读入] 事件 {len(ev):,} 行；宽表 {wide.shape}")

    # ---------------------------------------------------------------
    # 1. 先给每行打一个"身份证号" row_id，再按时间排序
    #    ★ 宽表的行顺序 = 原始顺序里所有领券行，用 row_id 对齐才不会把特征贴错行
    # ---------------------------------------------------------------
    ev = ev.reset_index(drop=True)
    ev["row_id"] = np.arange(len(ev), dtype="int64")
    coupon_row_ids = ev.loc[ev["is_coupon"] == 1, "row_id"].to_numpy()

    ev["ev_date"] = ev["date_received"].fillna(ev["consume_date"])
    ev = ev.sort_values("ev_date", kind="mergesort").reset_index(drop=True)
    ev["y"] = ev["label"].fillna(0.0).astype("float32")
    ev["is_recv"] = ev["is_coupon"].astype("float32")
    ev["is_cons"] = ev["consume_date"].notna().astype("float32")

    # ---------------------------------------------------------------
    # 2. 累计（只看过去）特征
    # ---------------------------------------------------------------
    print("[特征] 计算 point-in-time 累计特征……")

    def add_prior(keys, prefix):
        """
        对一组 key 生成：过去领券次数、过去核销次数、过去核销率、过去消费次数、距上次事件天数
        """
        kser = ev[keys[0]] if len(keys) == 1 else ev[keys].astype(str).agg("|".join, axis=1)
        grp = lambda col: ev.groupby(kser, observed=True)[col]  # noqa: E731
        ev[f"{prefix}_prior_recv"] = grp("is_recv").cumsum().groupby(kser).shift(1).fillna(0)
        ev[f"{prefix}_prior_redeem"] = grp("y").cumsum().groupby(kser).shift(1).fillna(0)
        ev[f"{prefix}_prior_cons"] = grp("is_cons").cumsum().groupby(kser).shift(1).fillna(0)
        ev[f"{prefix}_prior_rate"] = (
            ev[f"{prefix}_prior_redeem"] / ev[f"{prefix}_prior_recv"].replace(0, np.nan)
        ).fillna(0)
        gap = ev.groupby(kser, observed=True)["ev_date"].diff().dt.days
        ev[f"{prefix}_last_gap"] = gap.fillna(-1)

    add_prior(["user_id"], "u")
    add_prior(["user_id", "coupon_type"], "ut")
    add_prior(["merchant_id"], "m")
    add_prior(["coupon_id"], "c")
    add_prior(["user_id", "merchant_id"], "ui")

    kser_u = ev["user_id"]
    dsum = ev.groupby(kser_u, observed=True)["distance"].cumsum().groupby(kser_u).shift(1).fillna(0)
    ev["u_prior_dist_mean"] = (dsum / ev["u_prior_recv"].replace(0, np.nan)).fillna(ev["distance"])

    prior_cols = [c for c in ev.columns if "prior_" in c or c.endswith("_last_gap")]

    recv_rows = ev[ev["is_recv"] == 1].set_index("row_id")
    pri = recv_rows[prior_cols].reindex(coupon_row_ids).reset_index(drop=True)
    assert len(pri) == len(wide), f"行数不一致：{len(pri)} vs {len(wide)}"
    wide = wide.reset_index(drop=True)
    wide = pd.concat([wide, pri], axis=1)

    # ---------------------------------------------------------------
    # 3. 日期类特征
    # ---------------------------------------------------------------
    print("[特征] 日期类特征……")
    d = pd.to_datetime(wide["date_received"])
    wide["date_received"] = d
    wide["day_of_month"] = d.dt.day.astype("int8")
    wide["month"] = d.dt.month.astype("int8")
    wide["day_of_week"] = d.dt.dayofweek.astype("int8")
    wide["is_weekend"] = (d.dt.dayofweek >= 5).astype("int8")
    wide["is_month_start"] = (d.dt.day <= 10).astype("int8")
    wide["is_month_end"] = (d.dt.day >= 21).astype("int8")
    wide["days_since_start"] = (d - pd.Timestamp("2016-01-01")).dt.days.astype("int16")
    daily = wide.groupby(d.dt.date).size()
    wide["daily_recv_cnt"] = d.dt.date.map(daily).astype("int32")
    wide["is_manjian"] = (wide["coupon_type"] == "满减").astype("int8")
    wide["dist_diff"] = wide["distance"] - wide["u_prior_dist_mean"]

    # ---------------------------------------------------------------
    # 4. 切分训练 / 验证
    # ---------------------------------------------------------------
    train_end = pd.Timestamp(TRAIN_END)
    tr = wide[wide["date_received"] <= train_end].reset_index(drop=True)
    va = wide[wide["date_received"] > train_end].reset_index(drop=True)
    print(f"[切分] 训练集 {len(tr):,} 行（1/1-4/30），验证集 {len(va):,} 行（5/1-6/30）")
    print(f"        训练集核销率 {tr['label'].mean():.2%}，验证集核销率 {va['label'].mean():.2%}")

    # ---------------------------------------------------------------
    # 5. 挑出真正进模型的列
    # ---------------------------------------------------------------
    drop_cols = ["user_id", "merchant_id", "coupon_id", "coupon_type",
                 "date_received", "consume_date", "days_gap", "is_coupon", "label"]
    feats = [c for c in wide.columns if c not in drop_cols and wide[c].dtype != object]
    print(f"[特征] 最终进入模型的数值特征 {len(feats)} 个")

    tr.to_pickle(FEAT_TRAIN)
    va.to_pickle(FEAT_VALID)

    desc = {
        "discount_strength": "优惠力度（满减=减免/门槛；折扣=1-折扣率）",
        "min_spend": "满减门槛金额（折扣券为空）",
        "cut_amount": "满减金额（折扣券为空）",
        "discount_rate_num": "折扣率（仅折扣券有值）",
        "is_manjian": "是否满减券（1=满减，0=折扣券）",
        "distance": "用户与商家距离等级（0-10）",
        "daily_recv_cnt": "当天全平台发券量（热度）",
        "day_of_week": "星期几（0=周一）",
        "is_weekend": "是否周末",
        "days_since_start": "距 2016-01-01 的天数",
        "u_prior_rate": "该用户此前领券的核销率（只看过去）",
        "m_prior_rate": "该商家此前券的核销率（只看过去）",
        "c_prior_rate": "该券种此前核销率（只看过去）",
        "ui_prior_rate": "该用户在该店此前的核销率（只看过去）",
        "u_last_gap": "距该用户上一次行为多少天（-1=从没出现过）",
    }
    rows = []
    for c in feats:
        rows.append({
            "特征名": c,
            "说明": desc.get(c, "见 02 数据集成脚本的聚合口径"),
            "训练集均值": round(float(tr[c].mean()), 4),
            "缺失率": round(float(tr[c].isna().mean()), 4),
        })
    pd.DataFrame(rows).to_csv(TAB_DIR / "03_特征清单.csv", index=False, encoding="utf-8-sig")
    print(f"[保存] {FEAT_TRAIN.name} / {FEAT_VALID.name} / 03_特征清单.csv")


if __name__ == "__main__":
    main()
