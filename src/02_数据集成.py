# -*- coding: utf-8 -*-
"""
02_数据集成.py —— 多表构建 + 宽表合并（数据集成 Data Integration）
====================================================================
原始表只有一张"流水账"：谁、在哪个店、领了什么券、什么时候领、什么时候消费。
信息量太薄，模型学不到东西。所以这一步要做的是"集成"：

  从一张流水账，拆成 4 张维度表
    ① 用户画像表  user_dim       （一个人整体可不可靠）
    ② 商家画像表  merchant_dim   （一个店整体核销率高不高）
    ③ 优惠券画像表 coupon_dim    （这张券本身好不好用）
    ④ 用户×商家关系表 interaction（这个人在这个店的历史表现）
  再把它们按主键左连接到"领券样本"上 → 得到一张宽表

★ 防泄漏铁律：所有画像统计只用【训练期 2016-01-01 ~ 2016-04-30】的数据算出来，
  然后冻结成一张"字典"，再给验证期（5-6月）用。绝不能用 5-6 月的标签去算画像，
  否则就是"用未来预测未来"。

输入：data/clean/events.pkl
输出：data/clean/wide_table.pkl          合并后的宽表（领券样本）
      outputs/tables/02_集成汇总.csv      每张表的规模
运行：python src/02_数据集成.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLEAN_EVENTS, DATA_CLEAN, TAB_DIR, TRAIN_END  # noqa: E402


def main():
    print("=" * 70)
    print("步骤 2 / 数据集成：4 张维度表 + 宽表合并")
    print("=" * 70)

    ev = pd.read_pickle(CLEAN_EVENTS)
    print(f"[读入] 事件总表 {len(ev):,} 行")

    # ---------------------------------------------------------------
    # 0. 定义"样本"和"画像统计窗口"
    # ---------------------------------------------------------------
    samples = ev[ev["is_coupon"] == 1].copy()
    print(f"[样本] 领券样本 {len(samples):,} 行，核销率 {samples['label'].mean():.2%}")

    freeze_end = pd.Timestamp(TRAIN_END)
    hist = ev[(ev["date_received"].notna() & (ev["date_received"] <= freeze_end))].copy()
    print(f"[画像窗口] 2016-01-01 ~ {TRAIN_END}，共 {len(hist):,} 条领券记录用于建画像")

    consume = ev[ev["consume_date"].notna()].copy()

    # ---------------------------------------------------------------
    # ① 用户画像 user_dim
    # ---------------------------------------------------------------
    # ★ 这里只放"与标签无关"的活跃度统计。凡是跟"核销没核销"有关的统计
    #   （核销率、核销次数），一律留给第 3 步的 point-in-time 特征，否则训练集里
    #   "自己算自己"，AUC 会虚高到 0.98，一上验证集立刻崩盘。
    g = hist.groupby("user_id")
    user_dim = pd.DataFrame({
        "u_receive_cnt": g.size(),
        "u_coupon_types": g["coupon_type"].nunique(),
        "u_dist_mean": g["distance"].mean(),
        "u_first_recv": g["date_received"].min(),
        "u_last_recv": g["date_received"].max(),
    })
    user_dim = user_dim.reset_index()
    user_dim["u_recv_span"] = (user_dim["u_last_recv"] - user_dim["u_first_recv"]).dt.days
    user_dim = user_dim.drop(columns=["u_first_recv", "u_last_recv"])
    print(f"[①用户画像] {len(user_dim):,} 人，{user_dim.shape[1]-1} 个特征")

    # ---------------------------------------------------------------
    # ② 商家画像 merchant_dim
    # ---------------------------------------------------------------
    gm = hist.groupby("merchant_id")
    merchant_dim = pd.DataFrame({
        "m_receive_cnt": gm.size(),
        "m_user_cnt": gm["user_id"].nunique(),
        "m_dist_mean": gm["distance"].mean(),
    })
    gmc = consume.groupby("merchant_id")
    merchant_dim["m_consume_cnt"] = gmc.size()
    merchant_dim["m_consume_user_cnt"] = gmc["user_id"].nunique()
    merchant_dim = merchant_dim.reset_index()
    print(f"[②商家画像] {len(merchant_dim):,} 家店，{merchant_dim.shape[1]-1} 个特征")

    # ---------------------------------------------------------------
    # ③ 优惠券画像 coupon_dim
    # ---------------------------------------------------------------
    gk = hist.groupby("coupon_id")
    coupon_dim = pd.DataFrame({
        "c_receive_cnt": gk.size(),
        "c_dist_mean": gk["distance"].mean(),
        "c_user_cnt": gk["user_id"].nunique(),
        "c_merchant_cnt": gk["merchant_id"].nunique(),
    }).reset_index()
    print(f"[③券画像] {len(coupon_dim):,} 张券，{coupon_dim.shape[1]-1} 个特征")

    # ---------------------------------------------------------------
    # ④ 用户×商家关系 interaction
    # ---------------------------------------------------------------
    gi = hist.groupby(["user_id", "merchant_id"])
    inter = pd.DataFrame({"ui_receive_cnt": gi.size()}).reset_index()
    inter["ui_is_new_pair"] = 0  # 后面左连接时缺失的就是"新组合"
    print(f"[④用户×商家] {len(inter):,} 对关系")

    # ---------------------------------------------------------------
    # ⑤ 合并成宽表（left join，保留所有领券样本）
    # ---------------------------------------------------------------
    wide = samples.merge(user_dim, on="user_id", how="left")
    wide = wide.merge(merchant_dim, on="merchant_id", how="left")
    wide = wide.merge(coupon_dim, on="coupon_id", how="left")
    wide = wide.merge(inter, on=["user_id", "merchant_id"], how="left")

    wide["ui_is_new_pair"] = wide["ui_is_new_pair"].fillna(1).astype("int8")
    wide["u_is_new_user"] = wide["u_receive_cnt"].isna().astype("int8")
    wide["m_is_new_merchant"] = wide["m_receive_cnt"].isna().astype("int8")

    print(f"[合并] 宽表 {wide.shape[0]:,} 行 × {wide.shape[1]} 列")
    print("[冷启动] 新用户占比 {:.2%}，新商家占比 {:.2%}，新组合占比 {:.2%}".format(
        wide["u_is_new_user"].mean(), wide["m_is_new_merchant"].mean(), wide["ui_is_new_pair"].mean()))

    wide.to_pickle(DATA_CLEAN / "wide_table.pkl")

    summary = pd.DataFrame([
        {"表": "事件总表 events", "行数": len(ev), "列数": ev.shape[1], "说明": "清洗后流水账"},
        {"表": "①用户画像 user_dim", "行数": len(user_dim), "列数": user_dim.shape[1], "说明": "按 user_id 聚合"},
        {"表": "②商家画像 merchant_dim", "行数": len(merchant_dim), "列数": merchant_dim.shape[1], "说明": "按 merchant_id 聚合"},
        {"表": "③券画像 coupon_dim", "行数": len(coupon_dim), "列数": coupon_dim.shape[1], "说明": "按 coupon_id 聚合"},
        {"表": "④用户×商家 interaction", "行数": len(inter), "列数": inter.shape[1], "说明": "复合主键聚合"},
        {"表": "领券样本 samples", "行数": len(samples), "列数": samples.shape[1], "说明": "建模对象"},
        {"表": "集成宽表 wide", "行数": wide.shape[0], "列数": wide.shape[1], "说明": "4 表左连接结果"},
    ])
    summary.to_csv(TAB_DIR / "02_集成汇总.csv", index=False, encoding="utf-8-sig")
    print(f"[保存] {DATA_CLEAN/'wide_table.pkl'}  ({os.path.getsize(DATA_CLEAN/'wide_table.pkl')/1024**2:.1f} MB)")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
