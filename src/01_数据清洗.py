# -*- coding: utf-8 -*-
"""
01_数据清洗.py —— 数据清洗 + 质量监测
====================================================================
这一支脚本干三件事：
  1) 把原始 CSV 读进来（顺便用小类型省内存）
  2) 把"脏"的地方逐条修干净：日期非法、金额/折扣混在一个字段、重复行、
     领券与消费时间对不上、字段含义歧义
  3) 生成"数据质量报告"（CSV），把每一步改了多少行写清楚，方便写论文

输入：data/raw/ccf_offline_stage1_train.csv
输出：data/clean/events.pkl            清洗后的事件总表（一行 = 一次"领券"或"消费"）
      outputs/tables/01_数据质量报告.csv

运行：python src/01_数据清洗.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    RAW_TRAIN, CLEAN_EVENTS, CLEAN_QUALITY, TAB_DIR,
    REDEEM_WINDOW_DAYS,
)


# ------------------------------------------------------------------
# 小工具：把 20160528 / "20160528.0" / 20160500 这类数字日期变成真日期
# ------------------------------------------------------------------
def parse_numeric_date(series: pd.Series) -> pd.Series:
    """
    原始数据里的日期是"20160528.0"这种数字。它有两个坑：
      坑1：被读成浮点数，末尾多一个 .0
      坑2：出现 20160100 这种"0 日"，其实是脏数据（1月00日不存在）
    做法：转成整数 -> 补成 8 位字符串 -> 按 %Y%m%d 解析，解析失败的就变 NaT（空值）
    """
    num = pd.to_numeric(series, errors="coerce")
    s = num.dropna().astype("int64").astype(str).str.zfill(8)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    parsed = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    out.loc[parsed.index] = parsed
    return out


def parse_discount(dr: pd.Series):
    """
    折扣字段其实是两种业务规则混在一起，必须先拆开：
      · "150:20"  = 满 150 元减 20 元（满减券）
      · "0.9"     = 9 折（折扣券）
    返回 4 列：券类型 / 门槛金额 / 减免金额 / 折扣率
    """
    raw = dr.fillna("").astype(str).str.strip()
    is_manjian = raw.str.contains(":", regex=False)
    # —— 满减券 ——
    parts = raw.where(is_manjian).str.split(":", expand=True)
    min_spend = pd.to_numeric(parts[0], errors="coerce")
    cut_amount = pd.to_numeric(parts[1], errors="coerce")
    # —— 折扣券 ——
    rate = pd.to_numeric(raw.where(~is_manjian), errors="coerce")
    # 统一成一个"优惠力度"：满减=减多少/满多少；折扣=1-折扣率
    strength = np.where(
        is_manjian,
        cut_amount / min_spend.where(min_spend > 0),
        1 - rate,
    )
    coupon_type = np.where(is_manjian, "满减", np.where(raw.eq(""), "", "折扣"))
    return pd.Series(coupon_type, index=dr.index), min_spend, cut_amount, rate, pd.Series(strength, index=dr.index)


def main():
    print("=" * 70)
    print("步骤 1 / 数据清洗与质量监测")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. 读数据（指定小类型，1.75 亿字节的数据只占 110MB 内存）
    # ---------------------------------------------------------------
    dtypes = {
        "user_id": "int32",
        "merchant_id": "int32",
        "coupon_id": "float32",
        "distance": "float32",
    }
    df = pd.read_csv(RAW_TRAIN, dtype=dtypes)
    n_raw = len(df)
    print(f"[读入] 原始行数：{n_raw:,}   列：{list(df.columns)}")

    report = []

    def log(step, value, note):
        report.append({"清洗环节": step, "数值": value, "说明": note})

    log("原始行数", n_raw, "官方训练集 ccf_offline_stage1_train.csv")

    # ---------------------------------------------------------------
    # 2. 日期解析（坑：20160100 这类非法日期、.0 尾巴）
    # ---------------------------------------------------------------
    recv_raw = pd.to_numeric(df["date_received"], errors="coerce")
    cons_raw = pd.to_numeric(df["date"], errors="coerce")

    df["date_received"] = parse_numeric_date(df["date_received"])
    df["consume_date"] = parse_numeric_date(df["date"])

    n_recv_na = int(recv_raw.isna().sum())
    n_cons_na = int(cons_raw.isna().sum())
    n_recv_invalid = int(recv_raw.notna().sum() - df["date_received"].notna().sum())
    n_cons_invalid = int(cons_raw.notna().sum() - df["consume_date"].notna().sum())
    log("领券日期缺失/为空", n_recv_na, "无券记录或未领券，属正常业务缺失")
    log("领券日期非法被置空", n_recv_invalid, "如 20160100（1月00日不存在）")
    log("消费日期缺失/为空", n_cons_na, "领了券但没消费 = 未核销，属正常业务缺失")
    log("消费日期非法被置空", n_cons_invalid, "同上，按脏数据处理")

    # ---------------------------------------------------------------
    # 3. 折扣字段拆解
    # ---------------------------------------------------------------
    ctype, min_spend, cut_amount, rate, strength = parse_discount(df["discount_rate"])
    df["coupon_type"] = ctype
    df["min_spend"] = min_spend
    df["cut_amount"] = cut_amount
    df["discount_rate_num"] = rate
    df["discount_strength"] = strength

    log("满减券记录数", int((df["coupon_type"] == "满减").sum()), "字段形如 150:20")
    log("折扣券记录数", int((df["coupon_type"] == "折扣").sum()), "字段形如 0.9")
    log("无券记录数", int((df["coupon_type"] == "").sum()), "coupon_id 与 discount_rate 均为空")
    bad_disc = int(((df["cut_amount"] >= df["min_spend"]) & (df["min_spend"] > 0)).sum())
    bad_min = int((df["min_spend"] <= 0).sum())
    log("异常券：减免≥门槛", bad_disc, "保留但记录，占比极小，不影响建模")
    log("异常券：门槛≤0", bad_min, "同上")

    # ---------------------------------------------------------------
    # 4. 完全重复的行（同一个人同一天同一张券刷了多条）
    # ---------------------------------------------------------------
    dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log("删除完全重复行", dup, "同用户+同商家+同券+同日期完全一致，判为重复上报")

    # ---------------------------------------------------------------
    # 5. 时间逻辑校验：消费必须发生在领券之后
    # ---------------------------------------------------------------
    gap = (df["consume_date"] - df["date_received"]).dt.days
    neg_gap = int((gap < 0).sum())
    log("消费早于领券的异常行", neg_gap, "逻辑矛盾，label 直接判为 0（不算核销）")
    gap = gap.where(gap >= 0, np.nan)
    df["days_gap"] = gap

    # ---------------------------------------------------------------
    # 6. 打标签：领券后 REDEEM_WINDOW_DAYS 天内消费 = 1
    # ---------------------------------------------------------------
    has_coupon = df["coupon_id"].notna() & df["date_received"].notna()
    label = pd.Series(np.nan, index=df.index, dtype="float32")
    cond = has_coupon & (df["days_gap"].notna()) & (df["days_gap"] <= REDEEM_WINDOW_DAYS)
    label.loc[has_coupon] = 0.0
    label.loc[cond] = 1.0
    df["label"] = label
    df["is_coupon"] = has_coupon.astype("int8")

    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    log("可建模的领券样本数", pos + neg, "有券 + 有领券日期，这才是预测的目标人群")
    log("其中核销(label=1)", pos, f"领券后 {REDEEM_WINDOW_DAYS} 天内消费")
    log("其中未核销(label=0)", neg, "含超期消费与完全没消费")
    log("核销率", round(pos / max(pos + neg, 1), 4), "正负样本比例，决定后面是否要加权")

    # ---------------------------------------------------------------
    # 7. 只保留还有信息量的行
    # ---------------------------------------------------------------
    both_na = int((df["date_received"].isna() & df["consume_date"].isna()).sum())
    df = df[~(df["date_received"].isna() & df["consume_date"].isna())].reset_index(drop=True)
    log("删除日期全空的行", both_na, "无领券也无消费，属无效记录")

    n_dist_na = int(df["distance"].isna().sum())
    med = float(df["distance"].median())
    df["distance"] = df["distance"].fillna(med)
    log("距离缺失用中位数填补", n_dist_na, f"填补值 = {med}")

    log("清洗后总行数", len(df), "含领券行 + 纯消费行（纯消费行用于构建历史特征）")
    print(f"[清洗完成] 行数 {n_raw:,} -> {len(df):,}")

    # ---------------------------------------------------------------
    # 8. 保存
    # ---------------------------------------------------------------
    keep = [
        "user_id", "merchant_id", "coupon_id", "coupon_type",
        "min_spend", "cut_amount", "discount_rate_num", "discount_strength",
        "distance", "date_received", "consume_date", "days_gap",
        "label", "is_coupon",
    ]
    df = df[keep]
    df.to_pickle(CLEAN_EVENTS)
    print(f"[保存] {CLEAN_EVENTS}  ({CLEAN_EVENTS.stat().st_size/1024**2:.1f} MB)")

    rep = pd.DataFrame(report)
    rep.to_csv(CLEAN_QUALITY, index=False, encoding="utf-8-sig")
    print(f"[保存] {CLEAN_QUALITY}")

    summary = {
        "raw_rows": n_raw,
        "clean_rows": int(len(df)),
        "coupon_samples": pos + neg,
        "positive": pos,
        "negative": neg,
        "positive_rate": round(pos / max(pos + neg, 1), 6),
        "users": int(df["user_id"].nunique()),
        "merchants": int(df["merchant_id"].nunique()),
        "coupons": int(df["coupon_id"].nunique()),
        "date_min": str(df["date_received"].min().date()) if df["date_received"].notna().any() else "",
        "date_max": str(df["date_received"].max().date()) if df["date_received"].notna().any() else "",
    }
    with open(TAB_DIR / "01_数据摘要.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[摘要]", json.dumps(summary, ensure_ascii=False))

    print("\n数据质量报告预览：")
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
