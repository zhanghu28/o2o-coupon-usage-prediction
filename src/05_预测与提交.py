# -*- coding: utf-8 -*-
"""
05_预测与提交.py —— 生成提交文件 + 业务决策表
====================================================================
天池的提交格式是：User_id, Coupon_id, Date_received, Probability
即对每一条“某人某天领了某张券”，给出一个 0~1 的核销概率。

这一支做两件事：
  ① 用效果最好的模型（按官方分组 AUC 选）在验证集上出概率 → submission.csv
  ② 把“概率”翻译成业务语言：按概率分档，算出每一档的实际核销率

输入：data/features/valid_features.pkl + outputs/model/* + outputs/tables/06_模型评估指标.csv
输出：outputs/submission.csv、outputs/tables/10_概率分档决策表.csv
运行：python src/05_预测与提交.py
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEAT_VALID, MOD_DIR, SUBMIT_FILE, TAB_DIR  # noqa: E402

DROP = ["user_id", "merchant_id", "coupon_id", "coupon_type",
        "date_received", "consume_date", "days_gap", "is_coupon", "label"]


def main():
    print("=" * 70)
    print("步骤 5 / 预测与提交")
    print("=" * 70)

    va = pd.read_pickle(FEAT_VALID)
    feats = [c for c in va.columns if c not in DROP and va[c].dtype != object]

    metrics = pd.read_csv(TAB_DIR / "06_模型评估指标.csv")
    best = metrics.sort_values("官方口径_分组平均AUC", ascending=False).iloc[0]["模型"]
    print(f"[选模型] 按官方分组 AUC 选出的最优模型：{best}")

    if best == "LGB分组感知":
        model = lgb.Booster(model_file=str(MOD_DIR / "lightgbm_groupaware.txt"))
        ga_feats = model.feature_name()
        prob = model.predict(va[ga_feats].astype("float32").values)
    elif best == "LightGBM":
        model = lgb.Booster(model_file=str(MOD_DIR / "lightgbm.txt"))
        prob = model.predict(va[feats].astype("float32").values)
    else:
        pred_all = pd.read_csv(TAB_DIR / "07_验证集预测明细.csv")
        prob = pred_all[f"pred_{best}"].values

    y = va["label"].astype("int8").values
    print(f"[预测] 验证集 {len(prob):,} 条，平均概率 {prob.mean():.4f}，实际核销率 {y.mean():.4f}")

    sub = pd.DataFrame({
        "User_id": va["user_id"].values,
        "Coupon_id": va["coupon_id"].astype("int64").values,
        "Date_received": pd.to_datetime(va["date_received"]).dt.strftime("%Y%m%d"),
        "Probability": np.round(prob, 6),
    })
    sub = sub.sort_values("Probability", ascending=False).reset_index(drop=True)
    sub.to_csv(SUBMIT_FILE, index=False, encoding="utf-8")
    print(f"[保存] {SUBMIT_FILE}  （{len(sub):,} 行，可直接上传天池）")
    print(sub.head(5).to_string(index=False))

    q = pd.qcut(prob, 10, labels=False, duplicates="drop")
    dec = pd.DataFrame({"pred": prob, "y": y, "band": q})
    g = dec.groupby("band").agg(
        样本数=("y", "size"),
        平均预测概率=("pred", "mean"),
        实际核销率=("y", "mean"),
        核销人数=("y", "sum"),
    ).reset_index()
    g["实际/预测"] = g["实际核销率"] / g["平均预测概率"].replace(0, np.nan)
    g["档位"] = ["最低 10%", "10-20%", "20-30%", "30-40%", "40-50%",
                 "50-60%", "60-70%", "70-80%", "80-90%", "最高 10%"][:len(g)]
    g.round(4).to_csv(TAB_DIR / "10_概率分档决策表.csv", index=False, encoding="utf-8-sig")
    print("\n[业务决策表] 按预测概率分 10 档：")
    print(g[["档位", "样本数", "平均预测概率", "实际核销率"]].round(4).to_string(index=False))

    top = g.iloc[-1]
    bottom = g.iloc[0]
    lift = top["实际核销率"] / max(bottom["实际核销率"], 1e-9)
    print(f"\n[结论] 最高档核销率 {top['实际核销率']:.2%}，最低档 {bottom['实际核销率']:.2%}，"
          f"相差 {lift:.1f} 倍 —— 模型具备实际投放价值。")
    print(f"[校验] 全局 AUC = {roc_auc_score(y, prob):.4f}")

    with open(TAB_DIR / "10_提交摘要.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best, "rows": int(len(sub)),
                   "mean_prob": float(prob.mean()),
                   "top_band_rate": float(top["实际核销率"]),
                   "bottom_band_rate": float(bottom["实际核销率"])},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
