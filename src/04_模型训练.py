# -*- coding: utf-8 -*-
"""
04_模型训练.py —— 5 个模型 + 消融实验 + 官方口径评估
====================================================================
这一支是整个项目的心脏，做三件事：
  ① 训练 5 个模型并对比（冠军方案给的探索点 3：xgboost / lightgbm / gbdt 都试一遍）
        · LogisticRegression   线性基线（对照组）
        · LightGBM             工业界最常用的梯度提升树
        · XGBoost              冠军方案用的模型
        · HistGradientBoosting sklearn 自带 GBDT（不装第三方库也能跑）
        · LGB分组感知           本项目创新点：剔除券内常量特征
  ② 消融实验：特征一组一组加，看每一组到底贡献多少 AUC
        A 只用券面信息 → B 加用户/商家画像 → C 再加“只看过去”的时序特征（全量）
  ③ 用比赛官方口径评估：先按 Coupon_id 分组算 AUC，再对所有券求平均。

输入：data/features/train_features.pkl、valid_features.pkl
输出：outputs/tables/06_模型评估指标.csv、06_消融实验.csv、06_数据划分对比.csv、
      outputs/tables/07_验证集预测明细.csv、08/08b_特征重要性.csv、09_券级AUC_*.csv
      outputs/model/*.txt|json|joblib
运行：python src/04_模型训练.py
"""
import os
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    FEAT_TRAIN, FEAT_VALID, TAB_DIR, MOD_DIR, METRICS_FILE, PRED_FILE,
    LGB_PARAMS, LGB_ROUNDS, LGB_EARLY_STOP, XGB_PARAMS, XGB_ROUNDS, XGB_EARLY_STOP, SEED,
    VALID_SIZE_FOR_TUNING,
)

DROP = ["user_id", "merchant_id", "coupon_id", "coupon_type",
        "date_received", "consume_date", "days_gap", "is_coupon", "label"]


def get_xy(df, feats):
    return df[feats].astype("float32").values, df["label"].astype("int8").values


def group_auc(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, min_pos=1, min_neg=1):
    """
    比赛官方评估口径：
      对每个 Coupon_id 分组，组内算 AUC；只保留“正负样本都有”的组；最后对组的 AUC 取平均。
    返回 (平均AUC, 参与计算的券数量, 每个券的明细DataFrame)
    """
    tmp = pd.DataFrame({"coupon_id": df["coupon_id"].values, "y": y_true, "p": y_pred})
    rows = []
    for cid, g in tmp.groupby("coupon_id", observed=True):
        if g["y"].sum() < min_pos or (1 - g["y"]).sum() < min_neg:
            continue
        rows.append({"coupon_id": cid, "n": len(g), "pos": int(g["y"].sum()),
                     "auc": roc_auc_score(g["y"], g["p"])})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return float("nan"), 0, detail
    return float(detail["auc"].mean()), len(detail), detail


def lift_at(df, y_true, y_pred, k=0.1):
    """取预测分数最高的前 k 比例人群，看能捞到多少比例的核销人群（提升度）。"""
    n = int(len(df) * k)
    idx = np.argsort(-y_pred)[:n]
    captured = y_true[idx].sum() / max(y_true.sum(), 1)
    return float(captured / k)


def main():
    t0 = time.time()
    print("=" * 70)
    print("步骤 4 / 模型训练与评估")
    print("=" * 70)

    tr = pd.read_pickle(FEAT_TRAIN)
    va = pd.read_pickle(FEAT_VALID)
    feats = [c for c in tr.columns if c not in DROP and tr[c].dtype != object]
    print(f"[数据] 训练 {tr.shape} / 验证 {va.shape} / 特征 {len(feats)} 个")

    Xtr, ytr = get_xy(tr, feats)
    Xva, yva = get_xy(va, feats)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(tr))
    n_val = int(len(tr) * VALID_SIZE_FOR_TUNING)
    inner_tr, inner_va = idx[n_val:], idx[:n_val]

    results = []
    preds = {}

    # ==============================================================
    # 模型 1：逻辑回归
    # ==============================================================
    print("\n[1/5] 逻辑回归基线 ……")
    t = time.time()
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xtr_l = sc.fit_transform(imp.fit_transform(Xtr[inner_tr]))
    Xva_l = sc.transform(imp.transform(Xva))
    lr = LogisticRegression(max_iter=600, tol=1e-3, class_weight="balanced", solver="lbfgs", n_jobs=2)
    lr.fit(Xtr_l, ytr[inner_tr])
    p_lr = lr.predict_proba(Xva_l)[:, 1]
    preds["LogisticRegression"] = p_lr
    joblib.dump({"imputer": imp, "scaler": sc, "model": lr}, MOD_DIR / "logistic.joblib")
    print(f"      用时 {time.time()-t:.1f}s")

    # ==============================================================
    # 模型 2：LightGBM
    # ==============================================================
    print("[2/5] LightGBM ……")
    t = time.time()
    dtr = lgb.Dataset(Xtr[inner_tr], label=ytr[inner_tr], feature_name=feats)
    dva = lgb.Dataset(Xtr[inner_va], label=ytr[inner_va], feature_name=feats, reference=dtr)
    params = dict(LGB_PARAMS)
    params["scale_pos_weight"] = float((1 - ytr.mean()) / ytr.mean())
    lgb_model = lgb.train(
        params, dtr, num_boost_round=LGB_ROUNDS, valid_sets=[dva],
        callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    p_lgb = lgb_model.predict(Xva, num_iteration=lgb_model.best_iteration)
    preds["LightGBM"] = p_lgb
    lgb_model.save_model(str(MOD_DIR / "lightgbm.txt"))
    print(f"      最优轮数 {lgb_model.best_iteration}，用时 {time.time()-t:.1f}s")

    # ==============================================================
    # 模型 3：XGBoost
    # ==============================================================
    print("[3/5] XGBoost ……")
    t = time.time()
    xgb_params = dict(XGB_PARAMS)
    xgb_params["scale_pos_weight"] = float((1 - ytr.mean()) / ytr.mean())
    dtrx = xgb.DMatrix(Xtr[inner_tr], label=ytr[inner_tr], feature_names=feats)
    dvax = xgb.DMatrix(Xtr[inner_va], label=ytr[inner_va], feature_names=feats)
    xgb_model = xgb.train(
        xgb_params, dtrx, num_boost_round=XGB_ROUNDS,
        evals=[(dvax, "inner_valid")], early_stopping_rounds=XGB_EARLY_STOP, verbose_eval=False,
    )
    dvax_all = xgb.DMatrix(Xva, feature_names=feats)
    p_xgb = xgb_model.predict(dvax_all, iteration_range=(0, xgb_model.best_iteration + 1))
    preds["XGBoost"] = p_xgb
    xgb_model.save_model(str(MOD_DIR / "xgboost.json"))
    print(f"      最优轮数 {xgb_model.best_iteration}，用时 {time.time()-t:.1f}s")

    # ==============================================================
    # 模型 4：HistGradientBoosting
    # ==============================================================
    print("[4/5] HistGradientBoosting（sklearn 版 GBDT）……")
    t = time.time()
    hgb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=100, l2_regularization=1.0, random_state=SEED,
        class_weight="balanced", early_stopping=True, validation_fraction=0.2,
    )
    hgb.fit(Xtr[inner_tr], ytr[inner_tr])
    p_hgb = hgb.predict_proba(Xva)[:, 1]
    preds["HistGBDT"] = p_hgb
    joblib.dump(hgb, MOD_DIR / "hist_gbdt.joblib")
    print(f"      用时 {time.time()-t:.1f}s")

    # ==============================================================
    # 模型 5：分组感知版 LightGBM（本项目主要创新点）
    # ==============================================================
    print("[5/5] 分组感知版 LightGBM ……")
    t = time.time()
    coupon_const = [c for c in feats if c.startswith("c_")] + \
                   ["discount_strength", "min_spend", "cut_amount", "discount_rate_num", "is_manjian"]
    ga_feats = [c for c in feats if c not in coupon_const]
    gi_idx = [feats.index(c) for c in ga_feats]
    params_ga = dict(LGB_PARAMS)
    params_ga["scale_pos_weight"] = float((1 - ytr.mean()) / ytr.mean())
    m_ga = lgb.train(params_ga, lgb.Dataset(Xtr[inner_tr][:, gi_idx], label=ytr[inner_tr],
                                           feature_name=ga_feats),
                     num_boost_round=300)
    p_ga = m_ga.predict(Xva[:, gi_idx])
    preds["LGB分组感知"] = p_ga
    m_ga.save_model(str(MOD_DIR / "lightgbm_groupaware.txt"))
    pd.DataFrame({"特征": ga_feats,
                  "重要性": m_ga.feature_importance(importance_type="gain")}
                 ).sort_values("重要性", ascending=False).to_csv(
        TAB_DIR / "08b_特征重要性_分组感知.csv", index=False, encoding="utf-8-sig")
    print(f"      去掉 {len(coupon_const)} 个券内常量特征，剩 {len(ga_feats)} 个；用时 {time.time()-t:.1f}s")

    # ==============================================================
    # 评估
    # ==============================================================
    print("\n[评估] 开始按比赛口径评估 ……")
    coupon_auc_details = {}
    for name, p in preds.items():
        g_auc = roc_auc_score(yva, p)
        m_auc, n_coupon, detail = group_auc(va, yva, p)
        lift = lift_at(va, yva, p, k=0.1)
        results.append({
            "模型": name,
            "全局AUC": round(g_auc, 4),
            "官方口径_分组平均AUC": round(m_auc, 4),
            "参与评估的券数": n_coupon,
            "Top10%提升度": round(lift, 2),
        })
        coupon_auc_details[name] = detail
        print(f"  {name:<22} 全局AUC={g_auc:.4f}  分组平均AUC={m_auc:.4f}  Top10%提升={lift:.2f}")

    res = pd.DataFrame(results).sort_values("官方口径_分组平均AUC", ascending=False)
    res.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig")
    print("\n" + res.to_string(index=False))

    pred_df = va[["user_id", "merchant_id", "coupon_id", "date_received",
                  "label", "discount_strength", "distance"]].copy()
    for name, p in preds.items():
        pred_df[f"pred_{name}"] = p
    pred_df.to_csv(PRED_FILE, index=False, encoding="utf-8-sig")
    print(f"[保存] {PRED_FILE}")

    for name, detail in coupon_auc_details.items():
        detail.to_csv(TAB_DIR / f"09_券级AUC_{name}.csv", index=False, encoding="utf-8-sig")

    imp_df = pd.DataFrame({
        "特征": feats,
        "重要性": lgb_model.feature_importance(importance_type="gain"),
    }).sort_values("重要性", ascending=False)
    imp_df["重要性占比"] = imp_df["重要性"] / imp_df["重要性"].sum()
    imp_df.to_csv(TAB_DIR / "08_特征重要性.csv", index=False, encoding="utf-8-sig")
    print("\n[特征重要性 Top10]")
    print(imp_df.head(10).to_string(index=False))

    # ==============================================================
    # 消融实验
    # ==============================================================
    print("\n[消融] 特征分组增量实验 ……")

    base_feats = ["discount_strength", "min_spend", "cut_amount", "discount_rate_num",
                  "is_manjian", "distance", "day_of_month", "month", "day_of_week",
                  "is_weekend", "is_month_start", "is_month_end",
                  "days_since_start", "daily_recv_cnt"]
    time_feats = [c for c in feats if ("prior_" in c or c.endswith("_last_gap"))]
    profile_feats = [c for c in feats if c.startswith(("u_", "m_", "c_", "ui_")) and c not in time_feats]
    groups = {
        "A 仅券面+日期（基线）": base_feats,
        "B 加用户/商家/券画像": base_feats + profile_feats,
        "C 再加时序累计特征（全量）": feats,
    }
    abl_rows = []
    for gname, glist in groups.items():
        glist = [c for c in glist if c in feats]
        sub_idx = [feats.index(c) for c in glist]
        params2 = dict(LGB_PARAMS)
        params2["scale_pos_weight"] = float((1 - ytr.mean()) / ytr.mean())
        m2 = lgb.train(params2, lgb.Dataset(Xtr[:, sub_idx], label=ytr), num_boost_round=300)
        p2 = m2.predict(Xva[:, sub_idx])
        g_auc = roc_auc_score(yva, p2)
        m_auc, n_coupon, _ = group_auc(va, yva, p2)
        abl_rows.append({"特征组合": gname, "特征数": len(glist),
                         "全局AUC": round(g_auc, 4),
                         "官方口径_分组平均AUC": round(m_auc, 4)})
        print(f"  {gname:<24} 特征{len(glist):>3}个  全局AUC={g_auc:.4f}  分组AUC={m_auc:.4f}")
        del m2
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(TAB_DIR / "06_消融实验.csv", index=False, encoding="utf-8-sig")

    # ==============================================================
    # 数据划分实验
    # ==============================================================
    print("\n[数据划分] 随机划分 vs 时间划分 ……")
    allx = np.vstack([Xtr, Xva])
    ally = np.concatenate([ytr, yva])
    rng2 = np.random.default_rng(SEED)
    p2 = rng2.permutation(len(ally))
    n = int(len(ally) * 0.2)
    ridx_tr, ridx_va = p2[n:], p2[:n]
    params3 = dict(LGB_PARAMS)
    params3["scale_pos_weight"] = float((1 - ally[ridx_tr].mean()) / ally[ridx_tr].mean())
    m3 = lgb.train(params3, lgb.Dataset(allx[ridx_tr], label=ally[ridx_tr]), num_boost_round=300)
    p3 = m3.predict(allx[ridx_va])
    random_auc = roc_auc_score(ally[ridx_va], p3)
    params4 = dict(LGB_PARAMS)
    params4["scale_pos_weight"] = float((1 - ytr.mean()) / ytr.mean())
    m4 = lgb.train(params4, lgb.Dataset(Xtr, label=ytr), num_boost_round=300)
    p4 = m4.predict(Xva)
    time_auc = roc_auc_score(yva, p4)
    split_df = pd.DataFrame([
        {"划分方式": "随机划分（同一时间段内随机抽 20%）", "全局AUC": round(random_auc, 4),
         "结论": "分数虚高：模型见过同一批用户/商家在未来的表现"},
        {"划分方式": "时间划分（1-4月训练，5-6月验证）", "全局AUC": round(time_auc, 4),
         "结论": "更接近真实上线场景，是本次采用的方案"},
    ])
    split_df.to_csv(TAB_DIR / "06_数据划分对比.csv", index=False, encoding="utf-8-sig")
    print(split_df.to_string(index=False))

    print(f"\n[完成] 总用时 {time.time()-t0:.1f}s")
    print("[提示] 下一步：python src/05_预测与提交.py")


if __name__ == "__main__":
    main()
