# -*- coding: utf-8 -*-
"""
00_下载数据.py —— 一键把原始数据下回来（不用注册天池账号）
====================================================================
天池官网的数据需要登录才能下。本项目用的是同一份官方数据在 ModelScope
（阿里自家另一个开放平台）上的公开镜像，字段与天池原版完全一致：

    表头：user_id, merchant_id, coupon_id, discount_rate, distance, date_received, date
    行数：1,754,883 行（不含表头），66MB

运行：python src/00_下载数据.py
输出：data/raw/ccf_offline_stage1_train.csv
"""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_TRAIN  # noqa: E402

URL = ("https://modelscope.cn/api/v1/datasets/fustudent0038/voucher/repo"
       "?Revision=master&FilePath=offline_train.csv")


def main():
    if RAW_TRAIN.exists() and RAW_TRAIN.stat().st_size > 60 * 1024 * 1024:
        print(f"[跳过] 数据已存在：{RAW_TRAIN} （{RAW_TRAIN.stat().st_size/1024**2:.0f} MB）")
        return
    print(f"[下载] {URL}")
    tmp = RAW_TRAIN.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.replace(RAW_TRAIN)
    print(f"[完成] {RAW_TRAIN} （{RAW_TRAIN.stat().st_size/1024**2:.0f} MB）")


if __name__ == "__main__":
    main()
