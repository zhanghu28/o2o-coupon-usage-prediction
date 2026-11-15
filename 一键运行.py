# -*- coding: utf-8 -*-
"""
一键运行.py —— 从下载数据到出图表，一条命令跑完全流程
====================================================================
用法（在项目根目录）：
    python 一键运行.py            # 全流程
    python 一键运行.py 1 2 3      # 只跑第 1、2、3 步

每一步也可以单独运行，例如：python src/03_特征工程.py
"""
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "src")

STEPS = [
    ("00_下载数据.py", "下载原始数据"),
    ("01_数据清洗.py", "数据清洗 + 质量监测"),
    ("02_数据集成.py", "构建 4 张维度表并合并宽表"),
    ("03_特征工程.py", "时序特征 + 训练/验证切分"),
    ("04_模型训练.py", "训练 5 个模型 + 消融实验"),
    ("05_预测与提交.py", "生成提交文件 + 业务决策表"),
    ("06_可视化.py", "生成 12 张图表"),
    ("07_生成结论报告.py", "汇总结论"),
]


def main():
    only = set(sys.argv[1:])
    t0 = time.time()
    for i, (script, desc) in enumerate(STEPS, start=1):
        if only and str(i - 1) not in only and str(i) not in only:
            continue
        print("\n" + "=" * 70)
        print(f"[{i}/{len(STEPS)}] {desc}   ← {script}")
        print("=" * 70)
        code = subprocess.call([sys.executable, os.path.join(SRC, script)], cwd=BASE)
        if code != 0:
            print(f"[中断] {script} 运行失败（退出码 {code}），后面的步骤已停止。")
            sys.exit(code)
    print(f"\n全部完成，总用时 {time.time()-t0:.1f} 秒。结果在 outputs/ 目录。")


if __name__ == "__main__":
    main()
