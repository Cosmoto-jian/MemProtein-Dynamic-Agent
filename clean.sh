#!/usr/bin/env bash
# 一键清除所有"生成"出来的产物,回到干净状态。
# 删除:生成的模型输入(data/inputs)、仿真结果与图(data/results)、python 缓存。
# 保留:所有源码(memprotein/, cli.py, viz/)、原始数据(data/raw 的 pdb + tm)。
#
# 用法:  bash clean.sh

cd "$(dirname "$0")" || exit 1

echo "清除生成产物..."
rm -f data/inputs/*
rm -f data/results/*
find . -path ./.venv -prune -o -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null

echo "已清除:"
echo "  - data/inputs/  (生成的 MODEL / targetNode / mass / evector)"
echo "  - data/results/ (仿真结果 *.h5 + 分析图/数据)"
echo "  - __pycache__ 缓存"
echo "完成。重新生成:python cli.py run --pdb data/raw/<x>.pdb --tm data/raw/<x>_tm.txt"
