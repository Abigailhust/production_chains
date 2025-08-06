#!/usr/bin/env python3
"""
快速测试脚本 - 验证RPline类是否正常工作
"""

from rp import RPline
import matplotlib.pyplot as plt
import numpy as np
import time

def test_rpline():
    """测试RPline类的基本功能"""
    print("开始测试RPline...")
    
    # 使用更少的网格点来加快计算
    start_time = time.time()
    
    # 测试1: 快速版本
    print("创建快速版本 (n=50)...")
    ps_fast = RPline(n=50, delta=1.1, sbar=0.5)
    print(f"创建时间: {time.time() - start_time:.2f}秒")
    
    # 测试2: 计算交易阶段
    print("计算交易阶段...")
    stage_start = time.time()
    ts = ps_fast.compute_stages()
    print(f"交易阶段计算时间: {time.time() - stage_start:.2f}秒")
    print(f"交易阶段: {ts}")
    
    # 测试3: 绘图
    print("绘制价格函数...")
    plt.figure(figsize=(10, 6))
    ps_fast.plot_prices(plot_stages=True, label='快速版本')
    plt.title('价格函数 (快速测试版本)')
    plt.xlabel('规模 s')
    plt.ylabel('价格 p(s)')
    plt.grid(True)
    plt.legend()
    
    # 保存图片
    plt.savefig('quick_test_output.png', dpi=150, bbox_inches='tight')
    print("图片已保存为 quick_test_output.png")
    
    total_time = time.time() - start_time
    print(f"总计算时间: {total_time:.2f}秒")
    print("测试完成！")

def test_parameters():
    """测试不同参数的影响"""
    print("\n测试不同参数...")
    
    # 测试不同的网格密度
    grid_sizes = [20, 50, 100]
    times = []
    
    for n in grid_sizes:
        print(f"测试网格大小 n={n}...")
        start = time.time()
        ps = RPline(n=n, delta=1.1, sbar=0.5)
        end = time.time()
        times.append(end - start)
        print(f"  计算时间: {end - start:.2f}秒")
    
    print(f"\n网格大小 vs 计算时间:")
    for n, t in zip(grid_sizes, times):
        print(f"  n={n}: {t:.2f}秒")

if __name__ == "__main__":
    test_rpline()
    test_parameters() 