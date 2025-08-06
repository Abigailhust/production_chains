#!/usr/bin/env python3
"""
快速运行版本 - 使用优化的参数设置
"""

from rp import RPline
import matplotlib.pyplot as plt
import numpy as np
import time

def run_fast_example():
    """运行快速示例"""
    print("=== 快速运行示例 ===")
    
    # 使用优化的参数
    start_time = time.time()
    
    # 创建快速版本
    ps = RPline(
        n=100,        # 减少网格点数
        delta=1.1,    # 降低delta值
        sbar=0.5,     # 减少规模范围
        c_list=[lambda s, t, a: np.exp(5*(s-t)) - 1]  # 降低指数系数
    )
    
    print(f"创建时间: {time.time() - start_time:.2f}秒")
    
    # 计算交易阶段
    stage_start = time.time()
    ts = ps.compute_stages()
    print(f"交易阶段计算时间: {time.time() - stage_start:.2f}秒")
    print(f"交易阶段: {ts}")
    
    # 绘图
    plt.figure(figsize=(12, 8))
    
    # 主图：价格函数
    plt.subplot(2, 2, 1)
    ps.plot_prices(plot_stages=True, label='价格函数')
    plt.title('价格函数 (快速版本)')
    plt.xlabel('规模 s')
    plt.ylabel('价格 p(s)')
    plt.grid(True)
    plt.legend()
    
    # 子图1：t_star函数
    plt.subplot(2, 2, 2)
    ps.plot_t_star()
    plt.title('最优交易量 t*(s)')
    plt.xlabel('规模 s')
    plt.ylabel('交易量 t*(s)')
    plt.grid(True)
    
    # 子图2：ell_star函数
    plt.subplot(2, 2, 3)
    ps.plot_ell_star()
    plt.title('最优内部生产量 ℓ*(s)')
    plt.xlabel('规模 s')
    plt.ylabel('内部生产量 ℓ*(s)')
    plt.grid(True)
    
    # 子图3：成本函数
    plt.subplot(2, 2, 4)
    s_values = np.linspace(0, ps.sbar, 100)
    t_values = np.linspace(0, ps.sbar, 100)
    S, T = np.meshgrid(s_values, t_values)
    C = np.array([[ps._cost(s, t) for t in t_values] for s in s_values])
    
    plt.contourf(S, T, C, levels=20)
    plt.colorbar(label='成本')
    plt.title('成本函数 c(s,t)')
    plt.xlabel('规模 s')
    plt.ylabel('交易量 t')
    
    plt.tight_layout()
    plt.savefig('fast_trial_output.png', dpi=150, bbox_inches='tight')
    print("图片已保存为 fast_trial_output.png")
    
    total_time = time.time() - start_time
    print(f"总计算时间: {total_time:.2f}秒")
    print("✅ 快速示例运行完成！")

def compare_parameters():
    """比较不同参数的影响"""
    print("\n=== 参数比较 ===")
    
    # 测试不同的参数组合
    test_cases = [
        {"name": "快速版本", "n": 50, "delta": 1.1, "sbar": 0.5, "exp_coeff": 5},
        {"name": "中等版本", "n": 100, "delta": 1.2, "sbar": 1.0, "exp_coeff": 8},
        {"name": "标准版本", "n": 200, "delta": 1.2, "sbar": 1.0, "exp_coeff": 10}
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n测试 {case['name']}...")
        start = time.time()
        
        ps = RPline(
            n=case['n'],
            delta=case['delta'],
            sbar=case['sbar'],
            c_list=[lambda s, t, a, coeff=case['exp_coeff']: np.exp(coeff*(s-t)) - 1]
        )
        
        end = time.time()
        results.append({
            "name": case['name'],
            "time": end - start,
            "grid_size": case['n'],
            "delta": case['delta']
        })
        
        print(f"  计算时间: {end - start:.2f}秒")
        print(f"  网格大小: {case['n']}")
        print(f"  Delta: {case['delta']}")
    
    print(f"\n=== 比较结果 ===")
    for result in results:
        print(f"{result['name']}: {result['time']:.2f}秒 (n={result['grid_size']}, δ={result['delta']})")

if __name__ == "__main__":
    run_fast_example()
    compare_parameters() 