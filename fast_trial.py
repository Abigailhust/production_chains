#!/usr/bin/env python3
"""
Fast running version - using optimized parameters
"""

from rp import RPline
import matplotlib.pyplot as plt
import numpy as np
import time

def make_cost_three_args(theta=5.0,
                         gamma_p=0.8, rho_p=1.0,
                         gamma_c=0.2, rho_c=1.0,
                         mu=0.2, psi=1.5):
    """
    Return cost function c(s, t, a).
    - production cost: (exp(theta * x) - 1) * exp(-gamma_p * a * x**rho_p)
    - transaction cost: mu * t**psi * exp(-gamma_c * a * t**rho_c)
    Higher automation level a reduces costs.
    """
    def c(s, t, a):
        x = s - t
        x = np.clip(s - t, 1e-8, None)  # make sure >= 1e-8
        t = np.clip(t, 1e-8, None)
        prod = (np.exp(theta * x) - 1.0) * np.exp(-gamma_p * a * (x ** rho_p))
        coord = mu * (t ** psi) * np.exp(-gamma_c * a * (t ** rho_c))
        return prod + coord
    return c

def run_fast_example():
    """Run fast example"""
    print("=== Fast Running Example ===")
    
    # Use optimized parameters
    start_time = time.time()
    
    # Create cost function with automation reducing costs
    c3 = make_cost_three_args(
        theta=10.0,           # Production cost parameter
        gamma_p=0.8, rho_p=1.0,   # Automation reduces production costs
        gamma_c=0.3, rho_c=1.0,   # Automation reduces coordination costs
        mu=0.25, psi=1.4
    )
    
    # Create fast version
    ps = RPline(
        n=100,        # Reduce grid points
        delta=1.1,    # Lower delta value
        sbar=0.5,     # Reduce scale range
        a=0.5,        # Set automation level
        c_list=[c3]   # Use the new cost function
    )
    
    print(f"Creation time: {time.time() - start_time:.2f} seconds")
    
    # Calculate transaction stages
    stage_start = time.time()
    ts = ps.compute_stages()
    print(f"Transaction stage calculation time: {time.time() - stage_start:.2f} seconds")
    print(f"Transaction stages: {ts}")
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Main plot: Price function
    plt.subplot(2, 2, 1)
    ps.plot_prices(plot_stages=True, label='Price Function')
    plt.title('Price Function (Fast Version)')
    plt.xlabel('Scale s')
    plt.ylabel('Price p(s)')
    plt.grid(True)
    plt.legend()
    
    # Subplot 1: t_star function
    plt.subplot(2, 2, 2)
    ps.plot_t_star()
    plt.title('Optimal Transaction Volume t*(s)')
    plt.xlabel('Scale s')
    plt.ylabel('Transaction Volume t*(s)')
    plt.grid(True)
    
    # Subplot 2: ell_star function
    plt.subplot(2, 2, 3)
    ps.plot_ell_star()
    plt.title('Optimal Internal Production ℓ*(s)')
    plt.xlabel('Scale s')
    plt.ylabel('Internal Production ℓ*(s)')
    plt.grid(True)
    
    # Subplot 3: Cost function
    plt.subplot(2, 2, 4)
    s_values = np.linspace(0, ps.sbar, 100)
    t_values = np.linspace(0, ps.sbar, 100)
    S, T = np.meshgrid(s_values, t_values)
    C = np.array([[ps._cost(s, t) for t in t_values] for s in s_values])
    
    plt.contourf(S, T, C, levels=20)
    plt.colorbar(label='Cost')
    plt.title('Cost Function c(s,t)')
    plt.xlabel('Scale s')
    plt.ylabel('Transaction Volume t')
    
    plt.tight_layout()
    plt.savefig('fast_trial_output.png', dpi=150, bbox_inches='tight')
    print("Image saved as fast_trial_output.png")
    
    total_time = time.time() - start_time
    print(f"Total computation time: {total_time:.2f} seconds")
    print("✅ Fast example completed!")

def compare_parameters():
    """Compare the effects of different parameters"""
    print("\n=== Parameter Comparison ===")
    
    # Test different parameter combinations
    test_cases = [
        {"name": "Fast Version", "n": 50, "delta": 1.1, "sbar": 0.5, "theta": 8.0},
        {"name": "Medium Version", "n": 100, "delta": 1.2, "sbar": 1.0, "theta": 10.0},
        {"name": "Standard Version", "n": 200, "delta": 1.2, "sbar": 1.0, "theta": 12.0}
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\nTesting {case['name']}...")
        start = time.time()
        
        # Create cost function for this case
        c3 = make_cost_three_args(
            theta=case['theta'],
            gamma_p=0.8, rho_p=1.0,
            gamma_c=0.3, rho_c=1.0,
            mu=0.25, psi=1.4
        )
        
        ps = RPline(
            n=case['n'],
            delta=case['delta'],
            sbar=case['sbar'],
            a=0.5,  # Set automation level
            c_list=[c3]
        )
        
        end = time.time()
        results.append({
            "name": case['name'],
            "time": end - start,
            "grid_size": case['n'],
            "delta": case['delta']
        })
        
        print(f"  Computation time: {end - start:.2f} seconds")
        print(f"  Grid size: {case['n']}")
        print(f"  Delta: {case['delta']}")
    
    print(f"\n=== Comparison Results ===")
    for result in results:
        print(f"{result['name']}: {result['time']:.2f} seconds (n={result['grid_size']}, δ={result['delta']})")

def test_automation_levels():
    """Test different automation levels"""
    print("\n=== Automation Level Test ===")
    
    # Test different automation levels
    automation_levels = [0.0, 0.2, 0.5, 0.8, 1.0]
    
    for a in automation_levels:
        print(f"\nTesting automation level a = {a}")
        start = time.time()
        
        # Create cost function with automation reducing costs
        c3 = make_cost_three_args(
            theta=10.0,
            gamma_p=0.8, rho_p=1.0,
            gamma_c=0.3, rho_c=1.0,
            mu=0.25, psi=1.4
        )
        
        ps = RPline(
            n=100,
            delta=1.1,
            sbar=0.5,
            a=a,  # Set automation level
            c_list=[c3]
        )
        
        end = time.time()
        print(f"  Computation time: {end - start:.2f} seconds")
        print(f"  Automation level: {a}")
        
        # Calculate some key metrics
        ts = ps.compute_stages()
        print(f"  Number of transaction stages: {len(ts)}")
        if len(ts) > 1:
            print(f"  First transaction stage: {ts[1]:.4f}")

def visualize_automation_effects():
    """Visualize how automation affects costs and firm boundaries"""
    print("\n=== Automation Effects Visualization ===")
    
    # Create cost function
    c3 = make_cost_three_args(
        theta=10.0,
        gamma_p=0.8, rho_p=1.0,
        gamma_c=0.3, rho_c=1.0,
        mu=0.25, psi=1.4
    )
    
    # Test different automation levels
    automation_levels = [0.0, 0.3, 0.6, 0.9]
    colors = ['red', 'blue', 'green', 'purple']
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Cost functions for different automation levels
    plt.subplot(2, 3, 1)
    s_values = np.linspace(0, 1, 100)
    t_values = np.linspace(0, 1, 100)
    S, T = np.meshgrid(s_values, t_values)
    
    for i, a in enumerate(automation_levels):
        C = np.array([[c3(s, t, a) for t in t_values] for s in s_values])
        plt.contour(S, T, C, levels=10, colors=colors[i], alpha=0.7, label=f'a={a}')
    
    plt.title('Cost Functions at Different Automation Levels')
    plt.xlabel('Scale s')
    plt.ylabel('Transaction Volume t')
    plt.legend()
    plt.colorbar(label='Cost')
    
    # Plot 2: Price functions
    plt.subplot(2, 3, 2)
    for i, a in enumerate(automation_levels):
        ps = RPline(n=100, delta=1.1, sbar=1.0, a=a, c_list=[c3])
        plt.plot(ps.grid, ps.p, color=colors[i], label=f'a={a}')
    
    plt.title('Price Functions')
    plt.xlabel('Scale s')
    plt.ylabel('Price p(s)')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Number of transaction stages vs automation
    plt.subplot(2, 3, 3)
    automation_range = np.linspace(0, 1, 20)
    stage_counts = []
    
    for a in automation_range:
        ps = RPline(n=100, delta=1.1, sbar=1.0, a=a, c_list=[c3])
        ts = ps.compute_stages()
        stage_counts.append(len(ts))
    
    plt.plot(automation_range, stage_counts, 'b-', linewidth=2)
    plt.title('Firm Boundaries vs Automation')
    plt.xlabel('Automation Level a')
    plt.ylabel('Number of Transaction Stages')
    plt.grid(True)
    
    # Plot 4: Cost reduction effect
    plt.subplot(2, 3, 4)
    s_test = 0.5
    t_test = 0.2
    costs = [c3(s_test, t_test, a) for a in automation_range]
    cost_reduction = [(costs[0] - cost) / costs[0] * 100 for cost in costs]
    
    plt.plot(automation_range, cost_reduction, 'g-', linewidth=2)
    plt.title('Cost Reduction Effect')
    plt.xlabel('Automation Level a')
    plt.ylabel('Cost Reduction (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('automation_effects.png', dpi=150, bbox_inches='tight')
    print("Automation effects visualization saved as automation_effects.png")

if __name__ == "__main__":
    run_fast_example()
    compare_parameters()
    test_automation_levels()
    visualize_automation_effects() 