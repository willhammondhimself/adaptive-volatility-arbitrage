#!/usr/bin/env python3
"""
Benchmark: C++ Heston FFT pricing performance.

Compares C++ (pybind11) implementation across different workloads.

Usage:
    python benchmarks/latency_benchmark.py
    python benchmarks/latency_benchmark.py --iterations 1000 --save-plot docs/benchmark.png

Expected results (M1 MacBook Pro):
    C++ single option:      ~0.1ms
    C++ 40 strikes:         ~3ms
    C++ 800 prices (40x20): ~60ms
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Try to import C++ module
try:
    import heston_cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    print("C++ module not found. Build with: cd cpp_heston && pip install .")
    print("Benchmark requires the C++ module.")
    sys.exit(1)


# Default Heston parameters (typical S&P 500 calibration)
DEFAULT_PARAMS = heston_cpp.HestonParams(
    0.04,       # v0: 20% initial vol
    0.05,       # theta: 22.4% long-run vol
    2.0,        # kappa: Mean reversion speed
    0.3,        # sigma_v: Vol of vol
    -0.7,       # rho: Spot-variance correlation
    0.05,       # r: Risk-free rate
    0.02,       # q: Dividend yield
)


def benchmark_single_option(n_iterations):
    """Benchmark single option pricing."""
    S, K, T = 100.0, 100.0, 1.0

    start = time.perf_counter()
    for _ in range(n_iterations):
        price = heston_cpp.price_call(S, K, T, DEFAULT_PARAMS)
    elapsed = time.perf_counter() - start

    return elapsed, price


def benchmark_strike_grid(n_iterations, n_strikes=40):
    """Benchmark pricing across a strike grid."""
    S, T = 100.0, 1.0
    strikes = list(np.linspace(80, 120, n_strikes))

    start = time.perf_counter()
    for _ in range(n_iterations):
        prices = heston_cpp.price_strikes(S, strikes, T, DEFAULT_PARAMS, True)
    elapsed = time.perf_counter() - start

    return elapsed, prices


def benchmark_surface(n_iterations, n_strikes=40, n_maturities=20):
    """Benchmark full surface pricing (strikes x maturities)."""
    S = 100.0
    strikes = list(np.linspace(80, 120, n_strikes))
    maturities = np.linspace(0.25, 2.0, n_maturities)

    start = time.perf_counter()
    for _ in range(n_iterations):
        surface = []
        for T in maturities:
            prices = heston_cpp.price_strikes(S, strikes, T, DEFAULT_PARAMS, True)
            surface.append(prices)
    elapsed = time.perf_counter() - start

    return elapsed, surface


def run_benchmark(n_iterations=100):
    """Run full benchmark suite."""
    print("=" * 60)
    print("HESTON FFT C++ LATENCY BENCHMARK")
    print("=" * 60)
    print(f"Iterations: {n_iterations}")
    print(f"Feller condition satisfied: {DEFAULT_PARAMS.feller_satisfied()}")
    print()

    results = {}

    # Single option
    print("Benchmarking single option pricing...")
    t1, p1 = benchmark_single_option(n_iterations)
    results["single_option"] = t1
    print(f"  Total: {t1*1000:.1f}ms ({t1/n_iterations*1000:.4f}ms/price)")
    print(f"  Sample price: ${p1:.4f}")
    print()

    # Strike grid (40 strikes)
    print("Benchmarking 40-strike grid...")
    t2, p2 = benchmark_strike_grid(n_iterations, n_strikes=40)
    results["strike_grid"] = t2
    print(f"  Total: {t2*1000:.1f}ms ({t2/n_iterations*1000:.3f}ms/grid)")
    print(f"  Sample ATM price: ${p2[20]:.4f}")
    print()

    # Full surface (40x20 = 800 prices)
    print("Benchmarking 40x20 surface (800 prices)...")
    t3, p3 = benchmark_surface(n_iterations // 10 or 1)  # Fewer iterations for surface
    results["surface"] = t3
    actual_iters = n_iterations // 10 or 1
    print(f"  Total: {t3*1000:.1f}ms ({t3/actual_iters*1000:.1f}ms/surface)")
    print()

    # Summary
    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"{'Operation':<25} {'Time/Op':>12} {'Throughput':>15}")
    print("-" * 60)

    single_ms = t1 / n_iterations * 1000
    grid_ms = t2 / n_iterations * 1000
    surface_ms = t3 / (n_iterations // 10 or 1) * 1000

    print(f"{'Single option':<25} {single_ms:>10.4f}ms {1000/single_ms:>12.0f} ops/sec")
    print(f"{'40-strike grid':<25} {grid_ms:>10.3f}ms {1000/grid_ms*40:>12.0f} prices/sec")
    print(f"{'40x20 surface (800)':<25} {surface_ms:>10.1f}ms {1000/surface_ms*800:>12.0f} prices/sec")

    return results


def plot_results(results, save_path=None):
    """Generate performance chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    labels = ["Single\nOption", "40-Strike\nGrid", "800-Price\nSurface"]
    times = [
        results["single_option"] / 100 * 1000,  # Normalize to ms/op
        results["strike_grid"] / 100 * 1000,
        results["surface"] / 10 * 1000,
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, times, color=["#3498db", "#2ecc71", "#e74c3c"])

    ax.set_ylabel("Time per operation (ms)")
    ax.set_title("C++ Heston FFT Pricing Performance")
    ax.set_yscale("log")

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f"{t:.2f}ms", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Heston FFT C++ Latency Benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=100,
                        help="Number of iterations (default: 100)")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Save plot to file")
    args = parser.parse_args()

    results = run_benchmark(n_iterations=args.iterations)

    if args.save_plot:
        plot_results(results, save_path=args.save_plot)


if __name__ == "__main__":
    main()
