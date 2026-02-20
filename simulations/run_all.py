"""
Run all three semi-synthetic simulations and generate combined report.

Usage:
    python -m simulations.run_all
    python -m simulations.run_all --quick
"""

import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(description="Run all simulations")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: M=5, reduced n and epochs")
    args = parser.parse_args()

    report_dir = Path(__file__).parent.parent / "evals" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"sim_all_{timestamp}.txt"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    fh = open(report_file, "w")
    sys.stdout = Tee(sys.__stdout__, fh)

    print(f"{'='*60}")
    print(f"  SEMI-SYNTHETIC SIMULATION SUITE")
    print(f"  {datetime.now().isoformat()}")
    print(f"{'='*60}")

    all_results = {}

    # Sim 1: GRU + Poisson
    from simulations.sim_01_sequential import run_simulation as run_sim1
    if args.quick:
        metrics1, _ = run_sim1(M=5, n=5000, n_folds=20, epochs=50, patience=30)
    else:
        metrics1, _ = run_sim1(M=20, n=10000, n_folds=20, epochs=100, patience=30)
    all_results["GRU + Poisson"] = metrics1

    # Sim 2: CNN + Logit
    from simulations.sim_02_image import run_simulation as run_sim2
    if args.quick:
        metrics2, _ = run_sim2(M=5, n=5000, n_folds=20, epochs=50, patience=30)
    else:
        metrics2, _ = run_sim2(M=20, n=10000, n_folds=20, epochs=150, patience=30)
    all_results["CNN + Logit"] = metrics2

    # Sim 3: Transformer + Logit
    from simulations.sim_03_text import run_simulation as run_sim3
    if args.quick:
        metrics3, _ = run_sim3(M=5, n=5000, n_folds=20, epochs=100, patience=30)
    else:
        metrics3, _ = run_sim3(M=20, n=10000, n_folds=50, epochs=200, patience=50)
    all_results["SBERT + Logit"] = metrics3

    # Combined table
    print(f"\n{'='*70}")
    print(f"  COMBINED RESULTS")
    print(f"{'='*70}")
    print(f"  {'Simulation':<20} {'Coverage':>10} {'SE Ratio':>10} {'|Bias|':>10} {'z-mean':>10}")
    print(f"  {'-'*60}")
    for name, m in all_results.items():
        status = "PASS" if 0.90 <= m['coverage'] <= 0.99 else "FAIL"
        print(f"  {name:<20} {m['coverage']*100:>9.1f}% {m['se_ratio']:>10.3f} "
              f"{abs(m['bias']):>10.4f} {m['z_mean']:>10.3f}  {status}")
    print(f"{'='*70}")

    fh.close()
    sys.stdout = sys.__stdout__
    print(f"\nReport saved to: {report_file.resolve()}")


if __name__ == "__main__":
    main()
