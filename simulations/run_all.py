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
    # Override knobs: default None -> keep the existing per-sim defaults below.
    parser.add_argument("--M", type=int, default=None, help="Override MC reps for all sims")
    parser.add_argument("--n", type=int, default=None, help="Override sample size for all sims")
    parser.add_argument("--n-folds", type=int, default=None, help="Override n_folds for all sims")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all sims")
    parser.add_argument("--patience", type=int, default=None, help="Override patience for all sims")
    parser.add_argument("--n-repeats", type=int, default=1,
                        help="Repeated cross-fitting splits for sim 2 (median DML)")
    parser.add_argument("--lambda-method", type=str, default=None,
                        help="Lambda strategy override for sim 2")
    args = parser.parse_args()

    def ov(cli, default):
        """CLI value if provided, else the existing per-sim default."""
        return cli if cli is not None else default

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
    d1 = dict(M=5, n=5000, n_folds=20, epochs=50, patience=30) if args.quick \
        else dict(M=20, n=10000, n_folds=20, epochs=100, patience=30)
    metrics1, _ = run_sim1(
        M=ov(args.M, d1["M"]), n=ov(args.n, d1["n"]), n_folds=ov(args.n_folds, d1["n_folds"]),
        epochs=ov(args.epochs, d1["epochs"]), patience=ov(args.patience, d1["patience"]),
    )
    all_results["GRU + Poisson"] = metrics1

    # Sim 2: CNN + Logit (also takes n_repeats / lambda_method)
    from simulations.sim_02_image import run_simulation as run_sim2
    d2 = dict(M=5, n=5000, n_folds=20, epochs=50, patience=30) if args.quick \
        else dict(M=20, n=10000, n_folds=20, epochs=150, patience=30)
    metrics2, _ = run_sim2(
        M=ov(args.M, d2["M"]), n=ov(args.n, d2["n"]), n_folds=ov(args.n_folds, d2["n_folds"]),
        epochs=ov(args.epochs, d2["epochs"]), patience=ov(args.patience, d2["patience"]),
        n_repeats=args.n_repeats, lambda_method=args.lambda_method,
    )
    all_results["CNN + Logit"] = metrics2

    # Sim 3: Transformer + Logit
    from simulations.sim_03_text import run_simulation as run_sim3
    d3 = dict(M=5, n=5000, n_folds=20, epochs=100, patience=30) if args.quick \
        else dict(M=20, n=10000, n_folds=50, epochs=200, patience=50)
    metrics3, _ = run_sim3(
        M=ov(args.M, d3["M"]), n=ov(args.n, d3["n"]), n_folds=ov(args.n_folds, d3["n_folds"]),
        epochs=ov(args.epochs, d3["epochs"]), patience=ov(args.patience, d3["patience"]),
    )
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
