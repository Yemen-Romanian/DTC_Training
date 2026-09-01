"""Run a chain of training experiments back-to-back off one base config.

Each run is the same trainer invoked with different --set / --unset overrides, so a variation
costs a command-line flag rather than an edit to configs/training.toml:

    python run_experiments.py --config configs/training.toml --sweep training_params.seed=1,2,3,4,5

A regime is one invocation: --sweep varies the thing being replicated, while --set / --unset
pin what that whole chain holds fixed.

Runs execute sequentially, each in its own process. That is deliberate: the CUDA context and
the persistent DataLoader worker pool are torn down between runs (there is little commit-charge
headroom at 15 workers), the RNG and MLflow run state start clean, and a crash costs one run
rather than the chain.
"""
import argparse
import itertools
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
OUTPUT_DIR = REPO_ROOT / "output"


@dataclass
class Run:
    name: str
    config_path: Path
    sets: list = field(default_factory=list)
    unsets: list = field(default_factory=list)

    def command(self) -> list:
        cmd = [sys.executable, "-m", "models.trainers.train_tracker",
               "--config_path", str(self.config_path), "--run_name", self.name]
        for override in self.sets:
            cmd += ["--set", override]
        for key in self.unsets:
            cmd += ["--unset", key]
        return cmd


def slug(text) -> str:
    """Make a value safe for a directory name: 0.005 -> 0-005."""
    return re.sub(r'[^A-Za-z0-9]+', '-', str(text)).strip('-') or "value"


def resolve_config(path) -> Path:
    """Config paths are given relative to the repo root, which is where the runner is invoked."""
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def runs_from_inline(args) -> list:
    """Cartesian product over every --sweep axis, sharing the common --set/--unset."""
    axes = []
    for spec in args.sweep:
        key, separator, values = spec.partition('=')
        if not separator or not values:
            raise SystemExit(f"--sweep {spec!r} is not of the form key=v1,v2,v3")
        axes.append((key.strip(), [value.strip() for value in values.split(',')]))

    config_path = resolve_config(args.config)
    runs = []
    for combination in itertools.product(*[values for _, values in axes]):
        name = "__".join(f"{key.rsplit('.', 1)[-1]}_{slug(value)}"
                         for (key, _), value in zip(axes, combination))
        if args.name_prefix:
            name = f"{args.name_prefix}_{name}"
        sets = list(args.overrides) + [f"{key}={value}"
                                       for (key, _), value in zip(axes, combination)]
        runs.append(Run(name=name, config_path=config_path, sets=sets, unsets=list(args.removals)))
    return runs


def find_run_dir(run: Run) -> str:
    """The output/ folder this run produced. The trainer appends a timestamp to --run_name."""
    candidates = [path for path in OUTPUT_DIR.glob(f"{run.name}_*") if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime).name if candidates else "-"


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def print_summary(results) -> None:
    width = max([len(name) for name, _, _, _ in results] + [3])
    print("\n" + "=" * (width + 40))
    print(f"{'RUN'.ljust(width)}  {'STATUS':<12} {'TIME':>9}  OUTPUT")
    for name, status, elapsed, run_dir in results:
        print(f"{name.ljust(width)}  {status:<12} {format_duration(elapsed):>9}  {run_dir}")
    print("=" * (width + 40))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a chain of training experiments sequentially.",
        epilog="Example: --config configs/training.toml --sweep training_params.seed=1,2,3,4,5 "
               "--set training_params.epochs_num=30",
    )
    parser.add_argument('--config', type=str, required=True,
                        help="Base training config, relative to the repo root.")
    parser.add_argument('--sweep', action='append', default=[], metavar='KEY=V1,V2',
                        help="Vary a config key over a list of values; repeatable, and multiple "
                             "axes take the cartesian product.")
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                        help="Override applied to every run in the chain; repeatable.")
    parser.add_argument('--unset', dest='removals', action='append', default=[], metavar='KEY',
                        help="Key removed from every run in the chain; repeatable.")
    parser.add_argument('--name-prefix', type=str, help="Prepended to every run name.")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print the commands that would run, then exit.")
    parser.add_argument('--stop-on-failure', action='store_true',
                        help="Abort the chain on the first failing run (default: carry on).")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.sweep:
        parser.error("at least one --sweep KEY=V1,V2 is required")

    runs = runs_from_inline(args)

    print(f"{len(runs)} run(s) queued, working directory {SRC_DIR}")
    if args.dry_run:
        for index, run in enumerate(runs, start=1):
            print(f"\n[{index}/{len(runs)}] {run.name}")
            print("  " + subprocess.list2cmdline(run.command()))
        return 0

    results = []
    stop_after_this = False
    for index, run in enumerate(runs, start=1):
        print(f"\n{'=' * 70}\n[{index}/{len(runs)}] {run.name}\n{'=' * 70}", flush=True)
        started = time.monotonic()
        try:
            completed = subprocess.run(run.command(), cwd=SRC_DIR)
            status = "ok" if completed.returncode == 0 else f"exit {completed.returncode}"
        except KeyboardInterrupt:
            # Ctrl-C reaches the child too; stop the whole chain rather than moving to the next
            # run, which is never what an interrupt means here.
            status = "interrupted"
            stop_after_this = True
        elapsed = time.monotonic() - started
        results.append((run.name, status, elapsed, find_run_dir(run)))

        if status != "ok" and args.stop_on_failure:
            print(f"Stopping: {run.name} finished with '{status}' and --stop-on-failure is set.")
            stop_after_this = True
        if stop_after_this:
            results += [(remaining.name, "skipped", 0.0, "-") for remaining in runs[index:]]
            break

    print_summary(results)
    return 0 if all(status == "ok" for _, status, _, _ in results) else 1


if __name__ == '__main__':
    sys.exit(main())
