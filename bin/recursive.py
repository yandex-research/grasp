import argparse
import sys
from pathlib import Path

import bin.go
import lib

_DEFAULT_N_SEEDS = 10


def main(
    path: str | Path,
    n_seeds: int = _DEFAULT_N_SEEDS,
    *,
    dry: bool = False,
    continue_: bool = False,
    force: bool = False,
    mode: str,
):
    path = Path(path).resolve()
    configs = sorted(list(path.rglob(f"{mode}.toml")))
    print(f"Found {len(configs)} {mode}.toml configs at {path}\n")

    if dry:
        return

    for config in configs:
        bin.go.main(
            config,
            n_seeds=n_seeds,
            continue_=continue_,
            force=force,
        )


if __name__ == "__main__":
    lib.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--n_seeds", type=int, default=_DEFAULT_N_SEEDS)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--continue", action="store_true", dest="continue_")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mode", type=str, default="tuning")

    main(**vars(parser.parse_args(sys.argv[1:])))
