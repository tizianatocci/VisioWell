"""
Zero-shot comparison: run cell counting with different models on all experiments.

Usage:
    python main.py --model cellposesam
    python main.py --model microsam
    python main.py --model all
"""
import argparse
import gc
import os
import traceback

import matplotlib
matplotlib.use("Agg")                  # no figure windows when running from cmd
import matplotlib.pyplot as plt

from comparison_models.functions_comparison import (
    RUNNERS, init_model, runner, postprocess_and_save,
)
from comparison_models.name_experiments import experiments


def parse_args():
    p = argparse.ArgumentParser(description="Run cell counting models on all experiments.")
    p.add_argument(
        "--model",
        required=True,
        choices=list(RUNNERS.keys()) + ["all"],
        help="Model to run, or 'all' to run every available model one after the other.",
    )
    return p.parse_args()


def run_model(choice_model):
    dir_results = f"comparison_models/output/{choice_model}"
    os.makedirs(dir_results, exist_ok=True)

    ctx = init_model(choice_model)     # load the model once
    failures = []

    for exp in experiments:
        assert len(exp["name_series_list"]) == len(exp["name_well_list"])
        for name_series, name_well in zip(exp["name_series_list"], exp["name_well_list"]):
            print(f"[{choice_model}] {exp['name_experiment']} | {name_series} | {name_well}")
            try:
                counts = runner(ctx, exp["path_exp"], name_series, name_well)
                postprocess_and_save(counts, exp["name_experiment"], name_series, name_well, dir_results)
            except Exception:
                traceback.print_exc()
                failures.append((name_series, name_well))
            finally:
                plt.close("all")

    # free GPU memory before loading the next model
    del ctx
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    return failures


def main():
    args = parse_args()
    models = list(RUNNERS.keys()) if args.model == "all" else [args.model]

    all_failures = {}
    for choice_model in models:
        print(f"\n===== {choice_model} =====")
        all_failures[choice_model] = run_model(choice_model)

    print("\n===== SUMMARY =====")
    for choice_model, failures in all_failures.items():
        print(f"{choice_model}: {len(failures)} failed")
        for name_series, name_well in failures:
            print(f"    {name_series} | {name_well}")


if __name__ == "__main__":
    main()