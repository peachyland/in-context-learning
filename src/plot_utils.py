import os

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from eval import get_run_metrics, baseline_names, get_model_from_run
from models import build_model

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


relevant_model_names = {
    "linear_regression": [
        "Transformer-1-layer-1-head",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
    ],
    "sparse_linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
        "Lasso (alpha=0.01)",
    ],
    "decision_tree": [
        "Transformer",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Greedy Tree Learning",
        "XGBoost",
    ],
    "relu_2nn_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
}


def basic_plot(metrics, models=None, trivial=1.0, use_log=False):
    fig, ax = plt.subplots(1, 1)

    # print(metrics.keys())
    # print(models)
    # input("check")
    if models is not None:
        metrics = {k: metrics[k] for k in models}

    color = 0
    ax.axhline(trivial, ls="--", color="gray")
    for name, vs in metrics.items():
        ax.plot(vs["mean"], "-", label=name, color=palette[color % 10], lw=4)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        color += 1
    if use_log:
        plt.yscale('symlog', linthresh=1e-6)
    ax.set_xlabel("in-context examples", fontsize=24)
    ax.set_ylabel("squared error", fontsize=24)
    if use_log:
        ax.set_xlim(-1, len(low) + 0.1)
        ax.set_ylim(10e-4, 10)
    else:
        ax.set_xlim(-1, len(low) + 0.1)
        ax.set_ylim(-0.1, 1.25)

    # legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # fig.set_size_inches(4, 3)
    fig.set_size_inches(8, 6)
    # for line in legend.get_lines():
    #     line.set_linewidth(3)

    return fig, ax


def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None):
    # import pdb ; pdb.set_trace()
    all_metrics = {}
    for _, r in df.iterrows():
        # print("check1")
        if valid_row is not None and not valid_row(r):
            continue
        # print("check2")

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        metrics = get_run_metrics(run_path, skip_model_load=False, skip_baselines=True)

        # print(metrics)
        # input("check")
        # import pdb ; pdb.set_trace()

        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            for model_name, m in results.items():
                if "gpt2" in model_name in model_name:
                    model_name = r.model
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                else:
                    model_name = baseline_names(model_name)
                m_processed = {}
                n_dims = conf.model.n_dims

                # xlim = 2 * n_dims + 1
                xlim = 41
                if r.task in ["relu_2nn_regression", "decision_tree"]:
                    xlim = 200

                normalization = n_dims
                if r.task == "sparse_linear_regression":
                    normalization = int(r.kwargs.split("=")[-1])
                if r.task == "decision_tree":
                    normalization = 1

                for k, v in m.items():
                    v = v[:xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
            all_metrics[eval_name].update(processed_results)
    # print(all_metrics.keys())
    # import pdb ; pdb.set_trace()
    # input("check")
    return all_metrics

