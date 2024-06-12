from typing import Literal, Callable
import pandas as pd
import numpy as np

def data_mode_acronyms(data_mode):
    return (
        data_mode.replace("Agglo", "AG")
        .replace("MLP-MultiHead", "[FFN-SFT]")
        .replace("MLP", "FFN")
        .replace("GNN-MultiHead", "[GNN-SFT]")
        .replace("TF-MultiHead", "[TF-SFT]")
        .replace("Encoded", "ENC")
        .replace("Decoded", "DEC")
        .replace("Original", "ORI")
        .replace("Random Sample", "RS")
        .replace("Binary", "BIN")
        .replace(" / Centroid", "/SYN")
        .replace(" / Closest", "/REAL")
    )


def compare_group_key(group: pd.DataFrame, metric: str, strategy: str):
    if strategy == "max":
        return group[metric].max()
    elif strategy == "min":
        return group[metric].min()
    elif strategy == "mean":
        return group[metric].mean()
    elif isinstance(strategy, Callable):
        return strategy(group, metric)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def aggregate_perf(group, metric):
    return {
        f"{metric} Values": group[metric].tolist(),
        f"{metric} Mean": group[metric].mean(),
        f"{metric} Max": group[metric].max(),
        f"{metric} Min": group[metric].min(),
    }


def compute_groups(
    report: pd.DataFrame,
    targets: list[str],
    metric: str,
    exclude: list[str] = [],
    group_aggr: str = "mean",
) -> pd.DataFrame:

    # Base filter so we only look at the test set
    base_filter = report["Subset"] == "Test"

    # The rank group must at least be this large. if not we skip b/c we dont' have a complete rank
    min_group_size = report[targets].drop_duplicates().shape[0]

    # gather columns that represent the pipeline components
    other_components = list(
        set(report.columns)
        - set(
            [
                "Subset",
                "Score",
                "Default Train Time",
                "Opt Train Time",
                "Opt Train Time Total",
                "Inference Time",
                "Data Distill Time",
                "Data Mode",
                "Short Name",
                "Regret",
                "Scaled Regret",
                "Scaled Regret with RS",
            ]
        )
        - set(targets)
        - set(exclude)
    )

    # Save methods that actually do distillation
    distill_methods = set(report["Distill Method"].unique()) - set(
        ["Original", "Encoded", "Decoded", "Random Sample"]
    )

    targets_filter = True
    if "Distill Method" in targets:
        targets_filter &= report["Distill Method"].isin(distill_methods)

    filtered = report[base_filter & targets_filter].copy()

    print(f"Grouping {targets} by {metric}, grouped by: {other_components}")

    all_groups = []

    # if distill method is in the targets, we will pre-compute the special cases
    grouped = list(filtered.groupby(other_components))
    for gr_idx, (comps, rank_group) in enumerate(grouped):
        comps = dict(zip(other_components, comps))

        # if the rank group's targets are less than min_group_size, runs are not complete yet.
        if rank_group[targets].drop_duplicates().shape[0] < min_group_size:
            continue

        # take average
        all_groups += [
            {
                "Target": targ,
                "Metric": compare_group_key(
                    group=perf_per_dm,
                    metric=metric,
                    strategy=group_aggr,
                ),
                "Group ID": gr_idx,
                **aggregate_perf(perf_per_dm, "Score"),
                **aggregate_perf(perf_per_dm, "Regret"),
                **aggregate_perf(perf_per_dm, "Scaled Regret"),
                **aggregate_perf(perf_per_dm, "Scaled Regret with RS"),
                **dict(zip(targets, targ)),
            }
            for targ, perf_per_dm in rank_group.groupby(targets)
        ]
    all_groups = pd.DataFrame(all_groups)
    print("Computing Ranks")

    return all_groups


def compute_ranks(
    groups,
    direction: Literal["max", "min"] = "max",
    target: str = "Target",
    metric: str = "Metric",
):
    rankings = pd.DataFrame(
        [
            dict(
                zip(
                    rank_group[target],
                    rank_group[metric].rank(ascending=(direction != "max")) - 1,
                )
            )
            for _, rank_group in groups.groupby("Group ID")
        ]
    )
    return rankings


def compute_regret(raw_results: pd.DataFrame) -> pd.DataFrame:
    is_distill_pipeline = ~raw_results["Distill Method"].isin(
        ["Original", "Encoded", "Decoded"]
    )

    # need to duplicate baselines for each distill size
    results = pd.concat(
        [
            pd.concat(
                [
                    raw_results[~is_distill_pipeline].assign(**{"N": n})
                    for n in sorted(set(raw_results["N"].unique()) - set([0]))
                ]
            ),
            raw_results[is_distill_pipeline],
        ]
    )

    # calculate regret score
    results["Balanced Regret"] = -np.inf
    results["Regret"] = -np.inf
    for n, grouped in results[results["Subset"] == "Test"].groupby("N"):
        if "original" in grouped["Distill Method"].values:
            original_score = grouped[(grouped["Data Mode"] == "Mixed Original")][
                "Score"
            ].mean()
            results.loc[
                ((results["N"] == n) & (results["Subset"] == "Test")), "Regret"
            ] = (original_score - grouped["Score"]) / original_score

            if "Random Sample" in grouped["Distill Method"].values:
                random_score = grouped[(grouped["Data Mode"] == "Random Sample")][
                    "Score"
                ].mean()
                results.loc[
                    (results["N"] == n) & (results["Subset"] == "Test"),
                    "Balanced Regret",
                ] = (original_score - grouped["Score"]) / (
                    original_score - random_score + 1e-10
                )

    return results
