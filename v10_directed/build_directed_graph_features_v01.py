"""Convert v10 graph states into causal HAR-X features and placebo panels."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from v10_common import (
    MARKET_FEATURES,
    OWN_FEATURES,
    build_har_panel,
    graph_refit_calendar,
    make_wide_model_frame,
    normalize_keys,
    read_table,
    recover_date_table,
    require_columns,
    write_table,
)


CURRENT_SUFFIXES = [
    "incoming_nbr_logrv_d", "incoming_nbr_logrv_w", "incoming_nbr_logrv_m",
    "incoming_degree", "incoming_strength",
]
DYNAMIC_SUFFIXES = [
    "incoming_jaccard_distance", "delta_incoming_strength",
    "weighted_incoming_edge_change", "incoming_edge_turnover", "g_change",
]


def graph_learner_feasibility(
    root: Path, variant: str
) -> tuple[set[str], set[str], str | None]:
    """Return graph learners permitted to enter downstream feature construction.

    Amendment 06 applies only to the residual branch.  Raw behavior remains
    frozen and continues to require both prespecified directed learners.
    """
    if variant == "raw":
        return {"linear", "catboost"}, set(), None

    path = root / "graphs_v10" / "directed_learner_feasibility_residual.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing residual learner feasibility audit: {path}. "
            "Run residual graph assembly with Amendment 06 first."
        )
    table = pd.read_csv(path)
    require_columns(
        table,
        ["variant", "learner", "feasibility_passed", "downstream_status"],
        "residual graph learner feasibility",
    )
    if table["learner"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate residual learner feasibility rows")
    if set(table["variant"].astype(str)) != {"residual"}:
        raise RuntimeError("Residual learner feasibility audit has wrong variant")
    passed = table["feasibility_passed"].astype(str).str.lower().isin(["true", "1"])
    eligible = set(table.loc[passed, "learner"].astype(str))
    ineligible = set(table.loc[~passed, "learner"].astype(str))
    if not eligible:
        raise RuntimeError("No residual graph learner passed the frozen feasibility gates")
    if "catboost" not in eligible:
        raise RuntimeError("Amendment 06 requires eligible residual CatBoost graphs")
    if not table.loc[passed, "downstream_status"].astype(str).eq("eligible").all():
        raise RuntimeError("Residual learner feasibility status is internally inconsistent")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return eligible, ineligible, digest


def prepare_raw(root: Path):
    node = normalize_keys(read_table(root / "model_data" / "node_features_all82.parquet"))
    samples = normalize_keys(read_table(root / "model_data" / "forecast_samples.parquet"))
    market = normalize_keys(read_table(root / "market_factor_v10" / "market_rv.parquet"), secid=False)
    return node, samples, market


def prepare_residual(root: Path):
    base = normalize_keys(read_table(root / "model_data" / "node_features_all82.parquet"))
    irv = normalize_keys(read_table(root / "market_factor_v10" / "idiosyncratic_rv.parquet"))
    node = base.drop(columns=["log_rv", "rv_feature_valid"], errors="ignore").merge(
        irv[["TRADEDATE", "SECID", "log_idiosyncratic_rv", "irv_valid"]],
        on=["TRADEDATE", "SECID"], how="left", validate="one_to_one",
    ).rename(columns={"log_idiosyncratic_rv": "log_rv", "irv_valid": "rv_feature_valid"})
    dates = recover_date_table(node)
    next_dates = dates[["TRADEDATE", "segment_id"]].copy()
    next_dates["next_date"] = next_dates.groupby("segment_id")["TRADEDATE"].shift(-1)
    universe = normalize_keys(pd.read_csv(root / "universe_v03" / "dynamic_universe.csv"))
    samples = universe.merge(next_dates, on=["TRADEDATE", "segment_id"], how="left")
    y = irv[["TRADEDATE", "SECID", "idiosyncratic_rv", "irv_valid"]].rename(
        columns={"TRADEDATE": "next_date", "idiosyncratic_rv": "y_rv", "irv_valid": "y_valid"}
    )
    samples = samples.merge(y, on=["next_date", "SECID"], how="left")
    samples["sample_valid"] = samples["next_date"].notna() & samples["y_valid"].fillna(False)
    market = normalize_keys(read_table(root / "market_factor_v10" / "market_rv.parquet"), secid=False)
    return node, samples, market


def date_to_block(date_table: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    rows = []
    refit_lookup = dict(zip(pd.to_datetime(calendar["forecast_start"]), calendar["block_id"]))
    for segment_id, group in date_table.groupby("segment_id", sort=True):
        dates = list(pd.to_datetime(group["TRADEDATE"]).sort_values())
        for start in range(0, len(dates), 20):
            block_dates = dates[start : start + 20]
            block_id = int(refit_lookup[block_dates[0]])
            rows.extend({"TRADEDATE": d, "segment_id": int(segment_id), "block_id": block_id}
                        for d in block_dates)
    return pd.DataFrame(rows)


def state_map(edges: pd.DataFrame) -> dict[tuple[int, str], dict[str, tuple[float, float]]]:
    states = {}
    if edges.empty:
        return states
    for (block_id, target), group in edges.groupby(["block_id", "dst"], sort=False):
        states[(int(block_id), str(target))] = {
            str(row.src): (float(row.weight), float(row.raw_weight))
            for row in group.itertuples(index=False)
        }
    return states


def deranged_identity_edges(edges: pd.DataFrame, seed: int = 260820) -> pd.DataFrame:
    parts = []
    for block_id, group in edges.groupby("block_id", sort=True):
        nodes = sorted(set(group["src"].astype(str)) | set(group["dst"].astype(str)))
        rng = np.random.default_rng(seed + int(block_id))
        mapping = None
        for _ in range(5000):
            shuffled = rng.permutation(nodes)
            candidate = dict(zip(nodes, shuffled))
            if all(candidate[str(s)] != str(t) for s, t in zip(group["src"], group["dst"])):
                mapping = candidate
                break
        if mapping is None:
            raise RuntimeError(f"Could not build self-loop-free source permutation for block {block_id}")
        part = group.copy()
        part["src"] = part["src"].astype(str).map(mapping)
        part["learner"] = "catboost_identity"
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else edges.iloc[:0].copy()


def frozen_edges(edges: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Repeat the first valid target graph within each temporal segment."""
    block_segment = dict(zip(calendar["block_id"].astype(int), calendar["segment_id"].astype(int)))
    edges = edges.copy()
    edges["segment_id"] = edges["block_id"].astype(int).map(block_segment)
    first = {}
    first_block_by_key = {}
    for (segment_id, target), group in edges.groupby(["segment_id", "dst"], sort=True):
        first_block = int(group["block_id"].min())
        first[(int(segment_id), str(target))] = group[group["block_id"] == first_block].copy()
        first_block_by_key[(int(segment_id), str(target))] = first_block
    parts = []
    for block in calendar.itertuples(index=False):
        targets = [t for (segment, t) in first if segment == int(block.segment_id)]
        for target in targets:
            if int(block.block_id) < first_block_by_key[(int(block.segment_id), target)]:
                continue
            part = first[(int(block.segment_id), target)].copy()
            part["block_id"] = int(block.block_id)
            part["refit_date"] = block.refit_date
            part["learner"] = "catboost_frozen"
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else edges.iloc[:0].copy()


def graph_features(
    prefix: str,
    edges: pd.DataFrame,
    base: pd.DataFrame,
    date_blocks: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    states = state_map(edges)
    values = {
        pd.Timestamp(day): {
            str(row.SECID): (
                float(row.har_logrv_d), float(row.har_logrv_w), float(row.har_logrv_m)
            )
            for row in group.itertuples(index=False)
            if np.isfinite(row.har_logrv_d) and np.isfinite(row.har_logrv_w) and np.isfinite(row.har_logrv_m)
        }
        for day, group in base.groupby("TRADEDATE", sort=False)
    }
    block_lookup = dict(zip(pd.to_datetime(date_blocks["TRADEDATE"]), date_blocks["block_id"].astype(int)))
    previous = {}
    for segment_id, group in calendar.groupby("segment_id", sort=True):
        blocks = list(group.sort_values("forecast_start")["block_id"].astype(int))
        previous.update({current: prior for prior, current in zip(blocks, blocks[1:])})
    rows = []
    for row in base[["TRADEDATE", "SECID"]].itertuples(index=False):
        day, target = pd.Timestamp(row.TRADEDATE), str(row.SECID)
        block_id = block_lookup.get(day)
        if block_id is None:
            continue
        block_id = int(block_id)
        current = states.get((block_id, target), {})
        prior_id = previous.get(block_id)
        prior = states.get((prior_id, target), {}) if prior_id is not None else {}
        value_map = values.get(day, {})
        norm_sum = sum(w for w, _ in current.values())
        current_valid = bool(current and np.isclose(norm_sum, 1.0, atol=1e-8))
        nbr = [np.nan, np.nan, np.nan]
        if current_valid and all(source in value_map for source in current):
            for h in range(3):
                nbr[h] = sum(weight * value_map[source][h] for source, (weight, _) in current.items())
        strength = sum(raw for _, raw in current.values()) if current_valid else np.nan
        transition = bool(current_valid and prior)
        jaccard = delta_strength = change = turnover = np.nan
        if transition:
            cset, pset = set(current), set(prior)
            union = cset | pset
            intersection = cset & pset
            jaccard = 1.0 - len(intersection) / len(union)
            turnover = jaccard
            prior_strength = sum(raw for _, raw in prior.values())
            delta_strength = strength - prior_strength
            change = sum(abs(current.get(s, (0.0, 0.0))[0] - prior.get(s, (0.0, 0.0))[0]) for s in union)
        rows.append({
            "TRADEDATE": day, "SECID": target, "block_id": block_id,
            f"{prefix}_graph_valid": current_valid,
            f"{prefix}_transition_valid": transition,
            f"{prefix}_incoming_nbr_logrv_d": nbr[0],
            f"{prefix}_incoming_nbr_logrv_w": nbr[1],
            f"{prefix}_incoming_nbr_logrv_m": nbr[2],
            f"{prefix}_incoming_degree": float(len(current)) if current_valid else np.nan,
            f"{prefix}_incoming_strength": strength,
            f"{prefix}_incoming_jaccard_distance": jaccard,
            f"{prefix}_delta_incoming_strength": delta_strength,
            f"{prefix}_weighted_incoming_edge_change": change,
            f"{prefix}_incoming_edge_turnover": turnover,
            f"{prefix}_g_change": change,
        })
    return pd.DataFrame(rows)


def add_lag20(frame: pd.DataFrame) -> pd.DataFrame:
    dynamic = [f"catboost_{suffix}" for suffix in DYNAMIC_SUFFIXES]
    source = frame[["TRADEDATE", "SECID", "segment_id", *dynamic]].sort_values(
        ["SECID", "segment_id", "TRADEDATE"]
    )
    # Graph changes are constant within each block; shift by exactly 20 research dates.
    lag = source.groupby(["SECID", "segment_id"], sort=False)[dynamic].shift(20)
    lag.columns = [f"catboost_lag20_{suffix.removeprefix('catboost_')}" for suffix in dynamic]
    return pd.concat([frame, lag.reindex(frame.index)], axis=1)


def stress_features(market: pd.DataFrame) -> pd.DataFrame:
    m = market.sort_values(["segment_id", "TRADEDATE"]).copy()
    rows = []
    for _, group in m.groupby("segment_id", sort=True):
        values = group["market_log_rv"]
        past = values.shift(1)
        window = past.rolling(252, min_periods=126)
        median = window.median()
        # ``Rolling.median`` ignores missing market observations once
        # ``min_periods`` is met.  The MAD calculation must use the same
        # missing-value convention.  Plain ``np.median`` returns NaN whenever
        # even one missing observation remains in the 252-date window and can
        # therefore erase stress_z for hundreds of otherwise eligible dates.
        mad = window.apply(
            lambda x: float(
                np.nanmedian(np.abs(x - np.nanmedian(x)))
            ),
            raw=True,
        )
        z = ((values - median) / mad.where(mad > 0)).clip(-5, 5)
        threshold = z.shift(1).rolling(252, min_periods=126).quantile(0.80)
        part = group[["TRADEDATE"]].copy()
        part["stress_z"] = z.to_numpy()
        part["stress_threshold80"] = threshold.to_numpy()
        part["stress_indicator"] = (z > threshold).to_numpy()
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def merge_graph_panel(
    output: pd.DataFrame, panel: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    """Merge one graph feature panel while keeping a single block assignment."""
    keys = ["TRADEDATE", "SECID"]
    if panel.duplicated(keys).any():
        raise RuntimeError(f"{prefix}: duplicate graph feature keys")
    if "block_id" not in panel:
        raise RuntimeError(f"{prefix}: graph feature panel has no block_id")
    if "block_id" in output:
        expected = output[[*keys, "block_id"]].rename(
            columns={"block_id": "expected_block_id"}
        )
        observed = panel[[*keys, "block_id"]].rename(
            columns={"block_id": "observed_block_id"}
        )
        check = expected.merge(observed, on=keys, how="left", validate="one_to_one")
        assigned = check["expected_block_id"].notna()
        missing = assigned & check["observed_block_id"].isna()
        mismatch = (
            assigned
            & check["observed_block_id"].notna()
            & (check["expected_block_id"] != check["observed_block_id"])
        )
        if missing.any() or mismatch.any():
            raise RuntimeError(
                f"{prefix}: inconsistent graph block assignment "
                f"(missing={int(missing.sum())}, mismatch={int(mismatch.sum())})"
            )
        panel = panel.drop(columns="block_id")
    return output.merge(panel, on=keys, how="left", validate="one_to_one")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data_v02"))
    p.add_argument("--variant", choices=["raw", "residual"], default="raw")
    p.add_argument("--min-labels-per-date", type=int, default=30)
    p.add_argument("--write-csv", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    eligible_learners, ineligible_learners, feasibility_sha256 = (
        graph_learner_feasibility(a.root, a.variant)
    )
    node, samples, market = prepare_raw(a.root) if a.variant == "raw" else prepare_residual(a.root)
    dates = recover_date_table(node)
    market = market.drop(columns=["segment_id"], errors="ignore").merge(
        dates[["TRADEDATE", "segment_id"]], on="TRADEDATE", how="inner", validate="one_to_one"
    )
    har = build_har_panel(node, market)
    base, _ = make_wide_model_frame(har, samples)
    calendar = graph_refit_calendar(dates)
    blocks = date_to_block(dates, calendar)
    suffix = "" if a.variant == "raw" else "_residual"
    graph_dir = a.root / "graphs_v10"
    specs = []
    if a.variant == "raw":
        specs.append(("pearson", read_table(graph_dir / "pearson_blocked_edges.parquet")))
    cat = None
    if "linear" in eligible_learners:
        linear = read_table(graph_dir / f"linear_directed_edges{suffix}.parquet")
        specs.append(("linear", linear))
    if "catboost" in eligible_learners:
        cat = read_table(graph_dir / f"catboost_directed_edges{suffix}.parquet")
        specs.append(("catboost", cat))
    if a.variant == "raw":
        if cat is None:
            raise RuntimeError("Raw CatBoost graph unexpectedly unavailable")
        specs.extend([
            ("catboost_identity", deranged_identity_edges(cat)),
            ("catboost_frozen", frozen_edges(cat, calendar)),
        ])
    output = base.copy()
    for prefix, edges in specs:
        panel = graph_features(prefix, edges, har, blocks, calendar)
        output = merge_graph_panel(output, panel, prefix)
    if a.variant == "raw":
        output = add_lag20(output)
    output = output.merge(stress_features(market), on="TRADEDATE", how="left", validate="many_to_one")

    real_prefixes = [
        learner for learner in ("linear", "catboost")
        if learner in eligible_learners
    ] + (["pearson"] if a.variant == "raw" else [])
    required = [*OWN_FEATURES, *MARKET_FEATURES]
    for prefix in real_prefixes:
        required.extend(f"{prefix}_{suffix_}" for suffix_ in CURRENT_SUFFIXES + DYNAMIC_SUFFIXES)
    finite = np.isfinite(output[required]).all(axis=1)
    valid = (
        output["sample_valid"].fillna(False)
        & (output["y_rv"] > 0)
        & (output["har_w_count"] >= 5)
        & (output["har_m_count"] >= 22)
        & (output["mkt_w_count"] >= 5)
        & (output["mkt_m_count"] >= 22)
        & finite
    )
    counts = output.loc[valid].groupby("TRADEDATE").size()
    valid_dates = set(counts[counts >= a.min_labels_per_date].index)
    output["directed_sample_valid_pre_date"] = valid
    output["directed_sample_valid"] = valid & output["TRADEDATE"].isin(valid_dates)
    if output.duplicated(["TRADEDATE", "SECID"]).any():
        raise RuntimeError("Duplicate directed feature keys")
    for learner in ineligible_learners:
        forbidden = [column for column in output if column.startswith(f"{learner}_")]
        if forbidden:
            raise RuntimeError(
                f"Ineligible residual learner {learner} leaked into features: "
                f"{forbidden[:5]}"
            )

    name = "directed_graph_features.parquet" if a.variant == "raw" else "residual_directed_graph_features.parquet"
    write_table(output, a.root / "model_data" / name, csv=a.write_csv)
    audit = pd.DataFrame([
        {"metric": "rows", "value": len(output)},
        {"metric": "dates", "value": output["TRADEDATE"].nunique()},
        {"metric": "valid_rows", "value": int(output["directed_sample_valid"].sum())},
        {"metric": "valid_dates", "value": output.loc[output["directed_sample_valid"], "TRADEDATE"].nunique()},
        {"metric": "nonfinite_required_on_valid", "value": int((~np.isfinite(output.loc[output["directed_sample_valid"], required])).to_numpy().sum())},
        {"metric": "graph_learners_included", "value": "|".join(sorted(eligible_learners))},
        {"metric": "graph_learners_excluded", "value": "|".join(sorted(ineligible_learners))},
        {"metric": "learner_feasibility_sha256", "value": feasibility_sha256 or "not_applicable_raw"},
        {"metric": "protocol_amendment", "value": "V10 Amendment 06" if a.variant == "residual" else "none"},
    ])
    audit.to_csv(a.root / "model_data" / f"directed_feature_audit{suffix}.csv", index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
