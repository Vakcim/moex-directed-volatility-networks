"""Build v10 CatBoost-, ElasticNet- and blocked-Pearson graph shards.

The expensive learners are resumable at ``block_id x target SECID``.  Use
``--stage select`` (optionally one ``--config-id`` at a time), then
``--stage finalize-selection``, ``--stage build`` and finally ``--stage assemble``.
No forecast-block observation enters graph training or importance estimation.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from v10_common import (
    DEVELOPMENT_END,
    MARKET_FEATURES,
    OWN_FEATURES,
    TOP_K,
    build_har_panel,
    calibration_scale,
    catboost_grid,
    dates_between,
    deterministic_shifts,
    graph_refit_calendar,
    json_dump,
    linear_grid,
    make_wide_model_frame,
    normalize_keys,
    qlike,
    read_table,
    recover_date_table,
    require_columns,
    source_column,
    write_table,
)


MIN_TRAIN_ROWS = 400
MIN_IMPORTANCE_ROWS = 80
MIN_SOURCE_COVERAGE = 0.80
MIN_GRAPH_NODES_PER_COVERED_DATE = 32
MIN_MEDIAN_GRAPH_NODES = 35
MIN_GRAPH_VALID_DATE_RATE = 0.90


def build_all82_target_samples(
    node: pd.DataFrame,
    target: pd.DataFrame,
    target_value: str,
    target_valid: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal all-82 training labels while retaining top-40 membership.

    Dynamic-universe membership determines which targets are forecast at a
    refit date.  It must not truncate the target's earlier train/importance
    history; doing so makes the frozen 400/80 row gates infeasible for recent
    entrants to the top-40 universe.
    """
    require_columns(node, ["TRADEDATE", "SECID", "segment_id", "active"], "node panel")
    require_columns(target, ["TRADEDATE", "SECID", target_value, target_valid], "target panel")

    date_table = recover_date_table(node)
    next_dates = date_table[["TRADEDATE", "segment_id"]].copy()
    next_dates["next_date"] = next_dates.groupby("segment_id")["TRADEDATE"].shift(-1)

    samples = node[["TRADEDATE", "segment_id", "SECID", "active"]].copy()
    if samples.duplicated(["TRADEDATE", "SECID"]).any():
        raise RuntimeError("node panel has duplicate TRADEDATE/SECID keys")
    samples = samples.rename(columns={"active": "target_active"})
    samples["target_active"] = (
        samples["target_active"].astype("boolean").fillna(False).astype(bool)
    )
    samples = samples.merge(
        next_dates,
        on=["TRADEDATE", "segment_id"],
        how="left",
        validate="many_to_one",
    )

    labels = target[["TRADEDATE", "SECID", target_value, target_valid]].copy()
    if labels.duplicated(["TRADEDATE", "SECID"]).any():
        raise RuntimeError("target panel has duplicate TRADEDATE/SECID keys")
    labels = labels.rename(columns={
        "TRADEDATE": "next_date",
        target_value: "y_rv",
        target_valid: "y_valid",
    })
    samples = samples.merge(
        labels,
        on=["next_date", "SECID"],
        how="left",
        validate="many_to_one",
    )
    samples["y_rv"] = pd.to_numeric(samples["y_rv"], errors="coerce")
    label_valid = samples["y_valid"].astype("boolean").fillna(False).astype(bool)
    samples["sample_valid"] = (
        samples["next_date"].notna()
        & label_valid
        & np.isfinite(samples["y_rv"])
        & (samples["y_rv"] > 0)
    )
    return samples, date_table


def active_targets_at_refit(model: pd.DataFrame, refit_date) -> set[str]:
    require_columns(model, ["TRADEDATE", "SECID", "target_active"], "graph model panel")
    mask = (
        (model["TRADEDATE"] == pd.Timestamp(refit_date))
        & model["target_active"].astype("boolean").fillna(False).astype(bool)
    )
    return set(model.loc[mask, "SECID"].astype(str))


def find_split(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        root / "benchmark_v05_paired_graph_history" / "split_dates.csv",
        root / "benchmark_v05_dynamic_history_placebos" / "split_dates.csv",
        root / "benchmark_v04_placebo" / "split_dates.csv",
        root / "benchmark_v03_residual" / "split_dates.csv",
        root / "benchmark_v02_dynamic" / "split_dates.csv",
        root / "benchmark_v01" / "split_dates.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No frozen split_dates.csv; pass --split-dates")


def prepare_raw(root: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    node = normalize_keys(read_table(root / "model_data" / "node_features_all82.parquet"))
    rv = normalize_keys(read_table(root / "model_data" / "rv_all82.parquet"))
    samples, date_table = build_all82_target_samples(
        node, rv, target_value="rv_10m_core", target_valid="rv_valid"
    )
    market = normalize_keys(read_table(root / "market_factor_v10" / "market_rv.parquet"), secid=False)
    har = build_har_panel(node, market)
    model, secids = make_wide_model_frame(har, samples)
    return model, secids, date_table


def prepare_residual(root: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    base = normalize_keys(read_table(root / "model_data" / "node_features_all82.parquet"))
    irv = normalize_keys(read_table(root / "market_factor_v10" / "idiosyncratic_rv.parquet"))
    require_columns(irv, ["idiosyncratic_rv", "log_idiosyncratic_rv", "irv_valid"], "IRV")
    node = base.drop(columns=["log_rv", "rv_feature_valid"], errors="ignore").merge(
        irv[["TRADEDATE", "SECID", "log_idiosyncratic_rv", "irv_valid"]],
        on=["TRADEDATE", "SECID"], how="left", validate="one_to_one",
    ).rename(columns={"log_idiosyncratic_rv": "log_rv", "irv_valid": "rv_feature_valid"})
    samples, date_table = build_all82_target_samples(
        node,
        irv,
        target_value="idiosyncratic_rv",
        target_valid="irv_valid",
    )
    market = normalize_keys(read_table(root / "market_factor_v10" / "market_rv.parquet"), secid=False)
    har = build_har_panel(node, market)
    model, secids = make_wide_model_frame(har, samples)
    return model, secids, date_table


def prepare(root: Path, variant: str):
    return prepare_raw(root) if variant == "raw" else prepare_residual(root)


def audit_target_panel(
    a, model: pd.DataFrame, source_ids: list[str], date_table: pd.DataFrame,
    calendar: pd.DataFrame,
) -> None:
    """Fail fast when all-82 history cannot support top-40 graph targets."""
    rows = []
    targets = sorted(model["SECID"].astype(str).unique())
    blocks = calendar[calendar["windows_valid"]]
    for block in blocks.itertuples(index=False):
        active = active_targets_at_refit(model, block.refit_date)
        if len(active) != 40:
            raise RuntimeError(
                f"block {block.block_id}: expected 40 active targets, found {len(active)}"
            )
        for target in targets:
            if target not in active:
                continue
            train, importance = block_frames(
                model, date_table, pd.Series(block._asdict()), target
            )
            sources = eligible_sources(train, importance, source_ids, target)
            feasible = bool(
                len(train) >= MIN_TRAIN_ROWS
                and len(importance) >= MIN_IMPORTANCE_ROWS
                and sources
            )
            rows.append({
                "variant": a.variant,
                "block_id": int(block.block_id),
                "refit_date": block.refit_date,
                "target": target,
                "train_rows": len(train),
                "importance_rows": len(importance),
                "eligible_sources": len(sources),
                "feasible": feasible,
            })
    detail = pd.DataFrame(rows)
    suffix = "" if a.variant == "raw" else "_residual"
    detail.to_csv(
        a.root / "graphs_v10" / f"target_panel_feasibility{suffix}.csv",
        index=False,
    )
    summary = detail.groupby("block_id", as_index=False).agg(
        active_targets=("target", "size"),
        feasible_targets=("feasible", "sum"),
        median_train_rows=("train_rows", "median"),
        median_importance_rows=("importance_rows", "median"),
        median_eligible_sources=("eligible_sources", "median"),
    )
    summary["forecast_dates"] = summary["block_id"].map(
        calendar.set_index("block_id")["forecast_dates"].astype(int)
    )
    valid_date_rate = float(
        summary.loc[
            summary["feasible_targets"] >= MIN_GRAPH_NODES_PER_COVERED_DATE,
            "forecast_dates",
        ].sum()
        / summary["forecast_dates"].sum()
    )
    median_targets = float(summary["feasible_targets"].median())
    summary.to_csv(
        a.root / "graphs_v10" / f"target_panel_feasibility_by_block{suffix}.csv",
        index=False,
    )
    print(summary.to_string(index=False))
    print(
        f"panel feasibility: valid_date_rate={valid_date_rate:.4f}, "
        f"median_targets={median_targets:.1f}"
    )
    if (
        valid_date_rate < MIN_GRAPH_VALID_DATE_RATE
        or median_targets < MIN_MEDIAN_GRAPH_NODES
    ):
        raise RuntimeError(
            "all-82 target panel fails Amendment 02 graph coverage gates"
        )


def configurations(learner: str):
    return catboost_grid() if learner == "catboost" else linear_grid()


class FittedLearner:
    def __init__(self, learner: str, params: dict):
        self.learner = learner
        self.params = params
        self.model = None
        self.medians = None
        self.scale = None
        self.converged = True
        self.fit_warnings: list[str] = []

    def fit(self, x: pd.DataFrame, y_log: np.ndarray, y_rv: np.ndarray):
        if self.learner == "catboost":
            try:
                from catboost import CatBoostRegressor
            except ImportError as exc:
                raise RuntimeError("Install catboost: pip install catboost") from exc
            self.model = CatBoostRegressor(
                **self.params,
                loss_function="RMSE",
                random_seed=260818,
                thread_count=1,
                allow_writing_files=False,
                verbose=False,
            )
            self.model.fit(x, y_log)
            train_log = np.asarray(self.model.predict(x), float)
        else:
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import ElasticNet
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                ElasticNet(**self.params, random_state=260818, selection="cyclic"),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.model.fit(x, y_log)
            convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
            self.converged = len(convergence) == 0
            self.fit_warnings = [str(item.message) for item in convergence]
            train_log = np.asarray(self.model.predict(x), float)
        raw = np.exp(np.clip(train_log, -30.0, 5.0))
        self.scale = calibration_scale(y_rv, raw)
        return self

    def predict_rv(self, x: pd.DataFrame) -> np.ndarray:
        log = np.asarray(self.model.predict(x), float)
        return np.maximum(np.exp(np.clip(log, -30.0, 5.0)) * self.scale, 1e-12)


def block_frames(
    model: pd.DataFrame,
    date_table: pd.DataFrame,
    block: pd.Series,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dates = dates_between(
        date_table, int(block.segment_id), block.train_start, block.train_end
    )
    importance_dates = dates_between(
        date_table, int(block.segment_id), block.importance_start, block.importance_end
    )
    target_rows = model[model["SECID"] == target]
    train = target_rows[target_rows["TRADEDATE"].isin(train_dates)].copy()
    importance = target_rows[target_rows["TRADEDATE"].isin(importance_dates)].copy()
    base = [*OWN_FEATURES, *MARKET_FEATURES]
    for frame in (train, importance):
        valid = (
            frame["sample_valid"].fillna(False)
            & np.isfinite(frame["y_rv"])
            & (frame["y_rv"] > 0)
            & np.isfinite(frame[base]).all(axis=1)
        )
        frame.drop(frame.index[~valid], inplace=True)
    return train, importance


def eligible_sources(
    train: pd.DataFrame, importance: pd.DataFrame, source_ids: list[str], target: str
) -> list[str]:
    eligible = []
    for source in source_ids:
        if source == target:
            continue
        columns = [source_column(source, h) for h in ("d", "w", "m")]
        coverage = np.isfinite(importance[columns]).all(axis=1).mean()
        train_observed = np.isfinite(train[columns]).any(axis=0).all()
        if coverage >= MIN_SOURCE_COVERAGE and train_observed:
            eligible.append(source)
    return eligible


def feature_columns(sources: list[str]) -> list[str]:
    return [*OWN_FEATURES, *MARKET_FEATURES] + [
        source_column(source, horizon)
        for source in sources for horizon in ("d", "w", "m")
    ]


def valid_selection_blocks(calendar: pd.DataFrame, split: pd.DataFrame) -> set[int]:
    split = normalize_keys(split, secid=False)
    split_map = dict(zip(split["TRADEDATE"], split["split"].astype(str).str.lower()))
    selected = set()
    for row in calendar[calendar["windows_valid"]].itertuples(index=False):
        dates = pd.date_range(row.importance_start, row.importance_end, freq="D")
        research = [d for d in dates if d in split_map]
        if sum(split_map[d] == "val" for d in research) >= MIN_IMPORTANCE_ROWS:
            selected.add(int(row.block_id))
    return selected


def run_selection(
    a, model: pd.DataFrame, source_ids: list[str], date_table: pd.DataFrame, calendar: pd.DataFrame
) -> None:
    split = pd.read_csv(find_split(a.root, a.split_dates))
    split = normalize_keys(split, secid=False)
    val_dates = set(split.loc[split["split"].astype(str).str.lower() == "val", "TRADEDATE"])
    allowed_blocks = valid_selection_blocks(calendar, split)
    configs = configurations(a.learner)
    if a.config_id:
        configs = [c for c in configs if c.config_id == a.config_id]
        if not configs:
            raise RuntimeError(f"Unknown config-id {a.config_id}")
    out = a.root / "graphs_v10" / "selection_parts" / f"variant={a.variant}" / f"learner={a.learner}"
    out.mkdir(parents=True, exist_ok=True)
    targets = sorted(model["SECID"].unique()) if not a.targets else a.targets.split(",")
    for config in configs:
        path = out / f"{config.config_id}.csv"
        existing = pd.read_csv(path) if path.exists() and not a.force else pd.DataFrame()
        done = set(zip(existing.get("block_id", []), existing.get("target", [])))
        rows = existing.to_dict("records")
        for block in calendar[calendar["block_id"].isin(allowed_blocks)].itertuples(index=False):
            active_targets = active_targets_at_refit(model, block.refit_date)
            block_targets = [target for target in targets if target in active_targets]
            for target in block_targets:
                if (block.block_id, target) in done:
                    continue
                train, importance = block_frames(model, date_table, pd.Series(block._asdict()), target)
                importance = importance[importance["TRADEDATE"].isin(val_dates)].copy()
                sources = eligible_sources(train, importance, source_ids, target)
                if len(train) < MIN_TRAIN_ROWS or len(importance) < MIN_IMPORTANCE_ROWS or not sources:
                    rows.append({"config_id": config.config_id, "block_id": block.block_id,
                                 "target": target, "valid": False, "fit_status": "feasibility_gate",
                                 "qlike": np.nan,
                                 "train_rows": len(train), "importance_rows": len(importance),
                                 "sources": len(sources)})
                else:
                    columns = feature_columns(sources)
                    fitted = FittedLearner(a.learner, config.params).fit(
                        train[columns], train["y_log_rv"].to_numpy(), train["y_rv"].to_numpy()
                    )
                    if not fitted.converged:
                        rows.append({"config_id": config.config_id, "block_id": block.block_id,
                                     "target": target, "valid": False, "fit_status": "nonconverged",
                                     "qlike": np.nan, "train_rows": len(train),
                                     "importance_rows": len(importance), "sources": len(sources)})
                    else:
                        pred = fitted.predict_rv(importance[columns])
                        loss = float(qlike(importance["y_rv"].to_numpy(), pred).mean())
                        rows.append({"config_id": config.config_id, "block_id": block.block_id,
                                     "target": target, "valid": True, "fit_status": "ok",
                                     "qlike": loss, "train_rows": len(train),
                                     "importance_rows": len(importance), "sources": len(sources)})
                pd.DataFrame(rows).to_csv(path, index=False)
            block_rows = [row for row in rows if row.get("config_id") == config.config_id
                          and int(row.get("block_id", -1)) == int(block.block_id)]
            status_counts = pd.Series([row.get("fit_status", "unknown") for row in block_rows]).value_counts()
            status_text = ", ".join(f"{key}={value}" for key, value in status_counts.items())
            print(f"selection {a.learner}/{config.config_id}: block {block.block_id} ({status_text})")


def finalize_selection(a) -> None:
    base = a.root / "graphs_v10" / "selection_parts" / f"variant={a.variant}"
    rows = []
    for learner in ("catboost", "linear"):
        files = sorted((base / f"learner={learner}").glob("*.csv"))
        expected = {x.config_id for x in configurations(learner)}
        found = {p.stem for p in files}
        if found != expected:
            raise RuntimeError(f"{learner} selection incomplete: missing={sorted(expected-found)}")
        frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        if "fit_status" not in frame:
            frame["fit_status"] = np.where(
                frame["valid"].astype(str).str.lower().isin(["true", "1"]),
                "ok",
                "feasibility_gate",
            )
        else:
            missing_status = frame["fit_status"].isna()
            frame.loc[missing_status, "fit_status"] = np.where(
                frame.loc[missing_status, "valid"].astype(str).str.lower().isin(["true", "1"]),
                "ok",
                "feasibility_gate",
            )
        summary = frame.groupby("config_id", as_index=False).agg(
            mean_validation_qlike=("qlike", "mean"),
            total_rows=("fit_status", "size"),
            feasible_fits=("fit_status", lambda x: int((x != "feasibility_gate").sum())),
            valid_fits=("fit_status", lambda x: int((x == "ok").sum())),
            nonconverged_fits=("fit_status", lambda x: int((x == "nonconverged").sum())),
            validation_rows=("importance_rows", "sum")
        )
        summary["learner"] = learner
        summary["selection_eligible"] = (
            (summary["valid_fits"] > 0)
            & (summary["nonconverged_fits"] == 0)
        )
        if not summary["selection_eligible"].any():
            raise RuntimeError(f"No converged {learner} configuration remains")
        summary["selection_score"] = summary["mean_validation_qlike"].where(
            summary["selection_eligible"], np.inf
        )
        summary = summary.sort_values(["selection_score", "config_id"])
        summary["selected"] = False
        summary.loc[summary.index[0], "selected"] = True
        rows.append(summary)
    output = pd.concat(rows, ignore_index=True)
    suffix = "" if a.variant == "raw" else "_residual"
    output.to_csv(a.root / "graphs_v10" / f"hyperparameter_selection{suffix}.csv", index=False)
    print(output[output["selected"]].to_string(index=False))


def selected_config(root: Path, variant: str, learner: str):
    suffix = "" if variant == "raw" else "_residual"
    path = root / "graphs_v10" / f"hyperparameter_selection{suffix}.csv"
    table = pd.read_csv(path)
    selected_mask = table["selected"].astype(str).str.lower().isin(["true", "1"])
    row = table[(table["learner"] == learner) & selected_mask]
    if len(row) != 1:
        raise RuntimeError(f"Expected one selected {learner} config in {path}")
    config_id = str(row.iloc[0]["config_id"])
    return next(c for c in configurations(learner) if c.config_id == config_id)


def importance_edges(
    learner: str, config, train: pd.DataFrame, importance: pd.DataFrame,
    sources: list[str], block_id: int, target: str
) -> tuple[list[dict], dict]:
    columns = feature_columns(sources)
    fitted = FittedLearner(learner, config.params).fit(
        train[columns], train["y_log_rv"].to_numpy(), train["y_rv"].to_numpy()
    )
    if not fitted.converged:
        return [], {
            "block_id": block_id,
            "target": target,
            "learner": learner,
            "graph_valid": False,
            "reason": "selected_configuration_nonconverged",
            "eligible_sources": len(sources),
            "train_rows": len(train),
            "importance_rows": len(importance),
            "config_id": config.config_id,
        }
    base_pred = fitted.predict_rv(importance[columns])
    base_loss = float(qlike(importance["y_rv"].to_numpy(), base_pred).mean())
    shifts = deterministic_shifts(importance["TRADEDATE"].nunique())
    raw = []
    for source in sources:
        group = [source_column(source, h) for h in ("d", "w", "m")]
        losses = []
        ordered = importance.sort_values("TRADEDATE").copy()
        for shift in shifts:
            altered = ordered[columns].copy()
            altered.loc[:, group] = np.roll(altered[group].to_numpy(), shift, axis=0)
            pred = fitted.predict_rv(altered)
            losses.append(float(qlike(ordered["y_rv"].to_numpy(), pred).mean()))
        weight = max(0.0, float(np.mean(losses) - base_loss))
        raw.append((source, weight, float(np.std(losses))))
    positive = sorted((x for x in raw if x[1] > 0), key=lambda x: (-x[1], x[0]))[:TOP_K]
    total = sum(x[1] for x in positive)
    edges = [{"block_id": block_id, "src": source, "dst": target,
              "raw_weight": weight, "weight": weight / total,
              "permutation_loss_sd": sd, "learner": learner}
             for source, weight, sd in positive] if total > 0 else []
    diag = {"block_id": block_id, "target": target, "learner": learner,
            "base_importance_qlike": base_loss, "eligible_sources": len(sources),
            "positive_sources": sum(w > 0 for _, w, _ in raw), "saved_edges": len(edges),
            "graph_valid": bool(edges), "train_rows": len(train),
            "importance_rows": len(importance), "config_id": config.config_id}
    return edges, diag


def run_build(a, model, source_ids, date_table, calendar) -> None:
    config = selected_config(a.root, a.variant, a.learner)
    blocks = calendar[calendar["windows_valid"]]
    if a.block_id is not None:
        blocks = blocks[blocks["block_id"] == a.block_id]
    targets = sorted(model["SECID"].unique()) if not a.targets else a.targets.split(",")
    base = a.root / "graphs_v10" / "parts" / f"variant={a.variant}" / f"learner={a.learner}"
    for block in blocks.itertuples(index=False):
        active = active_targets_at_refit(model, block.refit_date)
        for target in targets:
            if target not in active:
                continue
            target_dir = base / f"block_id={block.block_id:04d}" / f"target={target}"
            edge_path = target_dir / "edges.parquet"
            diag_path = target_dir / "diagnostic.json"
            if diag_path.exists() and not a.force:
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            train, importance = block_frames(model, date_table, pd.Series(block._asdict()), target)
            sources = eligible_sources(train, importance, source_ids, target)
            if len(train) >= MIN_TRAIN_ROWS and len(importance) >= MIN_IMPORTANCE_ROWS and sources:
                edges, diag = importance_edges(
                    a.learner, config, train, importance, sources, block.block_id, target
                )
            else:
                edges = []
                diag = {"block_id": block.block_id, "target": target, "learner": a.learner,
                        "graph_valid": False, "reason": "feasibility_gate",
                        "train_rows": len(train), "importance_rows": len(importance),
                        "eligible_sources": len(sources), "config_id": config.config_id}
            diag.update({
                "train_start": block.train_start,
                "train_end": block.train_end,
                "importance_start": block.importance_start,
                "importance_end": block.importance_end,
                "forecast_start": block.forecast_start,
                "forecast_end": block.forecast_end,
                "three_clock_valid": bool(
                    pd.Timestamp(block.train_end) < pd.Timestamp(block.importance_start)
                    and pd.Timestamp(block.importance_end) < pd.Timestamp(block.forecast_start)
                ),
            })
            if edges:
                frame = pd.DataFrame(edges)
                frame["refit_date"] = block.refit_date
                frame["segment_id"] = block.segment_id
                frame.to_parquet(edge_path, index=False)
            json_dump(diag_path, diag)
            print(f"build {a.learner}: block={block.block_id} target={target} edges={len(edges)}")


def build_pearson(a, model, date_table, calendar) -> None:
    returns = normalize_keys(read_table(a.root / "model_data" / "returns_history_all82.parquet"))
    universe = normalize_keys(pd.read_csv(a.root / "universe_v03" / "dynamic_universe.csv"))
    rows = []
    for block in calendar.itertuples(index=False):
        active = sorted(universe.loc[universe["TRADEDATE"] == block.refit_date, "SECID"].astype(str))
        segment_dates = date_table[(date_table["segment_id"] == block.segment_id)
                                   & (date_table["TRADEDATE"] < block.refit_date)].tail(60)["TRADEDATE"]
        if len(segment_dates) < 60:
            continue
        window = returns[returns["TRADEDATE"].isin(segment_dates)]
        wide = window.pivot(index="TRADEDATE", columns="SECID", values="r").reindex(columns=active)
        corr = wide.corr(min_periods=50).abs()
        chosen = set()
        for target in active:
            scores = corr[target].drop(index=target, errors="ignore").dropna().sort_values(ascending=False)
            for source in scores.head(TOP_K).index:
                chosen.add(tuple(sorted((str(source), target))))
        directed = []
        for left, right in sorted(chosen):
            weight = float(corr.loc[left, right])
            directed.extend([(left, right, weight), (right, left, weight)])
        temp = pd.DataFrame(directed, columns=["src", "dst", "raw_weight"])
        if temp.empty:
            continue
        temp["weight"] = temp["raw_weight"] / temp.groupby("dst")["raw_weight"].transform("sum")
        temp["block_id"] = block.block_id
        temp["refit_date"] = block.refit_date
        temp["segment_id"] = block.segment_id
        temp["learner"] = "pearson"
        rows.append(temp)
    output = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    write_table(output, a.root / "graphs_v10" / "pearson_blocked_edges.parquet")
    print(f"Pearson blocked edges: {len(output):,}")


def assemble(a, calendar: pd.DataFrame, model: pd.DataFrame) -> None:
    suffix = "" if a.variant == "raw" else "_residual"
    all_quality = []
    all_diag = []
    learner_feasibility = []
    for learner in ("catboost", "linear"):
        base = a.root / "graphs_v10" / "parts" / f"variant={a.variant}" / f"learner={learner}"
        edge_files = sorted(base.glob("block_id=*/target=*/edges.parquet"))
        diag_files = sorted(base.glob("block_id=*/target=*/diagnostic.json"))
        if not diag_files:
            raise RuntimeError(f"No {learner}/{a.variant} graph shards")
        edges = pd.concat([pd.read_parquet(p) for p in edge_files], ignore_index=True) if edge_files else pd.DataFrame()
        write_table(edges, a.root / "graphs_v10" / f"{learner}_directed_edges{suffix}.parquet")
        diagnostics = pd.DataFrame([json.loads(p.read_text()) for p in diag_files])
        if "three_clock_valid" not in diagnostics or not diagnostics["three_clock_valid"].astype(bool).all():
            raise RuntimeError(f"{learner}: three-clock chronology audit failed")
        expected = []
        for block in calendar[calendar["windows_valid"]].itertuples(index=False):
            targets = sorted(active_targets_at_refit(model, block.refit_date))
            expected.extend((int(block.block_id), target) for target in targets)
        found = set(zip(diagnostics["block_id"].astype(int), diagnostics["target"].astype(str)))
        missing = sorted(set(expected) - found)
        if missing:
            raise RuntimeError(f"{learner}/{a.variant}: {len(missing)} graph shards missing; first={missing[:5]}")
        if not edges.empty:
            if (edges["src"].astype(str) == edges["dst"].astype(str)).any():
                raise RuntimeError(f"{learner}: self-loop detected")
            if not np.isfinite(edges[["raw_weight", "weight"]]).all().all():
                raise RuntimeError(f"{learner}: non-finite edge weight")
            counts = edges.groupby(["block_id", "dst"]).size()
            if (counts > TOP_K).any():
                raise RuntimeError(f"{learner}: more than {TOP_K} incoming edges")
            sums = edges.groupby(["block_id", "dst"])["weight"].sum()
            if not np.allclose(sums.to_numpy(), 1.0, rtol=0, atol=1e-8):
                raise RuntimeError(f"{learner}: incoming weights do not sum to one")
        diagnostics["variant"] = a.variant
        all_diag.append(diagnostics)
        q = diagnostics.groupby("block_id", as_index=False).agg(
            graph_valid_nodes=("graph_valid", "sum"), targets_attempted=("target", "size"),
            median_train_rows=("train_rows", "median"), median_importance_rows=("importance_rows", "median")
        )
        q["learner"] = learner
        q["variant"] = a.variant
        q["forecast_dates"] = q["block_id"].map(
            calendar.set_index("block_id")["forecast_dates"].astype(int)
        )
        valid_date_rate = float(
            q.loc[
                q["graph_valid_nodes"] >= MIN_GRAPH_NODES_PER_COVERED_DATE,
                "forecast_dates",
            ].sum()
            / q["forecast_dates"].sum()
        )
        median_nodes = float(q["graph_valid_nodes"].median())
        feasibility_passed = bool(
            valid_date_rate >= MIN_GRAPH_VALID_DATE_RATE
            and median_nodes >= MIN_MEDIAN_GRAPH_NODES
        )
        failed_gates = []
        if valid_date_rate < MIN_GRAPH_VALID_DATE_RATE:
            failed_gates.append("valid_date_rate")
        if median_nodes < MIN_MEDIAN_GRAPH_NODES:
            failed_gates.append("median_graph_nodes")
        learner_feasibility.append({
            "variant": a.variant,
            "learner": learner,
            "valid_date_rate": valid_date_rate,
            "minimum_valid_date_rate": MIN_GRAPH_VALID_DATE_RATE,
            "median_graph_nodes": median_nodes,
            "minimum_median_graph_nodes": MIN_MEDIAN_GRAPH_NODES,
            "graph_states_attempted": int(len(diagnostics)),
            "graph_states_valid": int(diagnostics["graph_valid"].astype(bool).sum()),
            "feasibility_passed": feasibility_passed,
            "failed_gates": "|".join(failed_gates),
            "downstream_status": (
                "eligible"
                if feasibility_passed
                else "feasibility_failed_pre_forecast"
            ),
        })
        if not feasibility_passed and a.variant == "raw":
            # Preserve the frozen raw-graph protocol: every prespecified raw
            # learner must pass its graph-construction feasibility gate.
            raise RuntimeError(
                f"{learner}/{a.variant} feasibility gate failed: "
                f"valid_date_rate={valid_date_rate:.4f}, median_nodes={median_nodes:.1f}"
            )
        if not feasibility_passed and a.variant == "residual":
            # Amendment 06: residual feasibility is assessed per learner.
            # A failed learner is retained for auditability but must not enter
            # downstream forecast-loss analysis.  Thresholds and edges are not
            # altered after observing graph-construction coverage.
            warnings.warn(
                f"{learner}/{a.variant} excluded from downstream analysis: "
                f"valid_date_rate={valid_date_rate:.4f}, "
                f"median_nodes={median_nodes:.1f}",
                RuntimeWarning,
            )
        q["feasibility_passed"] = feasibility_passed
        q["downstream_status"] = (
            "eligible"
            if feasibility_passed
            else "feasibility_failed_pre_forecast"
        )
        q["valid_date_rate_overall"] = valid_date_rate
        q["median_graph_covered_nodes_overall"] = median_nodes
        all_quality.append(q)

    feasibility = pd.DataFrame(learner_feasibility)
    feasibility_path = (
        a.root / "graphs_v10" / f"directed_learner_feasibility{suffix}.csv"
    )
    feasibility.to_csv(feasibility_path, index=False)
    eligible_learners = sorted(
        feasibility.loc[feasibility["feasibility_passed"], "learner"].astype(str)
    )
    ineligible_learners = sorted(
        feasibility.loc[~feasibility["feasibility_passed"], "learner"].astype(str)
    )
    if a.variant == "residual" and not eligible_learners:
        raise RuntimeError(
            "No residual learner passed the frozen graph-construction "
            "feasibility gates"
        )
    quality_path = a.root / "graphs_v10" / f"directed_graph_quality{suffix}.csv"
    pd.concat(all_quality, ignore_index=True).to_csv(quality_path, index=False)
    pd.concat(all_diag, ignore_index=True).to_csv(
        a.root / "graphs_v10" / f"directed_importance_diagnostics{suffix}.csv", index=False
    )
    calendar.to_csv(a.root / "graphs_v10" / "graph_refit_dates.csv", index=False)
    json_dump(a.root / "graphs_v10" / f"graph_design{suffix}.json", {
        "variant": a.variant, "graph_train_dates": 504, "importance_dates": 126,
        "refit_dates": 20, "top_k": TOP_K, "importance_shifts": 20,
        "importance_seed": 260819, "minimum_shift_dates": 20,
        "catboost_seed": 260818, "development_end": str(DEVELOPMENT_END.date()),
        "target_history_scope": "all82_causal_history",
        "forecast_target_scope": "dynamic_top40_at_refit",
        "minimum_graph_nodes_per_covered_date": MIN_GRAPH_NODES_PER_COVERED_DATE,
        "minimum_median_graph_nodes": MIN_MEDIAN_GRAPH_NODES,
        "minimum_graph_valid_date_rate": MIN_GRAPH_VALID_DATE_RATE,
        "feasibility_gate_scope": (
            "per_learner_at_least_one_required"
            if a.variant == "residual"
            else "per_learner_all_required"
        ),
        "downstream_eligible_learners": eligible_learners,
        "downstream_ineligible_learners": ineligible_learners,
        "pearson_defined_for_variant": a.variant == "raw",
        "protocol_amendment": (
            "V10 Amendment 06: residual learner feasibility"
            if a.variant == "residual"
            else None
        ),
    })
    print("\nLearner feasibility:")
    print(feasibility.to_string(index=False))
    print("\nGraph quality by block:")
    print(pd.concat(all_quality, ignore_index=True).to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data_v02"))
    p.add_argument("--variant", choices=["raw", "residual"], default="raw")
    p.add_argument(
        "--stage",
        choices=["audit-panel", "select", "finalize-selection", "build", "pearson", "assemble"],
        required=True,
    )
    p.add_argument("--learner", choices=["catboost", "linear"], default="catboost")
    p.add_argument("--config-id", default=None)
    p.add_argument("--block-id", type=int, default=None)
    p.add_argument("--targets", default=None)
    p.add_argument("--split-dates", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    model, source_ids, date_table = prepare(a.root, a.variant)
    calendar = graph_refit_calendar(date_table)
    (a.root / "graphs_v10").mkdir(parents=True, exist_ok=True)
    calendar.to_csv(a.root / "graphs_v10" / "graph_refit_dates.csv", index=False)
    if a.stage == "audit-panel":
        audit_target_panel(a, model, source_ids, date_table, calendar)
    elif a.stage == "select":
        run_selection(a, model, source_ids, date_table, calendar)
    elif a.stage == "finalize-selection":
        finalize_selection(a)
    elif a.stage == "build":
        run_build(a, model, source_ids, date_table, calendar)
    elif a.stage == "pearson":
        if a.variant != "raw":
            raise RuntimeError("Pearson blocked is defined only for raw returns")
        build_pearson(a, model, date_table, calendar)
    else:
        assemble(a, calendar, model)


if __name__ == "__main__":
    main()
