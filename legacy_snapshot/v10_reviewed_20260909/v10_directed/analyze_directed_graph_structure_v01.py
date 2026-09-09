"""Exploratory structural audit of frozen V10 directed volatility graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v10_common import normalize_keys, read_table


GRAPH_FILES = {
    "raw_catboost": "catboost_directed_edges.parquet",
    "residual_catboost": "catboost_directed_edges_residual.parquet",
    "pearson_raw": "pearson_blocked_edges.parquet",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_edges(path: Path, graph: str) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Graph file missing: {path}")
    frame = read_table(path).copy()
    required = {"block_id", "src", "dst", "weight", "raw_weight"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{graph}: missing columns {missing}")
    frame["block_id"] = pd.to_numeric(frame["block_id"], errors="raise").astype(int)
    frame["src"] = frame["src"].astype(str)
    frame["dst"] = frame["dst"].astype(str)
    for column in ("weight", "raw_weight"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.duplicated(["block_id", "src", "dst"]).any():
        raise RuntimeError(f"{graph}: duplicate block-src-dst edges")
    if (frame["src"] == frame["dst"]).any():
        raise RuntimeError(f"{graph}: self-loop detected")
    if not np.isfinite(frame[["weight", "raw_weight"]]).all().all():
        raise RuntimeError(f"{graph}: nonfinite weights")
    if (frame[["weight", "raw_weight"]] < 0).any().any():
        raise RuntimeError(f"{graph}: negative weights")
    sums = frame.groupby(["block_id", "dst"])["weight"].sum().to_numpy()
    if not np.allclose(sums, 1.0, rtol=0, atol=1e-8):
        raise RuntimeError(f"{graph}: normalized incoming weights do not sum to one")
    frame["graph"] = graph
    return frame


def edge_map(frame: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {
        (str(row.src), str(row.dst)): float(row.weight)
        for row in frame.itertuples(index=False)
    }


def set_similarity(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    lm, rm = edge_map(left), edge_map(right)
    ls, rs = set(lm), set(rm)
    union = ls | rs
    intersection = ls & rs
    if not union:
        return {
            "edge_intersection": 0,
            "edge_union": 0,
            "edge_jaccard": np.nan,
            "retention_left": np.nan,
            "retention_right": np.nan,
            "weighted_jaccard": np.nan,
            "weight_spearman_union": np.nan,
        }
    left_weights = np.array([lm.get(edge, 0.0) for edge in union])
    right_weights = np.array([rm.get(edge, 0.0) for edge in union])
    denominator = np.maximum(left_weights, right_weights).sum()
    rank_left = pd.Series(left_weights).rank(method="average")
    rank_right = pd.Series(right_weights).rank(method="average")
    return {
        "edge_intersection": len(intersection),
        "edge_union": len(union),
        "edge_jaccard": len(intersection) / len(union),
        "retention_left": len(intersection) / len(ls) if ls else np.nan,
        "retention_right": len(intersection) / len(rs) if rs else np.nan,
        "weighted_jaccard": (
            np.minimum(left_weights, right_weights).sum() / denominator
            if denominator > 0
            else np.nan
        ),
        "weight_spearman_union": rank_left.corr(rank_right),
    }


def target_overlap(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    left_targets = set(left["dst"])
    right_targets = set(right["dst"])
    targets = sorted(left_targets & right_targets)
    nodes = set(left["src"]) | set(left["dst"]) | set(right["src"]) | set(right["dst"])
    jaccards = []
    observed = 0.0
    expected = 0.0
    for target in targets:
        ls = set(left.loc[left["dst"] == target, "src"])
        rs = set(right.loc[right["dst"] == target, "src"])
        union = ls | rs
        if union:
            jaccards.append(len(ls & rs) / len(union))
        observed += len(ls & rs)
        candidates = max(len(nodes - {target}), 1)
        expected += len(ls) * len(rs) / candidates
    return {
        "common_targets": len(targets),
        "mean_target_jaccard": float(np.mean(jaccards)) if jaccards else np.nan,
        "observed_target_edge_overlap": observed,
        "random_overlap_expectation": expected,
        "overlap_enrichment": observed / expected if expected > 0 else np.nan,
    }


def graph_block_summary(graphs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_rows = []
    target_rows = []
    for graph, frame in graphs.items():
        for block_id, block in frame.groupby("block_id", sort=True):
            edges = set(zip(block["src"], block["dst"]))
            nodes = set(block["src"]) | set(block["dst"])
            reciprocal = sum((dst, src) in edges for src, dst in edges)
            target = block.groupby("dst", as_index=False).agg(
                incoming_degree=("src", "size"),
                incoming_raw_strength=("raw_weight", "sum"),
                maximum_normalized_weight=("weight", "max"),
            )
            hhi = block.assign(weight_sq=block["weight"] ** 2).groupby("dst")[
                "weight_sq"
            ].sum()
            target["incoming_weight_hhi"] = target["dst"].map(hhi)
            target["effective_incoming_degree"] = 1.0 / target["incoming_weight_hhi"]
            target["graph"] = graph
            target["block_id"] = int(block_id)
            target_rows.append(target)
            out_weight = block.groupby("src")["weight"].sum()
            shares = out_weight / out_weight.sum()
            block_rows.append(
                {
                    "graph": graph,
                    "block_id": int(block_id),
                    "segment_id": (
                        int(block["segment_id"].iloc[0])
                        if "segment_id" in block and block["segment_id"].notna().any()
                        else np.nan
                    ),
                    "refit_date": (
                        pd.to_datetime(block["refit_date"]).min()
                        if "refit_date" in block
                        else pd.NaT
                    ),
                    "nodes": len(nodes),
                    "targets": block["dst"].nunique(),
                    "sources": block["src"].nunique(),
                    "edges": len(block),
                    "density": len(block) / (len(nodes) * (len(nodes) - 1))
                    if len(nodes) > 1
                    else np.nan,
                    "reciprocity": reciprocal / len(edges) if edges else np.nan,
                    "median_incoming_hhi": target["incoming_weight_hhi"].median(),
                    "median_effective_incoming_degree": target[
                        "effective_incoming_degree"
                    ].median(),
                    "median_maximum_normalized_weight": target[
                        "maximum_normalized_weight"
                    ].median(),
                    "median_incoming_raw_strength": target[
                        "incoming_raw_strength"
                    ].median(),
                    "outgoing_influence_hhi": float((shares**2).sum()),
                }
            )
    return pd.DataFrame(block_rows), pd.concat(target_rows, ignore_index=True)


def temporal_stability(graphs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for graph, frame in graphs.items():
        block_meta = (
            frame.groupby("block_id", as_index=False)
            .agg(
                segment_id=("segment_id", "first")
                if "segment_id" in frame
                else ("block_id", lambda _: 0),
                refit_date=("refit_date", "min")
                if "refit_date" in frame
                else ("block_id", "first"),
            )
            .sort_values(["segment_id", "block_id"])
        )
        for _, segment in block_meta.groupby("segment_id", sort=True):
            ids = segment["block_id"].astype(int).tolist()
            for previous, current in zip(ids, ids[1:]):
                left = frame[frame["block_id"] == previous]
                right = frame[frame["block_id"] == current]
                row = {
                    "graph": graph,
                    "previous_block_id": previous,
                    "block_id": current,
                    "segment_id": segment.loc[
                        segment["block_id"] == current, "segment_id"
                    ].iloc[0],
                    "refit_date": segment.loc[
                        segment["block_id"] == current, "refit_date"
                    ].iloc[0],
                }
                row.update(set_similarity(left, right))
                row.update(target_overlap(left, right))
                rows.append(row)
    return pd.DataFrame(rows)


def cross_graph_overlap(graphs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    comparisons = (
        ("raw_vs_residual_catboost", "raw_catboost", "residual_catboost"),
        ("raw_catboost_vs_pearson", "raw_catboost", "pearson_raw"),
        ("residual_catboost_vs_pearson", "residual_catboost", "pearson_raw"),
    )
    rows = []
    for comparison, left_name, right_name in comparisons:
        left, right = graphs[left_name], graphs[right_name]
        blocks = sorted(set(left["block_id"]) & set(right["block_id"]))
        for block_id in blocks:
            lblock = left[left["block_id"] == block_id]
            rblock = right[right["block_id"] == block_id]
            row = {
                "comparison": comparison,
                "left_graph": left_name,
                "right_graph": right_name,
                "block_id": block_id,
            }
            row.update(set_similarity(lblock, rblock))
            row.update(target_overlap(lblock, rblock))
            rows.append(row)
    return pd.DataFrame(rows)


def node_influence(graphs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for graph, frame in graphs.items():
        blocks = sorted(frame["block_id"].unique())
        states = []
        for block_id, block in frame.groupby("block_id", sort=True):
            nodes = sorted(set(block["src"]) | set(block["dst"]))
            outgoing = block.groupby("src").agg(
                out_degree=("dst", "size"),
                outgoing_weight=("weight", "sum"),
                outgoing_raw_weight=("raw_weight", "sum"),
            )
            incoming = block.groupby("dst").agg(
                in_degree=("src", "size"), incoming_raw_strength=("raw_weight", "sum")
            )
            state = pd.DataFrame({"SECID": nodes}).set_index("SECID")
            state = state.join(outgoing, how="left").join(incoming, how="left").fillna(0)
            state["top5_outgoing"] = False
            top = state.nlargest(min(5, len(state)), "outgoing_weight").index
            state.loc[top, "top5_outgoing"] = True
            state["block_id"] = int(block_id)
            states.append(state.reset_index())
        all_states = pd.concat(states, ignore_index=True)
        summary = all_states.groupby("SECID", as_index=False).agg(
            blocks_present=("block_id", "nunique"),
            mean_out_degree=("out_degree", "mean"),
            mean_outgoing_weight=("outgoing_weight", "mean"),
            median_outgoing_weight=("outgoing_weight", "median"),
            mean_outgoing_raw_weight=("outgoing_raw_weight", "mean"),
            mean_in_degree=("in_degree", "mean"),
            mean_incoming_raw_strength=("incoming_raw_strength", "mean"),
            top5_outgoing_blocks=("top5_outgoing", "sum"),
        )
        summary["graph_blocks"] = len(blocks)
        summary["top5_outgoing_frequency"] = (
            summary["top5_outgoing_blocks"] / summary["blocks_present"]
        )
        summary["graph"] = graph
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def stable_edges(graphs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for graph, frame in graphs.items():
        summary = frame.groupby(["src", "dst"], as_index=False).agg(
            blocks_present=("block_id", "nunique"),
            first_block=("block_id", "min"),
            last_block=("block_id", "max"),
            mean_weight=("weight", "mean"),
            median_weight=("weight", "median"),
            mean_raw_weight=("raw_weight", "mean"),
        )
        summary["graph_blocks"] = frame["block_id"].nunique()
        summary["unconditional_block_frequency"] = (
            summary["blocks_present"] / summary["graph_blocks"]
        )
        summary["graph"] = graph
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def sector_assortativity(
    graphs: dict[str, pd.DataFrame], sector_path: Path | None
) -> pd.DataFrame:
    if sector_path is None:
        return pd.DataFrame()
    if not sector_path.exists():
        raise RuntimeError(f"Sector map missing: {sector_path}")
    sector = pd.read_csv(sector_path)
    lookup = {column.lower(): column for column in sector.columns}
    if "secid" not in lookup or "sector" not in lookup:
        raise RuntimeError("Sector map must contain SECID and sector columns")
    sector = sector[[lookup["secid"], lookup["sector"]]].rename(
        columns={lookup["secid"]: "SECID", lookup["sector"]: "sector"}
    )
    sector["SECID"] = sector["SECID"].astype(str)
    sector = sector.dropna(subset=["sector"]).drop_duplicates("SECID")
    mapping = sector.set_index("SECID")["sector"]
    rows = []
    for graph, frame in graphs.items():
        work = frame.copy()
        work["src_sector"] = work["src"].map(mapping)
        work["dst_sector"] = work["dst"].map(mapping)
        work = work.dropna(subset=["src_sector", "dst_sector"])
        work["same_sector"] = work["src_sector"] == work["dst_sector"]
        for block_id, block in work.groupby("block_id", sort=True):
            rows.append(
                {
                    "graph": graph,
                    "block_id": int(block_id),
                    "covered_edges": len(block),
                    "edge_fraction_same_sector": block["same_sector"].mean(),
                    "weight_fraction_same_sector": block.loc[
                        block["same_sector"], "weight"
                    ].sum()
                    / block["weight"].sum(),
                }
            )
    return pd.DataFrame(rows)


def gain_structure_relation(
    root: Path, block_summary: pd.DataFrame, stability: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = root / "benchmark_v10_directed_development_residual"
    differential_path = out / "daily_loss_differentials_residual.csv"
    feature_path = root / "model_data" / "residual_directed_graph_features.parquet"
    if not differential_path.exists() or not feature_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    daily = normalize_keys(pd.read_csv(differential_path), secid=False)
    features = normalize_keys(read_table(feature_path))
    mapping = features[["TRADEDATE", "block_id"]].dropna().drop_duplicates()
    counts = mapping.groupby("TRADEDATE")["block_id"].nunique()
    if (counts > 1).any():
        raise RuntimeError("More than one residual graph block assigned to a date")
    mapping = mapping.drop_duplicates("TRADEDATE")
    if "stress_z" in features:
        stress = features[["TRADEDATE", "stress_z"]].drop_duplicates()
        mapping = mapping.merge(stress, on="TRADEDATE", how="left", validate="one_to_one")
    joined = daily.merge(mapping, on="TRADEDATE", how="inner", validate="one_to_one")
    gain_column = "d_RH1_current_graph_increment"
    if gain_column not in joined:
        raise RuntimeError(f"Residual daily differential missing {gain_column}")
    aggregations = {
        "forecast_dates": ("TRADEDATE", "nunique"),
        "mean_current_graph_gain": (gain_column, "mean"),
        "median_current_graph_gain": (gain_column, "median"),
        "positive_date_fraction": (gain_column, lambda x: float((x > 0).mean())),
    }
    if "stress_z" in joined:
        aggregations["mean_stress_z"] = ("stress_z", "mean")
    by_block = joined.groupby("block_id", as_index=False).agg(**aggregations)
    structure = block_summary[block_summary["graph"] == "residual_catboost"].copy()
    transition = stability[stability["graph"] == "residual_catboost"][
        ["block_id", "edge_jaccard", "weighted_jaccard", "mean_target_jaccard"]
    ]
    by_block = by_block.merge(structure, on="block_id", how="left", validate="one_to_one")
    by_block = by_block.merge(transition, on="block_id", how="left", validate="one_to_one")
    candidates = [
        "reciprocity",
        "median_incoming_hhi",
        "median_effective_incoming_degree",
        "median_incoming_raw_strength",
        "outgoing_influence_hhi",
        "edge_jaccard",
        "weighted_jaccard",
        "mean_target_jaccard",
        "mean_stress_z",
    ]
    rows = []
    for column in candidates:
        if column not in by_block:
            continue
        valid = by_block[["mean_current_graph_gain", column]].dropna()
        rows.append(
            {
                "outcome": "mean_current_graph_gain_M1_minus_M6",
                "structural_variable": column,
                "blocks": len(valid),
                "spearman_correlation": valid.rank().corr().iloc[0, 1]
                if len(valid) >= 3
                else np.nan,
                "analysis_status": "descriptive_exploratory_block_level",
            }
        )
    return by_block, pd.DataFrame(rows)


def save_figures(
    output: Path,
    stability: pd.DataFrame,
    overlap: pd.DataFrame,
    nodes: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for graph, group in stability.groupby("graph", sort=True):
        ax.plot(group["block_id"], group["mean_target_jaccard"], marker="o", label=graph)
    ax.set(xlabel="Refit block", ylabel="Mean target-level edge Jaccard", ylim=(0, 1))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "temporal_edge_stability.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for comparison, group in overlap.groupby("comparison", sort=True):
        ax.plot(group["block_id"], group["mean_target_jaccard"], marker="o", label=comparison)
    ax.set(xlabel="Refit block", ylabel="Mean target-level edge Jaccard", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "cross_graph_overlap.png", dpi=180)
    plt.close(fig)

    selected = nodes[nodes["graph"].isin(["raw_catboost", "residual_catboost"])].copy()
    selected = (
        selected.sort_values(["graph", "mean_outgoing_weight"], ascending=[True, False])
        .groupby("graph", as_index=False)
        .head(10)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
    for ax, (graph, group) in zip(axes, selected.groupby("graph", sort=True)):
        group = group.sort_values("mean_outgoing_weight")
        ax.barh(group["SECID"], group["mean_outgoing_weight"])
        ax.set_title(graph)
        ax.set_xlabel("Mean total outgoing normalized weight")
    fig.tight_layout()
    fig.savefig(figure_dir / "top_outgoing_influence_nodes.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_v02"))
    parser.add_argument("--sector-map", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_dir = args.root / "graphs_v10"
    output = args.root / "graph_structure_v10"
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: graph_dir / filename for name, filename in GRAPH_FILES.items()}
    graphs = {name: load_edges(path, name) for name, path in paths.items()}
    common_blocks = sorted(
        set.intersection(*(set(frame["block_id"]) for frame in graphs.values()))
    )
    if len(common_blocks) < 2:
        raise RuntimeError("Fewer than two graph blocks are common to all graph families")

    block_summary, target_summary = graph_block_summary(graphs)
    stability = temporal_stability(graphs)
    block_summary_common = block_summary[
        block_summary["block_id"].isin(common_blocks)
    ].copy()
    stability_common = stability[
        stability["block_id"].isin(common_blocks)
        & stability["previous_block_id"].isin(common_blocks)
    ].copy()
    overlap = cross_graph_overlap(graphs)
    nodes = node_influence(graphs)
    stable = stable_edges(graphs)
    sectors = sector_assortativity(graphs, args.sector_map)
    gain_blocks, gain_correlations = gain_structure_relation(
        args.root, block_summary, stability
    )

    tables = {
        "graph_block_summary.csv": block_summary,
        "graph_block_summary_common_blocks.csv": block_summary_common,
        "target_concentration.csv": target_summary,
        "temporal_stability.csv": stability,
        "temporal_stability_common_blocks.csv": stability_common,
        "cross_graph_overlap.csv": overlap,
        "node_influence_summary.csv": nodes,
        "stable_edges.csv": stable,
        "forecast_gain_by_block.csv": gain_blocks,
        "forecast_gain_structure_correlations.csv": gain_correlations,
    }
    if not sectors.empty:
        tables["sector_assortativity.csv"] = sectors
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)

    if not args.no_figures:
        save_figures(output, stability_common, overlap, nodes)

    design = {
        "analysis": "V10 directed graph structural audit",
        "analysis_status": "exploratory_post_development",
        "forecasts_refit_or_modified": False,
        "primary_forecast_hypotheses_modified": False,
        "graphs": GRAPH_FILES,
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "common_block_scope": common_blocks,
        "primary_structural_comparisons_use_common_blocks": True,
        "sector_map": str(args.sector_map) if args.sector_map else None,
        "absolute_raw_residual_qlike_compared": False,
    }
    (output / "graph_structure_design.json").write_text(
        json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    graph_summary_full = block_summary.groupby("graph", as_index=False).agg(
        blocks=("block_id", "nunique"),
        median_nodes=("nodes", "median"),
        median_edges=("edges", "median"),
        median_reciprocity=("reciprocity", "median"),
        median_incoming_hhi=("median_incoming_hhi", "median"),
    )
    graph_summary = block_summary_common.groupby("graph", as_index=False).agg(
        blocks=("block_id", "nunique"),
        median_nodes=("nodes", "median"),
        median_edges=("edges", "median"),
        median_reciprocity=("reciprocity", "median"),
        median_incoming_hhi=("median_incoming_hhi", "median"),
    )
    temporal_summary = stability_common.groupby("graph", as_index=False).agg(
        transitions=("block_id", "size"),
        mean_target_jaccard=("mean_target_jaccard", "mean"),
        median_target_jaccard=("mean_target_jaccard", "median"),
        mean_weighted_jaccard=("weighted_jaccard", "mean"),
    )
    cross_summary = overlap.groupby("comparison", as_index=False).agg(
        blocks=("block_id", "size"),
        mean_target_jaccard=("mean_target_jaccard", "mean"),
        median_overlap_enrichment=("overlap_enrichment", "median"),
        mean_weighted_jaccard=("weighted_jaccard", "mean"),
    )
    summary = [
        "# V10 directed graph structural audit",
        "",
        "Status: exploratory post-development. No forecasts were refit.",
        "",
        f"Primary comparisons use the {len(common_blocks)} blocks common to all graph families.",
        "",
        "## Block structure — common-block scope",
        "",
        graph_summary.to_markdown(index=False),
        "",
        "## Coverage audit — each graph's full available scope",
        "",
        graph_summary_full.to_markdown(index=False),
        "",
        "## Temporal stability — common-block scope",
        "",
        temporal_summary.to_markdown(index=False),
        "",
        "## Cross-graph overlap",
        "",
        cross_summary.to_markdown(index=False),
    ]
    if not gain_correlations.empty:
        summary.extend(
            [
                "",
                "## Structure and frozen M1-to-M6 forecast gain",
                "",
                gain_correlations.to_markdown(index=False),
            ]
        )
    if sectors.empty:
        summary.extend(
            [
                "",
                "Sector assortativity was not computed because no explicit SECID-to-sector map was supplied.",
            ]
        )
    (output / "STRUCTURAL_GRAPH_RESULTS_V10.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    print(f"\n===== GRAPH STRUCTURE — COMMON {len(common_blocks)}-BLOCK SCOPE =====")
    print(graph_summary.to_string(index=False))
    print(f"\n===== TEMPORAL STABILITY — COMMON {len(common_blocks)}-BLOCK SCOPE =====")
    print(temporal_summary.to_string(index=False))
    print("\n===== CROSS-GRAPH OVERLAP =====")
    print(cross_summary.to_string(index=False))
    if not gain_correlations.empty:
        print("\n===== STRUCTURE VS M1->M6 GAIN (DESCRIPTIVE) =====")
        print(gain_correlations.to_string(index=False))


if __name__ == "__main__":
    main()
