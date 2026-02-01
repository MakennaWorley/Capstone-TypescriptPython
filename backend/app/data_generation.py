"""
Probabilistic Ancestral Inference — Data Generator (msprime + tskit)

What this script does:
1) Simulates ancestry with msprime.sim_ancestry (the stochastic backbone).
2) Overlays mutations with msprime.sim_mutations (observable genotypes).
3) Extracts genotypes as:
   - Haploid matrix: (sites x sample_genomes)
   - Diploid dosage matrix: (sites x individuals), dosage in {0,1,2} for typical biallelic sites.
4) Applies controlled masking to create "observed" data.
5) Saves:
   - *.trees (tree sequence for ground truth + reproducibility)
   - *.truth_genotypes.csv
   - *.observed_genotypes.csv
   - *.sites.csv
   - *.sample_metadata.csv
   - *.run_metadata.json (reproducible: rerun from this)
   - optional: *.run_metadata.txt (human readable)

Meta replay features:
- --meta-in path.json  => loads params from previous run metadata JSON
- --meta-out path.json => chooses meta output path
- Merge behavior when both meta and CLI specify params:
    Default: CLI wins (meta provides defaults, CLI overrides)
    Option:  --meta-wins (meta overrides CLI)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import msprime
import tskit


# -----------------------------
# Configs
# -----------------------------

@dataclass(frozen=True)
class SimConfig:
    # Sampling / population parameters
    n_diploid_samples: int = 200
    Ne: int = 10_000
    ploidy: int = 2

    # Genome / simulation parameters
    sequence_length: int = 100_000
    recombination_rate: float = 1e-8
    mutation_rate: float = 1e-8

    # Output control
    seed: int = 42
    masking_rate: float = 0.20
    output_dir: str = "datasets"
    name: str = "run1"

    # Variant filtering
    min_variants: int = 100
    max_retries: int = 10

    # Optional outputs
    write_meta_txt: bool = True


# -----------------------------
# Utils
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def now_utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def add_to_file(text: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "datasets.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")

# -----------------------------
# Meta replay
# -----------------------------

def config_to_dict(cfg: SimConfig) -> Dict[str, Any]:
    """Dataclass -> JSON-ready dict."""
    d = asdict(cfg)
    # Ensure JSON types
    for k, v in d.items():
        if isinstance(v, (np.integer, np.floating)):
            d[k] = v.item()
    return d

def dict_to_config(d: Dict[str, Any]) -> SimConfig:
    """Dict -> SimConfig, ignoring unknown keys."""
    fields = set(SimConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in d.items() if k in fields}
    return SimConfig(**filtered)

def merge_configs(cli_cfg: SimConfig, meta_cfg: SimConfig, *, meta_wins: bool) -> SimConfig:
    """
    Merge meta config with CLI config.
    If meta_wins=True: meta overrides CLI
    Else: CLI overrides meta (default behavior; meta provides defaults)
    """
    cli = config_to_dict(cli_cfg)
    meta = config_to_dict(meta_cfg)

    if meta_wins:
        merged = {**cli, **meta}
    else:
        merged = {**meta, **cli}

    return dict_to_config(merged)


# -----------------------------
# Helpers
# -----------------------------

def simulate_tree_sequence(cfg: SimConfig) -> tskit.TreeSequence:
    """
    Step A: ancestry (coalescent + recombination)
    Step B: mutations (observables)
    """
    ts_ancestry = msprime.sim_ancestry(
        samples=cfg.n_diploid_samples,
        ploidy=cfg.ploidy,
        population_size=cfg.Ne,
        sequence_length=cfg.sequence_length,
        recombination_rate=cfg.recombination_rate,
        random_seed=cfg.seed,
    )

    ts_mut = msprime.sim_mutations(
        ts_ancestry,
        rate=cfg.mutation_rate,
        random_seed=cfg.seed + 1,
        model="binary"
    )

    return ts_mut

def simulate_with_min_variants(cfg: SimConfig) -> Tuple[tskit.TreeSequence, SimConfig]:
    """
    Retry with shifted seeds until ts.num_sites >= cfg.min_variants, up to max_retries.
    Returns (ts, cfg_used).
    """
    last_ts = None
    last_cfg = cfg

    for attempt in range(cfg.max_retries + 1):
        seed_try = cfg.seed + attempt * 1000
        cfg_try = dict_to_config({**config_to_dict(cfg), "seed": seed_try})
        ts = simulate_tree_sequence(cfg_try)
        last_ts, last_cfg = ts, cfg_try

        if ts.num_sites >= cfg.min_variants:
            return ts, cfg_try

    # Fall back to last attempt if min_variants not reached
    return last_ts, last_cfg


# -----------------------------
# Masking
# -----------------------------

def genotype_matrix(ts: tskit.TreeSequence) -> np.ndarray:
    """Genotypes: (num_sites x num_sample_genomes) allele indices."""
    return ts.genotype_matrix()

def haploid_to_diploid_dosage(G_hap: np.ndarray, ploidy: int) -> np.ndarray:
    """
    Convert haploid genotypes to diploid dosage (sites x individuals).
    Assumes msprime diploid ordering: [ind0_h0, ind0_h1, ind1_h0, ind1_h1, ...]
    """
    if ploidy != 2:
        raise ValueError("Diploid dosage conversion assumes ploidy=2.")
    if G_hap.shape[1] % 2 != 0:
        raise ValueError("Expected even number of sample genomes for diploids.")
    return G_hap[:, 0::2] + G_hap[:, 1::2]

def mask(X: np.ndarray, masking_rate: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mask entries randomly at masking_rate; return (masked, observed_mask).
    observed_mask=True means value kept (observed).
    """
    Xf = X.astype(float, copy=True)
    observed_mask = rng.random(Xf.shape) >= masking_rate
    Xf[~observed_mask] = np.nan
    return Xf, observed_mask

def sites_table(ts: tskit.TreeSequence) -> pd.DataFrame:
    """Site index + position + ancestral + derived states (for debugging / joins)."""
    rows = []
    for site in ts.sites():
        derived = []
        for mut in site.mutations:
            derived.append(mut.derived_state)
        rows.append(
            {
                "site_index": int(site.id),
                "position": int(site.position),
                "ancestral_state": site.ancestral_state,
                "derived_states": ",".join(sorted(set(derived))) if derived else "",
            }
        )
    return pd.DataFrame(rows)

def sample_metadata(ts: tskit.TreeSequence, ploidy: int) -> pd.DataFrame:
    """Map sample genomes to individuals and haplotypes."""
    sample_nodes = ts.samples()
    data = []
    for sample_index, node_id in enumerate(sample_nodes):
        data.append(
            {
                "sample_index": int(sample_index),
                "node_id": int(node_id),
                "individual_index": int(sample_index // ploidy),
                "hap_within_individual": int(sample_index % ploidy),
            }
        )
    return pd.DataFrame(data)


# -----------------------------
# IO paths + metadata writing
# -----------------------------

def build_paths(cfg: SimConfig) -> Dict[str, str]:
    """Centralize output naming."""
    ensure_dir(cfg.output_dir)
    base = os.path.join(cfg.output_dir, cfg.name)

    return {
        "trees": f"{base}.trees",
        "truth_csv": f"{base}.truth_genotypes.csv",
        "observed_csv": f"{base}.observed_genotypes.csv",
        "sites_csv": f"{base}.sites.csv",
        "sample_metadata_csv": f"{base}.sample_metadata.csv",
        "meta_json": f"{base}.run_metadata.json",
        "meta_txt": f"{base}.run_metadata.txt",
    }

def write_meta_txt(path: str, cfg_used: SimConfig, derived: Dict[str, Any], outputs: Dict[str, str]) -> None:
    """Optional human-readable meta, similar to make_msprime_families style."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Run metadata (human readable)\n")
        f.write(f"created_at={now_utc_iso()}\n\n")
        f.write("[params]\n")
        for k, v in sorted(config_to_dict(cfg_used).items()):
            f.write(f"{k}={v}\n")
        f.write("\n[derived]\n")
        for k, v in sorted(derived.items()):
            f.write(f"{k}={v}\n")
        f.write("\n[outputs]\n")
        for k, v in sorted(outputs.items()):
            f.write(f"{k}={v}\n")


# -----------------------------
# Simulation
# -----------------------------

def run_generation(cfg: SimConfig, *, meta_in: Optional[str] = None, meta_out: Optional[str] = None) -> Dict[str, str]:
    """
    Generate a dataset and write outputs.
    Returns output paths.
    """

    # Simulate (retry if too few variants)
    ts, cfg_used = simulate_with_min_variants(cfg)

    # Extract genotypes
    G_hap = genotype_matrix(ts)
    G_dip = haploid_to_diploid_dosage(G_hap, ploidy=cfg_used.ploidy)

    # Mask at diploid-individual level
    rng = np.random.default_rng(cfg_used.seed + 999)
    G_obs, obs_mask = mask(G_dip, cfg_used.masking_rate, rng=rng)

    # Build tables
    df_sites = sites_table(ts)
    df_samples = sample_metadata(ts, ploidy=cfg_used.ploidy)

    # DataFrames with consistent column naming
    individual_cols = [f"ind_{i:04d}" for i in range(G_dip.shape[1])]

    df_truth = pd.DataFrame(G_dip, columns=individual_cols)
    df_obs = pd.DataFrame(G_obs, columns=individual_cols)

    # Prepend site columns
    df_truth = pd.concat([df_sites, df_truth], axis=1)
    df_obs = pd.concat([df_sites, df_obs], axis=1)

    # Output paths
    outputs = build_paths(cfg_used)
    if meta_out is not None:
        outputs["meta_json"] = meta_out
        # keep txt aligned if user explicitly named json
        if outputs["meta_txt"].endswith(".run_metadata.txt"):
            # only override txt if the default naming convention was used
            pass

    # Persist tree sequence for ground truth
    ts.dump(outputs["trees"])

    # Persist CSVs
    df_truth.to_csv(outputs["truth_csv"], index=False)
    df_obs.to_csv(outputs["observed_csv"], index=False)
    df_sites.to_csv(outputs["sites_csv"], index=False)
    df_samples.to_csv(outputs["sample_metadata_csv"], index=False)

    # Derived/run info (very helpful for reproducibility + debugging)
    derived = {
        "ts_num_sites": int(ts.num_sites),
        "ts_num_mutations": int(ts.num_mutations),
        "ts_num_trees": int(ts.num_trees),
        "ts_num_samples": int(ts.num_samples),
        "sequence_length": float(ts.sequence_length),
        "haploid_genotypes_shape": [int(G_hap.shape[0]), int(G_hap.shape[1])],
        "diploid_dosage_shape": [int(G_dip.shape[0]), int(G_dip.shape[1])],
        "masking_rate": float(cfg_used.masking_rate),
        "observed_nonmasking_fraction": float(np.isfinite(G_obs).mean()),
    }

    meta_payload = {
        "created_at": now_utc_iso(),
        "script": os.path.basename(__file__),
        "meta_in": meta_in,
        "params": config_to_dict(cfg_used),   # <-- the key round-trip piece
        "derived": derived,
        "outputs": outputs,
    }

    # Save JSON meta (round-trippable)
    write_json(outputs["meta_json"], meta_payload)

    # Optional TXT meta (human readable)
    if cfg_used.write_meta_txt:
        write_meta_txt(outputs["meta_txt"], cfg_used, derived, outputs)

    # Console summary
    print("Generation complete.")
    print(f"   Trees:    {outputs['trees']}")
    print(f"   Truth:    {outputs['truth_csv']}")
    print(f"   Observed: {outputs['observed_csv']}")
    print(f"   Meta:     {outputs['meta_json']}")
    print(f"   Sites:    {outputs['sites_csv']}")
    print(f"   Samples:  {outputs['sample_metadata_csv']}")
    print(f"   Variants: {ts.num_sites}")

    return outputs

# -----------------------------
# Checks
# -----------------------------

def checks(truth_csv: str, observed_csv: str) -> None:
    truth = pd.read_csv(truth_csv)
    obs = pd.read_csv(observed_csv)

    assert len(truth) == len(obs), "Row mismatch between truth and observed"
    assert (truth["site_index"].values == obs["site_index"].values).all(), "Site index mismatch"

    genotype_cols = [c for c in truth.columns if c.startswith("ind_")]
    truth_vals = truth[genotype_cols].to_numpy()
    obs_vals = obs[genotype_cols].to_numpy()

    masking_fraction = np.isnan(obs_vals).mean()
    print(f"[check] Observed masking fraction: {masking_fraction:.3f}")

    uniq = np.unique(truth_vals)
    print(f"[check] Unique truth dosages (first few): {uniq[:10]} (total unique={len(uniq)})")

# -----------------------------
# CLI / main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate msprime stochastic genotype datasets with reproducible meta replay.")

    # Core params (mirror SimConfig fields)
    ap.add_argument("--n-diploid-samples", type=int, default=SimConfig.n_diploid_samples)
    ap.add_argument("--Ne", type=int, default=SimConfig.Ne)
    ap.add_argument("--ploidy", type=int, default=SimConfig.ploidy)

    ap.add_argument("--sequence-length", type=int, default=SimConfig.sequence_length)
    ap.add_argument("--recombination-rate", type=float, default=SimConfig.recombination_rate)
    ap.add_argument("--mutation-rate", type=float, default=SimConfig.mutation_rate)

    ap.add_argument("--seed", type=int, default=None, help="Random seed; if omitted, a random seed is chosen and recorded in meta.")
    ap.add_argument("--masking-rate", type=float, default=SimConfig.masking_rate)
    ap.add_argument("--output-dir", type=str, default=SimConfig.output_dir)
    ap.add_argument("--name", type=str, default=SimConfig.name)

    ap.add_argument("--min-variants", type=int, default=SimConfig.min_variants)
    ap.add_argument("--max-retries", type=int, default=SimConfig.max_retries)

    ap.add_argument("--write-meta-txt", action="store_true", default=SimConfig.write_meta_txt)
    ap.add_argument("--no-write-meta-txt", action="store_false", dest="write_meta_txt")

    # Meta replay flags
    ap.add_argument("--meta-in", type=str, default=None, help="JSON meta file to reproduce a previous run.")
    ap.add_argument("--meta-out", type=str, default=None, help="Explicit JSON meta output path (otherwise auto-named).")

    # Merge behavior
    merge = ap.add_mutually_exclusive_group()
    merge.add_argument("--cli-wins", action="store_true", default=True,
                       help="When using --meta-in, CLI overrides meta (default).")
    merge.add_argument("--meta-wins", action="store_true", default=False,
                       help="When using --meta-in, meta overrides CLI.")

    ap.add_argument("--checks", action="store_true", default=False,
                    help="Run lightweight integrity checks on saved CSVs.")

    return ap.parse_args()

def args_to_config(args: argparse.Namespace) -> SimConfig:
    # If no seed provided, choose random seed and record it
    seed = args.seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**32))

    return SimConfig(
        n_diploid_samples=args.n_diploid_samples,
        Ne=args.Ne,
        ploidy=args.ploidy,
        sequence_length=args.sequence_length,
        recombination_rate=args.recombination_rate,
        mutation_rate=args.mutation_rate,
        seed=seed,
        masking_rate=args.masking_rate,
        output_dir=args.output_dir,
        name=args.name,
        min_variants=args.min_variants,
        max_retries=args.max_retries,
        write_meta_txt=args.write_meta_txt,
    )

def create_data() -> None:
    args = parse_args()

    # Build CLI config first
    cli_cfg = args_to_config(args)

    # If meta-in is provided, load it and merge
    meta_in = args.meta_in
    meta_cfg = None
    if meta_in:
        meta = read_json(meta_in)
        meta_cfg = dict_to_config(meta.get("params", meta))
        # Merge meta with CLI
        cfg = merge_configs(cli_cfg, meta_cfg, meta_wins=args.meta_wins)
    else:
        cfg = cli_cfg

    # Run generation
    outputs = run_generation(cfg, meta_in=meta_in, meta_out=args.meta_out)

    # Add to .txt file
    add_to_file(args.name, args.output_dir)

    # Optional quick checksgener
    if args.checks:
        checks(outputs["truth_csv"], outputs["observed_csv"])


if __name__ == "__main__":
    create_data()

# -----------------------------
# API
# -----------------------------

def build_config_from_params(params: Dict[str, Any]) -> "SimConfig":
    """
    Builds a SimConfig from incoming params (e.g., API request JSON).
    - Starts from SimConfig defaults.
    - Overlays keys from params if they exist on SimConfig.
    - Ensures we have a seed (generate if missing/None).
    """

    valid_fields = SimConfig.__dataclass_fields__.keys()

    # 1. Filter the incoming params
    filtered_params = {k: v for k, v in params.items() if k in valid_fields and v is not None}

    # 2. Check if a seed was provided; if not, generate one (matching CLI behavior)
    if "seed" not in filtered_params or filtered_params["seed"] is None:
        filtered_params["seed"] = int(np.random.SeedSequence().entropy % (2**32))

    # 3. Create the frozen object
    return SimConfig(**filtered_params)

def create_data_from_params(
    params: Dict[str, Any],
    *,
    meta_in: Optional[str] = None,
    meta_out: Optional[str] = None,
    meta_wins: bool = False
) -> Dict[str, Any]:
    """
    Programmatic equivalent of create_data(), designed for API usage.

    params: dict from request body
    meta_in: optional path to a meta JSON to replay
    meta_out: optional path to write meta JSON (if you do metadata logging)
    meta_wins: if True and meta_in provided, meta overrides params
    """
    if meta_in:
        loaded = read_json(meta_in)
        if meta_wins:
            # meta overwrites request params
            merged = {**params, **loaded}
        else:
            # request params overwrite meta
            merged = {**loaded, **params}
        params = merged

    cfg = build_config_from_params(params)

    if meta_out:
        meta_payload = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg.__dict__
        write_json(meta_out, meta_payload)

    outputs = run_generation(cfg)

    add_to_file(cfg.name, cfg.output_dir)

    return {
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg.__dict__,
        "outputs": outputs,
    }
