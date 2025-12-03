import os, re, argparse
import numpy as np
import pandas as pd
import msprime, tskit

DEFAULT_OUT = "families"
DEFAULT_TREE = "trees"

# ----------------------------- utils -----------------------------
def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","1","y"): return True
    if v in ("no","false","f","0","n"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s or "run"

# ------------------------- simulation core ------------------------
def simulate_ancestry(n_diploid: int,
                      seq_len: float,
                      recomb: float,
                      Ne: int,
                      seed: int) -> tskit.TreeSequence:
    ts = msprime.sim_ancestry(
        samples=n_diploid,
        ploidy=2,
        sequence_length=seq_len,
        population_size=Ne,
        recombination_rate=recomb,
        model="dtwf",
        random_seed=seed
    )
    return ts

def add_single_trait_site(ts: tskit.TreeSequence,
                          position: float,
                          pA: float,
                          seed: int) -> tskit.TreeSequence:
    """
    Force a single biallelic site at `position` and imprint derived allele
    onto sample nodes with frequency ~ pA (independent per haplotype).
    """
    rng = np.random.default_rng(seed)
    tables = ts.dump_tables()

    site_id = tables.sites.add_row(position=position, ancestral_state="0")
    # Assign derived allele to each sample node with probability pA
    sample_nodes = ts.samples()
    draw = rng.binomial(1, pA, size=len(sample_nodes))
    for node_id, d in zip(sample_nodes, draw):
        if d == 1:
            tables.mutations.add_row(site=site_id, node=node_id, derived_state="1")
    return tables.tree_sequence()

def diploid_dosage_vector(ts: tskit.TreeSequence) -> np.ndarray:
    """
    Return diploid dosages (0,1,2) at the first site, length = n_diploid.
    """
    G = ts.genotype_matrix()   # shape: (num_sites, num_sample_nodes)
    if ts.num_sites == 0:
        raise RuntimeError("Tree sequence has no sites.")
    hap = G[0, :]              # first (and only) site
    n_dip = ts.num_samples // 2
    if hap.size != 2*n_dip:
        raise RuntimeError("Unexpected haploid sample count.")
    return hap.reshape(n_dip, 2).sum(axis=1)  # (n_dip,)

def make_families(dosages_parents: np.ndarray,
                  n_families: int,
                  kids_per_family: int,
                  seed: int) -> pd.DataFrame:
    """
    Pair parents randomly into families, then simulate kids' dosage by
    Bernoulli draws with p = parent_dosage/2 from each parent.
    """
    rng = np.random.default_rng(seed)
    n_parents = 2 * n_families
    if len(dosages_parents) < n_parents:
        raise ValueError("Not enough parents in the tree sequence.")
    idx = np.arange(len(dosages_parents))
    rng.shuffle(idx)
    parent_pairs = idx[:n_parents].reshape(n_families, 2)

    rows = []
    for fam_id, (p1, p2) in enumerate(parent_pairs):
        a1, a2 = dosages_parents[p1], dosages_parents[p2]  # 0/1/2
        sib_count = kids_per_family
        for k in range(sib_count):
            kid = rng.binomial(1, a1/2) + rng.binomial(1, a2/2)   # 0/1/2
            rows.append(dict(
                family_id=fam_id,
                kid_index=k,
                p1_dosage=int(a1),
                p2_dosage=int(a2),
                kid_dosage=int(kid)
            ))
    return pd.DataFrame(rows)

def apply_masking(df: pd.DataFrame, mask_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m1 = rng.random(len(df)) < mask_rate
    m2 = rng.random(len(df)) < mask_rate
    out = df.copy()
    out["p1_obs"] = out["p1_dosage"].astype(float)
    out["p2_obs"] = out["p2_dosage"].astype(float)
    out.loc[m1, "p1_obs"] = np.nan
    out.loc[m2, "p2_obs"] = np.nan
    out["p1_missing"] = m1.astype(int)
    out["p2_missing"] = m2.astype(int)
    return out

def label_binary_trait(df: pd.DataFrame,
                       model: str,
                       penetrance: float,
                       seed: int) -> pd.DataFrame:
    """
    model: 'dominant' or 'recessive'
    penetrance: probability of expression when genotype is risk-carrying.
    """
    rng = np.random.default_rng(seed)
    if model == "dominant":
        p = np.where(df["kid_dosage"] >= 1, penetrance, 1 - penetrance)
    elif model == "recessive":
        p = np.where(df["kid_dosage"] == 2, penetrance, 1 - penetrance)
    else:
        raise ValueError("model must be 'dominant' or 'recessive'")
    y = (rng.random(len(df)) < p).astype(int)
    out = df.copy()
    out["y"] = y
    out["penetrance"] = penetrance
    out["model"] = model
    return out

# ------------------------------ CLI -------------------------------
def main():
    ap = argparse.ArgumentParser(description="MVP family dataset generator using msprime.")
    # naming / outputs
    ap.add_argument("--name", type=str, default=None,
                    help="Base name; files saved under datasets/<name>_*.csv and optionally .trees")
    # UPDATED: Changed from 'out' to 'base-out' to generate both 'masked' and 'unmasked' names
    ap.add_argument("--base-out", type=str, default=DEFAULT_OUT, help="Base name for output CSVs (e.g., 'families' will create 'families_masked_out.csv' and 'families_unmasked_out.csv')")
    ap.add_argument("--save-ts", type=str, default=None, help="Optional explicit .trees output name")
    ap.add_argument("--trees", type=str2bool, nargs="?", const=True, default=False,
                    help="If set, also save the .trees (auto-named if --save-ts not given).")

    # population / locus
    ap.add_argument("--families", type=int, default=5000, help="Number of families")
    ap.add_argument("--kids-per-family", type=int, default=2, help="Children per family")
    ap.add_argument("--pA", type=float, default=0.3, help="Risk-allele frequency at the trait locus (per haplotype)")
    ap.add_argument("--penetrance", type=float, default=0.9, help="Trait penetrance for risk genotypes")
    ap.add_argument("--model", type=str, choices=["dominant","recessive"], default="dominant",
                    help="Genetic model for labeling")

    # masking / tech
    ap.add_argument("--mask-rate", type=float, default=0.6, help="Probability of masking each parent's genotype")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--Ne", type=int, default=10_000, help="Effective population size for ancestry sim")
    ap.add_argument("--seq-len", type=float, default=1e5, help="Sequence length (bp)")
    ap.add_argument("--recomb", type=float, default=1e-8, help="Recombination rate per bp")
    args = ap.parse_args()

    # seed
    if args.seed is None:
        args.seed = int(np.random.SeedSequence().entropy % (2**32))
        print(f"[info] No --seed provided. Using random seed: {args.seed}")
    else:
        print(f"[info] Using --seed {args.seed}")

    # names & dirs
    base = slugify(args.name) if args.name else "msprime"
    prefix = f"{base}"
    out_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(out_dir, exist_ok=True)

    # UPDATED: Generate three output filenames
    out_base = slugify(args.base_out)
    out_masked_csv = os.path.join(out_dir, f"{prefix}_{out_base}_masked_out.csv")
    out_unmasked_csv = os.path.join(out_dir, f"{prefix}_{out_base}_unmasked_out.csv")
    out_meta_txt = os.path.join(out_dir, f"{prefix}_{out_base}_meta.txt") # NEW METADATA PATH
    
    out_trees = None
    if args.save_ts:
        out_trees = os.path.join(out_dir, f"{prefix}_{slugify(args.save_ts)}.trees")
    elif args.trees:
        out_trees = os.path.join(out_dir, f"{prefix}_{DEFAULT_TREE}.trees")

    # ---- 1) ancestry (diploid) ----
    n_parents = 2 * args.families
    ts0 = simulate_ancestry(
        n_diploid=n_parents,
        seq_len=args.seq_len,
        recomb=args.recomb,
        Ne=args.Ne,
        seed=args.seed
    )

    # ---- 2) force single trait site with allele freq ~ pA ----
    pos = int(args.seq_len // 2)
    ts = add_single_trait_site(ts0, position=pos, pA=args.pA, seed=args.seed + 1)

    # ---- 3) build parent dosages (0/1/2 at the trait locus) ----
    parent_dos = diploid_dosage_vector(ts)  # len = n_parents

    # ---- 4) families & kids (simulated) ----
    fam_df = make_families(
        dosages_parents=parent_dos,
        n_families=args.families,
        kids_per_family=args.kids_per_family,
        seed=args.seed + 2
    )

    # ---- 5) label binary phenotype ----
    fam_df = label_binary_trait(fam_df,
                                model=args.model,
                                penetrance=args.penetrance,
                                seed=args.seed + 4)

    # ---- 6) Prepare UNMASKED DataFrame (Ground Truth) ----
    # This DataFrame contains the final, correct integer dosage values.
    fam_df_unmasked = fam_df[["family_id", "kid_index", 
                              "p1_dosage", "p2_dosage", "kid_dosage", "y"]].copy()
    
    # IMPORTANT: Convert parent dosages to float for masking/NaN later.
    fam_df_unmasked["p1_dosage"] = fam_df_unmasked["p1_dosage"].astype(float)
    fam_df_unmasked["p2_dosage"] = fam_df_unmasked["p2_dosage"].astype(float)
    
    # ---- 7) Prepare MASKED DataFrame (Model Input) ----
    fam_df_masked = fam_df_unmasked.copy()
    
    # Apply masking directly to the parent dosage columns of the masked copy
    rng = np.random.default_rng(args.seed + 3)
    m1 = rng.random(len(fam_df_masked)) < args.mask_rate
    m2 = rng.random(len(fam_df_masked)) < args.mask_rate
    
    # Mask the parent dosages in the 'masked' copy
    fam_df_masked.loc[m1, "p1_dosage"] = np.nan
    fam_df_masked.loc[m2, "p2_dosage"] = np.nan

    # ---- 8) save ----
    # Save the unmasked data
    fam_df_unmasked.to_csv(out_unmasked_csv, index=False)
    # Save the masked data
    fam_df_masked.to_csv(out_masked_csv, index=False)
    
    # NEW: Save Metadata
    with open(out_meta_txt, "w") as f:
        f.write(f"# Metadata for Family Simulation Run: {prefix}_{out_base}\n\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"families={args.families}\n")
        f.write(f"kids_per_family={args.kids_per_family}\n")
        f.write(f"mask_rate={args.mask_rate}\n")
        f.write(f"risk_allele_freq_pA={args.pA}\n")
        f.write(f"trait_model={args.model}\n")
        f.write(f"trait_penetrance={args.penetrance}\n")
        f.write(f"Ne={args.Ne}\n")
        f.write(f"seq_len={args.seq_len}\n")
        f.write(f"recomb={args.recomb}\n")

    if out_trees:
        ts.dump(out_trees)

    print("\n## ✅ Output Summary ##")
    print(f"[ok] Saved unmasked dataset: {out_unmasked_csv}")
    print(f"[ok] Saved masked dataset: {out_masked_csv}")
    print(f"[ok] Saved metadata: {out_meta_txt}")
    if out_trees:
        print(f"[ok] Saved tree sequence: {out_trees}")
    print(f"[info] Rows: {len(fam_df_unmasked)}, Families: {args.families}, Kids/family: {args.kids_per_family}")

if __name__ == "__main__":
    main()