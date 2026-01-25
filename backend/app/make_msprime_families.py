"""
Supports:
- msprime ancestry + recombination: Simulates founder haplotypes using the DTWF model.
- Independent nuclear families: Generates parent-child trios/siblings rather than a single population-wide pedigree.
- Multi-site genotypes: Supports a designated Trait Locus (S0) and multiple additional Marker Loci.
- Binary phenotype labeling: Assigns 'y' based on dominant or recessive models with adjustable penetrance.
- Linked Parent Masking: If a parent's trait dosage is masked, all their associated marker dosages are also masked (simulating missing individual data).
- Wide-format export: Optimized for sibling/parent-based Bayesian and HMM model inputs.
- Reproducible seeds: Global seed management for consistent stochastic simulation.
"""

import os, re, argparse
import numpy as np
import pandas as pd
import msprime, tskit

DEFAULT_OUT = "families"
DEFAULT_TREE = "trees"
TRAIT_SITE = 0

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

def add_mutations(ts: tskit.TreeSequence,
                  mu: float,
                  seed: int) -> tskit.TreeSequence:
    """
    Adds mutations uniformly at rate mu.
    """
    ts_mut = msprime.sim_mutations(ts, rate=mu, model="binary", random_seed=seed)
    if ts_mut.num_sites == 0:
        # We handle this case by deleting all sites later, but we warn the user
        print(f"[warning] Mutation rate mu={mu} resulted in no sites. Please increase mu or seq_len.")
    return ts_mut

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

def diploid_dosage_matrix(ts: tskit.TreeSequence) -> np.ndarray:
    """
    Return diploid dosages (0,1,2) for ALL sites.
    shape: (num_diploid, num_sites)
    """
    G = ts.genotype_matrix()   # shape: (num_sites, num_sample_nodes)
    if ts.num_sites == 0:
        raise RuntimeError("Tree sequence has no sites.")
    n_dip = ts.num_samples // 2
    n_sites = ts.num_sites
    # Reshape (n_sites, 2*n_dip) -> (n_dip, 2, n_sites). Sum over ploidy (axis=1)
    return G.T.reshape(n_dip, 2, n_sites).sum(axis=1) # (n_dip, n_sites)

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

def make_families_wide(dosages_parents: np.ndarray, # Shape: (N_all_parents, N_sites)
                       n_families: int,
                       kids_per_family: int,
                       seed: int) -> pd.DataFrame:
    """
    Pair parents randomly into families and simulate kid dosages for all sites 
    assuming independent segregation *per site*.
    """
    rng = np.random.default_rng(seed)
    n_sites = dosages_parents.shape[1]
    n_parents_all = dosages_parents.shape[0]
    n_parents_needed = 2 * n_families

    if n_parents_all < n_parents_needed:
        raise ValueError("Not enough parents in the tree sequence.")
    
    # Select and pair parents
    idx = np.arange(n_parents_all)
    rng.shuffle(idx)
    parent_pairs_idx = idx[:n_parents_needed].reshape(n_families, 2)
    
    rows = []
    
    for fam_id, (p1_idx, p2_idx) in enumerate(parent_pairs_idx):
        # Dosage vectors for p1 and p2 across all sites
        a1_vec = dosages_parents[p1_idx, :]
        a2_vec = dosages_parents[p2_idx, :]
        
        # Base row data
        base_row = {"family_id": fam_id}
        
        # Add parent dosages (static per family)
        for s in range(n_sites):
            base_row[f"p1_dosage_S{s}"] = int(a1_vec[s])
            base_row[f"p2_dosage_S{s}"] = int(a2_vec[s])
            
        # Simulate kids
        for k in range(kids_per_family):
            kid_row = base_row.copy()
            kid_row["kid_index"] = k
            
            # Draw 1 allele from p1 and 1 from p2 for all sites simultaneously
            draw1 = rng.binomial(1, a1_vec / 2.0)
            draw2 = rng.binomial(1, a2_vec / 2.0)
            kid_vec = draw1 + draw2
            
            # Add kid dosages
            for s in range(n_sites):
                kid_row[f"kid_dosage_S{s}"] = int(kid_vec[s])
            
            rows.append(kid_row)

    return pd.DataFrame(rows)

def apply_masking(df_unmasked: pd.DataFrame, 
                  mask_rate: float, 
                  seed: int) -> pd.DataFrame:
    """
    Applies masking to the entire set of columns associated with a parent if their 
    S0 dosage is masked. Ensures masked and unmasked headers are identical.
    """
    rng = np.random.default_rng(seed)
    df_masked = df_unmasked.copy()
    
    # 1. Identify rows where P1 or P2 are masked at the S0 locus
    m1 = rng.random(len(df_masked)) < mask_rate
    m2 = rng.random(len(df_masked)) < mask_rate
    
    # 2. Get the list of all parent-related dosage columns
    p1_cols = [col for col in df_masked.columns if col.startswith("p1_dosage_S")]
    p2_cols = [col for col in df_masked.columns if col.startswith("p2_dosage_S")]
    
    # 3. Convert all target columns to float to allow for NaN values
    all_dosage_cols = p1_cols + p2_cols
    for col in all_dosage_cols:
        df_masked[col] = df_masked[col].astype(float)
        
    # 4. Apply NaN mask across ALL associated columns
    # If m1 is True for a row, set ALL P1 dosages to NaN for that row.
    df_masked.loc[m1, p1_cols] = np.nan
    # If m2 is True for a row, set ALL P2 dosages to NaN for that row.
    df_masked.loc[m2, p2_cols] = np.nan
    
    return df_masked

def label_binary_trait(df: pd.DataFrame,
                       model: str,
                       penetrance: float,
                       seed: int) -> pd.DataFrame:
    """
    Labels the binary phenotype 'y' based on the kid's dosage at the TRAIT_SITE (S0).
    """
    rng = np.random.default_rng(seed)
    
    # Use the kid's dosage at the designated trait site (S0)
    trait_col = f"kid_dosage_S{TRAIT_SITE}"
    if trait_col not in df.columns:
        raise ValueError(f"Trait site column {trait_col} not found in DataFrame.")

    kid_dosage = df[trait_col]
    
    if model == "dominant":
        p = np.where(kid_dosage >= 1, penetrance, 1 - penetrance)
    elif model == "recessive":
        p = np.where(kid_dosage == 2, penetrance, 1 - penetrance)
    else:
        raise ValueError("model must be 'dominant' or 'recessive'")
        
    y = (rng.random(len(df)) < p).astype(int)
    out = df.copy()
    out["y"] = y
    out["penetrance"] = penetrance
    out["model"] = model
    return out

def cleanup_df(df: pd.DataFrame, n_sites: int) -> pd.DataFrame:
    """Removes model/penetrance and renames columns for final output."""
    
    # 1. Drop extraneous columns
    if 'penetrance' in df.columns: df = df.drop(columns=['penetrance'])
    if 'model' in df.columns: df = df.drop(columns=['model'])
    
    # 2. Rename columns to the standardized output format
    renames = {}
    for s in range(n_sites):
        if s == TRAIT_SITE:
            renames[f"p1_dosage_S{s}"] = "p1_dosage_T"
            renames[f"p2_dosage_S{s}"] = "p2_dosage_T"
            renames[f"kid_dosage_S{s}"] = "kid_dosage_T"
        else:
            renames[f"p1_dosage_S{s}"] = f"p1_dosage_M{s}"
            renames[f"p2_dosage_S{s}"] = f"p2_dosage_M{s}"
            renames[f"kid_dosage_S{s}"] = f"kid_dosage_M{s}"
            
    df = df.rename(columns=renames)
    
    # 3. Define final column order
    n_markers = n_sites - 1
    
    # Base columns
    final_cols = ['family_id', 'kid_index', 'y']
    
    # Trait Locus columns
    final_cols.extend(['p1_dosage_T', 'p2_dosage_T', 'kid_dosage_T'])
    
    # Marker columns (p1, p2, kid for each marker)
    for m in range(1, n_markers + 1):
        final_cols.extend([f'p1_dosage_M{m}', f'p2_dosage_M{m}', f'kid_dosage_M{m}'])
        
    # Ensure all columns exist before subsetting
    final_cols = [col for col in final_cols if col in df.columns]

    return df[final_cols]

# ------------------------------ CLI -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Multi-site family dataset generator using msprime.")
    # naming / outputs
    ap.add_argument("--name", type=str, default=None,
                    help="Base name; files saved under datasets/<name>_*.csv")
    ap.add_argument("--base-out", type=str, default=DEFAULT_OUT, help="Base name for output CSVs (e.g., 'families' will create 'families_masked_out.csv' and 'families_unmasked_out.csv')")
    ap.add_argument("--save-ts", type=str, default=None, help="Optional explicit .trees output name")
    ap.add_argument("--trees", type=str2bool, nargs="?", const=True, default=False,
                    help="If set, also save the .trees (auto-named if --save-ts not given).")

    # population / locus
    ap.add_argument("--families", type=int, default=100, help="Number of families")
    ap.add_argument("--kids-per-family", type=int, default=2, help="Children per family")
    ap.add_argument("--n-markers", type=int, default=5, help="Number of additional marker loci to simulate")
    ap.add_argument("--penetrance", type=float, default=0.9, help="Trait penetrance for risk genotypes")
    ap.add_argument("--model", type=str, choices=["dominant","recessive"], default="dominant",
                    help="Genetic model for labeling")

    # masking / tech
    ap.add_argument("--mask-rate", type=float, default=0.6, help="Probability of masking each parent's dosage at the TRAIT LOCUS (S0)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--Ne", type=int, default=10_000, help="Effective population size for ancestry sim")
    ap.add_argument("--seq-len", type=float, default=1e5, help="Sequence length (bp)")
    ap.add_argument("--recomb", type=float, default=1e-8, help="Recombination rate per bp")
    ap.add_argument("--mu", type=float, default=1e-6, help="Mutation rate (to generate sites)")

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

    # Designate Trait Locus (S0) and total sites
    N_SITES = args.n_markers + 1

    # Generate output filenames
    out_base = slugify(args.base_out)
    out_unmasked_csv = os.path.join(out_dir, f"{prefix}_{out_base}_unmasked_out.csv")
    out_masked_csv = os.path.join(out_dir, f"{prefix}_{out_base}_masked_out.csv")
    out_meta_txt = os.path.join(out_dir, f"{prefix}_{out_base}_meta.txt")
    
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

    # ---- 2) Add mutations and subset sites ----
    ts_mut = add_mutations(ts0, mu=args.mu, seed=args.seed + 1)
    
    # Determine sites to keep and delete to truncate the tree sequence
    sites_to_keep = np.arange(min(ts_mut.num_sites, N_SITES))
    sites_to_delete = np.arange(sites_to_keep.size, ts_mut.num_sites) # Indices of sites to remove

    if sites_to_delete.size > 0:
        # FIX: Call delete_sites using positional argument (no 'sites=' keyword)
        ts = ts_mut.delete_sites(sites_to_delete) 
    else:
        # If no sites need deleting, use ts_mut directly
        ts = ts_mut
    
    if ts.num_sites < N_SITES:
        N_SITES = ts.num_sites
        args.n_markers = N_SITES - 1
        
    if N_SITES < 1:
        raise RuntimeError("No sites could be generated. Cannot proceed.")
        
    print(f"[info] Using {ts.num_sites} sites (Trait Locus S{TRAIT_SITE} + {ts.num_sites - 1} Markers)")

    # ---- 3) build parent dosage matrix (n_parents x n_sites) ----
    parent_dos_matrix = diploid_dosage_matrix(ts)

    # ---- 4) families & kids (simulated) ----
    fam_df = make_families_wide(
        dosages_parents=parent_dos_matrix,
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
    fam_df_unmasked_final = cleanup_df(fam_df.copy(), N_SITES)
    
    # ---- 7) Prepare MASKED DataFrame (Model Input) ----
    # Apply masking to S0, which propagates to all marker columns for the parent
    fam_df_masked_base = apply_masking(fam_df.copy(), 
                                  mask_rate=args.mask_rate, 
                                  seed=args.seed + 3)
    
    # Clean up and format the masked data to match the unmasked data headers
    fam_df_masked_final = cleanup_df(fam_df_masked_base, N_SITES)

    # ---- 8) save ----
    fam_df_unmasked_final.to_csv(out_unmasked_csv, index=False)
    fam_df_masked_final.to_csv(out_masked_csv, index=False)
    
    # Save Metadata
    with open(out_meta_txt, "w") as f:
        f.write(f"# Metadata for Family Simulation Run: {prefix}_{out_base}\n\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"families={args.families}\n")
        f.write(f"kids_per-family={args.kids_per_family}\n")
        f.write(f"mask_rate={args.mask_rate}\n")
        f.write(f"trait_site_index={TRAIT_SITE}\n")
        f.write(f"n_markers={args.n_markers}\n")
        f.write(f"total_sites={N_SITES}\n")
        f.write(f"trait_model={args.model}\n")
        f.write(f"trait_penetrance={args.penetrance}\n")
        f.write(f"Ne={args.Ne}\n")
        f.write(f"seq_len={args.seq_len}\n")
        f.write(f"recomb={args.recomb}\n")
        f.write(f"mu={args.mu}\n")

    if out_trees:
        ts.dump(out_trees)

    print("\n## ✅ Output Summary ##")
    print(f"[ok] Trait Locus: S{TRAIT_SITE}, Marker Loci: S1 to S{N_SITES-1}")
    print(f"[ok] Saved unmasked dataset: {out_unmasked_csv}")
    print(f"[ok] Saved masked dataset: {out_masked_csv}")
    print(f"[ok] Saved metadata: {out_meta_txt}")
    if out_trees:
        print(f"[ok] Saved tree sequence: {out_trees}")
    print(f"[info] Rows: {len(fam_df_unmasked_final)}, Families: {args.families}, Kids/family: {args.kids_per_family}")

if __name__ == "__main__":
    main()