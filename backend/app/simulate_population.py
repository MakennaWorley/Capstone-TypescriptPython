"""
Full-project genetic simulation pipeline for
"Probabilistic Ancestral Inference from Incomplete Genetic Data".

Supports:
- msprime ancestry + recombination
- forced multi-locus Mendelian genotypes (DGRP-inspired allele frequencies)
- multi-generation forward pedigree building
- masking of ancestors
- metadata export for Bayesian / HMM / GNN models
- reproducible seeds
- CLI like your msprime script
"""

import os, re, argparse
import numpy as np
import pandas as pd
import msprime, tskit

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def slugify(s):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s or "run"

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

# ---------------------------------------------------------------------
# Ancestry + variant simulation
# ---------------------------------------------------------------------
def simulate_ancestry_msprime(
    n_diploid: int,
    seq_len: float,
    recomb: float,
    Ne: int,
    seed: int
) -> tskit.TreeSequence:

    ts = msprime.sim_ancestry(
        samples=n_diploid,
        ploidy=2,
        population_size=Ne,
        sequence_length=seq_len,
        recombination_rate=recomb,
        model="dtwf",
        random_seed=seed
    )
    return ts

def add_multilocus_mutations(
    ts: tskit.TreeSequence,
    mu: float,
    seed: int
) -> tskit.TreeSequence:

    ts = msprime.sim_mutations(
        ts,
        rate=mu,
        model=msprime.SLiMMutationModel(type=1),
        random_seed=seed
    )
    return ts

# ---------------------------------------------------------------------
# Extract diploid genotype matrix
# ---------------------------------------------------------------------
def diploid_matrix(ts: tskit.TreeSequence):
    """Return diploid genotype matrix: shape (N_diploid × M_variants)."""
    G_hap = ts.genotype_matrix().T   # (hap_samples × M)
    n_dip = ts.num_samples // 2
    if G_hap.shape[0] != 2 * n_dip:
        raise RuntimeError("Mismatched haploid sample count.")
    G = G_hap.reshape(n_dip, 2, -1).sum(axis=1)  # (N_dip, M)
    return G

# ---------------------------------------------------------------------
# Multi-generation pedigree builder
# ---------------------------------------------------------------------
def build_pedigree(G0, generations, children_per_mating, seed):
    """
    G0: genotype matrix of founders (N0 × M)
    generations: number of forward-time generations
    children_per_mating: integer
    Returns:
        individuals_df
        pedigree_edges (parent→child)
        genotype_list (per generation)
    """

    rng = np.random.default_rng(seed)

    # Store info
    all_genotypes = [G0]
    pedigree_rows = []
    individual_rows = []

    # Track IDs
    cur_id = 0
    # Assign IDs to founders
    n_founders = G0.shape[0]
    founder_ids = np.arange(cur_id, cur_id + n_founders)
    cur_id += n_founders

    # Record metadata for founders
    for ind, gid in zip(founder_ids, range(n_founders)):
        individual_rows.append({
            "id": ind,
            "generation": 0,
            "founder": 1
        })

    parent_ids = founder_ids
    parent_genos = G0

    for gen in range(1, generations + 1):
        # Randomly pair parents
        rng.shuffle(parent_ids)
        pairs = parent_ids.reshape(-1, 2)

        next_gen_genos = []
        next_gen_ids = []

        for p1, p2 in pairs:
            g1 = parent_genos[parent_ids.tolist().index(p1)]
            g2 = parent_genos[parent_ids.tolist().index(p2)]

            for _ in range(children_per_mating):
                # Mendelian for each locus
                kid = np.array([
                    rng.choice([0, 1], p=[
                        np.clip(1 - g1[l] / 2.0, 0.0, 1.0),
                        np.clip(g1[l] / 2.0, 0.0, 1.0)
                    ]) +
                    rng.choice([0, 1], p=[
                        np.clip(1 - g2[l] / 2.0, 0.0, 1.0),
                        np.clip(g2[l] / 2.0, 0.0, 1.0)
                    ])
                    for l in range(len(g1))
                ], dtype=np.int8)

                next_gen_genos.append(kid)

                # assign ID
                kid_id = cur_id
                cur_id += 1
                next_gen_ids.append(kid_id)

                # pedigree edge entries
                pedigree_rows.append({"parent": p1, "child": kid_id})
                pedigree_rows.append({"parent": p2, "child": kid_id})

                # record individual metadata
                individual_rows.append({
                    "id": kid_id,
                    "generation": gen,
                    "founder": 0
                })

        parent_ids = np.array(next_gen_ids)
        parent_genos = np.array(next_gen_genos)
        all_genotypes.append(parent_genos)

    indiv_df = pd.DataFrame(individual_rows)
    ped_df = pd.DataFrame(pedigree_rows)
    return indiv_df, ped_df, all_genotypes

# ---------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------
def apply_masking_to_ancestors(df, mask_rate, seed):
    rng = np.random.default_rng(seed)
    is_masked = rng.random(len(df)) < mask_rate
    out = df.copy()
    out.loc[is_masked, "dosage"] = np.nan
    return out

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Full genetic simulator (msprime + pedigree).")

    # Filenames
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--save-trees", type=str2bool, nargs="?", const=True, default=False)

    # Genetic params
    ap.add_argument("--N0", type=int, default=2000, help="Number of founder diploids")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--children", type=int, default=2)
    ap.add_argument("--seq-len", type=float, default=5e5)
    ap.add_argument("--Ne", type=int, default=10000)
    ap.add_argument("--recomb", type=float, default=1e-8)
    ap.add_argument("--mu", type=float, default=1e-8)

    # Masking
    ap.add_argument("--mask-rate", type=float, default=0.6)

    # Seed
    ap.add_argument("--seed", type=int, default=None)

    args = ap.parse_args()

    if args.seed is None:
        args.seed = int(np.random.SeedSequence().entropy % (2**32))
        print(f"[INFO] seed={args.seed}")
    rng = np.random.default_rng(args.seed)

    name = slugify(args.name) if args.name else "sim"

    outdir = "datasets"
    os.makedirs(outdir, exist_ok=True)

    # -----------------------------------------------------------------
    # 1) Simulate ancestry + mutations
    # -----------------------------------------------------------------
    ts = simulate_ancestry_msprime(
        n_diploid=args.N0,
        seq_len=args.seq_len,
        recomb=args.recomb,
        Ne=args.Ne,
        seed=args.seed
    )
    ts = add_multilocus_mutations(ts, mu=args.mu, seed=args.seed + 1)

    if args.save_trees:
        ts.dump(os.path.join(outdir, f"{name}.trees"))

    # -----------------------------------------------------------------
    # 2) Extract founder genotype matrix
    # -----------------------------------------------------------------
    G0 = diploid_matrix(ts)   # (N0 × M)

    # -----------------------------------------------------------------
    # 3) Build multi-gen pedigree
    # -----------------------------------------------------------------
    indiv_df, ped_df, geno_list = build_pedigree(
        G0=G0,
        generations=args.generations,
        children_per_mating=args.children,
        seed=args.seed + 2
    )

    # Flatten genotypes into per-individual rows
    geno_flat = []
    for G in geno_list:
        for g in G:
            geno_flat.append(g)
    geno_flat = np.array(geno_flat)

    # Attach genotype summary (dosage sums per locus)
    M = geno_flat.shape[1]
    geno_df = pd.DataFrame(geno_flat, columns=[f"locus_{i}" for i in range(M)])
    geno_df["id"] = indiv_df["id"].values

    # -----------------------------------------------------------------
    # 4) Masking ancestors
    # -----------------------------------------------------------------
    # Summarize per-individual: dosage sum across loci (for simple Bayesian/HMM)
    indiv_df["dosage"] = geno_df[[f"locus_{i}" for i in range(M)]].sum(axis=1)

    masked_df = apply_masking_to_ancestors(
        indiv_df,
        mask_rate=args.mask_rate,
        seed=args.seed + 3
    )

    # -----------------------------------------------------------------
    # 5) Save
    # -----------------------------------------------------------------
    # Base name for outputs
    base_filename = os.path.join(outdir, f"{name}")

    # Output paths for masked/unmasked summary data (based on total dosage)
    unmasked_out_csv = f"{base_filename}_unmasked_out.csv"
    masked_out_csv = f"{base_filename}_masked_out.csv"

    # Save the original individual data (which includes the true 'dosage') 
    # as the UNMASKED file.
    indiv_df.to_csv(unmasked_out_csv, index=False) 
    
    # Save the masked individual data (which includes the 'observed' column) 
    # as the MASKED file.
    masked_df.to_csv(masked_out_csv, index=False)
    
    # Keep the other outputs for completeness, though they are technically 
    # redundant or for specialized use.
    ped_df.to_csv(f"{base_filename}_pedigree.csv", index=False)
    geno_df.to_csv(f"{base_filename}_genotypes.csv", index=False)
    
    meta_out = f"{base_filename}_meta.txt"

    # Metadata for reproducibility
    with open(meta_out, "w") as f:
        f.write(f"seed={args.seed}\n")
        f.write(f"N0={args.N0}\n")
        f.write(f"gens={args.generations}\n")
        f.write(f"children={args.children}\n")
        f.write(f"mask_rate={args.mask_rate}\n")
        f.write(f"Ne={args.Ne}\n")
        f.write(f"seq_len={args.seq_len}\n")
        f.write(f"recomb={args.recomb}\n")
        f.write(f"mu={args.mu}\n")

    print("[OK] Simulation complete.")
    print(f"Saved unmasked data: {unmasked_out_csv} (Columns: id, generation, founder, dosage)")
    print(f"Saved masked data: {masked_out_csv} (Columns: id, generation, founder, dosage, masked, observed)")
    print(f"Saved: {f'{base_filename}_pedigree.csv'}")
    print(f"Saved: {f'{base_filename}_genotypes.csv'}")
    print(f"Saved: {meta_out}")


if __name__ == "__main__":
    main()
