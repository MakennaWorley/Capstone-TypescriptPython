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
   - *.truth_genotypes.csv
   - *.observed_genotypes.csv
   - *.pedigree.csv
   - *.run_metadata.json (reproducible: rerun from this)
   - *.trees (tree sequence for ground truth + reproducibility)
   - *.pedigree.svg


Meta replay features:
- --meta-in path.json  => loads params from previous run metadata JSON
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import msprime
import numpy as np
import pandas as pd
import tskit
from graphviz import Digraph

# -----------------------------
# Configs
# -----------------------------


@dataclass(frozen=True)
class SimConfig:
	# Sampling / population parameters
	n_diploid_samples: int = 250
	Ne: int = 500
	ploidy: int = 2

	# Genome / simulation parameters
	sequence_length: int = 100
	mutation_rate: float = 1e-8

	# Initial Diversity
	founder_recessive_chance: float = 0.05

	# Ancestry
	n_generations: int = 5
	samples_per_generation: int = 50

	# Output control
	seed: int = 42
	masking_rate: float = 0.20
	output_dir: str = 'datasets'
	full_data: bool = False
	name: str = 'DEFAULT_NAME'


# -----------------------------
# Utils
# -----------------------------


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: dict) -> None:
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: str) -> dict:
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)


def now_utc_iso() -> str:
	return datetime.utcnow().isoformat() + 'Z'


def add_to_file(text: str, output_dir: str) -> None:
	os.makedirs(output_dir, exist_ok=True)
	path = os.path.join(output_dir, 'datasets.txt')
	with open(path, 'a', encoding='utf-8') as f:
		f.write(text.rstrip('\n') + '\n')


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
	if 'prefix' in d and 'name' not in d:
		d['name'] = d['prefix']

	fields = set(SimConfig.__dataclass_fields__.keys())
	filtered = {k: v for k, v in d.items() if k in fields}
	return SimConfig(**filtered)


# -----------------------------
# Helpers
# -----------------------------


def build_random_pedigree(cfg: SimConfig) -> tskit.TableCollection:
	"""
	Builds an explicit multi-generation pedigree where every non-founder has 2 parents,
	and the whole cohort shares a small set of founders (so it's one connected family forest,
	not 200 unrelated people).
	"""
	rng = np.random.default_rng(cfg.seed + 12345)

	# How many individuals per generation
	n_per_gen = int(cfg.samples_per_generation) if cfg.samples_per_generation is not None else int(cfg.n_diploid_samples)

	# Make the oldest generation SMALL so everyone is related
	n_founders = max(2, n_per_gen // 10)

	pb = msprime.PedigreeBuilder()

	# generation_ids[t] = list of individual ids at time t (t=0 are "children"/present)
	generation_ids = {}

	# Oldest generation = founders (no parents)
	t_oldest = cfg.n_generations - 1
	generation_ids[t_oldest] = [pb.add_individual(time=float(t_oldest), parents=None, is_sample=False) for _ in range(n_founders)]

	# Work forward toward the present (t decreases)
	# Each generation has n_per_gen individuals, each with two parents from t+1
	for t in range(t_oldest - 1, -1, -1):
		parents = generation_ids[t + 1]

		# Create parent pairs (couples). Ensure we have enough pairs.
		rng.shuffle(parents)
		couples = []
		couples = [(parents[i], parents[i + 1]) for i in range(0, len(parents) - 1, 2)]
		if not couples:
			couples = [(parents[0], parents[0])]

		# Make sure every parent is used at least once:
		# assign children round-robin over couples.
		k = len(couples)
		gen_ids = []
		for i in range(n_per_gen):
			mom, dad = couples[i % k]
			gen_ids.append(pb.add_individual(time=float(t), parents=[mom, dad], is_sample=(t == 0)))
		generation_ids[t] = gen_ids

	# Important: give the pedigree a sequence length so sim_ancestry can run
	return pb.finalise(sequence_length=cfg.sequence_length)


def promote_individual_nodes_to_samples(ts: tskit.TreeSequence, *, keep_existing=True) -> tskit.TreeSequence:
	"""
	Mark nodes belonging to *all individuals* as sample nodes AFTER simulation.
	This avoids msprime's limitation (no internal samples in pedigree),
	but still lets us export genotypes for ancestors via genotype_matrix().
	"""
	tables = ts.dump_tables()

	if not keep_existing:
		tables.nodes.flags[:] = tables.nodes.flags & ~tskit.NODE_IS_SAMPLE

	# Mark nodes referenced by individuals as sample nodes
	for ind in ts.individuals():
		for n in ind.nodes:
			tables.nodes.flags[n] |= tskit.NODE_IS_SAMPLE

	# Ensure tables are properly sorted and indexed before building tree sequence
	tables.sort()
	tables.build_index()

	return tables.tree_sequence()


def simulate_tree_sequence(cfg: SimConfig) -> tskit.TreeSequence:
	"""
	Simulates ancestry through an explicit pedigree so ind.parents is real.
	Then overlays mutations.
	"""
	pedigree = build_random_pedigree(cfg)

	ts = msprime.sim_ancestry(initial_state=pedigree, model='fixed_pedigree', recombination_rate=cfg.mutation_rate, random_seed=cfg.seed)

	ts = promote_individual_nodes_to_samples(ts)

	# 1. Manually Inject Founder Mutations (Initial Diversity)
	tables = ts.dump_tables()
	rng = np.random.default_rng(cfg.seed + 777)

	founder_nodes = []
	max_time = max(ind.time for ind in ts.individuals())

	for ind in ts.individuals():
		if ind.time == max_time:
			founder_nodes.extend(ind.nodes)

	# 2. Initialize Sites and Inject Founder Alleles
	mutation_count = 0
	for pos in range(cfg.sequence_length):
		site_id = tables.sites.add_row(position=pos, ancestral_state='0')

		for node_id in founder_nodes:
			# Check if this founder node inherits the recessive allele at this site
			if rng.random() < cfg.founder_recessive_chance:
				tables.mutations.add_row(site=site_id, node=node_id, derived_state='1')
				mutation_count += 1

	# Sort and build tree sequence with our mutations
	tables.sort()
	ts = tables.tree_sequence()

	return ts


def draw_pedigree_svg(ts: tskit.TreeSequence, base_path: str, obs_matrix: np.ndarray) -> None:
	"""
	Draw a pedigree-style family tree (individuals as nodes, parent->child edges),
	layered by generation time. Colors nodes using obs_matrix masking at site 0:
		- known: filled blue
		- masked: dotted outline blue
	"""
	if ts.num_individuals > 1000:
		print('   Skipping Pedigree SVG: Population size too large for visualization.')
		return

	ped = []
	for ind in ts.individuals():
		p0 = ind.parents[0] if len(ind.parents) > 0 else -1
		p1 = ind.parents[1] if len(ind.parents) > 1 else -1
		ped.append((ind.id, int(ind.time), int(p0), int(p1)))

	# Decide styling based on whether the individual's column is masked at site 0
	# (If you changed observed to nullable ints, handle pd.NA too; but NaN works here.)
	num_individuals = obs_matrix.shape[1]
	is_masked = [False] * num_individuals

	if obs_matrix.shape[0] > 0:
		for ind_id in range(num_individuals):
			column = obs_matrix[:, ind_id]
			# An individual is considered masked if their entire column is NaN
			# We use np.all(np.isnan(...)) to handle the stochastically dropped columns
			if np.all(np.isnan(column.astype(float))):
				is_masked[ind_id] = True

	g = Digraph('pedigree', format='svg')
	g.attr(rankdir='TB', splines='polyline', nodesep='0.35', ranksep='0.75')
	g.attr('node', shape='circle', width='0.35', fixedsize='true', fontsize='10')

	# Create subgraphs per generation to force layering
	times = sorted({t for _, t, _, _ in ped})
	for t in times:
		with g.subgraph(name=f'rank_t{t}') as sg:
			sg.attr(rank='same')
			for ind_id, ind_time, _, _ in ped:
				if ind_time != t:
					continue

				if is_masked[ind_id]:
					# Dotted outline for masked/missing individuals
					sg.node(f'i{ind_id}', label=str(ind_id), style='dashed', color='#3498db', fontcolor='#3498db')
				else:
					# Solid fill for observed individuals
					sg.node(f'i{ind_id}', label=str(ind_id), style='filled', fillcolor='#3498db', color='#3498db', fontcolor='white')

	# Parent -> child edges
	for child_id, _, p0, p1 in ped:
		if p0 != -1:
			g.edge(f'i{p0}', f'i{child_id}')
		if p1 != -1:
			g.edge(f'i{p1}', f'i{child_id}')

	out_path = f'{base_path}.pedigree'
	g.render(filename=out_path, cleanup=True)
	print(f'   Pedigree SVG saved: {out_path}')


# -----------------------------
# Masking
# -----------------------------


def genotype_matrix(ts: tskit.TreeSequence) -> np.ndarray:
	"""Genotypes: (num_sites x num_sample_genomes) allele indices."""
	return ts.genotype_matrix()


def haploid_to_diploid_dosage(ts: tskit.TreeSequence, cfg: SimConfig) -> np.ndarray:
	"""
	Creates a dense matrix for the entire sequence length.
	Encoding follows msprime/tskit defaults (derived allele count):
	0 = homozygous ancestral (0/0)
	1 = heterozygous (0/1)
	2 = homozygous derived (1/1)
	Missing (-1) is defaulted to 0.
	"""
	all_nodes = []
	for ind in ts.individuals():
		if len(ind.nodes) == 2:
			all_nodes.extend([int(ind.nodes[0]), int(ind.nodes[1])])

	# Genotype matrix for *these* nodes (sites x chosen_nodes)
	G_hap = ts.genotype_matrix(samples=all_nodes)
	num_inds = ts.num_individuals
	G_dip_full = np.zeros((cfg.sequence_length, num_inds), dtype=np.int8)

	# node_id -> column index in G_hap
	node_to_column = {node_id: col_idx for col_idx, node_id in enumerate(all_nodes)}

	# Iterate through every site
	for site_idx, site in enumerate(ts.sites()):
		site_pos = int(site.position)
		if site_pos >= cfg.sequence_length:
			continue  # Site outside matrix

		for ind in ts.individuals():
			if len(ind.nodes) != 2:
				continue

			n0, n1 = int(ind.nodes[0]), int(ind.nodes[1])
			c0, c1 = node_to_column[n0], node_to_column[n1]

			a0 = max(int(G_hap[site_idx, c0]), 0)
			a1 = max(int(G_hap[site_idx, c1]), 0)

			dosage = a0 + a1
			G_dip_full[site_pos, ind.id] = dosage

	return G_dip_full


def mask(X: np.ndarray, masking_rate: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Masks entire individuals (columns) at the masking_rate.
	If an individual is masked, their entire vertical column becomes NaN.
	"""
	Xf = X.astype(float, copy=True)
	num_individuals = Xf.shape[1]

	# Generate a 1D mask: True means keep (observed), False means drop (masked)
	# This samples once per individual
	individual_observed = rng.random(num_individuals) >= masking_rate

	# Apply the 1D mask across the 2D matrix
	# If individual_observed[i] is False, Xf[:, i] becomes NaN
	Xf[:, ~individual_observed] = np.nan

	# Return the masked matrix and the mask itself
	full_mask = np.tile(individual_observed, (Xf.shape[0], 1))
	return Xf, full_mask


def sites_table(cfg: SimConfig) -> pd.DataFrame:
	"""Creates a table for every single individual in the sequence"""
	return pd.DataFrame({'index': np.arange(cfg.sequence_length)})


def pedigree_table(ts: tskit.TreeSequence) -> pd.DataFrame:
	"""Map sample genomes to individuals and haplotypes."""
	rows = []
	for ind in ts.individuals():
		p0 = ind.parents[0] if len(ind.parents) > 0 else -1
		p1 = ind.parents[1] if len(ind.parents) > 1 else -1
		rows.append({'individual_id': ind.id, 'time': ind.time, 'parent_0_id': p0, 'parent_1_id': p1, 'num_nodes': len(ind.nodes)})
	return pd.DataFrame(rows)


def write_genotypes_by_generation(ts, G_dip, df_sites, output_dir, name_prefix):
	ped = pedigree_table(ts)
	for t in sorted(ped['time'].unique()):
		ids = ped.loc[ped['time'] == t, 'individual_id'].to_list()
		cols = [f'ind_{i:04d}' for i in ids]

		df = pd.DataFrame(G_dip[:, ids], columns=cols)
		df = pd.concat([df_sites, df], axis=1)

		out = os.path.join(output_dir, f'{name_prefix}.gen_t{int(t)}.truth_genotypes.csv')
		df.to_csv(out, index=False)


# -----------------------------
# IO paths + metadata writing
# -----------------------------


def build_paths(cfg: SimConfig) -> Dict[str, str]:
	"""Centralize output naming."""
	ensure_dir(cfg.output_dir)
	base = os.path.join(cfg.output_dir, cfg.name)

	return {
		'trees': f'{base}.trees',
		'truth_csv': f'{base}.truth_genotypes.csv',
		'observed_csv': f'{base}.observed_genotypes.csv',
		'pedigree_csv': f'{base}.pedigree.csv',
		'meta_json': f'{base}.run_metadata.json',
	}


# -----------------------------
# Simulation
# -----------------------------


def run_generation(cfg: SimConfig, *, meta_in: Optional[str] = None) -> Dict[str, str]:
	"""
	Generate a dataset and write outputs.
	Returns output paths.
	"""
	ts = simulate_tree_sequence(cfg)

	# Extract genotypes
	G_dip = haploid_to_diploid_dosage(ts, cfg)

	# Build tables
	df_sites = sites_table(cfg)
	df_pedigree = pedigree_table(ts)
	# write_genotypes_by_generation(ts=ts, G_dip=G_dip, df_sites=df_sites, output_dir=cfg.output_dir, name_prefix=cfg.name)

	# Mask at diploid-individual level
	rng = np.random.default_rng(cfg.seed + 999)
	G_obs, _ = mask(G_dip, cfg.masking_rate, rng=rng)
	draw_pedigree_svg(ts, os.path.join(cfg.output_dir, cfg.name), G_obs)

	# DataFrames with consistent column naming
	individual_cols = [f'i_{i:04d}' for i in range(G_dip.shape[1])]

	df_truth = pd.DataFrame(G_dip, columns=individual_cols)
	df_obs = pd.DataFrame(G_obs, columns=individual_cols)

	genotype_cols = [c for c in df_obs.columns if c.startswith('i_')]
	df_obs[genotype_cols] = df_obs[genotype_cols].astype('Int8')

	# Prepend site columns
	df_truth = pd.concat([df_sites, df_truth], axis=1)
	df_obs = pd.concat([df_sites, df_obs], axis=1)

	# Output paths
	outputs = build_paths(cfg)

	# Persist tree sequence for ground truth
	ts.dump(outputs['trees'])

	# Persist CSVs
	df_truth.to_csv(outputs['truth_csv'], index=False)
	df_obs.to_csv(outputs['observed_csv'], index=False)
	df_pedigree.to_csv(outputs['pedigree_csv'], index=False)

	# Derived/run info
	derived = {
		'ts_num_sites': int(ts.num_sites),
		'ts_num_mutations': int(ts.num_mutations),
		'ts_num_trees': int(ts.num_trees),
		'ts_num_samples': int(ts.num_samples),
		'sequence_length': float(ts.sequence_length),
		'diploid_dosage_shape': [int(G_dip.shape[0]), int(G_dip.shape[1])],
		'masking_rate': float(cfg.masking_rate),
		'observed_nonmasking_fraction': float(np.isfinite(G_obs).mean()),
	}

	meta_payload = {
		'created_at': now_utc_iso(),
		'script': os.path.basename(__file__),
		'meta_in': meta_in,
		'params': config_to_dict(cfg),
		'derived': derived,
		'outputs': outputs,
	}

	# Save JSON meta
	write_json(outputs['meta_json'], meta_payload)

	# Console summary
	print('Generation complete.')
	print(f'   Trees:    {outputs["trees"]}')
	print(f'   Truth:    {outputs["truth_csv"]}')
	print(f'   Observed: {outputs["observed_csv"]}')
	print(f'   Meta:     {outputs["meta_json"]}')
	print(f'   Pedigree: {outputs["pedigree_csv"]}')
	print(f'   Variants: {ts.num_sites}')

	return outputs


# -----------------------------
# Checks
# -----------------------------


def checks(truth_csv: str, observed_csv: str) -> None:
	truth = pd.read_csv(truth_csv)
	obs = pd.read_csv(observed_csv)

	assert len(truth) == len(obs), 'Row mismatch between truth and observed'
	assert (truth['site_index'].values == obs['site_index'].values).all(), 'Site index mismatch'

	genotype_cols = [c for c in truth.columns if c.startswith('ind_')]
	truth_vals = truth[genotype_cols].to_numpy()
	obs_vals = obs[genotype_cols].to_numpy()

	masking_fraction = np.isnan(obs_vals).mean()
	print(f'[check] Observed masking fraction: {masking_fraction:.3f}')

	uniq = np.unique(truth_vals)
	print(f'[check] Unique truth dosages (first few): {uniq[:10]} (total unique={len(uniq)})')


# -----------------------------
# CLI / main
# -----------------------------


def parse_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser(description='Generate msprime stochastic genotype datasets with reproducible meta replay.')

	# Core params (mirror SimConfig fields)
	ap.add_argument('--n-diploid-samples', type=int, default=SimConfig.n_diploid_samples)
	ap.add_argument('--Ne', type=int, default=SimConfig.Ne)
	ap.add_argument('--ploidy', type=int, default=SimConfig.ploidy)

	ap.add_argument('--sequence-length', type=int, default=SimConfig.sequence_length)
	ap.add_argument('--mutation-rate', type=float, default=SimConfig.mutation_rate)

	ap.add_argument('--founder-recessive-chance', type=float, default=0.05)

	ap.add_argument('--n_generations', type=int, default=SimConfig.n_generations)
	ap.add_argument('--samples_per_generation', type=float, default=SimConfig.samples_per_generation)

	ap.add_argument('--seed', type=int, default=None, help='Random seed; if omitted, a random seed is chosen and recorded in meta.')
	ap.add_argument('--masking-rate', type=float, default=SimConfig.masking_rate)
	ap.add_argument('--output-dir', type=str, default=SimConfig.output_dir)
	ap.add_argument('--full-data', action='store_true', help='Generate Train/Val/Test splits.', default=SimConfig.full_data)
	ap.add_argument('--name', type=str, default=SimConfig.name)

	ap.add_argument('--meta-in', type=str, default=None, help='JSON meta file to reproduce a previous run.')

	ap.add_argument('--checks', action='store_true', default=False, help='Run lightweight integrity checks on saved CSVs.')

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
		mutation_rate=args.mutation_rate,
		founder_recessive_chance=args.founder_recessive_chance,
		n_generations=args.n_generations,
		samples_per_generation=args.samples_per_generation,
		seed=seed,
		masking_rate=args.masking_rate,
		output_dir=args.output_dir,
		full_data=args.full_data,
		name=args.name,
	)


def create_data() -> None:
	args = parse_args()

	# If meta-in is provided, load it and merge
	if args.meta_in:
		meta = read_json(args.meta_in)
		# Handle the case where 'params' might be nested or flat
		params_dict = meta.get('params', meta)
		cfg = dict_to_config(params_dict)
	else:
		cfg = args_to_config(args)

	# Run generation
	if args.full_data:
		splits = ['training', 'validation', 'testing']

		for i, split in enumerate(splits):
			# Create a unique config for each split
			split_cfg = dict_to_config({**asdict(cfg), 'name': f'{cfg.name}.{split}', 'seed': cfg.seed + (i * 1000)})
			run_generation(split_cfg)
			add_to_file(f'{cfg.name}.{split}', cfg.output_dir)
	else:
		# Standard single run
		outputs = run_generation(cfg)
		add_to_file(args.name, cfg.output_dir)

	# Optional quick checksgener
	if args.checks:
		checks(outputs['truth_csv'], outputs['observed_csv'])


if __name__ == '__main__':
	create_data()


# -----------------------------
# API
# -----------------------------


def build_config_from_params(params: Dict[str, Any]) -> 'SimConfig':
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
	if 'seed' not in filtered_params or filtered_params['seed'] is None:
		filtered_params['seed'] = int(np.random.SeedSequence().entropy % (2**32))

	# 3. Create the frozen object
	return SimConfig(**filtered_params)


def create_data_from_params(params: Dict[str, Any], *, meta_in: Optional[str] = None) -> Dict[str, Any]:
	"""
	Programmatic equivalent of create_data(), designed for API usage.

	params: dict from request body
	meta_in: optional path to a meta JSON to replay
	meta_wins: if True and meta_in provided, meta overrides params
	"""
	if meta_in:
		meta = read_json(meta_in)
		# Handle the case where 'params' might be nested or flat
		params_dict = meta.get('params', meta)
		cfg = dict_to_config(params_dict)
	else:
		cfg = build_config_from_params(params)

	if cfg.full_data:
		splits = ['training', 'validation', 'testing']

		for i, split in enumerate(splits):
			split_cfg = dict_to_config({**asdict(cfg), 'name': f'{cfg.name}.{split}', 'seed': cfg.seed + (i * 1000)})
			run_generation(split_cfg)
			add_to_file(f'{cfg.name}.{split}', cfg.output_dir)

		return {'config': asdict(cfg), 'outputs': cfg.name, 'mode': 'stratified_triplet'}

	else:
		run_generation(cfg)
		add_to_file(cfg.name, cfg.output_dir)
		return {'config': asdict(cfg), 'outputs': cfg.name, 'mode': 'single_dataset'}
