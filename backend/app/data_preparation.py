from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple

import functions
import numpy as np
import pandas as pd

# -----------------------------
# Classes
# -----------------------------


@dataclass(frozen=True)
class PrepConfig:
	dataset_name: str
	train_frac: float = 0.70
	val_frac: float = 0.15
	test_frac: float = 0.15
	seed: int = 123

	# How far out in the pedigree graph to look for relatives
	max_hops: int = 2

	# If True, only create targets for individuals whose observed column is fully missing
	only_predict_masked: bool = True


# -----------------------------
# IO
# -----------------------------


def load_dataset_frames(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Load truth, observed, pedigree CSVs for a dataset prefix using functions.py helpers.
	"""
	files = functions.get_all_dataset_files(dataset_name)  # raises DashboardFilesMissing if anything missing

	truth = pd.read_csv(StringIO(files['truth_genotypes_csv']))
	obs = pd.read_csv(StringIO(files['observed_genotypes_csv']))
	ped = pd.read_csv(StringIO(files['pedigree_csv']))

	return truth, obs, ped


def genotype_columns(df: pd.DataFrame) -> List[str]:
	return [c for c in df.columns if c.startswith('i_')]


# -----------------------------
# Pedigree graph + family components
# -----------------------------


def build_adjacency_from_pedigree(ped: pd.DataFrame) -> Dict[int, Set[int]]:
	"""
	Build an undirected adjacency list from pedigree edges (parent <-> child).
	"""
	required = {'individual_id', 'parent_0_id', 'parent_1_id'}
	missing = required - set(ped.columns)
	if missing:
		raise ValueError(f'Pedigree missing columns: {sorted(missing)}')

	ped = ped.copy()
	ped['individual_id'] = ped['individual_id'].astype(int)
	ped['parent_0_id'] = ped['parent_0_id'].astype(int)
	ped['parent_1_id'] = ped['parent_1_id'].astype(int)

	adj: Dict[int, Set[int]] = {}

	def link(a: int, b: int) -> None:
		if a == -1 or b == -1:
			return
		adj.setdefault(a, set()).add(b)
		adj.setdefault(b, set()).add(a)

	for _, r in ped.iterrows():
		child = int(r['individual_id'])
		p0 = int(r['parent_0_id'])
		p1 = int(r['parent_1_id'])
		adj.setdefault(child, set())
		if p0 != -1:
			adj.setdefault(p0, set())
			link(child, p0)
		if p1 != -1:
			adj.setdefault(p1, set())
			link(child, p1)

	return adj


def connected_components(adj: Dict[int, Set[int]]) -> List[Set[int]]:
	seen: Set[int] = set()
	comps: List[Set[int]] = []

	for node in adj.keys():
		if node in seen:
			continue
		stack = [node]
		comp: Set[int] = set()
		while stack:
			cur = stack.pop()
			if cur in seen:
				continue
			seen.add(cur)
			comp.add(cur)
			for nxt in adj.get(cur, set()):
				if nxt not in seen:
					stack.append(nxt)
		comps.append(comp)

	return comps


def split_families(components: List[Set[int]], cfg: PrepConfig) -> Dict[str, List[Set[int]]]:
	rng = np.random.default_rng(cfg.seed)
	comps = components[:]
	rng.shuffle(comps)

	n = len(comps)
	n_train = int(n * cfg.train_frac)
	n_val = int(n * cfg.val_frac)
	# remainder -> test
	train = comps[:n_train]
	val = comps[n_train : n_train + n_val]
	test = comps[n_train + n_val :]

	return {'train': train, 'val': val, 'test': test}


# -----------------------------
# Feature engineering: relatives within K hops
# -----------------------------


def k_hop_neighborhood(adj: Dict[int, Set[int]], start: int, k: int) -> Set[int]:
	"""
	Returns nodes within <= k hops of `start`, excluding `start`.
	"""
	if k <= 0:
		return set()

	frontier = {start}
	visited = {start}
	for _ in range(k):
		nxt: Set[int] = set()
		for u in frontier:
			nxt |= adj.get(u, set())
		nxt -= visited
		visited |= nxt
		frontier = nxt

	visited.remove(start)
	return visited


def is_fully_missing_individual(obs_df: pd.DataFrame, ind_id: int) -> bool:
	col = functions._col_name(ind_id)
	if col not in obs_df.columns:
		return False  # not in data -> don't treat as masked target
	return obs_df[col].isna().all()


def make_example_for_target(
	truth_df: pd.DataFrame, obs_df: pd.DataFrame, adj: Dict[int, Set[int]], target_id: int, max_hops: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
	"""
	Builds:
	  X = per-site summary of relatives' observed genotypes
	  y = target truth genotype vector (per site)

	X per site = [mean_dosage, frac_observed, count_relatives]
	Shape: (num_sites, 3)
	y shape: (num_sites,)
	"""
	geno_cols_truth = genotype_columns(truth_df)
	geno_cols_obs = genotype_columns(obs_df)

	tcol = functions._col_name(target_id)
	if tcol not in geno_cols_truth:
		return None

	relatives = sorted(k_hop_neighborhood(adj, target_id, max_hops))
	rel_cols = [functions._col_name(r) for r in relatives if functions._col_name(r) in geno_cols_obs]

	if len(rel_cols) == 0:
		return None

	rel_mat = obs_df[rel_cols].to_numpy(dtype=float)  # (sites, relatives)
	# Observed mask
	obs_mask = np.isfinite(rel_mat)
	count_obs = obs_mask.sum(axis=1)  # per site

	# avoid division by zero:
	mean_dosage = np.where(count_obs > 0, np.nan_to_num(rel_mat, nan=0.0).sum(axis=1) / count_obs, 0.0)
	frac_observed = count_obs / rel_mat.shape[1]
	count_relatives = np.full(rel_mat.shape[0], rel_mat.shape[1], dtype=float)

	X = np.stack([mean_dosage, frac_observed, count_relatives], axis=1)  # (sites, 3)

	y = truth_df[tcol].to_numpy(dtype=np.int8)  # (sites,)
	return X, y


def build_split_examples(
	truth_df: pd.DataFrame, obs_df: pd.DataFrame, adj: Dict[int, Set[int]], families: List[Set[int]], cfg: PrepConfig, ped_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Returns:
	  X_all: (n_examples, n_sites, n_features)
	  y_all: (n_examples, n_sites)
	"""
	X_list: List[np.ndarray] = []
	y_list: List[np.ndarray] = []
	g_list: List[np.ndarray] = []

	for fam in families:
		for target_id in fam:
			if cfg.only_predict_masked and not is_fully_missing_individual(obs_df, target_id):
				continue
			ex = make_example_for_target(truth_df, obs_df, adj, target_id, cfg.max_hops)
			if ex is None:
				continue
			X, y = ex
			X_list.append(X)
			y_list.append(y)

			gen = ped_df.loc[ped_df['individual_id'] == target_id, 'time'].values[0]
			g_list.append(gen)

	if not X_list:
		# Return empty arrays with consistent ranks
		n_sites = truth_df.shape[0]
		return np.zeros((0, n_sites, 3), dtype=float), np.zeros((0, n_sites), dtype=np.int8)

	X_all = np.stack(X_list, axis=0)
	y_all = np.stack(y_list, axis=0)
	g_all = np.array(g_list, dtype=int)
	return X_all, y_all, g_all


# -----------------------------
# Main API
# -----------------------------


def prepare_data(cfg: PrepConfig) -> Dict[str, np.ndarray]:
	"""
	Application-mode preparation.

	Loads ONE dataset (e.g. "test1.testing") and returns model-ready arrays.
	No train/val/test splitting here.
	"""
	truth, obs, ped = load_dataset_frames(cfg.dataset_name)

	# Families/components for context
	adj = build_adjacency_from_pedigree(ped)
	comps = connected_components(adj)

	# Use ALL components in this dataset
	X, y, g = build_split_examples(truth, obs, adj, comps, cfg, ped)

	return {'X': X, 'y': y, 'groups': g}


def prepare_data_triplet(base_name: str, cfg: PrepConfig) -> Dict[str, Dict[str, np.ndarray]]:
	"""
	Loads three datasets: base.training, base.validation, base.testing
	No family splitting inside each dataset.
	"""
	out: Dict[str, Dict[str, np.ndarray]] = {}

	for split_key, suffix in [('train', 'training'), ('val', 'validation'), ('test', 'testing')]:
		ds_name = f'{base_name}.{suffix}'

		truth, obs, ped = load_dataset_frames(ds_name)

		adj = build_adjacency_from_pedigree(ped)
		comps = connected_components(adj)

		X, y, g = build_split_examples(truth, obs, adj, comps, cfg, ped)

		out[split_key] = {'X': X, 'y': y, 'groups': g, 'dataset': ds_name}

	return out


if __name__ == '__main__':
	dataset = 'testing'

	# If it's something like "test1.training", strip suffix to get base_name
	if dataset.endswith(('.training', '.validation', '.testing')):
		base_name = dataset.rsplit('.', 1)[0]
	else:
		base_name = dataset

	cfg = PrepConfig(dataset_name='unused')

	data = prepare_data_triplet(base_name, cfg)

	for split_name, split_data in data.items():
		print(split_name, split_data['X'].shape, split_data['y'].shape)
