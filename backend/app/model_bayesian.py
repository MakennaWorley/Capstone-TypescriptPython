from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import multiprocessing
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import arviz as az
import data_preparation
import matplotlib
import model_graph_functions
import numpy as np
import pymc as pm
from sklearn.model_selection import KFold

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------


def ensure_dir(path: str | Path) -> None:
	Path(path).mkdir(parents=True, exist_ok=True)


def flatten_examples(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""
	X: (n_examples, n_sites, n_features)
	y: (n_examples, n_sites)
	-> (n_examples*n_sites, n_features), (n_examples*n_sites,)
	"""
	if X.ndim != 3:
		raise ValueError(f'Expected X rank-3, got {X.shape}')
	if y.ndim != 2:
		raise ValueError(f'Expected y rank-2, got {y.shape}')
	if X.shape[:2] != y.shape[:2]:
		raise ValueError(f'X/y mismatch: X={X.shape}, y={y.shape}')

	Xf = X.reshape(-1, X.shape[-1]).astype(np.float32, copy=False)
	yf = y.reshape(-1).astype(np.float32, copy=False)
	return Xf, yf


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	mu = X.mean(axis=0)
	sd = X.std(axis=0)
	sd = np.where(sd == 0, 1.0, sd)
	return (X - mu) / sd, mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
	return (X - mu) / sd


def coerce_dosage_classes(y: np.ndarray) -> np.ndarray:
	"""
	y can be float-ish dosage; this coerces to int {0,1,2}.
	"""
	y_int = np.rint(y).astype(np.int64)
	y_int = np.clip(y_int, 0, 2)
	return y_int


# -----------------------------
# Persistence helpers
# -----------------------------


def model_paths(models_dir: str | Path, base_name: str, model_tag: str) -> Dict[str, Path]:
	d = Path(models_dir)
	return {
		'dir': d,
		'idata': d / f'{base_name}.{model_tag}.idata.nc',
		'meta': d / f'{base_name}.{model_tag}.meta.json',
		'graph_val': d / f'{base_name}.{model_tag}.val_plot.png',
		'graph_test': d / f'{base_name}.{model_tag}.test_plot.png',
		'graph_cm': d / f'{base_name}.{model_tag}.cm_plot.png',
	}


def save_common_meta(paths: Dict[str, Path], payload: Dict[str, Any]) -> None:
	ensure_dir(paths['dir'])
	paths['meta'].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def load_meta(paths: Dict[str, Path]) -> Dict[str, Any]:
	return json.loads(paths['meta'].read_text(encoding='utf-8'))


def _run_fold_parallel(args):
	"""Helper function to run a single fold in a separate process."""
	fold_idx, train_idx, val_idx, X_all, y_all, groups_all, model_kind = args

	X_t, X_v = X_all[train_idx], X_all[val_idx]
	y_t, y_v = y_all[train_idx], y_all[val_idx]
	g_t = groups_all[train_idx]

	# Force cores=1 to avoid nested multiprocessing issues
	if model_kind == 'linear':
		fold_model = BayesianLinearDosageRegressor(chains=4, draws=500, tune=500, cores=1)
	else:
		fold_model = BayesianCategoricalDosageClassifier(chains=4, draws=500, tune=500, cores=1)

	fold_model.fit(X_t, y_t, groups=g_t)

	y_pred = fold_model.predict(X_v)
	mse = np.mean((y_v - y_pred) ** 2)
	return fold_idx, mse


# -----------------------------
# Model 1: Bayesian Linear Dosage Regression
# -----------------------------


class BayesianLinearDosageRegressor:
	def __init__(
		self,
		*,
		draws: int = 1000,
		tune: int = 1000,
		chains: int = 4,
		target_accept: float = 0.9,
		random_seed: int = 123,
		clip_to_dosage_range: bool = True,
		cores: int = 4,
	):
		self.draws = draws
		self.tune = tune
		self.chains = chains
		self.target_accept = target_accept
		self.random_seed = random_seed
		self.clip_to_dosage_range = clip_to_dosage_range
		self.cores = cores

		self.idata: Optional[az.InferenceData] = None
		self.feature_mean_: Optional[np.ndarray] = None
		self.feature_std_: Optional[np.ndarray] = None

		self._coef_mean: Optional[np.ndarray] = None
		self._intercept_mean: Optional[float] = None

	@property
	def tag(self) -> str:
		return 'bayes_linear'

	def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> 'BayesianLinearDosageRegressor':
		X = np.asarray(X, dtype=np.float32)
		y = np.asarray(y, dtype=np.float32)

		Xz, mu, sd = standardize_fit(X)

		self.feature_mean_ = mu
		self.feature_std_ = sd
		n_groups = len(np.unique(groups))

		with pm.Model():
			# 1. Define Hierarchical Priors (Population Level) FIRST
			mu_alpha = pm.Normal('mu_alpha', mu=0.0, sigma=1.0)
			sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1.0)

			# 2. Define Group-specific intercepts (Ancestry branches)
			intercepts = pm.Normal('intercepts', mu=mu_alpha, sigma=sigma_alpha, shape=n_groups)

			# 3. Define Global coefficients and noise
			coef = pm.Normal('coef', mu=0.0, sigma=1.0, shape=Xz.shape[1])
			sigma = pm.HalfNormal('sigma', sigma=1.0)

			# 4. Calculation: intercepts[groups] broadcasts to flattened site-level data
			mu_y = intercepts[groups] + pm.math.dot(Xz, coef)
			pm.Normal('y', mu=mu_y, sigma=sigma, observed=y)

			self.idata = pm.sample(
				draws=self.draws,
				tune=self.tune,
				chains=self.chains,
				target_accept=self.target_accept,
				random_seed=self.random_seed,
				return_inferencedata=True,
				progressbar=False,
				# Added this to prevent EOFErrror on macbook
				cores=self.cores,
			)

		self._coef_mean = self.idata.posterior['coef'].mean(axis=(0, 1)).values
		self._mu_alpha_mean = float(self.idata.posterior['mu_alpha'].mean())
		return self

	def predict(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		Xz = standardize_apply(X, self.feature_mean_, self.feature_std_)

		# Extract the group-specific intercepts
		if self.idata is not None and groups is not None:
			group_intercepts = self.idata.posterior['intercepts'].mean(axis=(0, 1)).values
			# Map each input to its specific group intercept
			intercept = group_intercepts[groups]
			mask = (groups >= len(group_intercepts)) | (groups < 0)
			intercept[mask] = self._mu_alpha_mean
		else:
			# Fallback to the global population mean (mu_alpha)
			intercept = self._mu_alpha_mean

		yhat = intercept + Xz @ self._coef_mean
		return np.clip(yhat, 0.0, 2.0) if self.clip_to_dosage_range else yhat

	def get_calibration_data(self):
		if self.idata is None:
			raise RuntimeError('Model must be fit first.')
		with pm.Model():  # Re-create model context for PPC
			ppc = pm.sample_posterior_predictive(self.idata)
		return ppc

	def save(self, paths: Dict[str, Path], extra_meta: Dict[str, Any]) -> None:
		if self.idata is None:
			raise RuntimeError('No idata to save.')
		ensure_dir(paths['dir'])
		az.to_netcdf(self.idata, paths['idata'])

		payload = {
			'type': 'BayesianLinearDosageRegressor',
			'feature_mean': self.feature_mean_.tolist(),
			'feature_std': self.feature_std_.tolist(),
			'posterior_means': {'coef': self._coef_mean.tolist(), 'mu_alpha': self._mu_alpha_mean},
			'params': {
				'draws': self.draws,
				'tune': self.tune,
				'chains': self.chains,
				'target_accept': self.target_accept,
				'random_seed': self.random_seed,
				'clip_to_dosage_range': self.clip_to_dosage_range,
			},
			'extra': extra_meta,
		}
		save_common_meta(paths, payload)

	@classmethod
	def load(cls, paths: Dict[str, Path]) -> 'BayesianLinearDosageRegressor':
		meta = load_meta(paths)
		m = cls(**meta['params'])
		m.idata = az.from_netcdf(paths['idata'])

		m.feature_mean_ = np.array(meta['feature_mean'], dtype=np.float32)
		m.feature_std_ = np.array(meta['feature_std'], dtype=np.float32)
		m._coef_mean = np.array(meta['posterior_means']['coef'], dtype=np.float32)
		m._mu_alpha_mean = float(meta['posterior_means']['mu_alpha'])
		return m


# -----------------------------
# Model 2: Bayesian Categorical Dosage (Softmax)
# -----------------------------


class BayesianCategoricalDosageClassifier:
	"""
	Multinomial logistic regression with softmax over 3 classes (0/1/2).
	We return:
	  - predict_proba(X): (n, 3)
	  - predict_class(X): (n,)
	  - predict(X): expected dosage E[y] for compatibility with regression plotting
	"""

	def __init__(self, *, draws: int = 1000, tune: int = 1000, chains: int = 4, target_accept: float = 0.9, random_seed: int = 123, cores: int = 4):
		self.draws = draws
		self.tune = tune
		self.chains = chains
		self.target_accept = target_accept
		self.random_seed = random_seed
		self.cores = cores

		self.idata: Optional[az.InferenceData] = None
		self.feature_mean_: Optional[np.ndarray] = None
		self.feature_std_: Optional[np.ndarray] = None

		# posterior means for fast inference
		self._W_mean: Optional[np.ndarray] = None  # (k, 3)
		self._b_mean: Optional[np.ndarray] = None  # (3,)

	@property
	def tag(self) -> str:
		return 'bayes_softmax3'

	def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> 'BayesianCategoricalDosageClassifier':
		X = np.asarray(X, dtype=np.float32)
		y_int = coerce_dosage_classes(np.asarray(y, dtype=np.float32))

		Xz, mu, sd = standardize_fit(X)

		self.feature_mean_ = mu
		self.feature_std_ = sd
		n_groups = len(np.unique(groups))
		C = 3

		with pm.Model():
			# 1. Define Hierarchical Priors for category intercepts
			mu_b = pm.Normal('mu_b', mu=0.0, sigma=1.0, shape=C)
			sigma_b = pm.HalfNormal('sigma_b', sigma=1.0, shape=C)

			# 2. Define Group-level intercepts per category
			b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=(n_groups, C))
			W = pm.Normal('W', mu=0.0, sigma=1.0, shape=(Xz.shape[1], C))

			# 3. Logits calculation using the group indices
			logits = b[groups] + pm.math.dot(Xz, W)
			pm.Categorical('y', logit_p=logits, observed=y_int)

			self.idata = pm.sample(
				draws=self.draws,
				tune=self.tune,
				chains=self.chains,
				target_accept=self.target_accept,
				random_seed=self.random_seed,
				return_inferencedata=True,
				cores=self.cores,
			)

		self._W_mean = self.idata.posterior['W'].mean(axis=(0, 1)).values
		self._mu_b_mean = self.idata.posterior['mu_b'].mean(axis=(0, 1)).values
		return self

	def predict_proba(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		Xz = standardize_apply(X, self.feature_mean_, self.feature_std_)

		if self.idata is not None and groups is not None:
			# b shape is (n_groups, 3)
			group_b = self.idata.posterior['b'].mean(axis=(0, 1)).values
			intercept = np.take(group_b, groups, axis=0, mode='clip')
			mask = (groups >= len(group_b)) | (groups < 0)
			intercept[mask] = self._mu_b_mean
		else:
			# Fallback to global category means
			intercept = self._mu_b_mean  # shape (3,)

		logits = intercept + Xz @ self._W_mean
		expz = np.exp(logits - logits.max(axis=1, keepdims=True))
		return (expz / expz.sum(axis=1, keepdims=True)).astype(np.float32)

	def get_calibration_data(self):
		if self.idata is None:
			raise RuntimeError('Model must be fit first.')
		with pm.Model():
			ppc = pm.sample_posterior_predictive(self.idata)
		return ppc

	def predict_class(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		p = self.predict_proba(X, groups=groups)
		return np.argmax(p, axis=1).astype(np.int64)

	def predict(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		Expected dosage E[y] = sum_c c * p(c)
		This makes it compatible with regression metrics/plots.
		"""
		p = self.predict_proba(X, groups=groups)
		classes = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		return (p * classes[None, :]).sum(axis=1)

	def save(self, paths: Dict[str, Path], extra_meta: Dict[str, Any]) -> None:
		if self.idata is None:
			raise RuntimeError('No idata to save.')
		ensure_dir(paths['dir'])
		az.to_netcdf(self.idata, paths['idata'])

		payload = {
			'type': 'BayesianCategoricalDosageClassifier',
			'tag': self.tag,
			'feature_mean': self.feature_mean_.tolist(),
			'feature_std': self.feature_std_.tolist(),
			'posterior_means': {'W': self._W_mean.tolist(), 'mu_b': self._mu_b_mean.tolist()},
			'params': {
				'draws': self.draws,
				'tune': self.tune,
				'chains': self.chains,
				'target_accept': self.target_accept,
				'random_seed': self.random_seed,
			},
			'extra': extra_meta,
		}
		save_common_meta(paths, payload)

	@classmethod
	def load(cls, paths: Dict[str, Path]) -> 'BayesianCategoricalDosageClassifier':
		meta = load_meta(paths)
		m = cls(**meta['params'])
		m.idata = az.from_netcdf(paths['idata'])

		m.feature_mean_ = np.array(meta['feature_mean'], dtype=np.float32)
		m.feature_std_ = np.array(meta['feature_std'], dtype=np.float32)
		m._W_mean = np.array(meta['posterior_means']['W'], dtype=np.float32)
		m._mu_b_mean = np.array(meta['posterior_means']['mu_b'], dtype=np.float32)
		return m


# -----------------------------
# Pipeline
# -----------------------------


class _NoRefitProxy:
	"""
	graph_model_functions.evaluate_and_graph_clf calls .fit().
	We wrap a fitted model so fit() is a no-op.
	"""

	def __init__(self, fitted_model):
		self._m = fitted_model

	def fit(self, X, y):
		return self

	def predict(self, X):
		return self._m.predict(X)


def train_with_cross_val(base_name, model_kind, prep_cfg, n_splits=5, models_dir='models'):
	"""
	Performs K-Fold CV using the entirety of a single file.
	"""
	# Load the whole file as one block
	X_all, y_all, groups_all = load_whole_dataset(base_name, prep_cfg)
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

	fold_args = []
	for i, (t_idx, v_idx) in enumerate(kf.split(X_all)):
		fold_args.append((i, t_idx, v_idx, X_all, y_all, groups_all, model_kind))

	print(f'\n--- Parallel CV on {base_name} ({n_splits} Folds) ---')

	# Use built-in multiprocessing Pool
	with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
		results = list(executor.map(_run_fold_parallel, fold_args))

	for fold_idx, mse in sorted(results):
		print(f'Fold {fold_idx + 1} MSE: {mse:.4f}')

	# Return model fit on the whole file
	final_model = BayesianLinearDosageRegressor(cores=1) if model_kind == 'linear' else BayesianCategoricalDosageClassifier(cores=1)
	final_model.fit(X_all, y_all, groups=groups_all)
	return final_model


def load_whole_dataset(base_name: str, prep_cfg: data_preparation.PrepConfig):
	"""
	Loads a dataset file and flattens it completely without looking for
	train/val/test sub-splits.
	"""
	# Create a NEW config instance because the original is frozen
	# This copies all settings from prep_cfg but updates the dataset_name
	current_cfg = dataclasses.replace(prep_cfg, dataset_name=base_name)

	# Pass the updated config to prepare_data
	data = data_preparation.prepare_data(current_cfg)

	X_raw = data['X']
	y_raw = data['y']

	X, y = flatten_examples(X_raw, y_raw)

	if 'groups' in data:
		n_sites = X_raw.shape[1]
		groups = np.repeat(data['groups'].flatten().astype(int), n_sites)
	else:
		groups = np.zeros(X.shape[0], dtype=int)

	return X, y, groups


def run_custom_pipeline(train_file, val_file, test_file, model_kind='linear'):
	prep_cfg = data_preparation.PrepConfig()

	# 1. TRAINING: Use all data in the training file
	X_train, y_train, g_train = load_whole_dataset(train_file, prep_cfg)
	if model_kind == 'linear':
		model = BayesianLinearDosageRegressor(cores=1)
	else:
		model = BayesianCategoricalDosageClassifier(cores=1)

	print(f'Phase 1: Training on {train_file}')
	model.fit(X_train, y_train, groups=g_train)

	# 2. VALIDATION: Use cross-validation on the validation file to update the model
	print(f'Phase 2: Updating model with CV on {val_file}')
	# We pass the existing model or refit using the logic in train_with_cross_val
	X_val, y_val, g_val = load_whole_dataset(val_file, prep_cfg)
	model.fit(X_val, y_val, groups=g_val)  # Or your KFold logic

	# 3. TESTING: Purely unseen data
	X_test, y_test, _ = load_whole_dataset(test_file, prep_cfg)
	print(f'Phase 3: Testing on {test_file}')
	predictions = model.predict(X_test)

	# Calculate final metrics...
	return predictions


def train_eval_one(
	train_base: str,
	val_base: str,
	test_base: str,
	model_kind: str,
	*,
	prep_cfg: Optional[data_preparation.PrepConfig] = None,
	models_dir: str | Path = 'models',
	force_retrain: bool = False,
	draws: int = 1000,
	tune: int = 1000,
	chains: int = 4,
	target_accept: float = 0.9,
	seed: int = 123,
) -> Dict[str, Any]:
	if prep_cfg is None:
		prep_cfg = data_preparation.PrepConfig(dataset_name='unused')

	# 1. Setup Model Types
	if model_kind == 'linear':
		ModelCls = BayesianLinearDosageRegressor
		model_tag = 'Bayes Linear'
	elif model_kind == 'softmax3':
		ModelCls = BayesianCategoricalDosageClassifier
		model_tag = 'Bayes Softmax 3'
	else:
		raise ValueError("model_kind must be 'linear' or 'softmax3'")

	paths = model_paths(models_dir, train_base, model_tag)

	# 2. PHASE 1: INITIAL TRAINING (Using the entire training file)
	X_train, y_train, groups_train = load_whole_dataset(train_base, prep_cfg)

	if (not force_retrain) and paths['meta'].exists() and paths['idata'].exists():
		print(f'Loading existing model from {paths["meta"]}')
		model = ModelCls.load(paths)
		trained = False
	else:
		print(f'--- Phase 1: Training on {train_base} ---')
		X_resampled, y_resampled, groups_resampled = data_preparation.resample_training_data(X_train, y_train, groups_train)

		model = ModelCls(draws=draws, tune=tune, chains=chains, target_accept=target_accept, random_seed=seed)

		# Fit using the balanced dataset
		model.fit(X_resampled, y_resampled, groups=groups_resampled)
		trained = True

	# 3. PHASE 2: CROSS-VALIDATION UPDATE (Using the entire validation file)
	print(f'--- Phase 2: Updating with CV on {val_base} ---')
	# This calls your KFold logic on the dedicated validation file
	model = train_with_cross_val(val_base, model_kind, prep_cfg)

	model.save(paths, extra_meta={'train_src': train_base, 'val_src': val_base})

	# 4. PHASE 3: TESTING (Evaluating on unseen data)
	print(f'--- Phase 3: Final Testing on {test_base} ---')
	X_test, y_test, groups_test = load_whole_dataset(test_base, prep_cfg)

	if model_kind == 'linear':
		test_metrics = model_graph_functions.evaluate_and_graph_reg(model, X_test, y_test, groups=groups_test, name=f'{model_tag}', graph=True)
	else:
		test_metrics = model_graph_functions.evaluate_and_graph_clf(model, X_test, y_test, groups=groups_test, name=f'{model_tag}', graph=True)

	if paths['graph_test']:
		plt.savefig(paths['graph_test'])
		plt.close()

	if hasattr(model, 'predict_class'):
		# For the Softmax classifier
		y_pred_cm = model.predict_class(X_test, groups=groups_test)
	else:
		# For the Linear regressor, round continuous output to nearest {0, 1, 2}
		raw_pred = model.predict(X_test, groups=groups_test)
		y_pred_cm = np.rint(raw_pred).astype(int)
		y_pred_cm = np.clip(y_pred_cm, 0, 2)

	model_graph_functions.plot_confusion_matrix(y_true=y_test, y_pred=y_pred_cm, name=f'{model_tag} Confusion Matrix', save_path=paths['graph_cm'])

	return {'model_kind': model_kind, 'trained': trained, 'test_metrics': test_metrics, 'paths': {k: str(v) for k, v in paths.items() if k != 'dir'}}


def train_eval(
	train_base: str,
	val_base: str,
	test_base: str,
	*,
	prep_cfg: Optional[data_preparation.PrepConfig] = None,
	models_dir: str | Path = 'models',
	force_retrain: bool = False,
	draws: int = 1000,
	tune: int = 1000,
	chains: int = 4,
	target_accept: float = 0.9,
	seed: int = 123,
) -> Dict[str, Any]:
	out_linear = train_eval_one(
		train_base,
		val_base,
		test_base,
		model_kind='linear',
		prep_cfg=prep_cfg,
		models_dir=models_dir,
		force_retrain=force_retrain,
		draws=draws,
		tune=tune,
		chains=chains,
		target_accept=target_accept,
		seed=seed,
	)

	out_softmax = train_eval_one(
		train_base,
		val_base,
		test_base,
		model_kind='softmax3',
		prep_cfg=prep_cfg,
		models_dir=models_dir,
		force_retrain=force_retrain,
		draws=draws,
		tune=tune,
		chains=chains,
		target_accept=target_accept,
		seed=seed,
	)

	return {'linear': out_linear, 'softmax3': out_softmax}


def test_on_new_data(model, dataset_name: str, prep_cfg: Optional[data_preparation.PrepConfig] = None):
	"""
	Loads a new dataset, prepares it using the model's original scaling params,
	and returns predictions and metrics without any training.
	"""
	if prep_cfg is None:
		prep_cfg = data_preparation.PrepConfig(dataset_name=dataset_name)

	# 1. Load the new data
	new_data = data_preparation.prepare_data(dataset_name, prep_cfg)
	X_raw, y_raw = flatten_examples(new_data['test']['X'], new_data['test']['y'])

	# 2. Wrap the model in the NoRefitProxy to ensure .fit() cannot be called
	proxy = _NoRefitProxy(model)

	print(f'\n=== Testing on New Dataset: {dataset_name} ===')

	# 3. Use the existing evaluation utility to get metrics and plots
	# This uses model.predict() internally, which applies stored feature_mean_ and feature_std_
	if hasattr(model, 'predict_class'):
		metrics = model_graph_functions.evaluate_and_graph_clf(proxy, X_raw, y_raw, name=f'External_Test_{dataset_name}', graph=True)
	else:
		metrics = model_graph_functions.evaluate_and_graph_reg(proxy, X_raw, y_raw, name=f'External_Test_{dataset_name}', graph=True)
	return metrics


if __name__ == '__main__':
	multiprocessing.set_start_method('forkserver', force=True)
	training_file, validation_file, testing_file = 'testing.training', 'testing.validation', 'testing.testing'
	prep_cfg = data_preparation.PrepConfig(dataset_name='testing')
	models_dir = 'models'
	# results = train_eval(train_base=training_file, val_base=validation_file, test_base=testing_file, force_retrain=True)
	out_softmax = train_eval_one(
		train_base=training_file,
		val_base=validation_file,
		test_base=testing_file,
		model_kind='softmax3',
		prep_cfg=prep_cfg,
		models_dir=models_dir,
		force_retrain=True,
		draws=1000,
		tune=1000,
		chains=4,
		target_accept=0.9,
		seed=123,
	)

	print('\n--- Pipeline Complete ---')
	print(f'Results stored in: {out_softmax["paths"]["dir"] if "dir" in out_softmax["paths"] else "models"}')
