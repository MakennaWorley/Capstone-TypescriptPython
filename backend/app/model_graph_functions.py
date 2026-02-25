import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, precision_recall_curve, r2_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


def evaluate_and_graph_clf(model, X, y, name, graph, **kwargs):
	"""
	Evaluates a pre-trained classifier on the provided data and generates
	diagnostic plots (ROC and PR curves). No internal splitting or fitting.
	"""
	# 1. Generate Predictions using the pre-trained model
	y = np.asarray(y)
	n_classes = 3
	classes = [0, 1, 2]

	y_pred = model.predict_class(X, **kwargs) if hasattr(model, 'predict_class') else model.predict(X)

	# Handle probability extraction for different model types
	if hasattr(model, 'predict_proba'):
		# This assumes binary-style probability or uses the decision values.
		y_score = model.predict_proba(X)
	else:
		# Fallback to standard predict if no probability/decision method exists
		y_score = label_binarize(y_pred, classes=classes)

	# 2. Calculate Metrics for the provided data
	acc = accuracy_score(y, y_pred)

	# ROC/PR metrics only work out-of-the-box for binary tasks in this implementation
	auc_macro = roc_auc_score(y, y_score, multi_class='ovr', average='macro')

	print(f'--- {name} ---')
	print(f'Accuracy: {acc:.4f} | Macro-AUC: {auc_macro:.4f}')
	print('-' * 30)

	# 3. Diagnostic Plots
	if graph:
		# Binarize y for plotting individual curves
		y_bin = label_binarize(y, classes=classes)
		fig, axes = plt.subplots(1, 2, figsize=(14, 6))
		colors = ['blue', 'red', 'green']

		for i, color in zip(range(n_classes), colors):
			# ROC Curve per class
			fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
			axes[0].plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (AUC = {roc_auc_score(y_bin[:, i], y_score[:, i]):.2f})')

			# PR Curve per class
			prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
			axes[1].plot(rec, prec, color=color, lw=2, label=f'Class {i}')

		# Finalize ROC Plot
		axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
		axes[0].set_title(f'Multiclass ROC: {name}')
		axes[0].set_xlabel('False Positive Rate')
		axes[0].set_ylabel('True Positive Rate')
		axes[0].legend(loc='lower right')
		axes[0].grid(True, alpha=0.3)

		# Finalize PR Plot
		axes[1].set_title(f'Multiclass PR: {name}')
		axes[1].set_xlabel('Recall')
		axes[1].set_ylabel('Precision')
		axes[1].legend(loc='upper right')
		axes[1].grid(True, alpha=0.3)

		plt.tight_layout()

	return {'model': name, 'accuracy': acc, 'auc_macro': auc_macro}


def evaluate_and_graph_reg(model, X, y, name, graph, **kwargs):
	"""
	Evaluates a pre-trained model on the provided data and generates
	diagnostic plots. No internal splitting or fitting occurs.
	"""
	# 1. Generate Predictions using the pre-trained model
	y = np.asarray(y)
	y_pred = model.predict(X, **kwargs)
	y_pred = np.asarray(y_pred)

	mask = np.isfinite(y) & np.isfinite(y_pred)
	if mask.sum() == 0:
		raise ValueError('No finite y/y_pred values to evaluate.')
	if mask.sum() != len(y):
		y = y[mask]
		y_pred = y_pred[mask]

	# 2. Calculate Metrics for the provided data
	mse = mean_squared_error(y, y_pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y, y_pred)

	print(f'--- {name} ---')
	print(f'RMSE: {rmse:.4f} | R²: {r2:.4f}')
	print('-' * 30)

	# 3. Diagnostic Plots
	if graph:
		residuals = y - y_pred
		fig, axs = plt.subplots(2, 2, figsize=(12, 10))

		# 1) Predicted vs True (with Jitter for discrete dosage)
		y_jitter = y + np.random.normal(0, 0.05, size=len(y))
		axs[0, 0].scatter(y_jitter, y_pred, alpha=0.3, s=10)

		# Range for lines (use true-value range, but ensure a non-zero span)
		x_min = float(np.min(y))
		x_max = float(np.max(y))
		if np.isclose(x_min, x_max):
			x_min -= 0.5
			x_max += 0.5

		x_line = np.linspace(x_min, x_max, 200)

		# Add regression line for the provided data
		try:
			if len(np.unique(y)) >= 2:
				slope, intercept, *_ = stats.linregress(y, y_pred)
				axs[0, 0].plot(x_line, slope * x_line + intercept, color='blue', label='Regression Line')
			else:
				slope = intercept = None
		except Exception:
			slope = intercept = None

		# Identity line (y=x)
		axs[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Identity Line')
		axs[0, 0].set_xlabel('True Values')
		axs[0, 0].set_ylabel('Predicted Values')
		axs[0, 0].set_title(f'Predicted vs True ({name})\n$R^2 = {r2:.3f}$')
		axs[0, 0].legend()

		# 2) Residual Histogram
		sns.histplot(residuals, kde=True, bins=20, ax=axs[0, 1])
		axs[0, 1].set_title('Residual Distribution')
		axs[0, 1].set_xlabel('Residual')
		axs[0, 1].set_ylabel('Count')

		# 3) Residuals vs Fitted
		axs[1, 0].scatter(y_pred, residuals, alpha=0.5)
		axs[1, 0].axhline(0, color='red', linestyle='--')
		axs[1, 0].set_xlabel('Fitted Values')
		axs[1, 0].set_ylabel('Residuals')
		axs[1, 0].set_title('Residuals vs Fitted')

		# 4) QQ Plot
		stats.probplot(residuals, dist='norm', plot=axs[1, 1])
		axs[1, 1].set_title('QQ Plot')

		plt.tight_layout()

	return {'model': name, 'rmse': rmse, 'r2': r2}


def plot_confusion_matrix(y_true, y_pred, name, save_path):
	"""
	Generates and saves a confusion matrix heatmap for genetic dosage (0, 1, 2).
	"""
	y_true_int = np.rint(y_true).astype(int)
	y_pred_int = np.rint(y_pred).astype(int)

	cm = confusion_matrix(y_true_int, y_pred_int, labels=[0, 1, 2])

	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])

	plt.title(f'Confusion Matrix: {name}')
	plt.xlabel('Predicted Dosage')
	plt.ylabel('True Dosage')
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close()
