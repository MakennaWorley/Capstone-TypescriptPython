import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import accuracy_score, average_precision_score, mean_squared_error, precision_recall_curve, r2_score, roc_auc_score, roc_curve


def evaluate_and_graph_clf(model, X, y, name, graph):
	"""
	Evaluates a pre-trained classifier on the provided data and generates
	diagnostic plots (ROC and PR curves). No internal splitting or fitting.
	"""
	# 1. Predictions and Probabilities
	y_pred = model.predict_class(X) if hasattr(model, 'predict_class') else model.predict(X)

	# Handle probability extraction for different model types
	if hasattr(model, 'predict_proba'):
		# For multi-class (like your softmax3), you might need specific class logic.
		# This assumes binary-style probability or uses the decision values.
		probs = model.predict_proba(X)
		y_prob = probs[:, 1] if probs.ndim > 1 and probs.shape[1] > 1 else probs
	else:
		y_prob = model.decision_function(X)

	# 2. Calculate Metrics for the provided data
	acc = accuracy_score(y, y_pred)
	# Note: For multi-class AUC, ensure multi_class='ovr' or similar is used if needed
	auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0
	ap = average_precision_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0

	print(f'--- {name} ---')
	print(f'Accuracy: {acc:.4f} | AUC: {auc:.4f} | Avg Precision: {ap:.4f}')
	print('-' * 30)

	# 3. Plots
	if graph:
		fpr, tpr, _ = roc_curve(y, y_prob)
		precision, recall, _ = precision_recall_curve(y, y_prob)

		fig, axes = plt.subplots(1, 2, figsize=(14, 6))

		# --- Plot 1: ROC Curve ---
		axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.3f}')
		axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
		axes[0].set_xlim([0.0, 1.0])
		axes[0].set_ylim([0.0, 1.05])
		axes[0].set_xlabel('False Positive Rate')
		axes[0].set_ylabel('True Positive Rate')
		axes[0].set_title(f'ROC Curve: {name}')
		axes[0].legend(loc='lower right')
		axes[0].grid(True, alpha=0.3)

		# --- Plot 2: Precision-Recall Curve ---
		no_skill = y.mean() if len(y) > 0 else 0
		axes[1].plot(recall, precision, color='green', lw=2, label=f'Avg Precision = {ap:.3f}')
		axes[1].plot([0, 1], [no_skill, no_skill], color='navy', linestyle='--', label='No Skill')
		axes[1].set_xlim([0.0, 1.0])
		axes[1].set_ylim([0.0, 1.05])
		axes[1].set_xlabel('Recall')
		axes[1].set_ylabel('Precision')
		axes[1].set_title(f'PR Curve: {name}')
		axes[1].legend(loc='upper right')
		axes[1].grid(True, alpha=0.3)

		plt.tight_layout()

	return {'model': name, 'accuracy': acc, 'auc': auc, 'average_precision': ap}


def evaluate_and_graph_reg(model, X, y, name, graph):
	"""
	Evaluates a pre-trained model on the provided data and generates
	diagnostic plots. No internal splitting or fitting occurs.
	"""
	# 1. Generate Predictions using the pre-trained model
	y_pred = model.predict(X)

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

		# 1) Predicted vs True
		axs[0, 0].scatter(y, y_pred, alpha=0.5)
		# Add regression line for the provided data
		slope, intercept = np.polyfit(y, y_pred, 1)
		axs[0, 0].plot(y, slope * y + intercept, color='blue', label='Regression Line')
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
