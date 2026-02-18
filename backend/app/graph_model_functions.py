import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import accuracy_score, average_precision_score, mean_squared_error, precision_recall_curve, r2_score, roc_auc_score, roc_curve


def evaluate_and_graph_clf(model, X_train, y_train, X_test, y_test, name, graph):
	model.fit(X_train, y_train)
	y_pred_test = model.predict(X_test)
	y_pred_train = model.predict(X_train)

	if hasattr(model, 'predict_proba'):
		y_prob_test = model.predict_proba(X_test)[:, 1]
		y_prob_train = model.predict_proba(X_train)[:, 1]
	else:
		y_prob_test = model.decision_function(X_test)
		y_prob_train = model.decision_function(X_train)

	test_acc = accuracy_score(y_test, y_pred_test)
	test_auc = roc_auc_score(y_test, y_prob_test)
	test_ap = average_precision_score(y_test, y_prob_test)

	train_acc = accuracy_score(y_train, y_pred_train)
	train_auc = roc_auc_score(y_train, y_prob_train)

	print(f'--- {name} ---')
	print(f'Train Accuracy: {train_acc:.4f} | Train AUC: {train_auc:.4f}')
	print(f'Test  Accuracy: {test_acc:.4f} | Test  AUC: {test_auc:.4f}')

	if (train_acc - test_acc) > 0.05:
		print('⚠️ Warning: Signs of Overfitting (Train is much better than Test)')
	else:
		print('✅ Model seems balanced')
	print('-' * 30)

	fpr, tpr, _ = roc_curve(y_test, y_prob_test)
	precision, recall, _ = precision_recall_curve(y_test, y_prob_test)

	if graph:
		fig, axes = plt.subplots(1, 2, figsize=(14, 6))

		# --- Plot 1: ROC Curve ---
		axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {test_auc:.3f}')
		axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
		axes[0].set_xlim([0.0, 1.0])
		axes[0].set_ylim([0.0, 1.05])
		axes[0].set_xlabel('False Positive Rate')
		axes[0].set_ylabel('True Positive Rate')
		axes[0].set_title(f'ROC Curve: {name}')
		axes[0].legend(loc='lower right')
		axes[0].grid(True, alpha=0.3)

		# --- Plot 2: Precision-Recall Curve ---
		# "No Skill" baseline is just the percentage of positive cases
		no_skill = y_test.mean()

		axes[1].plot(recall, precision, color='green', lw=2, label=f'Avg Precision = {test_ap:.3f}')
		axes[1].plot([0, 1], [no_skill, no_skill], color='navy', linestyle='--', label='No Skill')
		axes[1].set_xlim([0.0, 1.0])
		axes[1].set_ylim([0.0, 1.05])
		axes[1].set_xlabel('Recall')
		axes[1].set_ylabel('Precision')
		axes[1].set_title(f'PR Curve: {name}')
		axes[1].legend(loc='upper right')
		axes[1].grid(True, alpha=0.3)

		plt.tight_layout()

	return {'model': name, 'accuracy': test_acc, 'auc': test_auc, 'average_precision': test_ap, 'train_accuracy': train_acc, 'train_auc': train_auc}


def evaluate_and_graph_reg(model, X_train, y_train, X_test, y_test, name, graph):
	model.fit(X_train, y_train)

	# Predictions
	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)

	# --- Metrics (same as evaluate_and_graph_reg) ---
	mse_train = mean_squared_error(y_train, y_pred_train)
	mse_test = mean_squared_error(y_test, y_pred_test)

	rmse_train = np.sqrt(mse_train)
	rmse_test = np.sqrt(mse_test)

	r2_train = r2_score(y_train, y_pred_train)
	r2_test = r2_score(y_test, y_pred_test)

	print(f'--- {name} ---')
	print(f'Train RMSE: {rmse_train:.4f} | Train R²: {r2_train:.4f}')
	print(f'Test  RMSE: {rmse_test:.4f} | Test  R²: {r2_test:.4f}')

	if (r2_train - r2_test) > 0.05:
		print('⚠️ Warning: Signs of Overfitting (Train R² much higher than Test R²)')
	else:
		print('✅ Model seems reasonably balanced')
	print('-' * 30)

	# --- Plots (equivalent to plot_full_diagnostics) ---
	if graph:
		residuals = y_test - y_pred_test
		r2 = r2_test

		fig, axs = plt.subplots(2, 2, figsize=(12, 10))

		# 1) Predicted vs True
		axs[0, 0].scatter(y_test, y_pred_test, alpha=0.5)
		slope, intercept = np.polyfit(y_test, y_pred_test, 1)
		axs[0, 0].plot(y_test, slope * y_test + intercept, color='blue', label='Regression Line')
		axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Identity Line')
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
		axs[1, 0].scatter(y_pred_test, residuals, alpha=0.5)
		axs[1, 0].axhline(0, color='red', linestyle='--')
		axs[1, 0].set_xlabel('Fitted Values')
		axs[1, 0].set_ylabel('Residuals')
		axs[1, 0].set_title('Residuals vs Fitted')

		# 4) QQ Plot
		stats.probplot(residuals, dist='norm', plot=axs[1, 1])
		axs[1, 1].set_title('QQ Plot')

		plt.tight_layout()

	return {'model': name, 'rmse_train': rmse_train, 'rmse_test': rmse_test, 'r2_train': r2_train, 'r2_test': r2_test}
