import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# --- Configuration ---
SEED = 42

# --- Utilities ---
def get_vars(df, exclude_cols=None):
    """
    Identify variable types based on dtype + unique counts. 
    (Recreated from s5e12.ipynb)
    """
    if exclude_cols is None:
        exclude_cols = []

    # --------------------
    # Base variable groups
    # --------------------
    numerical_vars = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_vars = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Remove excluded columns from each list
    numerical_vars = [c for c in numerical_vars if c not in exclude_cols]
    categorical_vars = [c for c in categorical_vars if c not in exclude_cols]

    # --------------------
    # Continuous vs Discrete
    # --------------------
    continuous_vars = []
    discrete_vars = []

    n_rows = len(df)

    for col in numerical_vars:
        n_unique = df[col].nunique(dropna=True)

        # Heuristic from notebook: Check unique count <= 10 or (integer type AND unique count is < 1% of rows)
        if (n_unique <= 10) or (df[col].dtype.kind in "iu" and n_unique / n_rows < 0.01):
            discrete_vars.append(col)
        else:
            continuous_vars.append(col)

    # --------------------
    # Binary detection (among discrete vars)
    # --------------------
    binary_vars = []
    nonbinary_discrete = discrete_vars.copy()

    for col in nonbinary_discrete.copy():
        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)

        if n_unique == 2:
            binary_vars.append(col)
            discrete_vars.remove(col)

    # --------------------
    # Final "true" groups
    # --------------------
    true_numerical_vars = continuous_vars + discrete_vars
    true_categorical_vars = categorical_vars + binary_vars
    all_vars = true_numerical_vars + true_categorical_vars

    # --------------------
    # Return everything
    # --------------------
    return {
        "true_numerical_vars": true_numerical_vars,
        "true_categorical_vars": true_categorical_vars,
        "all_vars": all_vars
    }

# --- EDA Dashboard ---
@st.cache_data
def display_eda(data):
    """Displays EDA for the new genetic dosage data."""
    st.header("Data Overview & Summary Statistics")

    st.subheader("Initial Data Sample (First 5 Rows)")
    st.dataframe(data.head(), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(data.describe(include="all").T, use_container_width=True)
    st.markdown("---")

    # --- Feature Distributions (Dosages) ---
    st.header("Dosage Feature Distributions")
    dosage_cols = [col for col in data.columns if 'dosage' in col]

    if dosage_cols:
        # Use a loop to plot distributions of all dosage columns
        num_plots = len(dosage_cols)
        cols_per_row = 3
        num_rows = (num_plots + cols_per_row - 1) // cols_per_row

        fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(4 * cols_per_row, 4 * num_rows))
        plt.suptitle("Dosage Distributions (Counts of Alleles 0, 1, or 2)", y=1.02)

        # Flatten axes array for easy iteration
        axes = axes.flatten() if num_rows > 1 or cols_per_row > 1 else [axes]

        for i, col in enumerate(dosage_cols):
            sns.countplot(x=data[col], ax=axes[i], palette="viridis")
            axes[i].set_title(f"{col}", fontsize=10)
            axes[i].set_xlabel("Dosage (0, 1, 2)")
            axes[i].set_ylabel("Count")

        # Hide unused subplots
        for j in range(num_plots, len(axes)):
            if j < len(axes): # Safety check
                fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        st.pyplot(fig, clear_figure=True)
        st.markdown("---")
    else:
        st.info("No dosage columns found for distribution plotting.")

    # --- Target Variable Distribution ---
    if 'y' in data.columns:
        st.header("Target Variable $Y$ Distribution")
        fig_y, ax_y = plt.subplots(figsize=(5, 3))
        sns.countplot(x=data["y"], ax=ax_y, palette="coolwarm")
        ax_y.set_title("Count of Target $Y$ (0 or 1)")
        ax_y.set_xlabel("Y Status")
        ax_y.set_ylabel("Count")
        st.pyplot(fig_y, clear_figure=True)
        st.markdown(f"Count of $Y=0$: {len(data[data['y'] == 0])}")
        st.markdown(f"Count of $Y=1$: {len(data[data['y'] == 1])}")
        st.markdown("---")

@st.cache_data
def display_categorical_pairs(data):
    """
    Displays a grid of plots showing pairwise relationships between categorical features (dosages).
    - Diagonal: Individual feature distribution (Count Plot).
    - Off-Diagonal: Pairwise relationship (Heatmap of row-normalized contingency table).
    """
    st.header("Pairwise Dosage Feature Relationships")
    st.text("For right now, I am only graphing the first 3 columns since this is a time-consuming process and I haven't fully"
    "explored the data to pull the most interesting/important columns for this stat.")
    
    # Identify dosage columns as the 'categorical' variables for this context
    all_dosage_cols = [col for col in data.columns if 'dosage' in col]

    # FOR POC ONLY
    categorical_vars = all_dosage_cols[:3]
    
    if not categorical_vars or len(categorical_vars) < 2:
        st.info("Insufficient dosage columns (less than 2) for pairwise relationship plotting.")
        return

    n_categorical_vars = len(categorical_vars)
    
    # Set up the figure and axes
    # We use a smaller size for the plot to fit better in Streamlit, adjusting row/col size based on count
    plot_size = min(3.5, 20 / n_categorical_vars) # Cap size to avoid huge figures
    fig, axes = plt.subplots(n_categorical_vars, n_categorical_vars,
                             figsize=(n_categorical_vars * plot_size, n_categorical_vars * plot_size))
    
    plt.suptitle("Pairwise Dosage Relationships (Dosages 0, 1, 2)", y=1.01, fontsize=16)

    # Ensure axes is always a 2D array, even if n_categorical_vars is 1
    if n_categorical_vars == 1:
        axes = np.array([[axes]])
    elif n_categorical_vars > 1 and axes.ndim == 1:
        # Handle the case for n=2 where subplots returns a 1D array of 2
        axes = axes.reshape(1, -1) if n_categorical_vars == 2 else axes
        
    for i in range(n_categorical_vars):
        for j in range(n_categorical_vars):
            var1 = categorical_vars[i]
            var2 = categorical_vars[j]
            
            # --- Diagonal (i == j): Individual Distribution (Count Plot) ---
            if i == j:
                sns.countplot(
                    y=data[var1].astype('category'), # Treat as category for better plotting
                    ax=axes[i, j],
                    hue=data[var1].astype('category'),
                    palette="Pastel1",
                    order=data[var1].value_counts().index,
                    legend=False
                )
                axes[i, j].set_title(f"Distribution of **{var1}**", fontsize=10)
                axes[i, j].set_ylabel("")
                axes[i, j].set_xlabel("Count")
            
            # --- Off-Diagonal (i != j): Pairwise Relationship (Heatmap) ---
            else:
                # Create a contingency table (cross-tabulation)
                # Normalize='index' (row-wise) shows the distribution of var2 *within* each category of var1
                contingency_table = pd.crosstab(data[var1], data[var2], normalize='index')
                
                # Plot the contingency table as a heatmap
                sns.heatmap(
                    contingency_table,
                    annot=True,
                    fmt=".2f",
                    cmap="YlGnBu",
                    cbar=False,
                    ax=axes[i, j],
                    linewidths=.5,
                    linecolor='gray'
                )
                axes[i, j].set_title(f"**{var1}** vs **{var2}** (Row Normalized)", fontsize=10)
                axes[i, j].set_ylabel(var1)
                axes[i, j].set_xlabel(var2)

    # Improve layout spacing
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    st.pyplot(fig, clear_figure=True)
    st.markdown("---")

@st.cache_data
def display_feature_vs_target(data, target_name='y'):
    """
    Displays box plots and count plots for dosage features vs. the target variable.
    Assumes dosage columns exist and are pre-identified.
    """
    if target_name not in data.columns:
        st.error(f"Target column '{target_name}' is missing from the data.")
        return

    st.header(f"Dosage Features vs. Target Variable ({target_name})")
    
    # Identify dosage columns as the features (X)
    dosage_cols = [col for col in data.columns if 'dosage' in col]
    num_plots = len(dosage_cols)

    if not dosage_cols:
        st.info("No dosage columns found for feature vs. target plotting.")
        return

    cols_per_row = 3
    num_rows = (num_plots + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(4 * cols_per_row, 4 * num_rows))
    plt.suptitle(f"Allele Dosage Counts by Target {target_name} Status", y=1.02)

    # Flatten axes array for easy iteration and handle single plot case
    axes = axes.flatten() if num_rows > 1 or cols_per_row > 1 else [axes]

    for i, feature_name in enumerate(dosage_cols):
        ax = axes[i]
        
        # Use countplot for categorical/discrete data
        sns.countplot(
            x=data[feature_name],
            hue=data[target_name],
            data=data,
            palette="Pastel2",
            order=data[feature_name].value_counts().index,
            ax=ax
        )
        ax.set_title(f'{feature_name}', fontsize=10)
        ax.set_xlabel("Dosage (0, 1, 2)")
        ax.set_ylabel("Count")
        ax.legend(title=target_name)

    # Hide unused subplots
    for j in range(num_plots, len(axes)):
        if j < len(axes): # Safety check
            fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    st.pyplot(fig, clear_figure=True)
    st.markdown("---")

# --- Preprocessing ---
@st.cache_data
def preprocess_data(data, target_col='y'):
    """
    Applies the full preprocessing pipeline from the notebook (split, resample, scale, encode).
    The input 'data' should be the full training dataset.
    (Recreated from s5e12.ipynb)
    """
    if target_col not in data.columns:
        st.error(f"Target column '{target_col}' not found in data.")
        return None, None, None, None

    # 1. Separate features and target using custom logic
    vars_dict = get_vars(data, exclude_cols=[target_col])
    X = data[vars_dict["all_vars"]]
    y = data[target_col]

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    
    # Reset index for clean resampling operation
    X_train_resampled = X_train.reset_index(drop=True)
    y_train_resampled = y_train.reset_index(drop=True)
    train_data = pd.concat([X_train_resampled, y_train_resampled], axis=1)

    # 3. Handle data imbalance (Oversampling on TRAINING set only)
    st.info(f"Original Training Counts: {y_train.value_counts().to_dict()} (Skewed, applying oversampling)")
    
    majority = train_data[train_data[target_col] == 1]
    minority = train_data[train_data[target_col] == 0]

    minority_upsampled = resample(minority,
                                  replace=True,
                                  n_samples=len(majority),
                                  random_state=SEED)

    train_data_balanced = pd.concat([majority, minority_upsampled])
    # Shuffle the balanced data
    train_data_balanced = train_data_balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)

    X_train_resampled = train_data_balanced.drop(target_col, axis=1)
    y_train_resampled = train_data_balanced[target_col]
    
    st.success(f"Balanced Training Counts: {y_train_resampled.value_counts().to_dict()}")

    # 4. Scaling Numerical Features
    scaler = StandardScaler()
    numerical_cols = vars_dict["true_numerical_vars"]

    # Fit scaler only on resampled training numerical features
    X_train_scaled_num = scaler.fit_transform(X_train_resampled[numerical_cols])
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled_num,
        columns=numerical_cols,
        index=X_train_resampled.index
    )

    # Transform test numerical features
    X_test_scaled_num = scaler.transform(X_test[numerical_cols])
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled_num,
        columns=numerical_cols,
        index=X_test.index
    )

    # 5. One-Hot Encoding Categorical Features
    categorical_cols = vars_dict["true_categorical_vars"]
    
    # Use the original split X_train/X_test to define the categorical columns to encode.
    # Important: Re-create the encoding process based on the *resampled* training set.
    X_train_encoded = pd.get_dummies(X_train_resampled[categorical_cols], drop_first=True)
    X_test_encoded = pd.get_dummies(X_test[categorical_cols], drop_first=True)

    # Align columns, filling missing new columns with 0 (as done in the notebook with join='left')
    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded,
        join='left', # Matches notebook logic: ensures X_test only has columns that appeared in X_train
        axis=1,
        fill_value=0
    )
    
    # 6. Concatenate numerical (scaled) and categorical (encoded) features
    X_train_final = pd.concat([X_train_scaled_df, X_train_encoded], axis=1)
    X_test_final = pd.concat([X_test_scaled_df, X_test_encoded], axis=1)
    
    st.info(f"Final feature shape (Train): {X_train_final.shape}, (Test): {X_test_final.shape}")

    # Return resampled training data and original test data (as in notebook split)
    return X_train_final, X_test_final, y_train_resampled, y_test

# --- Model Training and Evaluation ---
def train_and_evaluate_clf(model_class, X_train, y_train, X_test, y_test, name, **kwargs):
    """
    Fits a classifier, evaluates its performance, and generates ROC/PR curves.
    (Adapted from s5e12.ipynb's evaluate_and_graph function)
    """
    # Instantiate the model with kwargs (like max_iter, reg_param)
    model_params = kwargs.copy()
    
    # Check if the model class's constructor supports 'random_state'
    if 'random_state' in model_class().get_params():
        # If it supports random_state AND it's not already in kwargs (e.g., from the model list)
        if 'random_state' not in model_params:
            model_params['random_state'] = SEED
    
    # Instantiate the model with the prepared parameters
    model = model_class(**model_params)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Calculate probabilities/decision function
    if hasattr(model, "predict_proba"):
        # Predict_proba returns (n_samples, n_classes), need only the positive class [:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]
        y_prob_train = model.predict_proba(X_train)[:, 1]
    elif hasattr(model, "decision_function"):
        # Use decision_function for models like Perceptron/SVC when they don't have predict_proba
        y_prob_test = model.decision_function(X_test)
        y_prob_train = model.decision_function(X_train)
        # Decision function outputs need to be normalized for AUC/AP,
        # but ROC_AUC_SCORE can handle non-normalized decision scores.
    elif hasattr(model, 'coef_'):
                    feature_names = X_train.columns.tolist() 
                    if model_class in [Perceptron, LogisticRegression, LDA]:
                        st.write("**Feature Coefficients**:")
                        # coef_ is 2D for binary classification
                        coef_unmasked_df = pd.DataFrame(model.coef_[0], index=feature_names, columns=['Coefficient'])
                        st.dataframe(coef_unmasked_df.sort_values(by='Coefficient', ascending=False), use_container_width=True)
    else:
        # Fallback for models without proba or decision function (rare for classification)
        st.warning(f"Model {name} does not support probability output for AUC/PR calculation.")
        # Cannot calculate AUC/AP without probabilities or decision scores.
        test_auc = np.nan
        train_auc = np.nan
        test_ap = np.nan
        
        # Calculate standard accuracy only
        test_acc = accuracy_score(y_test, y_pred_test)
        train_acc = accuracy_score(y_train, y_pred_train)
        
        st.subheader(f"Model: {name}")
        st.metric("Test Accuracy", f"{test_acc:.4f}", delta=None)
        st.info("ROC/PR plots skipped due to lack of probability/decision function output.")
        
        results = {
            "model": name,
            "accuracy": test_acc,
            "auc": test_auc,
            "average_precision": test_ap,
            "train_accuracy": train_acc,
            "train_auc": train_auc,
            "model_instance": model
        }
        return results

    # Calculate Metrics
    test_acc = accuracy_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_prob_test)
    test_ap = average_precision_score(y_test, y_prob_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    train_auc = roc_auc_score(y_train, y_prob_train)
    
    # --- Streamlit Display ---
    st.subheader(f"Model: {name}")
    col1, col2, col3, col4 = st.columns(4)

    # Display basic metrics
    col1.metric("Test Accuracy", f"{test_acc:.4f}", delta=None)
    col2.metric("Test AUC", f"{test_auc:.4f}", delta=None)
    col3.metric("Train AUC", f"{train_auc:.4f}", delta=None)

    # Overfitting Check
    overfitting_gap = train_auc - test_auc
    if overfitting_gap > 0.05:
        col4.warning(f"⚠️ Overfitting Warning: Gap of {overfitting_gap:.4f}. Train AUC much higher than Test AUC.")
    else:
        col4.info("✅ Model performance is balanced")

    # --- Plotting ROC and PR Curves ---
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob_test)
    
    # "No Skill" baseline is just the percentage of positive cases (for PR curve)
    no_skill = y_test.mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {test_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'ROC Curve: {name}')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    axes[1].plot(recall, precision, color='green', lw=2, label=f'Avg Precision = {test_ap:.3f}')
    axes[1].plot([0, 1], [no_skill, no_skill], color='navy', linestyle='--', label=f'No Skill (AP={no_skill:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'PR Curve: {name}')
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    
    # Store results to be returned
    results = {
        "model": name,
        "accuracy": test_acc,
        "auc": test_auc,
        "average_precision": test_ap,
        "train_accuracy": train_acc,
        "train_auc": train_auc,
        "model_instance": model
    }
    
    return results
