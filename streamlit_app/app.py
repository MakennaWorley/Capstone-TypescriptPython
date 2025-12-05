import os
import requests
import streamlit as st
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from io import StringIO

from utils import display_categorical_pairs, display_eda, display_feature_vs_target, preprocess_data, train_and_evaluate_clf, SEED
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

load_dotenv()
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Streamlit + FastAPI", layout="wide")
st.title("Proof of Concept: Probabilistic Ancestral Inference from Incomplete Genetic Data")
st.caption(f"Backend fastAPI Docker container at: {API_BASE}")

st.text("The core problem in genetics is that we rarely have complete data—sequencing every ancestor "
"is often impossible due to cost or sample degradation. This leaves critical gaps in our family trees, "
"limiting our ability to predict hereditary traits and model population history."
"\n\nMy goal is to develop a computational system to reconstruct these missing ancestral genotypes "
"using probabilistic machine learning models, creating a rigorous tool for studies that rely on partial "
"genetic data.")

# --- Test FastAPI ---
if st.button("Ping FastAPI"):
    try:
        r = requests.get(f"{API_BASE}/api/hello", timeout=5)
        st.success(r.json())
    except Exception as e:
        st.error(f"Error: {e}")

# --- Load CSV Data ---
if st.button("Fetch masked data!"):
    with st.spinner('Fetching simulated genetic data... this may take a moment.'):
        try:
            url = f"{API_BASE}/poc/data/A_five_masked_out/csv"
            r = requests.get(url, timeout=5)
            r.raise_for_status()

            csv_str = r.content.decode("utf-8")
            masked_df = pd.read_csv(StringIO(csv_str))

            st.success("Data loaded!")

            # Save in session state
            st.session_state["masked"] = masked_df

        except Exception as e:
            st.error(f"Error: {e}")

# --- Use the Data ---
if "masked" in st.session_state:
    masked_df = st.session_state["masked"]

    st.header("Masked Analysis & Visualizations")

    st.subheader("Data Preview")
    st.dataframe(masked_df)

    # Summary
    st.subheader("Summary Statistics")
    st.write(masked_df.describe())

# --- Load CSV Data ---
if st.button("Fetch unmasked data!"):
    with st.spinner('Loading simulated genetic data... this may take a moment.'):
        if "unmasked_df" not in st.session_state:
            try:
                url = f"{API_BASE}/poc/data/X_ten_unmasked_out/csv"
                
                # Use a cached function to fetch data to avoid re-fetching on *every* rerun
                @st.cache_data(ttl=600) # Cache the fetch for 10 minutes
                def fetch_unmasked_data(url):
                    r = requests.get(url, timeout=5)
                    r.raise_for_status()
                    csv_str = r.content.decode("utf-8")
                    return pd.read_csv(StringIO(csv_str))

                unmasked_df = fetch_unmasked_data(url)
                st.session_state["unmasked_df"] = unmasked_df
                st.success("Unmasked data loaded!")

            except Exception as e:
                st.error(f"Error fetching unmasked data: {e}")

# --- Use the Data ---
if "unmasked_df" in st.session_state:
    unmasked_df = st.session_state["unmasked_df"]

    st.header("Unmasked Data Analysis & Visualizations")

    data_tab1, data_tab2, data_tab3 = st.tabs(["Data Preprocessing", "Exploratory Data Analysis", "Classification Models"])

    # --- Preprocessing ---
    with data_tab1:
        st.subheader("Data Preview")
        st.dataframe(unmasked_df)

        # Summary
        st.subheader("Summary Statistics")
        st.write(unmasked_df.describe())

        st.subheader("Data Preprocessing")
        
        # 3. Call the preprocess_data function
        processed_data = preprocess_data(unmasked_df)
        
        if processed_data is not None:
            X_train, X_test, y_clf_train, y_clf_test = processed_data
            
            st.success(f"Data successfully preprocessed and scaled.")
            
            st.subheader("Training Data: Scaled Features")
            st.dataframe(X_train)
            
            st.subheader("Trainging Data: Target Distribution")
            st.dataframe(y_clf_train.value_counts())

    # --- EDA ---
    with data_tab2:
        st.subheader("Exploratory Data Analysis")
        # 4. Call the display_eda function
        display_eda(unmasked_df)
        
        display_categorical_pairs(unmasked_df)
        
        display_feature_vs_target(unmasked_df, target_name='y')

    # --- Classification ---
    with data_tab3:
        st.subheader("Classification Models (Predicting Target Y)")
        
        if processed_data is not None:
            # Re-run models only if preprocessing was successful
            X_train, X_test, y_clf_train, y_clf_test = processed_data

            clf_models = [
                ("PLA Pocket", Perceptron, X_train, y_clf_train, X_test, y_clf_test, {'max_iter': 1000}),
                ("Logistic Regression", LogisticRegression, X_train, y_clf_train, X_test, y_clf_test, {'solver': 'liblinear'}),
                ("Softmax/LR (Multinomial)", LogisticRegression, X_train, y_clf_train, X_test, y_clf_test, {'multi_class': 'multinomial', 'solver': 'lbfgs', 'max_iter': 1000}),
                ("LDA", LDA, X_train, y_clf_train, X_test, y_clf_test, {}),
                ("QDA", QDA, X_train, y_clf_train, X_test, y_clf_test, {'reg_param': 0.1}),
                ("Naive Bayes (Gaussian)", GaussianNB, X_train, y_clf_train, X_test, y_clf_test, {}),
                ("Naive Bayes (Bernoulli)", BernoulliNB, X_train, y_clf_train, X_test, y_clf_test, {}),
                ("Decision Tree (Gini)", DecisionTreeClassifier, X_train, y_clf_train, X_test, y_clf_test, {'criterion': 'gini', 'random_state': SEED}),
                ("Decision Tree (Entropy)", DecisionTreeClassifier, X_train, y_clf_train, X_test, y_clf_test, {'criterion': 'entropy', 'random_state': SEED}),
                ("Random Forest", RandomForestClassifier, X_train, y_clf_train, X_test, y_clf_test, {'random_state': SEED}),
                ("Gradient Boosting", GradientBoostingClassifier, X_train, y_clf_train, X_test, y_clf_test, {'random_state': SEED}),
            ]
            
            clf_results_display = []

            for name, model_class, X_tr, y_tr, X_te, y_te, params in clf_models:
                
                # 5. Call the train_and_evaluate_clf function
                # train_and_evaluate_clf handles all plotting and metric display internally.
                # It returns a dictionary of results.
                results = train_and_evaluate_clf(
                    model_class, X_tr, y_tr, X_te, y_te, name, **params
                )
                
                # Extract model instance for coefficient display
                model = results['model_instance']

                # Display Feature Coefficients (Only for linear models)
                if hasattr(model, 'coef_'):
                    feature_names = X_train.columns.tolist() 
                    if model_class in [Perceptron, LogisticRegression, LDA]:
                        st.write("**Feature Coefficients**:")
                        # coef_ is 2D for binary classification
                        coef_unmasked_df = pd.DataFrame(model.coef_[0], index=feature_names, columns=['Coefficient'])
                        st.dataframe(coef_unmasked_df.sort_values(by='Coefficient', ascending=False), use_container_width=True)

                elif hasattr(model, 'feature_importances_'):
                    # Tree-based Models (Decision Tree, Random Forest, Gradient Boosting)
                    feature_names = X_train.columns.tolist()
                    st.write("**Feature Importance**:")
                    importance_df = pd.DataFrame(model.feature_importances_, index=feature_names, columns=['Importance'])
                    st.dataframe(importance_df.sort_values(by='Importance', ascending=False).head(10), use_container_width=True)

                # Append results for the summary table
                clf_results_display.append(
                    {'Model': results['model'],
                     'Test Accuracy': f"{results['accuracy']:.4f}",
                     'Test AUC': f"{results['auc']:.4f}"})
                st.markdown("---")
            
            # Display results summary in sidebar
            with st.sidebar:
                st.title("Classification Summary")
                st.dataframe(pd.DataFrame(clf_results_display), hide_index=True, use_container_width=True)

        else:
            st.error("Cannot run classification models: Preprocessing failed.")
