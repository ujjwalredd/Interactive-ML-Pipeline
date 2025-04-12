import streamlit as st
import pandas as pd
import numpy as np
import time
from modules.data_processor import DataProcessor
from modules.eda import EDA
from modules.feature_engineering import FeatureEngineering
from modules.model_trainer import ModelTrainer
from modules.model_evaluation import ModelEvaluator

# Set page config
st.set_page_config(
    page_title="ML Model Development Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'categorical_features' not in st.session_state:
    st.session_state.categorical_features = []
if 'numerical_features' not in st.session_state:
    st.session_state.numerical_features = []
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}

# Main title
st.title("🤖 ML Model Development Platform")
st.write("An interactive platform for training, evaluating, and comparing machine learning models")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a step:",
    ["Data Upload", "Data Analysis", "Feature Engineering", "Model Training", "Model Evaluation"]
)

# Data Upload Section
if app_mode == "Data Upload":
    st.header("📤 Data Upload and Preprocessing")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Process the uploaded file
        processor = DataProcessor()
        
        try:
            # Show a loading message while processing
            with st.spinner("Loading and processing your data..."):
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.session_state.data = data
                st.success(f"Data loaded successfully! Shape: {data.shape}")
                
                # Display data overview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Display data info
                st.subheader("Data Information")
                buffer = processor.get_data_info(data)
                st.text(buffer)
                
                # Select target variable
                st.subheader("Select Target Variable")
                target_col = st.selectbox("Choose the target column", data.columns)
                
                if st.button("Confirm Target Selection"):
                    st.session_state.target = target_col
                    
                    # Automatically identify feature types
                    categorical_cols, numerical_cols = processor.identify_feature_types(data, target_col)
                    st.session_state.categorical_features = categorical_cols
                    st.session_state.numerical_features = numerical_cols
                    st.session_state.features = categorical_cols + numerical_cols
                    
                    # Determine problem type
                    problem_type = processor.determine_problem_type(data[target_col])
                    st.session_state.problem_type = problem_type
                    
                    st.success(f"Target selected: {target_col}")
                    st.info(f"Problem type detected: {problem_type}")
                    
                    # Display feature types
                    st.subheader("Feature Types")
                    st.write("Categorical features:", ", ".join(categorical_cols) if categorical_cols else "None")
                    st.write("Numerical features:", ", ".join(numerical_cols) if numerical_cols else "None")
                
                # Data preprocessing options if target is selected
                if st.session_state.target:
                    st.subheader("Data Preprocessing")
                    
                    # Handle missing values
                    st.write("Handle Missing Values")
                    handle_missing = st.checkbox("Handle Missing Values", value=True)
                    missing_strategy = None
                    if handle_missing:
                        missing_strategy = st.selectbox(
                            "Strategy for numerical features",
                            ["mean", "median", "mode", "remove_rows", "constant"]
                        )
                        if missing_strategy == "constant":
                            missing_const_value = st.number_input("Fill value", value=0)
                    
                    # Handle outliers
                    st.write("Handle Outliers")
                    handle_outliers = st.checkbox("Handle Outliers", value=False)
                    outlier_strategy = None
                    if handle_outliers:
                        outlier_strategy = st.selectbox(
                            "Strategy", 
                            ["clip", "remove", "none"]
                        )
                    
                    # Scaling options
                    st.write("Feature Scaling")
                    scaling_method = st.selectbox(
                        "Scaling Method", 
                        ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
                    )
                    
                    # Encoding options
                    st.write("Categorical Encoding")
                    encoding_method = st.selectbox(
                        "Encoding Method", 
                        ["OneHotEncoder", "LabelEncoder", "None"]
                    )
                    
                    # Process data button
                    if st.button("Process Data"):
                        with st.spinner("Processing data..."):
                            processed_data = processor.preprocess_data(
                                data,
                                target_col,
                                categorical_features=st.session_state.categorical_features,
                                numerical_features=st.session_state.numerical_features,
                                handle_missing=handle_missing,
                                missing_strategy=missing_strategy,
                                handle_outliers=handle_outliers,
                                outlier_strategy=outlier_strategy,
                                scaling_method=scaling_method if scaling_method != "None" else None,
                                encoding_method=encoding_method if encoding_method != "None" else None
                            )
                            st.session_state.processed_data = processed_data
                            st.success("Data processing completed!")
                            
                            # Show processed data
                            st.subheader("Processed Data Preview")
                            st.dataframe(processed_data.head())
                            
                            # Show shape
                            st.write(f"Processed data shape: {processed_data.shape}")
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

# Data Analysis Section
elif app_mode == "Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the 'Data Upload' section")
    else:
        eda = EDA()
        data = st.session_state.data
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(data.describe().T)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        corr_plot = eda.plot_correlation(data)
        st.pyplot(corr_plot)
        
        # Target distribution
        if st.session_state.target:
            st.subheader(f"Target Variable ({st.session_state.target}) Distribution")
            target_plot = eda.plot_target_distribution(data, st.session_state.target, st.session_state.problem_type)
            st.pyplot(target_plot)
        
        # Feature analysis
        st.subheader("Feature Analysis")
        st.write("Select a feature to analyze:")
        feature_to_analyze = st.selectbox("Feature", data.columns)
        
        if feature_to_analyze:
            # Detect if the feature is numerical or categorical
            is_numeric = pd.api.types.is_numeric_dtype(data[feature_to_analyze])
            
            if is_numeric:
                st.write(f"Distribution of {feature_to_analyze}")
                fig = eda.plot_numerical_feature(data, feature_to_analyze, st.session_state.target)
                st.pyplot(fig)
            else:
                st.write(f"Distribution of {feature_to_analyze}")
                fig = eda.plot_categorical_feature(data, feature_to_analyze, st.session_state.target)
                st.pyplot(fig)
        
        # Missing values analysis
        st.subheader("Missing Values Analysis")
        missing_plot = eda.plot_missing_values(data)
        st.pyplot(missing_plot)
        
        # Feature importance if processed data exists
        if st.session_state.processed_data is not None and st.session_state.target:
            st.subheader("Feature Importance")
            importance_fig = eda.plot_feature_importance(
                st.session_state.processed_data, 
                st.session_state.target,
                st.session_state.problem_type
            )
            st.pyplot(importance_fig)

# Feature Engineering Section
elif app_mode == "Feature Engineering":
    st.header("🔧 Feature Engineering")
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the 'Data Upload' section")
    else:
        fe = FeatureEngineering()
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        st.subheader("Create New Features")
        
        # Feature transformation
        st.write("Transform Features")
        transform_options = ["None", "Log Transform", "Square Root", "Square", "Cube", "Box-Cox"]
        
        col1, col2 = st.columns(2)
        with col1:
            feature_to_transform = st.selectbox("Select feature to transform", st.session_state.numerical_features)
        with col2:
            transform_type = st.selectbox("Select transformation", transform_options)
        
        if st.button("Apply Transformation") and feature_to_transform and transform_type != "None":
            try:
                transformed_data = fe.transform_feature(data, feature_to_transform, transform_type)
                st.session_state.processed_data = transformed_data
                st.success(f"Applied {transform_type} to {feature_to_transform}")
                st.dataframe(transformed_data.head())
            except Exception as e:
                st.error(f"Error applying transformation: {str(e)}")
        
        # Feature interaction
        st.write("Feature Interaction")
        col1, col2, col3 = st.columns(3)
        with col1:
            feature1 = st.selectbox("Select first feature", st.session_state.numerical_features, key="feat1")
        with col2:
            feature2 = st.selectbox("Select second feature", 
                                   [f for f in st.session_state.numerical_features if f != feature1], 
                                   key="feat2")
        with col3:
            interaction_type = st.selectbox("Type of interaction", ["Multiply", "Add", "Subtract", "Divide", "Ratio"])
        
        if st.button("Create Interaction Feature") and feature1 and feature2:
            try:
                new_data = fe.create_interaction_feature(data, feature1, feature2, interaction_type)
                st.session_state.processed_data = new_data
                new_feature_name = f"{feature1}_{interaction_type.lower()}_{feature2}"
                st.success(f"Created new feature: {new_feature_name}")
                st.dataframe(new_data.head())
                
                # Update numerical features in session state
                if new_feature_name not in st.session_state.numerical_features:
                    st.session_state.numerical_features.append(new_feature_name)
                    st.session_state.features = st.session_state.categorical_features + st.session_state.numerical_features
            except Exception as e:
                st.error(f"Error creating interaction feature: {str(e)}")
        
        # Polynomial features
        st.write("Polynomial Features")
        poly_features = st.multiselect("Select features for polynomial expansion", st.session_state.numerical_features)
        poly_degree = st.slider("Degree", min_value=2, max_value=5, value=2)
        
        if st.button("Generate Polynomial Features") and poly_features:
            try:
                poly_data = fe.create_polynomial_features(data, poly_features, poly_degree)
                st.session_state.processed_data = poly_data
                st.success(f"Generated polynomial features of degree {poly_degree}")
                st.dataframe(poly_data.head())
                
                # Update features list
                for col in poly_data.columns:
                    if col not in st.session_state.features and col != st.session_state.target:
                        st.session_state.numerical_features.append(col)
                st.session_state.features = st.session_state.categorical_features + st.session_state.numerical_features
            except Exception as e:
                st.error(f"Error generating polynomial features: {str(e)}")
        
        # Feature selection
        st.subheader("Feature Selection")
        selection_method = st.selectbox(
            "Feature selection method", 
            ["None", "SelectKBest", "Recursive Feature Elimination", "Feature Importance"]
        )
        
        if selection_method != "None":
            if selection_method == "SelectKBest":
                n_features = st.slider("Number of features to select", 
                                      min_value=1, 
                                      max_value=len(st.session_state.features), 
                                      value=min(5, len(st.session_state.features)))
                score_func = st.selectbox("Scoring function", ["f_classif", "chi2", "f_regression", "mutual_info_classif"])
                
                if st.button("Select Features"):
                    try:
                        with st.spinner("Selecting features..."):
                            X = st.session_state.processed_data.drop(columns=[st.session_state.target])
                            y = st.session_state.processed_data[st.session_state.target]
                            selected_data, selected_features = fe.select_k_best_features(
                                X, y, n_features, score_func, st.session_state.problem_type
                            )
                            
                            # Add target back to the selected data
                            selected_data[st.session_state.target] = y
                            
                            st.session_state.processed_data = selected_data
                            st.success(f"Selected {len(selected_features)} features")
                            st.write("Selected features:", ", ".join(selected_features))
                            
                            # Update features list
                            st.session_state.features = [f for f in st.session_state.features if f in selected_features]
                            st.session_state.numerical_features = [f for f in st.session_state.numerical_features if f in selected_features]
                            st.session_state.categorical_features = [f for f in st.session_state.categorical_features if f in selected_features]
                    except Exception as e:
                        st.error(f"Error selecting features: {str(e)}")
                        
            elif selection_method == "Recursive Feature Elimination":
                n_features = st.slider("Number of features to select", 
                                      min_value=1, 
                                      max_value=len(st.session_state.features), 
                                      value=min(5, len(st.session_state.features)))
                
                if st.button("Select Features"):
                    try:
                        with st.spinner("Selecting features using RFE..."):
                            X = st.session_state.processed_data.drop(columns=[st.session_state.target])
                            y = st.session_state.processed_data[st.session_state.target]
                            selected_data, selected_features = fe.rfe_select_features(
                                X, y, n_features, st.session_state.problem_type
                            )
                            
                            # Add target back to the selected data
                            selected_data[st.session_state.target] = y
                            
                            st.session_state.processed_data = selected_data
                            st.success(f"Selected {len(selected_features)} features")
                            st.write("Selected features:", ", ".join(selected_features))
                            
                            # Update features list
                            st.session_state.features = [f for f in st.session_state.features if f in selected_features]
                            st.session_state.numerical_features = [f for f in st.session_state.numerical_features if f in selected_features]
                            st.session_state.categorical_features = [f for f in st.session_state.categorical_features if f in selected_features]
                    except Exception as e:
                        st.error(f"Error selecting features with RFE: {str(e)}")
                        
            elif selection_method == "Feature Importance":
                threshold = st.slider("Importance threshold", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
                
                if st.button("Select Features"):
                    try:
                        with st.spinner("Selecting features based on importance..."):
                            X = st.session_state.processed_data.drop(columns=[st.session_state.target])
                            y = st.session_state.processed_data[st.session_state.target]
                            selected_data, selected_features, importances = fe.importance_based_selection(
                                X, y, threshold, st.session_state.problem_type
                            )
                            
                            # Add target back to the selected data
                            selected_data[st.session_state.target] = y
                            
                            st.session_state.processed_data = selected_data
                            st.success(f"Selected {len(selected_features)} features")
                            
                            # Display feature importances
                            importance_df = pd.DataFrame({
                                'Feature': selected_features,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False)
                            
                            st.dataframe(importance_df)
                            
                            # Update features list
                            st.session_state.features = [f for f in st.session_state.features if f in selected_features]
                            st.session_state.numerical_features = [f for f in st.session_state.numerical_features if f in selected_features]
                            st.session_state.categorical_features = [f for f in st.session_state.categorical_features if f in selected_features]
                    except Exception as e:
                        st.error(f"Error selecting features based on importance: {str(e)}")
        
        # PCA Dimensionality Reduction
        st.subheader("Dimensionality Reduction (PCA)")
        
        n_components = st.slider("Number of components", 
                                min_value=1, 
                                max_value=len(st.session_state.numerical_features), 
                                value=min(3, len(st.session_state.numerical_features)))
        
        if st.button("Apply PCA"):
            try:
                with st.spinner("Applying PCA..."):
                    # Only apply PCA to numerical features
                    X = st.session_state.processed_data[st.session_state.numerical_features]
                    reduced_data, explained_variance = fe.apply_pca(X, n_components)
                    
                    # Create a new DataFrame with PCA components and target
                    pca_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])
                    pca_df[st.session_state.target] = st.session_state.processed_data[st.session_state.target]
                    
                    # Keep categorical features if any
                    if st.session_state.categorical_features:
                        for cat in st.session_state.categorical_features:
                            pca_df[cat] = st.session_state.processed_data[cat]
                    
                    st.session_state.processed_data = pca_df
                    
                    # Update features list
                    new_numerical_features = [f'PC{i+1}' for i in range(n_components)]
                    st.session_state.numerical_features = new_numerical_features
                    st.session_state.features = st.session_state.categorical_features + new_numerical_features
                    
                    st.success(f"Applied PCA and reduced to {n_components} components")
                    
                    # Display explained variance
                    st.write("Explained variance ratio:", explained_variance)
                    total_var = sum(explained_variance)
                    st.write(f"Total explained variance: {total_var:.2%}")
                    
                    # Show the new DataFrame
                    st.dataframe(pca_df.head())
            except Exception as e:
                st.error(f"Error applying PCA: {str(e)}")

# Model Training Section
elif app_mode == "Model Training":
    st.header("🧠 Model Training")
    
    if st.session_state.processed_data is None:
        st.warning("Please process your data first in the 'Data Upload' or 'Feature Engineering' sections")
    else:
        trainer = ModelTrainer()
        
        st.subheader("Select Models to Train")
        
        # Problem type based model selection
        problem_type = st.session_state.problem_type
        
        if problem_type == "classification":
            models_to_train = st.multiselect(
                "Select classification models",
                ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", 
                 "Gradient Boosting", "K-Nearest Neighbors", "Naive Bayes"],
                ["Logistic Regression", "Random Forest"]
            )
        elif problem_type == "regression":
            models_to_train = st.multiselect(
                "Select regression models",
                ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree", 
                 "Random Forest", "SVR", "Gradient Boosting"],
                ["Linear Regression", "Random Forest"]
            )
        else:
            st.error("Unknown problem type. Please go back to the Data Upload section.")
            models_to_train = []
        
        # Cross-validation options
        st.subheader("Cross-Validation Settings")
        cv_folds = st.slider("Number of CV folds", min_value=2, max_value=10, value=5)
        random_state = st.number_input("Random state", value=42)
        
        # Hyperparameter tuning options
        st.subheader("Hyperparameter Tuning")
        enable_tuning = st.checkbox("Enable hyperparameter tuning", value=False)
        
        tuning_method = None
        n_iterations = None
        
        if enable_tuning:
            tuning_method = st.selectbox(
                "Tuning method", 
                ["RandomizedSearchCV", "GridSearchCV"]
            )
            
            if tuning_method == "RandomizedSearchCV":
                n_iterations = st.slider("Number of iterations", min_value=10, max_value=100, value=20)
            
            st.info("Hyperparameter tuning will increase training time but can improve model performance.")
        
        # Training button
        if st.button("Train Models"):
            if not models_to_train:
                st.warning("Please select at least one model to train")
            else:
                with st.spinner("Training models... This may take a while."):
                    progress_bar = st.progress(0)
                    trained_models = {}
                    
                    data = st.session_state.processed_data
                    X = data.drop(columns=[st.session_state.target])
                    y = data[st.session_state.target]
                    
                    for i, model_name in enumerate(models_to_train):
                        progress_bar.progress((i) / len(models_to_train))
                        st.write(f"Training {model_name}...")
                        
                        try:
                            if enable_tuning:
                                model, best_params, cv_score = trainer.train_model_with_tuning(
                                    X, y, model_name, problem_type, 
                                    cv=cv_folds, 
                                    random_state=random_state,
                                    tuning_method=tuning_method,
                                    n_iterations=n_iterations
                                )
                                trained_models[model_name] = {
                                    "model": model,
                                    "cv_score": cv_score,
                                    "best_params": best_params
                                }
                                st.write(f"✅ {model_name} trained with CV score: {cv_score:.4f}")
                                st.write(f"Best parameters: {best_params}")
                            else:
                                model, cv_score = trainer.train_model(
                                    X, y, model_name, problem_type, 
                                    cv=cv_folds, 
                                    random_state=random_state
                                )
                                trained_models[model_name] = {
                                    "model": model,
                                    "cv_score": cv_score,
                                    "best_params": None
                                }
                                st.write(f"✅ {model_name} trained with CV score: {cv_score:.4f}")
                        except Exception as e:
                            st.error(f"Error training {model_name}: {str(e)}")
                    
                    progress_bar.progress(1.0)
                    
                    # Save models to session state
                    st.session_state.models = trained_models
                    
                    # Find best model
                    if trained_models:
                        if problem_type == "classification":
                            best_model_name = max(trained_models, key=lambda x: trained_models[x]["cv_score"])
                        else:  # regression - lower error is better
                            best_model_name = min(trained_models, key=lambda x: trained_models[x]["cv_score"])
                        
                        st.session_state.best_model = best_model_name
                        st.success(f"Training completed! Best model: {best_model_name}")
                    else:
                        st.error("No models were successfully trained.")
        
        # Display trained models if available
        if st.session_state.models:
            st.subheader("Trained Models Summary")
            
            models_df = []
            for model_name, model_info in st.session_state.models.items():
                models_df.append({
                    "Model": model_name,
                    "CV Score": model_info["cv_score"],
                    "Best Parameters": str(model_info["best_params"]) if model_info["best_params"] else "N/A"
                })
            
            models_df = pd.DataFrame(models_df)
            if problem_type == "classification":
                models_df = models_df.sort_values("CV Score", ascending=False)
            else:  # regression
                models_df = models_df.sort_values("CV Score", ascending=True)
            
            st.dataframe(models_df)
            
            # Set the best model as selected if available
            if st.session_state.best_model:
                st.write(f"Best model: **{st.session_state.best_model}**")

# Model Evaluation Section
elif app_mode == "Model Evaluation":
    st.header("📏 Model Evaluation")
    
    if not st.session_state.models:
        st.warning("No trained models available. Please train models in the 'Model Training' section.")
    else:
        evaluator = ModelEvaluator()
        
        # Select model to evaluate
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox(
            "Select model to evaluate",
            model_names,
            index=model_names.index(st.session_state.best_model) if st.session_state.best_model in model_names else 0
        )
        
        # Get model from session state
        model_info = st.session_state.models[selected_model]
        model = model_info["model"]
        
        data = st.session_state.processed_data
        X = data.drop(columns=[st.session_state.target])
        y = data[st.session_state.target]
        
        # Test-train split options
        st.subheader("Evaluation Settings")
        test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random state", value=42)
        
        # Evaluation button
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model..."):
                try:
                    # Split data
                    X_train, X_test, y_train, y_test = evaluator.train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Evaluate model
                    evaluation_results = evaluator.evaluate_model(
                        model, X_train, X_test, y_train, y_test, st.session_state.problem_type
                    )
                    
                    # Store results in session state
                    st.session_state.evaluation_results[selected_model] = evaluation_results
                    
                    # Display results
                    st.subheader("Evaluation Results")
                    
                    if st.session_state.problem_type == "classification":
                        # Display classification metrics
                        metrics_df = pd.DataFrame({
                            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
                            "Train": [
                                evaluation_results["train_accuracy"],
                                evaluation_results["train_precision"],
                                evaluation_results["train_recall"],
                                evaluation_results["train_f1"],
                                evaluation_results["train_roc_auc"] if "train_roc_auc" in evaluation_results else None
                            ],
                            "Test": [
                                evaluation_results["test_accuracy"],
                                evaluation_results["test_precision"],
                                evaluation_results["test_recall"],
                                evaluation_results["test_f1"],
                                evaluation_results["test_roc_auc"] if "test_roc_auc" in evaluation_results else None
                            ]
                        })
                        st.dataframe(metrics_df)
                        
                        # Plot confusion matrix
                        st.subheader("Confusion Matrix")
                        cm_fig = evaluator.plot_confusion_matrix(evaluation_results["confusion_matrix"])
                        st.pyplot(cm_fig)
                        
                        # Plot ROC curve if available
                        if "roc_curve" in evaluation_results:
                            st.subheader("ROC Curve")
                            roc_fig = evaluator.plot_roc_curve(
                                evaluation_results["roc_curve"][0],
                                evaluation_results["roc_curve"][1],
                                evaluation_results["test_roc_auc"]
                            )
                            st.pyplot(roc_fig)
                        
                        # Plot precision-recall curve if available
                        if "precision_recall_curve" in evaluation_results:
                            st.subheader("Precision-Recall Curve")
                            pr_fig = evaluator.plot_precision_recall_curve(
                                evaluation_results["precision_recall_curve"][0],
                                evaluation_results["precision_recall_curve"][1],
                                evaluation_results["test_average_precision"]
                            )
                            st.pyplot(pr_fig)
                            
                        # Classification report
                        st.subheader("Classification Report")
                        st.text(evaluation_results["classification_report"])
                    
                    else:  # regression
                        # Display regression metrics
                        metrics_df = pd.DataFrame({
                            "Metric": ["MAE", "MSE", "RMSE", "R²"],
                            "Train": [
                                evaluation_results["train_mae"],
                                evaluation_results["train_mse"],
                                evaluation_results["train_rmse"],
                                evaluation_results["train_r2"]
                            ],
                            "Test": [
                                evaluation_results["test_mae"],
                                evaluation_results["test_mse"],
                                evaluation_results["test_rmse"],
                                evaluation_results["test_r2"]
                            ]
                        })
                        st.dataframe(metrics_df)
                        
                        # Plot actual vs predicted
                        st.subheader("Actual vs Predicted")
                        pred_fig = evaluator.plot_regression_predictions(
                            y_test, evaluation_results["test_predictions"]
                        )
                        st.pyplot(pred_fig)
                        
                        # Plot residuals
                        st.subheader("Residuals Plot")
                        resid_fig = evaluator.plot_residuals(
                            y_test, evaluation_results["test_predictions"]
                        )
                        st.pyplot(resid_fig)
                    
                    # Feature importance if available
                    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
                        st.subheader("Feature Importance")
                        importance_fig = evaluator.plot_feature_importance(model, X.columns)
                        st.pyplot(importance_fig)
                    
                    # Learning curve
                    st.subheader("Learning Curve")
                    learning_curve_fig = evaluator.plot_learning_curve(
                        model, X, y, st.session_state.problem_type, cv=5
                    )
                    st.pyplot(learning_curve_fig)
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
        
        # Compare models if multiple evaluations are available
        if len(st.session_state.evaluation_results) > 1:
            st.subheader("Model Comparison")
            
            comparison_df = []
            
            if st.session_state.problem_type == "classification":
                for model_name, results in st.session_state.evaluation_results.items():
                    comparison_df.append({
                        "Model": model_name,
                        "Accuracy": results["test_accuracy"],
                        "Precision": results["test_precision"],
                        "Recall": results["test_recall"],
                        "F1 Score": results["test_f1"],
                        "ROC AUC": results["test_roc_auc"] if "test_roc_auc" in results else None
                    })
                
                comparison_df = pd.DataFrame(comparison_df).sort_values("F1 Score", ascending=False)
            
            else:  # regression
                for model_name, results in st.session_state.evaluation_results.items():
                    comparison_df.append({
                        "Model": model_name,
                        "MAE": results["test_mae"],
                        "MSE": results["test_mse"],
                        "RMSE": results["test_rmse"],
                        "R²": results["test_r2"]
                    })
                
                comparison_df = pd.DataFrame(comparison_df).sort_values("RMSE")
            
            st.dataframe(comparison_df)
            
            # Plot comparison bar chart
            st.subheader("Model Performance Comparison")
            
            if st.session_state.problem_type == "classification":
                metric_to_plot = st.selectbox(
                    "Select metric to compare",
                    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
                )
            else:  # regression
                metric_to_plot = st.selectbox(
                    "Select metric to compare",
                    ["MAE", "MSE", "RMSE", "R²"]
                )
            
            comp_fig = evaluator.plot_model_comparison(comparison_df, "Model", metric_to_plot)
            st.pyplot(comp_fig)
        
        # Export options
        st.subheader("Export Results")
        
        export_options = st.multiselect(
            "Select what to export",
            ["Trained Model", "Evaluation Results", "Preprocessed Data"],
            ["Evaluation Results"]
        )
        
        if st.button("Export Selected Items"):
            if "Trained Model" in export_options:
                # Export model as pickle
                try:
                    model_pickle = evaluator.export_model(model)
                    st.download_button(
                        label="Download Model (pickle)",
                        data=model_pickle,
                        file_name=f"{selected_model.replace(' ', '_').lower()}_model.pkl",
                        mime="application/octet-stream"
                    )
                except Exception as e:
                    st.error(f"Error exporting model: {str(e)}")
            
            if "Evaluation Results" in export_options and selected_model in st.session_state.evaluation_results:
                # Export evaluation results as JSON
                try:
                    results_json = evaluator.export_results(st.session_state.evaluation_results[selected_model])
                    st.download_button(
                        label="Download Evaluation Results (JSON)",
                        data=results_json,
                        file_name=f"{selected_model.replace(' ', '_').lower()}_evaluation.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error exporting results: {str(e)}")
            
            if "Preprocessed Data" in export_options:
                # Export preprocessed data as CSV
                try:
                    csv_data = st.session_state.processed_data.to_csv(index=False)
                    st.download_button(
                        label="Download Preprocessed Data (CSV)",
                        data=csv_data,
                        file_name="preprocessed_data.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error exporting data: {str(e)}")
