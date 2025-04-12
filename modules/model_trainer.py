import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class ModelTrainer:
    """
    Class for training machine learning models and hyperparameter tuning.
    """
    
    def __init__(self):
        """Initialize model dictionaries for classification and regression."""
        # Define model constructors and default parameters for classification
        self.classification_models = {
            "Logistic Regression": {
                "model": LogisticRegression,
                "params": {"C": 1.0, "solver": "liblinear", "max_iter": 1000},
                "tuning_params": {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "solver": ["liblinear", "saga"],
                    "penalty": ["l1", "l2"]
                }
            },
            "Decision Tree": {
                "model": DecisionTreeClassifier,
                "params": {"max_depth": 10, "min_samples_split": 2},
                "tuning_params": {
                    "max_depth": [3, 5, 7, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier,
                "params": {"n_estimators": 100, "max_depth": 10},
                "tuning_params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "bootstrap": [True, False]
                }
            },
            "SVM": {
                "model": SVC,
                "params": {"C": 1.0, "kernel": "rbf", "probability": True},
                "tuning_params": {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "gamma": ["scale", "auto", 0.1, 0.01]
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier,
                "params": {"n_estimators": 100, "learning_rate": 0.1},
                "tuning_params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "subsample": [0.8, 0.9, 1.0]
                }
            },
            "K-Nearest Neighbors": {
                "model": KNeighborsClassifier,
                "params": {"n_neighbors": 5},
                "tuning_params": {
                    "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "p": [1, 2]
                }
            },
            "Naive Bayes": {
                "model": GaussianNB,
                "params": {},
                "tuning_params": {
                    "var_smoothing": np.logspace(0, -9, 10)
                }
            }
        }
        
        # Define model constructors and default parameters for regression
        self.regression_models = {
            "Linear Regression": {
                "model": LinearRegression,
                "params": {},
                "tuning_params": {
                    "fit_intercept": [True, False],
                    "normalize": [True, False]
                }
            },
            "Ridge Regression": {
                "model": Ridge,
                "params": {"alpha": 1.0},
                "tuning_params": {
                    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
                }
            },
            "Lasso Regression": {
                "model": Lasso,
                "params": {"alpha": 1.0},
                "tuning_params": {
                    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                    "selection": ["cyclic", "random"]
                }
            },
            "Decision Tree": {
                "model": DecisionTreeRegressor,
                "params": {"max_depth": 10, "min_samples_split": 2},
                "tuning_params": {
                    "max_depth": [3, 5, 7, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["mse", "friedman_mse", "mae"]
                }
            },
            "Random Forest": {
                "model": RandomForestRegressor,
                "params": {"n_estimators": 100, "max_depth": 10},
                "tuning_params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "bootstrap": [True, False]
                }
            },
            "SVR": {
                "model": SVR,
                "params": {"C": 1.0, "kernel": "rbf"},
                "tuning_params": {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "gamma": ["scale", "auto", 0.1, 0.01],
                    "epsilon": [0.05, 0.1, 0.2, 0.5]
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor,
                "params": {"n_estimators": 100, "learning_rate": 0.1},
                "tuning_params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "subsample": [0.8, 0.9, 1.0]
                }
            }
        }
    
    def train_model(self, X, y, model_name, problem_type, cv=5, random_state=42):
        """
        Train a model with default parameters.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Target variable
        model_name : str
            Name of the model to train
        problem_type : str
            'classification' or 'regression'
        cv : int, default=5
            Number of cross-validation folds
        random_state : int, default=42
            Random state for reproducibility
            
        Returns:
        --------
        tuple
            (model, cv_score)
            model is the trained model
            cv_score is the cross-validation score (accuracy for classification, negative MSE for regression)
        """
        # Select the appropriate model config based on problem type
        if problem_type == 'classification':
            models_dict = self.classification_models
            scoring = 'accuracy'
        else:  # regression
            models_dict = self.regression_models
            scoring = 'neg_mean_squared_error'
        
        # Check if model exists
        if model_name not in models_dict:
            raise ValueError(f"Model '{model_name}' not found for {problem_type} problems")
        
        # Get model class and params
        model_config = models_dict[model_name]
        model_class = model_config["model"]
        model_params = model_config["params"].copy()
        
        # Add random_state to params if the model accepts it
        if hasattr(model_class(), "random_state"):
            model_params["random_state"] = random_state
        
        # Instantiate and train the model
        model = model_class(**model_params)
        model.fit(X, y)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        cv_score = cv_scores.mean()
        
        return model, cv_score
    
    def train_model_with_tuning(self, X, y, model_name, problem_type, cv=5, random_state=42, 
                               tuning_method='RandomizedSearchCV', n_iterations=20):
        """
        Train a model with hyperparameter tuning.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Target variable
        model_name : str
            Name of the model to train
        problem_type : str
            'classification' or 'regression'
        cv : int, default=5
            Number of cross-validation folds
        random_state : int, default=42
            Random state for reproducibility
        tuning_method : str, default='RandomizedSearchCV'
            Method for hyperparameter tuning ('RandomizedSearchCV' or 'GridSearchCV')
        n_iterations : int, default=20
            Number of iterations for RandomizedSearchCV
            
        Returns:
        --------
        tuple
            (model, best_params, cv_score)
            model is the trained model with best parameters
            best_params is a dictionary of the best parameters
            cv_score is the cross-validation score for the best parameters
        """
        # Select the appropriate model config based on problem type
        if problem_type == 'classification':
            models_dict = self.classification_models
            scoring = 'accuracy'
        else:  # regression
            models_dict = self.regression_models
            scoring = 'neg_mean_squared_error'
        
        # Check if model exists
        if model_name not in models_dict:
            raise ValueError(f"Model '{model_name}' not found for {problem_type} problems")
        
        # Get model class and params
        model_config = models_dict[model_name]
        model_class = model_config["model"]
        base_params = model_config["params"].copy()
        tuning_params = model_config["tuning_params"]
        
        # Add random_state to params if the model accepts it
        if hasattr(model_class(), "random_state"):
            base_params["random_state"] = random_state
        
        # Instantiate the base model
        base_model = model_class(**base_params)
        
        # Set up hyperparameter tuning
        if tuning_method == 'RandomizedSearchCV':
            search = RandomizedSearchCV(
                base_model,
                param_distributions=tuning_params,
                n_iter=n_iterations,
                cv=cv,
                scoring=scoring,
                random_state=random_state,
                n_jobs=-1
            )
        else:  # GridSearchCV
            search = GridSearchCV(
                base_model,
                param_grid=tuning_params,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )
        
        # Perform hyperparameter tuning
        search.fit(X, y)
        
        # Get best model and parameters
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        return best_model, best_params, best_score
