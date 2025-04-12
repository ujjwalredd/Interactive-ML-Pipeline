import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

class FeatureEngineering:
    """
    Class for feature engineering operations including feature creation,
    transformation, and selection.
    """
    
    def transform_feature(self, data, feature, transform_type):
        """
        Apply transformation to a numerical feature.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        feature : str
            The feature to transform
        transform_type : str
            Type of transformation to apply
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with the transformed feature added
        """
        # Make a copy to avoid modifying the original data
        result_data = data.copy()
        
        # Ensure the feature exists and is numerical
        if feature not in result_data.columns:
            raise ValueError(f"Feature '{feature}' not found in the dataset")
            
        if not pd.api.types.is_numeric_dtype(result_data[feature]):
            raise ValueError(f"Feature '{feature}' is not numerical")
        
        # Apply the transformation
        if transform_type == "Log Transform":
            # Handle zero or negative values by adding small constant if needed
            min_val = result_data[feature].min()
            shift = 0 if min_val > 0 else abs(min_val) + 1e-6
            result_data[f"{feature}_log"] = np.log(result_data[feature] + shift)
            
        elif transform_type == "Square Root":
            min_val = result_data[feature].min()
            shift = 0 if min_val >= 0 else abs(min_val) + 1e-6
            result_data[f"{feature}_sqrt"] = np.sqrt(result_data[feature] + shift)
            
        elif transform_type == "Square":
            result_data[f"{feature}_squared"] = result_data[feature] ** 2
            
        elif transform_type == "Cube":
            result_data[f"{feature}_cubed"] = result_data[feature] ** 3
            
        elif transform_type == "Box-Cox":
            # Box-Cox requires positive values
            min_val = result_data[feature].min()
            shift = 0 if min_val > 0 else abs(min_val) + 1e-6
            transformed_data, _ = stats.boxcox(result_data[feature] + shift)
            result_data[f"{feature}_boxcox"] = transformed_data
            
        else:
            # If transform_type is not recognized, return the original data
            return result_data
        
        return result_data
    
    def create_interaction_feature(self, data, feature1, feature2, interaction_type):
        """
        Create interaction feature between two features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        feature1 : str
            First feature name
        feature2 : str
            Second feature name
        interaction_type : str
            Type of interaction ('Multiply', 'Add', 'Subtract', 'Divide', 'Ratio')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with the interaction feature added
        """
        # Make a copy to avoid modifying the original data
        result_data = data.copy()
        
        # Check if features exist
        for feat in [feature1, feature2]:
            if feat not in result_data.columns:
                raise ValueError(f"Feature '{feat}' not found in the dataset")
                
            if not pd.api.types.is_numeric_dtype(result_data[feat]):
                raise ValueError(f"Feature '{feat}' is not numerical")
        
        # Create the interaction feature
        if interaction_type == "Multiply":
            result_data[f"{feature1}_multiply_{feature2}"] = result_data[feature1] * result_data[feature2]
            
        elif interaction_type == "Add":
            result_data[f"{feature1}_add_{feature2}"] = result_data[feature1] + result_data[feature2]
            
        elif interaction_type == "Subtract":
            result_data[f"{feature1}_subtract_{feature2}"] = result_data[feature1] - result_data[feature2]
            
        elif interaction_type == "Divide":
            # Handle division by zero
            denominator = result_data[feature2].copy()
            # Replace zeros with a small value to avoid division by zero
            denominator = denominator.replace(0, 1e-10)
            result_data[f"{feature1}_divide_{feature2}"] = result_data[feature1] / denominator
            
        elif interaction_type == "Ratio":
            # Similar to divide but different naming
            denominator = result_data[feature2].copy()
            numerator = result_data[feature1].copy()
            
            # Ensure both are positive for ratio
            min_denom = denominator.min()
            min_num = numerator.min()
            
            if min_denom <= 0:
                denominator = denominator - min_denom + 1e-6
            if min_num <= 0:
                numerator = numerator - min_num + 1e-6
                
            result_data[f"{feature1}_ratio_{feature2}"] = numerator / denominator
        
        return result_data
    
    def create_polynomial_features(self, data, features, degree=2):
        """
        Create polynomial features from selected numerical features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        features : list
            List of feature names to use for polynomial features
        degree : int, default=2
            The degree of the polynomial features
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with polynomial features added
        """
        # Make a copy to avoid modifying the original data
        result_data = data.copy()
        
        # Check if features exist and are numerical
        for feat in features:
            if feat not in result_data.columns:
                raise ValueError(f"Feature '{feat}' not found in the dataset")
                
            if not pd.api.types.is_numeric_dtype(result_data[feat]):
                raise ValueError(f"Feature '{feat}' is not numerical")
        
        # Extract the selected features
        X_poly = result_data[features]
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(X_poly)
        
        # Get feature names
        feature_names = poly.get_feature_names_out(features)
        
        # Add polynomial features to the result dataframe
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=result_data.index)
        
        # Remove the original features from poly_df to avoid duplication
        original_features_in_poly = [col for col in poly_df.columns if col in features]
        poly_df = poly_df.drop(columns=original_features_in_poly)
        
        # Concatenate with the original data
        result_data = pd.concat([result_data, poly_df], axis=1)
        
        return result_data
    
    def select_k_best_features(self, X, y, k, score_func_name, problem_type):
        """
        Select k best features using statistical tests.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Target variable
        k : int
            Number of top features to select
        score_func_name : str
            Name of the scoring function to use
        problem_type : str
            'classification' or 'regression'
            
        Returns:
        --------
        tuple
            (X_new, selected_features)
            X_new is DataFrame with only selected features
            selected_features is list of selected feature names
        """
        # Set score function based on the name and problem type
        if problem_type == 'classification':
            if score_func_name == 'f_classif':
                score_func = f_classif
            elif score_func_name == 'chi2':
                # chi2 requires non-negative features
                if (X < 0).any().any():
                    raise ValueError("Chi-square test requires non-negative features")
                score_func = chi2
            elif score_func_name == 'mutual_info_classif':
                score_func = mutual_info_classif
            else:
                score_func = f_classif  # default
        else:  # regression
            if score_func_name == 'f_regression':
                score_func = f_regression
            elif score_func_name == 'mutual_info_regression':
                score_func = mutual_info_regression
            else:
                score_func = f_regression  # default
        
        # Apply feature selection
        selector = SelectKBest(score_func=score_func, k=k)
        X_new = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()]
        
        # Create a new DataFrame with only the selected features
        X_selected = X[selected_features]
        
        return X_selected, selected_features.tolist()
    
    def rfe_select_features(self, X, y, n_features, problem_type):
        """
        Select features using Recursive Feature Elimination.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Target variable
        n_features : int
            Number of features to select
        problem_type : str
            'classification' or 'regression'
            
        Returns:
        --------
        tuple
            (X_new, selected_features)
            X_new is DataFrame with only selected features
            selected_features is list of selected feature names
        """
        # Select the appropriate model based on problem type
        if problem_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # regression
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Apply RFE
        selector = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
        selector = selector.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.support_]
        
        # Create a new DataFrame with only the selected features
        X_selected = X[selected_features]
        
        return X_selected, selected_features.tolist()
    
    def importance_based_selection(self, X, y, threshold, problem_type):
        """
        Select features based on feature importance from Random Forest.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Target variable
        threshold : float
            Minimum importance threshold for selecting features
        problem_type : str
            'classification' or 'regression'
            
        Returns:
        --------
        tuple
            (X_new, selected_features, importances)
            X_new is DataFrame with only selected features
            selected_features is list of selected feature names
            importances is numpy array of importance values for selected features
        """
        # Select the appropriate model based on problem type
        if problem_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # regression
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Fit the model and get feature importances
        model.fit(X, y)
        importances = model.feature_importances_
        
        # Select features with importance above threshold
        mask = importances > threshold
        selected_features = X.columns[mask]
        selected_importances = importances[mask]
        
        # If no features meet the threshold, select at least one feature
        if len(selected_features) == 0:
            idx = np.argmax(importances)
            selected_features = [X.columns[idx]]
            selected_importances = [importances[idx]]
        
        # Create a new DataFrame with only the selected features
        X_selected = X[selected_features]
        
        return X_selected, selected_features.tolist(), selected_importances
    
    def apply_pca(self, X, n_components):
        """
        Apply Principal Component Analysis for dimensionality reduction.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data (numerical only)
        n_components : int
            Number of principal components to keep
            
        Returns:
        --------
        tuple
            (pca_result, explained_variance_ratio)
            pca_result is numpy array of transformed data
            explained_variance_ratio is numpy array of variance explained by each component
        """
        from sklearn.decomposition import PCA
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X)
        
        return pca_result, pca.explained_variance_ratio_
