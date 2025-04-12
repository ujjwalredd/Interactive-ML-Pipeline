import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
import io

class DataProcessor:
    """
    Class for data preprocessing operations including handling missing values,
    outliers, encoding, and scaling.
    """
    
    def get_data_info(self, data):
        """
        Generate a summary of the dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
            
        Returns:
        --------
        str
            A string containing the data information summary
        """
        buffer = io.StringIO()
        buffer.write(f"Dataset Shape: {data.shape}\n\n")
        
        # Data types and missing values
        info_df = pd.DataFrame({
            'Data Type': data.dtypes,
            'Missing Values': data.isnull().sum(),
            'Missing (%)': round(data.isnull().sum() / len(data) * 100, 2),
            'Unique Values': data.nunique()
        })
        
        buffer.write(info_df.to_string())
        buffer.write("\n\n")
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        buffer.write(f"Duplicate rows: {duplicates} ({round(duplicates/len(data)*100, 2)}%)\n")
        
        return buffer.getvalue()
    
    def identify_feature_types(self, data, target_col):
        """
        Automatically identify categorical and numerical features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        target_col : str
            The target column name
            
        Returns:
        --------
        tuple
            (categorical_cols, numerical_cols)
        """
        categorical_cols = []
        numerical_cols = []
        
        for col in data.columns:
            if col == target_col:
                continue
                
            # Check if it's categorical
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                categorical_cols.append(col)
            # Check if it's a numerical column with few unique values (potential categorical)
            elif data[col].nunique() < 10 and data[col].nunique() / len(data) < 0.05:
                categorical_cols.append(col)
            # Otherwise, treat as numerical
            else:
                numerical_cols.append(col)
        
        return categorical_cols, numerical_cols
    
    def determine_problem_type(self, target_series):
        """
        Determine if the problem is classification or regression based on the target variable.
        
        Parameters:
        -----------
        target_series : pd.Series
            The target variable
            
        Returns:
        --------
        str
            'classification' or 'regression'
        """
        # Check if target is categorical/discrete
        if target_series.dtype == 'object' or target_series.dtype.name == 'category':
            return 'classification'
        # If it's numeric but has few unique values, it's likely classification
        elif target_series.nunique() < 10 and target_series.nunique() / len(target_series) < 0.05:
            return 'classification'
        # Otherwise, treat as regression
        else:
            return 'regression'
    
    def preprocess_data(self, data, target_col, categorical_features=None, numerical_features=None, 
                        handle_missing=True, missing_strategy='mean', handle_outliers=False, 
                        outlier_strategy='clip', scaling_method=None, encoding_method=None):
        """
        Preprocess the data with specified operations.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        target_col : str
            The target column name
        categorical_features : list
            List of categorical feature names
        numerical_features : list
            List of numerical feature names
        handle_missing : bool
            Whether to handle missing values
        missing_strategy : str
            Strategy for handling missing values
        handle_outliers : bool
            Whether to handle outliers
        outlier_strategy : str
            Strategy for handling outliers
        scaling_method : str
            Method for scaling numerical features
        encoding_method : str
            Method for encoding categorical features
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed DataFrame
        """
        # Make a copy of the data to avoid modifying the original
        processed_data = data.copy()
        
        # If feature lists not provided, identify them
        if categorical_features is None or numerical_features is None:
            categorical_features, numerical_features = self.identify_feature_types(data, target_col)
        
        # Handle missing values
        if handle_missing:
            processed_data = self._handle_missing_values(processed_data, numerical_features, 
                                                         categorical_features, missing_strategy)
        
        # Handle outliers in numerical features
        if handle_outliers:
            processed_data = self._handle_outliers(processed_data, numerical_features, outlier_strategy)
        
        # Scale numerical features
        if scaling_method:
            processed_data = self._scale_features(processed_data, numerical_features, scaling_method)
        
        # Encode categorical features
        if encoding_method and categorical_features:
            processed_data = self._encode_features(processed_data, categorical_features, encoding_method)
        
        return processed_data
    
    def _handle_missing_values(self, data, numerical_features, categorical_features, strategy):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        numerical_features : list
            List of numerical feature names
        categorical_features : list
            List of categorical feature names
        strategy : str
            Strategy for handling missing values
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with handled missing values
        """
        # Check if there are any missing values
        if not data.isnull().any().any():
            return data
            
        result_data = data.copy()
        
        if strategy == 'remove_rows':
            # Remove rows with any missing values
            return result_data.dropna()
        
        # Handle numerical features
        if numerical_features:
            if strategy in ['mean', 'median', 'mode']:
                imputer = SimpleImputer(
                    strategy='mean' if strategy == 'mean' else 
                            'median' if strategy == 'median' else 'most_frequent'
                )
                result_data[numerical_features] = imputer.fit_transform(result_data[numerical_features])
            elif strategy == 'constant':
                # Default to 0 for constant imputation
                imputer = SimpleImputer(strategy='constant', fill_value=0)
                result_data[numerical_features] = imputer.fit_transform(result_data[numerical_features])
        
        # Handle categorical features - always use most frequent value
        if categorical_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            
            # Convert to object type to avoid issues with SimpleImputer
            for col in categorical_features:
                if col in result_data.columns:
                    result_data[col] = result_data[col].astype('object')
            
            result_data[categorical_features] = cat_imputer.fit_transform(result_data[categorical_features])
        
        return result_data
    
    def _handle_outliers(self, data, numerical_features, strategy):
        """
        Handle outliers in numerical features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        numerical_features : list
            List of numerical feature names
        strategy : str
            Strategy for handling outliers
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with handled outliers
        """
        result_data = data.copy()
        
        if not numerical_features:
            return result_data
            
        if strategy == 'clip':
            # Clip values outside 1.5 * IQR
            for col in numerical_features:
                if col in result_data.columns:
                    Q1 = result_data[col].quantile(0.25)
                    Q3 = result_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    result_data[col] = result_data[col].clip(lower_bound, upper_bound)
        
        elif strategy == 'remove':
            # Identify outliers
            mask = pd.Series(False, index=result_data.index)
            
            for col in numerical_features:
                if col in result_data.columns:
                    Q1 = result_data[col].quantile(0.25)
                    Q3 = result_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Update mask for rows with outliers
                    mask = mask | ((result_data[col] < lower_bound) | (result_data[col] > upper_bound))
            
            # Remove rows with outliers
            result_data = result_data[~mask]
        
        return result_data
    
    def _scale_features(self, data, numerical_features, method):
        """
        Scale numerical features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        numerical_features : list
            List of numerical feature names
        method : str
            Scaling method
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with scaled features
        """
        result_data = data.copy()
        
        if not numerical_features:
            return result_data
            
        if method == 'StandardScaler':
            scaler = StandardScaler()
        elif method == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif method == 'RobustScaler':
            scaler = RobustScaler()
        else:
            return result_data
            
        # Scale numerical features
        result_data[numerical_features] = scaler.fit_transform(result_data[numerical_features])
        
        return result_data
    
    def _encode_features(self, data, categorical_features, method):
        """
        Encode categorical features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        categorical_features : list
            List of categorical feature names
        method : str
            Encoding method
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with encoded features
        """
        result_data = data.copy()
        
        if not categorical_features:
            return result_data
            
        if method == 'OneHotEncoder':
            # Apply one-hot encoding
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
            # Make sure all columns are in the DataFrame
            valid_categorical = [col for col in categorical_features if col in result_data.columns]
            
            if not valid_categorical:
                return result_data
                
            encoded_data = encoder.fit_transform(result_data[valid_categorical])
            
            # Create DataFrame with encoded column names
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=[f"{col}_{val}" for i, col in enumerate(valid_categorical) 
                         for val in encoder.categories_[i]]
            )
            
            # Drop original categorical columns and add encoded ones
            result_data = result_data.drop(columns=valid_categorical)
            result_data = pd.concat([result_data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
            
        elif method == 'LabelEncoder':
            # Apply label encoding
            for col in categorical_features:
                if col in result_data.columns:
                    encoder = LabelEncoder()
                    result_data[col] = encoder.fit_transform(result_data[col].astype(str))
        
        return result_data
