import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle
import json
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)

class ModelEvaluator:
    """
    Class for evaluating machine learning models with various metrics and visualizations.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        # Set the default style for plots
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 10})
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Target variable
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        random_state : int, default=42
            Random state for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, problem_type):
        """
        Evaluate a trained model on training and test data.
        
        Parameters:
        -----------
        model : object
            Trained model
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        y_train : pd.Series
            Training target
        y_test : pd.Series
            Test target
        problem_type : str
            'classification' or 'regression'
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        results = {}
        
        # Evaluate based on problem type
        if problem_type == 'classification':
            # For classification problems
            
            # Get prediction probabilities if available
            if hasattr(model, "predict_proba"):
                y_train_proba = model.predict_proba(X_train)
                y_test_proba = model.predict_proba(X_test)
                
                # For binary classification, get the probability of the positive class
                if y_train_proba.shape[1] == 2:
                    y_train_proba = y_train_proba[:, 1]
                    y_test_proba = y_test_proba[:, 1]
            else:
                y_train_proba = None
                y_test_proba = None
            
            # Basic metrics
            results['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            results['test_accuracy'] = accuracy_score(y_test, y_test_pred)
            
            # If binary or multiclass with average
            results['train_precision'] = precision_score(y_train, y_train_pred, average='weighted')
            results['test_precision'] = precision_score(y_test, y_test_pred, average='weighted')
            
            results['train_recall'] = recall_score(y_train, y_train_pred, average='weighted')
            results['test_recall'] = recall_score(y_test, y_test_pred, average='weighted')
            
            results['train_f1'] = f1_score(y_train, y_train_pred, average='weighted')
            results['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
            
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_test, y_test_pred)
            
            # Classification report
            results['classification_report'] = classification_report(y_test, y_test_pred)
            
            # ROC curve and AUC (for binary classification)
            if y_train_proba is not None and len(np.unique(y_train)) == 2:
                # Only calculate AUC and ROC curve for binary classification with probability predictions
                fpr, tpr, _ = roc_curve(y_test, y_test_proba)
                results['roc_curve'] = (fpr, tpr)
                results['test_roc_auc'] = auc(fpr, tpr)
                
                # Also calculate for training set
                train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
                results['train_roc_auc'] = auc(train_fpr, train_tpr)
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
                results['precision_recall_curve'] = (precision, recall)
                results['test_average_precision'] = average_precision_score(y_test, y_test_proba)
                
                train_precision, train_recall, _ = precision_recall_curve(y_train, y_train_proba)
                results['train_average_precision'] = average_precision_score(y_train, y_train_proba)
            
        else:  # regression
            # For regression problems
            results['train_mae'] = mean_absolute_error(y_train, y_train_pred)
            results['test_mae'] = mean_absolute_error(y_test, y_test_pred)
            
            results['train_mse'] = mean_squared_error(y_train, y_train_pred)
            results['test_mse'] = mean_squared_error(y_test, y_test_pred)
            
            results['train_rmse'] = np.sqrt(results['train_mse'])
            results['test_rmse'] = np.sqrt(results['test_mse'])
            
            results['train_r2'] = r2_score(y_train, y_train_pred)
            results['test_r2'] = r2_score(y_test, y_test_pred)
            
            # Store predictions for plotting
            results['train_predictions'] = y_train_pred
            results['test_predictions'] = y_test_pred
        
        return results
    
    def plot_confusion_matrix(self, cm):
        """
        Plot a confusion matrix.
        
        Parameters:
        -----------
        cm : numpy.ndarray
            Confusion matrix from sklearn.metrics.confusion_matrix
            
        Returns:
        --------
        matplotlib.figure.Figure
            The confusion matrix heatmap
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot the confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            cbar=False,
            ax=ax
        )
        
        # Labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """
        Plot ROC curve.
        
        Parameters:
        -----------
        fpr : array
            False positive rate
        tpr : array
            True positive rate
        roc_auc : float
            Area under the ROC curve
            
        Returns:
        --------
        matplotlib.figure.Figure
            The ROC curve plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Labels, title and legend
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, precision, recall, avg_precision):
        """
        Plot precision-recall curve.
        
        Parameters:
        -----------
        precision : array
            Precision values
        recall : array
            Recall values
        avg_precision : float
            Average precision score
            
        Returns:
        --------
        matplotlib.figure.Figure
            The precision-recall curve plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot precision-recall curve
        ax.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        
        # Labels, title and legend
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        
        plt.tight_layout()
        return fig
    
    def plot_regression_predictions(self, y_true, y_pred):
        """
        Plot actual vs predicted values for regression.
        
        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted target values
            
        Returns:
        --------
        matplotlib.figure.Figure
            Scatter plot of actual vs predicted values
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot actual vs predicted
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        range_min = min(min(y_true), min(y_pred))
        range_max = max(max(y_true), max(y_pred))
        ax.plot([range_min, range_max], [range_min, range_max], 'r--')
        
        # Labels and title
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Values')
        
        # Add R² value to the plot
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, y_true, y_pred):
        """
        Plot residuals for regression models.
        
        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted target values
            
        Returns:
        --------
        matplotlib.figure.Figure
            Residual plots
        """
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs Predicted plot
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        
        # Residuals distribution
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, model, feature_names):
        """
        Plot feature importance for tree-based models or coefficients for linear models.
        
        Parameters:
        -----------
        model : object
            Trained model
        feature_names : array-like
            Names of features
            
        Returns:
        --------
        matplotlib.figure.Figure
            Feature importance plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Check if the model has feature_importances_ attribute (tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_type = "Feature Importance"
        # Check if the model has coef_ attribute (linear models)
        elif hasattr(model, 'coef_'):
            importances = model.coef_
            # If it's a 2D array (multiclass), take the mean across classes
            if importances.ndim > 1:
                importances = np.abs(importances).mean(axis=0)
            else:
                importances = np.abs(importances)
            importance_type = "Coefficient Magnitude"
        else:
            # Return empty plot if model has no interpretable feature importance
            ax.text(0.5, 0.5, "Model does not provide feature importance information", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title("Feature Importance Not Available")
            return fig
        
        # Sort importances
        indices = np.argsort(importances)[::-1]
        
        # Plot horizontal bars
        bars = ax.barh(range(len(indices)), importances[indices], align='center')
        
        # Add feature names as y-tick labels
        feature_names = np.array(feature_names)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(feature_names[indices])
        
        # Add value labels to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center')
        
        ax.set_title(f'{importance_type}')
        ax.set_xlabel('Importance')
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curve(self, model, X, y, problem_type, cv=5):
        """
        Plot learning curve to diagnose overfitting/underfitting.
        
        Parameters:
        -----------
        model : object
            Trained model
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Target variable
        problem_type : str
            'classification' or 'regression'
        cv : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        matplotlib.figure.Figure
            Learning curve plot
        """
        # Set scoring based on problem type
        if problem_type == 'classification':
            scoring = 'accuracy'
        else:  # regression
            scoring = 'neg_mean_squared_error'
        
        # Calculate learning curve values
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, scoring=scoring, 
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, 
                label='Training score')
        ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        
        ax.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, 
                label='Validation score')
        ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        
        # If regression, convert negative MSE to RMSE
        if problem_type == 'regression':
            # Convert negative MSE to RMSE for better readability
            ax.set_ylim(top=0)  # For negative MSE, lower is better
            ax.set_ylabel('Root Mean Squared Error')
            # Invert y-axis to show lower error at top
            ax.invert_yaxis()
        else:
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('Score')
        
        ax.set_xlabel('Training Examples')
        ax.set_title('Learning Curve')
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, comparison_df, model_col, metric_col):
        """
        Plot bar chart comparing models on a specific metric.
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            DataFrame with model comparison data
        model_col : str
            Column name for model names
        metric_col : str
            Column name for metric to compare
            
        Returns:
        --------
        matplotlib.figure.Figure
            Bar chart comparing models
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by the metric
        df_sorted = comparison_df.sort_values(metric_col)
        
        # Plot horizontal bars for better readability with many models
        bars = ax.barh(df_sorted[model_col], df_sorted[metric_col], color=sns.color_palette("viridis", len(df_sorted)))
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + width * 0.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', va='center')
        
        ax.set_xlabel(metric_col)
        ax.set_title(f'Model Comparison by {metric_col}')
        
        plt.tight_layout()
        return fig
    
    def export_model(self, model):
        """
        Export model to a pickle file.
        
        Parameters:
        -----------
        model : object
            Trained model to export
            
        Returns:
        --------
        bytes
            Pickled model as bytes
        """
        return pickle.dumps(model)
    
    def export_results(self, results):
        """
        Export evaluation results to JSON.
        
        Parameters:
        -----------
        results : dict
            Dictionary of evaluation results
            
        Returns:
        --------
        str
            JSON string of results
        """
        # Create a copy of results to avoid modifying the original
        export_results = {}
        
        # Convert numpy arrays to lists and remove non-serializable objects
        for key, value in results.items():
            # Skip classification report which is a string
            if key == 'classification_report':
                export_results[key] = value
            # Handle numpy arrays
            elif isinstance(value, np.ndarray):
                export_results[key] = value.tolist()
            # Handle tuples of numpy arrays (e.g., ROC curve)
            elif isinstance(value, tuple) and all(isinstance(i, np.ndarray) for i in value):
                export_results[key] = [i.tolist() for i in value]
            # Handle basic types that are JSON serializable
            elif isinstance(value, (int, float, str, bool)) or value is None:
                export_results[key] = value
        
        return json.dumps(export_results, indent=2)
