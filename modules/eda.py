import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class EDA:
    """
    Class for Exploratory Data Analysis and visualization functions.
    """
    
    def __init__(self):
        """Initialize the EDA class with default settings."""
        # Set the default style for plots
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 10})
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_correlation(self, data):
        """
        Create a correlation heatmap for numerical features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
            
        Returns:
        --------
        matplotlib.figure.Figure
            The correlation heatmap figure
        """
        # Select only numerical columns
        numerical_data = data.select_dtypes(include=['int64', 'float64'])
        
        # Create correlation matrix
        corr = numerical_data.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr, 
            mask=mask, 
            cmap=cmap, 
            vmax=1, 
            vmin=-1, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            annot=True, 
            fmt=".2f",
            ax=ax
        )
        
        plt.title('Feature Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def plot_target_distribution(self, data, target_col, problem_type):
        """
        Plot the distribution of the target variable.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        target_col : str
            Name of the target column
        problem_type : str
            'classification' or 'regression'
            
        Returns:
        --------
        matplotlib.figure.Figure
            The target distribution figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if problem_type == 'classification':
            # For classification, create a count plot
            target_counts = data[target_col].value_counts().sort_index()
            
            # Use bar plot for better control over appearance
            bars = ax.bar(
                target_counts.index.astype(str), 
                target_counts.values,
                color=sns.color_palette("husl", len(target_counts))
            )
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.1,
                    f'{height}',
                    ha='center', 
                    va='bottom',
                    fontsize=10
                )
            
            plt.title(f'Distribution of Target Variable: {target_col}', fontsize=14)
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            
            # Add percentage annotations
            total = sum(target_counts)
            for i, count in enumerate(target_counts):
                percentage = 100 * count / total
                ax.annotate(
                    f'{percentage:.1f}%',
                    xy=(i, count / 2),
                    ha='center',
                    va='center',
                    color='white' if percentage > 10 else 'black',
                    fontweight='bold'
                )
            
        else:  # Regression
            # For regression, create a histogram
            sns.histplot(data[target_col], kde=True, ax=ax)
            plt.title(f'Distribution of Target Variable: {target_col}', fontsize=14)
            plt.xlabel(target_col, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            
            # Add mean and median lines
            mean_val = data[target_col].mean()
            median_val = data[target_col].median()
            
            plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='g', linestyle='-.', label=f'Median: {median_val:.2f}')
            plt.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_numerical_feature(self, data, feature, target_col=None):
        """
        Plot the distribution of a numerical feature, optionally colored by target.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        feature : str
            The numerical feature to plot
        target_col : str, optional
            The target column for coloring
            
        Returns:
        --------
        matplotlib.figure.Figure
            The feature distribution figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if target_col and data[target_col].nunique() <= int(np.sqrt(len(data))):
            # If target is categorical with not too many unique values, use it for hue
            sns.histplot(
                data=data,
                x=feature,
                hue=target_col,
                kde=True,
                ax=ax,
                element="step"
            )
            plt.title(f'Distribution of {feature} by {target_col}', fontsize=14)
        else:
            # Otherwise just plot the distribution
            sns.histplot(data=data, x=feature, kde=True, ax=ax)
            
            # Add mean and median lines
            mean_val = data[feature].mean()
            median_val = data[feature].median()
            
            plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='g', linestyle='-.', label=f'Median: {median_val:.2f}')
            plt.legend()
            
            plt.title(f'Distribution of {feature}', fontsize=14)
        
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        
        # Add basic statistics as text
        stats_text = (
            f"Min: {data[feature].min():.2f}\n"
            f"Max: {data[feature].max():.2f}\n"
            f"Mean: {data[feature].mean():.2f}\n"
            f"Median: {data[feature].median():.2f}\n"
            f"Std Dev: {data[feature].std():.2f}"
        )
        
        plt.figtext(0.95, 0.6, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8),
                   horizontalalignment='right')
        
        return fig
    
    def plot_categorical_feature(self, data, feature, target_col=None):
        """
        Plot the distribution of a categorical feature, optionally split by target.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        feature : str
            The categorical feature to plot
        target_col : str, optional
            The target column for splitting
            
        Returns:
        --------
        matplotlib.figure.Figure
            The feature distribution figure
        """
        # Handle features with too many categories
        if data[feature].nunique() > 20:
            # Take the top 15 most common categories and group the rest as 'Others'
            top_categories = data[feature].value_counts().nlargest(15).index
            data_plot = data.copy()
            data_plot[feature] = data_plot[feature].apply(lambda x: x if x in top_categories else 'Others')
        else:
            data_plot = data
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if target_col and pd.api.types.is_numeric_dtype(data[target_col]) and data[target_col].nunique() > 5:
            # If target is numerical with more than 5 unique values, show mean target value per category
            aggregation = data_plot.groupby(feature)[target_col].agg(['mean', 'count'])
            aggregation = aggregation.sort_values('mean')
            
            # Plot with dual y-axis
            ax1 = ax
            ax2 = ax1.twinx()
            
            # Plot mean value as bars
            sns.barplot(x=aggregation.index, y=aggregation['mean'], ax=ax1, alpha=0.7)
            ax1.set_ylabel(f'Mean of {target_col}', color='b')
            ax1.tick_params(axis='y', colors='b')
            
            # Plot count as line
            ax2.plot(range(len(aggregation)), aggregation['count'], 'r-', marker='o')
            ax2.set_ylabel('Count', color='r')
            ax2.tick_params(axis='y', colors='r')
            
            plt.title(f'Mean {target_col} by {feature} Categories', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            
        else:
            # Standard count plot, with optional hue
            if target_col and data[target_col].nunique() <= 5:
                # Use target for splitting if it has few unique values
                order = data_plot[feature].value_counts().index
                sns.countplot(
                    data=data_plot, 
                    x=feature, 
                    hue=target_col,
                    order=order,
                    ax=ax
                )
                plt.title(f'Distribution of {feature} by {target_col}', fontsize=14)
            else:
                # Otherwise just plot the counts
                order = data_plot[feature].value_counts().index
                sns.countplot(data=data_plot, x=feature, order=order, ax=ax)
                plt.title(f'Distribution of {feature}', fontsize=14)
                
                # Add percentage labels
                total = len(data_plot)
                for i, p in enumerate(ax.patches):
                    percentage = 100 * p.get_height() / total
                    ax.annotate(
                        f'{percentage:.1f}%',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', 
                        va='bottom'
                    )
            
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def plot_missing_values(self, data):
        """
        Plot the percentage of missing values for each column.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
            
        Returns:
        --------
        matplotlib.figure.Figure
            The missing values figure
        """
        # Calculate missing values percentage
        missing_percentage = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)
        
        # Filter only columns with missing values
        missing_percentage = missing_percentage[missing_percentage > 0]
        
        if missing_percentage.empty:
            # No missing values, create an empty plot with a message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No missing values found in the dataset', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            plt.title('Missing Values Analysis', fontsize=14)
            return fig
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot horizontal bars
        bars = ax.barh(missing_percentage.index, missing_percentage, color='skyblue')
        
        # Add percentage labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width < 50 else width - 5
            label_color = 'black' if width < 50 else 'white'
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                    va='center', ha='left' if width < 50 else 'right', color=label_color)
        
        plt.title('Percentage of Missing Values by Feature', fontsize=14)
        plt.xlabel('Missing Values (%)', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self, data, target_col, problem_type):
        """
        Plot feature importance using a Random Forest model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        target_col : str
            The target column name
        problem_type : str
            'classification' or 'regression'
            
        Returns:
        --------
        matplotlib.figure.Figure
            The feature importance figure
        """
        # Prepare the data
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Select only numerical columns for simplicity
        X_numeric = X.select_dtypes(include=['int64', 'float64'])
        
        if X_numeric.shape[1] == 0:
            # No numerical features available
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No numerical features available for importance calculation', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            plt.title('Feature Importance', fontsize=14)
            return fig
        
        # Train a Random Forest model
        try:
            if problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
            model.fit(X_numeric, y)
            
            # Get feature importance
            importances = model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot horizontal bars for better readability with many features
            bars = ax.barh(
                range(len(indices)), 
                importances[indices],
                align='center',
                color=sns.color_palette("viridis", len(indices))
            )
            
            # Set tick labels
            plt.yticks(
                range(len(indices)), 
                [X_numeric.columns[i] for i in indices]
            )
            
            # Add importance percentage labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.001
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                        va='center', ha='left')
            
            plt.title('Feature Importance (Random Forest)', fontsize=14)
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            # If training fails, return a message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error computing feature importance: {str(e)}', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            plt.title('Feature Importance', fontsize=14)
            return fig
