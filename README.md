# ML Model Development Platform

An interactive machine learning model development platform for training, evaluating, and comparing machine learning models. This platform is designed to help data scientists and machine learning enthusiasts streamline their workflow from data preprocessing to model evaluation.

## Features

- **Data Analysis Dashboard**: Upload and analyze datasets with interactive visualizations
- **Exploratory Data Analysis (EDA)**: Automatically generate insightful visualizations of your data
- **Feature Engineering**: Create new features, transform existing ones, and select the most relevant features
- **ML Model Training**: Train various classification and regression models with a simple interface
- **Hyperparameter Tuning**: Optimize model parameters to improve performance
- **Model Evaluation**: Comprehensive evaluation metrics and visualizations to understand model performance
- **Model Comparison**: Compare multiple models to select the best one for your task
- **Results Export**: Export your trained models, evaluation results, and processed data

## Tech Stack

- **Streamlit**: Web interface for the application
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and utilities
- **Matplotlib & Seaborn**: Data visualization
- **NumPy**: Numerical operations

## Supported ML Algorithms

### Classification
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- K-Nearest Neighbors
- Naive Bayes

### Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Support Vector Regression (SVR)
- Gradient Boosting

## Workflow

1. **Data Upload**: Upload your dataset (CSV or Excel)
2. **Data Analysis**: Explore your data through visualizations and statistics
3. **Feature Engineering**: Create, transform, and select features
4. **Model Training**: Train machine learning models with optional hyperparameter tuning
5. **Model Evaluation**: Evaluate model performance with various metrics and visualizations
6. **Export Results**: Export your trained model, evaluation results, and processed data

## How to Run

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/interactive-ml-pipeline.git
cd interactive-ml-pipeline
```

2. Create and activate a virtual environment (optional but recommended):
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

### Running the Application

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. If port 5000 is already in use, specify a different port:
```bash
streamlit run app.py --server.port 8501
```

3. The application will open in your web browser. If it doesn't open automatically, you can access it at:
   - http://localhost:8501 (or whatever port you specified)

## Usage Guide

1. **Data Upload**:
   - Upload your CSV or Excel file
   - Select your target variable
   - Review automated data type detection
   - Apply preprocessing options as needed

2. **Data Analysis**:
   - Explore summary statistics
   - View correlation heatmaps
   - Analyze target distribution
   - Examine feature distributions
   - Check missing values

3. **Feature Engineering**:
   - Transform numerical features (log, square root, etc.)
   - Create interaction features
   - Generate polynomial features
   - Select the most important features

4. **Model Training**:
   - Choose from multiple algorithms
   - Split data into training and testing sets
   - Perform hyperparameter tuning
   - Train models with cross-validation

5. **Model Evaluation**:
   - Compare performance metrics
   - Visualize confusion matrices (for classification)
   - Analyze ROC and Precision-Recall curves
   - View feature importance
   - Examine residual plots (for regression)

## Troubleshooting

- **Port conflicts**: If you see "Port XXXX is already in use", specify a different port as shown above.
- **Missing dependencies**: Ensure all required packages are installed.
- **Memory issues**: For large datasets, consider sampling or using more efficient feature selection techniques.

## Contributing

Contributions to improve the platform are welcome! Please feel free to submit a Pull Request.
