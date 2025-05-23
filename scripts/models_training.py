'''This script is used to train and evaluate different classification machine learning models on the supply chain dataset.'''

# Standard Libraries
import os

# Data Handling & Computation
import pandas as pd
import numpy as np
import copy

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

# ==============================
# Directory Setup
# ==============================

# Define the directory name for saving images
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../images")

# Check if the directory exists, if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================
# Plot Styling & Customization
# ==============================

# Set a Minimalist Style
sns.set_style("whitegrid")

# Customize Matplotlib settings for a modern look
mpl.rcParams.update({
    'axes.edgecolor': 'grey',       
    'axes.labelcolor': 'black',     
    'xtick.color': 'black',         
    'ytick.color': 'black',         
    'text.color': 'black'           
})

# General color palette for plots
custom_colors = ["#8F2C78", "#1F4E79"]

# Colors for late and not late orders
non_risk_color = "#8F2C78" # Purple
risk_color = "#1F4E79" # Blue

# Define a custom colormap from light to dark shades of purple
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "custom_purple", ["#F5A7C4", "#8F2C78", "#5C0E2F"]
)

# ==============================
# Font Configuration
# ==============================

# Path to the custom font file
FONT_PATH = '../../scripts/fonts/Montserrat-Regular.ttf'

# Add the font to matplotlib's font manager
font_manager.fontManager.addfont(FONT_PATH)

# Set the font family to Montserrat
plt.rcParams['font.family'] = 'Montserrat'

def preprocess_data(X, transform_type):
    ''' Preprocess the data using different scaling techniques.
    
    Parameters:'
        -'X': DataFrame containing the features to be transformed.
        -'transform_type': Type of transformation to apply ('minmax', 'standardization', 'pca').'
    
    Returns:
        -'X_scaled': Transformed DataFrame.'
    '''

    # Transformation type MinMaxScaler
    if transform_type == 'minmax':
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    
    # Transformation type StandardScaler
    elif transform_type == 'standardization':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # Transformation type PCA
    elif transform_type == 'pca':
        pca = PCA(n_components=0.95)  # Keep 95% variance
        X_scaled = pca.fit_transform(X)
    
    else:
        raise ValueError("Invalid transformation type. Choose from ['minmax', 'standardization', 'pca'].")

    return X_scaled

def selecting_features(dataframe, target, corr_coef=0.009):
    ''' Select features based on correlation with the target variable.
    
    Parameters:
    
        -'dataframe': DataFrame containing the dataset.
        - target': Name of the target variable.
        -'corr_coef': Correlation coefficient threshold for feature selection.'
    
    Returns:
        - correlated_columns: List of features with correlation > corr_coef.''
    '''

    # Remove date column for the correlation matrix
    if "date" in dataframe.columns:
        dataframe = dataframe.drop(columns=['date'])

    # Correlation matrix with price
    correlation_with_target = dataframe.corrwith(dataframe[target]).sort_values(ascending=False)

    # Saving columns names with a correlation > 0.25
    correlated_columns = correlation_with_target[abs(correlation_with_target) >= corr_coef].index.tolist()
    correlated_columns = [col for col in correlated_columns if (col != target)]
    
    print(f"Features with correlation coefficient with target > than {round(corr_coef, 5)}")
    if correlated_columns != []:     
        print("Columns that will be used for the training:\n", correlated_columns, "\n") 

    return correlated_columns 

def select_training_set(dataframe, target, correlated_columns, test_size=0.2, transform = False, transform_type = 'pca'):
    ''' Select the training and testing sets based on the correlation with the target variable.
    
    Parameters:
        -'dataframe': DataFrame containing the dataset.
        -'target': Name of the target variable.
        -'correlated_columns': List of features with correlation > corr_coef.
        -'test_size': Proportion of the dataset to include in the test split (default is 0.2).
        -'transform': Boolean indicating whether to apply transformation (default is False).
        -'transform_type': Type of transformation to apply (default is 'minmax').
        
    Returns:
        -'X_train': Training features DataFrame.
        -'X_test': Testing features DataFrame.
        -'y_train': Training target DataFrame.
        -'y_test': Testing target DataFrame.
    '''   
    
    # Creating the training dataset
    X = dataframe[correlated_columns]
    y = dataframe[target]

    if transform == False:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        print(f"Applying transformation: {transform_type}...")
        X_transformed = preprocess_data(X, transform_type)
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=test_size, random_state=42)
        
    print(f"Test size {int(test_size * 100)}%:")
    print(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")
    

    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test, model_name, results_list=[], model_tree = "", estimator = 100):
    ''' Train a machine learning model and evaluate its performance.
    
    Parameters:
        -'X_train': Training features DataFrame.
        -'X_test': Testing features DataFrame.
        ''y_train': Training target DataFrame.
        -'y_test': Testing target DataFrame.
        - 'model_name': Machine learning model to be trained (e.g., DecisionTreeClassifier, RandomForestClassifier, etc.).
        -'results_list': List to store the results of the model evaluation (default is an empty list).
        -'model_tree': Pre-trained model (default is an empty string).
        -'estimator': Number of estimators for the model (default is 100).
    
    'Returns:
        -'results_list': List containing the results of the model evaluation.
        -'model_tree': Trained machine learning model.
    '''

    # Initializing and training the model
    if model_tree == "":
            if model_name == RandomForestClassifier:
                print(f"Random Forest Classifier initialized with {estimator} estimators.")
                model_tree = model_name(n_estimators=estimator, random_state=42)
            elif model_name == XGBClassifier:
                print(f"XGBClassifier initialized with {estimator} estimators.")
                model_tree = model_name(use_label_encoder=False, n_estimators=estimator, random_state=42)
            elif model_name == LogisticRegression:
                print(f"Logistic Regression initialized.")
                model_tree = model_name(random_state=42)
            elif model_name == ExtraTreesClassifier:
                print(f"Extra Trees Classifier initialized.")
                model_tree = model_name(random_state=42)
            elif model_name == DecisionTreeClassifier:
                print(f"Decision Tree Classifier initialized.")
                model_tree = model_name(random_state=42)

    model_tree.fit(X_train, y_train)
    
    # Making predictions
    y_pred = model_tree.predict(X_test)
   
    # Calculating metrics for the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")  # Change to 'macro' if multiclass
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print the metrics
    print(f"Model Metrics: | Accuracy = {round(accuracy, 4)} | Precision = {round(precision, 4)} | Recall = {round(recall, 4)} | F1-score = {round(f1, 4)} |")
    print(f"Confusion Matrix: \n{conf_matrix}\n")

    # Create a table with actual vs predicted values and their difference
    results_list.append({
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Confusion Matrix": conf_matrix
    })

    if accuracy >= 0.85:
        print("The model performs well! High accuracy and balance between precision & recall.\n")
    elif 0.65 <= accuracy < 0.85:
        print("The model is moderately good but could be improved.\n")
    else:
        print("The model performs poorly. Consider feature selection or hyperparameter tuning.\n")
    
    return results_list, model_tree

def create_train_test_splits_and_evaluate(dataframe, target, correlated_columns, model, transform = False, transform_type = 'pca'):
    ''' Create different train-test splits and evaluate the model performance.'
    
    Parameters:
        -'dataframe': DataFrame containing the dataset.
        -'target': Name of the target variable.
        -'correlated_columns': List of features with correlation > corr_coef.
        -'model': Machine learning model to be trained (e.g., DecisionTreeClassifier, RandomForestClassifier, etc.).
    
    Return:
        results_list: List containing the results of the model evaluation.
    '''

    test_sizes = [0.3, 0.2, 0.1]
    results_list = []
    
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = select_training_set(dataframe, target, correlated_columns, test_size, transform = transform, transform_type = transform_type)
        results_list, model_tree = train_model(X_train, X_test, y_train, y_test, model_name=model , results_list=results_list)
        results_list[-1]["test_size"] = test_size

    return results_list

def evaluate_different_correlations(dataframe, target, model, transform = False, transform_type = 'pca'):
    ''' Evaluate the model performance for different correlation coefficients.
    Parameters:
        -'dataframe': DataFrame containing the dataset.
        -'target': Name of the target variable.
        -'model': Machine learning model to be trained (e.g., DecisionTreeClassifier, RandomForestClassifier, etc.).
    
    Return:
        -'results_df: DataFrame containing the results of the model evaluation for different correlation coefficients.
    '''

    correlation_thresholds = [0.00001, 0.01, 0.02, 0.03, 0.04, 0.08, 0.2, 0.3, 0.4]  
    results = []

    for corr_coef in correlation_thresholds:

        # Select features based on correlation coefficient
        correlated_columns = selecting_features(dataframe, target, corr_coef)

        # Ensure there are enough features to proceed
        if not correlated_columns:
            print("No features meet this correlation threshold. Ending...\n")
            break

        # Train and evaluate models for different test sizes
        results_list = create_train_test_splits_and_evaluate(dataframe, target, correlated_columns, model, transform = transform, transform_type = transform_type)

        # Collecting results for the DataFrame
        for result in results_list:
            results.append({
                    "Correlation Coefficient ≥": corr_coef,
                    "Test Size (%)": result["test_size"],
                    "Accuracy": result["Accuracy"],
                    "Precision": result["Precision"],
                    "Recall": result["Recall"],
                    "F1-score": result["F1-score"]
            })

    # Convert collected results into a DataFrame
    results_df = pd.DataFrame(results)

    return results_df    

def cross_validate_model(dataframe, correlated_columns, target, model, n_splits=5, trained_model = False, transform=False, transform_type='minmax'):
    ''' Perform KFold cross-validation on the model.
    
    Parameters:
        -'dataframe': DataFrame containing the dataset.
        -'target': Name of the target variable.
        -'correlated_columns': List of features with correlation > corr_coef.
        -'model': Machine learning model to be trained (e.g., DecisionTreeClassifier, RandomForestClassifier, etc.).
        -'n_splits': Number of folds for KFold cross-validation (default is 5).
        -'transform': Boolean indicating whether to apply transformation (default is False).
        -'transform_type': Type of transformation to apply (default is 'minmax').
        -'trained_model': Boolean indicating whether to use a pre-trained model (default is False).
        
    'Returns:
        -'results_df': DataFrame containing the results of the cross-validation.'
        -'average': Dictionary containing the average accuracy scores for training and testing.'
    '''

    # Preparing the data
    X = dataframe[correlated_columns]
    y = dataframe[target]

    if transform == True:
        X = preprocess_data(X, transform_type)
    
    # Setting up KFold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []
    training_scores = []
    test_scores = []
    average = {}
    
    fold = 1
    for train_index, test_index in kf.split(X):

        # Splitting the data into training and testing sets for the fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Initialize and train the model
        if trained_model == True:
            model_tree = copy.deepcopy(model)
            model_tree.fit(X_train, y_train)
        else:
            model_tree = model(random_state=42)
            model_tree.fit(X_train, y_train)
        
        # Predict on the training and testing sets
        y_train_pred = model_tree.predict(X_train)
        y_test_pred = model_tree.predict(X_test)
        
        # Calculate accuracy scores for training and testing
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # F1-Score for a more robust metric in classification
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        training_scores.append(train_accuracy)
        test_scores.append(test_accuracy)

        # Save fold results to print and return later
        results.append({
            'Fold': fold,
            'Train Accuracy': round(train_accuracy, 4),
            'Test Accuracy': round(test_accuracy, 4),
            'Train F1': round(train_f1, 4),
            'Test F1': round(test_f1, 4)
        })
        
        fold += 1

    # Create a DataFrame to return the results of the cross-validation
    results_df = pd.DataFrame(results)

    # Calculate average scores across all folds
    avg_train_accuracy = round(sum(training_scores) / len(training_scores), 4)
    avg_test_accuracy = round(sum(test_scores) / len(test_scores), 4)

    # Append the results for this n_splits to the overall results
    average = {
            'n_splits': n_splits,
            'Average Train Accuracy': avg_train_accuracy,
            'Average Test Accuracy': avg_test_accuracy
        }

    # Print conclusion based on the accuracy scores
    print("Cross-Validation Results:")
    print(f"Number of folds: {n_splits}")
    print("Average Training Accuracy: ", avg_train_accuracy)
    print("Average Test Accuracy: ", avg_test_accuracy)
    
    # Conclusion based on the average test accuracy
    if avg_test_accuracy >= 0.75:
        print("The model performs well on unseen data, demonstrating high accuracy.\n")
    elif 0.5 <= avg_test_accuracy < 0.75:
        print("The model performs moderately well, but there's room for improvement in its generalization.\n")
    else:
        print("The model performs poorly on unseen data. Consider tuning hyperparameters or using a different approach.\n")
    
    
    return results_df, average

def perform_grid_search(dataframe, target, model, transform = False, transform_type = 'minmax'):
    ''' Perform grid search for hyperparameter tuning on the model.
    
    'Parameters:
        -'dataframe': DataFrame containing the dataset.
        -'target': Name of the target variable.'
        -'model': Machine learning model to be trained (e.g., DecisionTreeClassifier, RandomForestClassifier, etc.).
        -'transform': Boolean indicating whether to apply transformation (default is False).
        -'transform_type': Type of transformation to apply (default is 'minmax').

        'Returns:
        -'results_df': DataFrame containing the results of the grid search.
        -'average': Dictionary containing the average accuracy scores for training and testing.
    '''
        
    results_list = []

    n_splits_list = [5]
    test_sizes = [0.2]
    

    if model == RandomForestClassifier:
        print ('Random Forest Classifier selected.')
        correlation_thresholds = [0.00001, 0.04, 0.08, 0.2, 0.3]
        param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depth': [None, 10, 20, 30],  # Depth of trees
        'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
        'min_samples_leaf': [1, 2, 4]     # Minimum samples per leaf
        }
        print ('Hyperparameters to tune: ',param_grid)

    elif model == LogisticRegression:
        print ('Logistic Regression selected.')
        correlation_thresholds = [0.04, 0.08, 0.2]
        param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', None],  # Regularization type
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],  # Optimization algorithm
        'max_iter': [100, 200, 500, 1000],  # Maximum number of iterations
        'fit_intercept': [True, False],  # Whether to include an intercept
        'class_weight': [None, 'balanced'],  # Handle class imbalance
        'tol': [1e-4, 1e-3, 1e-2]  # Tolerance for stopping criteria
        }

        print ('Hyperparameters to tune: ',param_grid)

    elif model == DecisionTreeClassifier:
        print ('Decision Tree Classifier selected.')
        correlation_thresholds = [0.00001, 0.04, 0.2]
        param_grid = {
            'max_depth': [10, 15, 30, None],
            'min_samples_split': [2, 5, 7],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy', "log_loss"]
        }

        print ('Hyperparameters to tune: ',param_grid)

    elif model == ExtraTreesClassifier:
        print ('Extra Tree Classifier selected.')
        correlation_thresholds = [0.00001, 0.04, 0.2]
        param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [10, 15, 30, None],  # Depth of trees
        'min_samples_split': [2, 5, 7],  # Minimum samples to split a node
        'min_samples_leaf': [1, 2, 4],  # Minimum samples per leaf
        'criterion': ['gini', 'entropy', 'log_loss'],  # Split quality function
        'max_features': [None, 'sqrt', 'log2'],  # Number of features per split
        }

    elif model == XGBClassifier:
        print('XGBoost Classifier selected.')
        correlation_thresholds = [0.00001, 0.04, 0.3]
        param_grid = {
            'n_estimators': [50, 100, 200],       # Number of trees
            'max_depth': [3, 6, 10],              # Maximum tree depth
            'learning_rate': [0.01, 0.1, 0.2],    # Step size shrinkage
            'subsample': [0.6, 0.8, 1.0],         # Row subsampling
            'colsample_bytree': [0.6, 0.8, 1.0],  # Feature subsampling per tree
            'gamma': [0, 0.1, 0.3],               # Minimum loss reduction
            'reg_lambda': [1, 1.5, 2]             # L2 regularization
        }

        print ('Hyperparameters to tune: ',param_grid)

    model_tree = model()
     
    for corr_limit in correlation_thresholds:
        print("\n" + "="*50) 
        correlated_columns = selecting_features(dataframe, target, corr_coef=corr_limit)
        print("="*50)

        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = select_training_set(dataframe, target, correlated_columns, test_size, transform = transform, transform_type = transform_type)
       
            for cv in n_splits_list:
                
                print(f"\n  Number of folds {cv}")
                
                # Initialize GridSearchCV
                grid_search = GridSearchCV(model_tree, param_grid, cv=cv, scoring='accuracy', n_jobs=1)
    
                # Fit the model
                grid_search.fit(X_train, y_train)
    
                # Get results
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                results_list.append({
                    'Correlation Coef Limit': corr_limit,
                    'Test Size': test_size,
                    'Cross-validation fold numbers': cv,
                    'Best Parameters': best_params,
                    'Best Accuracy Score': round(best_score, 4)
                })

                print(f"\nBest Accuracy Score: {best_score:.4f}")
                print(f"Best Parameters: {best_params}")

                # Classification performance conclusions
                if best_score > 0.75:
                    print("Excellent model! The accuracy is high.\n")
                elif 0.50 <= best_score <= 0.75:
                    print("Acceptable model. Consider tuning further.\n")
                else:
                    print("Poor model! Needs improvement.\n")

    results_df = pd.DataFrame(results_list)
    
    return results_df

def feature_score(final_model, X_train):
    ''' Calculate and plot feature importance scores.
    
    Parameters:
        -'final_model': Trained machine learning model.
        -X_train': Training features DataFrame.
        
    'Returns:'
        -'importance_df': DataFrame containing feature importance scores.
    '''

    feature_importances = final_model.feature_importances_

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 7))
    plt.bar(importance_df['Feature'].head(10), importance_df['Importance'].head(10), color='#eeba30')
    plt.ylabel('Feature Importance', fontweight='bold', fontsize=12, color='black')
    plt.xlabel('Feature Name', fontweight='bold', fontsize=12, color='black')
    plt.title(f'Feature Importance {final_model.__class__.__name__}', fontweight='bold', fontsize=14, color='black')
    plt.xticks(ticks=range(len(importance_df['Feature'].head(10))), 
           labels=[label.replace("days_for_shipment_scheduled", "Shipment Days Sch").replace("shipping_mode_standard_class", "Shipment Mode Std").replace('_', ' ').title() for label in importance_df['Feature'].head(10)], 
           rotation=45, fontsize=10, color='black')
    plt.grid(color='black', linestyle='--', linewidth=0.5)

    plt.savefig(f"../../images/feature_importance_{final_model.__class__.__name__}.png", 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
    plt.show()

    return importance_df

def feature_selection(final_model, X_train, X_test, y_train, y_test, features = 5):
    ''' Perform Recursive Feature Elimination (RFE) for feature selection.'
    
    Parameters:
        -'final_model': Trained machine learning model.
        -'X_train': Training features DataFrame.
        -'y_train: Training target DataFrame.
        -'X_test: Testing features DataFrame.
        -'y_test: Testing target DataFrame.
        -'features: Number of features to select (default is 5).
        
    'Returns:
        -'correlated_columns': List of selected features.
        -'final_model': Trained machine learning model.
    '''

    # Initialize RFE and select the top 5 features
    rfe = RFE(estimator=final_model, n_features_to_select=features)
    X_train_rfe = rfe.fit_transform(X_train, y_train)

    # Fit the model with the selected features
    final_model.fit(X_train_rfe, y_train)

    # Test the model
    X_test_rfe = rfe.transform(X_test)
    y_pred = final_model.predict(X_test_rfe)

    # Evaluate the model
    print(f"Selected Features: {X_train.columns[rfe.support_]}")
    correlated_columns = X_train.columns[rfe.support_]

    # Calculating metrics for the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print the metrics
    print(f" Model Metrics: | Accuracy = {round(accuracy, 4)} | Precision = {round(precision, 4)} | Recall = {round(recall, 4)} | F1 Score = {round(f1, 4)}")
    print(f" Confusion Matrix:\n {conf_matrix}\n")

    # Provide insights about model performance
    if accuracy >= 0.75:
        print(" The model performs well! It has high accuracy.\n")
    elif 0.5 <= accuracy < 0.75:
        print("The model is moderately good, but there is room for improvement.\n")
    else:
        print("The model performs poorly. Consider tuning hyperparameters or using a different approach.\n")


    return correlated_columns, final_model, accuracy


def perform_grid_search_with_relevant(dataframe, correlated_columns, target, model, transform = False, transform_type = 'minmax'):
    ''' Perform grid search for hyperparameter tuning on the model with relevant features.'
    
    'Parameters:
        -'dataframe': DataFrame containing the dataset.
        -'target': Name of the target variable.
        -'correlated_columns': List of features with correlation > corr_coef.
        -'model': Machine learning model to be trained (e.g., DecisionTreeClassifier, RandomForestClassifier, etc.).
        -'transform': Boolean indicating whether to apply transformation (default is False).
        -'transform_type': Type of transformation to apply (default is 'minmax').
        
        'Returns:
        -'results_df': DataFrame containing the results of the grid search.
        -'average': Dictionary containing the average accuracy scores for training and testing.
    '''

    results_list = []

    n_splits_list = [5]
    test_sizes = [0.2]

    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    if model == RandomForestClassifier:
        print ('Random Forest Classifier selected.')

        param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depth': [None, 10, 20, 30],  # Depth of trees
        'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
        'min_samples_leaf': [1, 2, 4]     # Minimum samples per leaf
        }

        print ('Hyperparameters to tune: ',param_grid)

    elif model == LogisticRegression:
        print ('Logistic Regression selected.')

        param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', None],  # Regularization type
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],  # Optimization algorithm
        'max_iter': [100, 200, 500, 1000],  # Maximum number of iterations
        'fit_intercept': [True, False],  # Whether to include an intercept
        'class_weight': [None, 'balanced'],  # Handle class imbalance
        'tol': [1e-4, 1e-3, 1e-2]  # Tolerance for stopping criteria
        }

        print ('Hyperparameters to tune: ',param_grid)

    elif model == DecisionTreeClassifier:
        print ('Decision Tree Classifier selected.')

        param_grid = {
            'max_depth': [3, 5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy', "log_loss"]
        }

        print ('Hyperparameters to tune: ',param_grid)

    model_tree = model(random_seed=42)
     
    
    print("\n" + "="*50) 
    correlated_columns = correlated_columns
    print("="*50)

    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = select_training_set(dataframe, target, correlated_columns, test_size, transform = transform, transform_type = transform_type)
       
        for cv in n_splits_list:
                
            print(f"\n  Number of folds {cv}")
                
            # Initialize GridSearchCV
            grid_search = GridSearchCV(model_tree, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    
            # Fit the model
            grid_search.fit(X_train, y_train)
    
            # Get results
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            results_list.append({
                    'Test Size': test_size,
                    'Cross-validation fold numbers': cv,
                    'Best Parameters': best_params,
                    'Best Accuracy Score': round(best_score, 4)
            })

            print(f"\nBest Accuracy Score: {best_score:.4f}")
            print(f"Best Parameters: {best_params}")

            # Classification performance conclusions
            if best_score > 0.75:
                print("Excellent model! The accuracy is high.\n")
            elif 0.50 <= best_score <= 0.75:
                print("Acceptable model. Consider tuning further.\n")
            else:
                print("Poor model! Needs improvement.\n")

    results_df = pd.DataFrame(results_list)
    
    return results_df

def confusion_matrix_plot(final_model, X_test, y_test, correlated_columns):
    ''' Plot the confusion matrix for the model predictions.
    
    Parameters:
        -'y_test': Testing target DataFrame.
        -'X_test': Testing features DataFrame.
        -'correlated_columns': List of features with correlation > corr_coef.
        -'final_model': Trained machine learning model.
    '''
    
    # Predicted values
    y_pred = final_model.predict(X_test[correlated_columns])

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"../../images/confusion_matrix_{final_model.__class__.__name__}.png", 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)
    plt.show()

def roc_curve_plot(final_model, X_test, y_test, correlated_columns):
    ''' Plot the ROC curve for the model predictions.
    
    Parameters:
        -'y_test': Testing target DataFrame.
        -'X_test': Testing features DataFrame.
        -'correlated_columns': List of features with correlation > corr_coef.
        -'final_model': Trained machine learning model.
    '''
    
    # Predicted probabilities
    y_pred = final_model.predict(X_test[correlated_columns])

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred)  # y_probs are predicted probabilities
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color = 'green')
    plt.plot([0, 1], [0, 1], linestyle='--', color = 'red')  # Diagonal line for reference
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(f"../../images/roc_{final_model.__class__.__name__}.png", 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)

    plt.show()    