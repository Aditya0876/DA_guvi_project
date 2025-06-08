# Healthcare Data Analysis: Diabetes Prediction
# Complete Machine Learning Project

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

print("Healthcare Data Analysis: Diabetes Prediction")
print("=" * 50)

# Step 1: Load and Explore Dataset
def load_data():
    """
    Load the diabetes dataset
    Using Pima Indians Diabetes Database as example
    """
    # For demonstration, we'll create sample data structure
    # In real project, use: df = pd.read_csv('diabetes.csv')
    
    # Sample data creation (replace with actual data loading)
    np.random.seed(42)
    n_samples = 768
    
    data = {
        'Pregnancies': np.random.poisson(3, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples),
        'BloodPressure': np.random.normal(72, 12, n_samples),
        'SkinThickness': np.random.normal(26, 10, n_samples),
        'Insulin': np.random.normal(120, 80, n_samples),
        'BMI': np.random.normal(32, 7, n_samples),
        'DiabetesPedigreeFunction': np.random.gamma(0.5, 0.6, n_samples),
        'Age': np.random.randint(21, 81, n_samples),
        'Outcome': np.random.binomial(1, 0.35, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Clean negative values and add some realistic constraints
    df['Glucose'] = np.clip(df['Glucose'], 50, 200)
    df['BloodPressure'] = np.clip(df['BloodPressure'], 40, 120)
    df['SkinThickness'] = np.clip(df['SkinThickness'], 0, 60)
    df['Insulin'] = np.clip(df['Insulin'], 0, 400)
    df['BMI'] = np.clip(df['BMI'], 15, 50)
    
    return df

# Step 2: Data Exploration and Visualization
def explore_data(df):
    """
    Perform comprehensive data exploration
    """
    print("\n1. DATASET OVERVIEW")
    print("-" * 30)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n2. DATA TYPES AND INFO")
    print("-" * 30)
    print(df.info())
    
    print("\n3. STATISTICAL SUMMARY")
    print("-" * 30)
    print(df.describe())
    
    print("\n4. MISSING VALUES")
    print("-" * 30)
    print(df.isnull().sum())
    
    print("\n5. TARGET VARIABLE DISTRIBUTION")
    print("-" * 30)
    target_counts = df['Outcome'].value_counts()
    print(f"Non-Diabetic (0): {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
    print(f"Diabetic (1): {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")
    
    return df

def visualize_data(df):
    """
    Create comprehensive data visualizations
    """
    print("\n6. CREATING VISUALIZATIONS")
    print("-" * 30)
    
    # Set up the plotting area
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Diabetes Dataset - Exploratory Data Analysis', fontsize=16)
    
    # 1. Target distribution
    axes[0,0].pie(df['Outcome'].value_counts(), labels=['Non-Diabetic', 'Diabetic'], 
                  autopct='%1.1f%%', startangle=90)
    axes[0,0].set_title('Target Distribution')
    
    # 2. Age distribution by outcome
    df[df['Outcome']==0]['Age'].hist(ax=axes[0,1], alpha=0.7, label='Non-Diabetic', bins=20)
    df[df['Outcome']==1]['Age'].hist(ax=axes[0,1], alpha=0.7, label='Diabetic', bins=20)
    axes[0,1].set_title('Age Distribution by Outcome')
    axes[0,1].legend()
    
    # 3. BMI distribution by outcome
    df[df['Outcome']==0]['BMI'].hist(ax=axes[0,2], alpha=0.7, label='Non-Diabetic', bins=20)
    df[df['Outcome']==1]['BMI'].hist(ax=axes[0,2], alpha=0.7, label='Diabetic', bins=20)
    axes[0,2].set_title('BMI Distribution by Outcome')
    axes[0,2].legend()
    
    # 4. Glucose vs BMI scatter
    scatter = axes[1,0].scatter(df['Glucose'], df['BMI'], c=df['Outcome'], alpha=0.6)
    axes[1,0].set_xlabel('Glucose')
    axes[1,0].set_ylabel('BMI')
    axes[1,0].set_title('Glucose vs BMI')
    
    # 5. Correlation heatmap
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Feature Correlation Matrix')
    
    # 6. Box plot for glucose by outcome
    df.boxplot(column='Glucose', by='Outcome', ax=axes[1,2])
    axes[1,2].set_title('Glucose Levels by Outcome')
    
    # 7. Pregnancies distribution
    df['Pregnancies'].value_counts().sort_index().plot(kind='bar', ax=axes[2,0])
    axes[2,0].set_title('Pregnancies Distribution')
    axes[2,0].tick_params(axis='x', rotation=0)
    
    # 8. Feature distributions
    features = ['Glucose', 'BloodPressure', 'Insulin']
    for i, feature in enumerate(features[:2]):
        axes[2,1+i].hist(df[feature], bins=20, alpha=0.7)
        axes[2,1+i].set_title(f'{feature} Distribution')
    
    plt.tight_layout()
    plt.show()

# Step 3: Data Preprocessing
def preprocess_data(df):
    """
    Clean and preprocess the dataset
    """
    print("\n7. DATA PREPROCESSING")
    print("-" * 30)
    
    # Create a copy for preprocessing
    df_processed = df.copy()
    
    # Handle zero values that might be missing values
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in zero_columns:
        # Replace zeros with median (except for naturally zero values)
        if col in ['SkinThickness', 'Insulin']:  # These can naturally be zero
            continue
        df_processed[col] = df_processed[col].replace(0, df_processed[col].median())
    
    # Feature engineering
    df_processed['BMI_Category'] = pd.cut(df_processed['BMI'], 
                                        bins=[0, 18.5, 25, 30, 50], 
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    df_processed['Age_Group'] = pd.cut(df_processed['Age'], 
                                     bins=[0, 30, 50, 100], 
                                     labels=['Young', 'Middle', 'Senior'])
    
    # Encode categorical variables
    le = LabelEncoder()
    df_processed['BMI_Category_Encoded'] = le.fit_transform(df_processed['BMI_Category'])
    df_processed['Age_Group_Encoded'] = le.fit_transform(df_processed['Age_Group'])
    
    print("Preprocessing completed!")
    print(f"New shape: {df_processed.shape}")
    
    return df_processed

# Step 4: Feature Selection
def select_features(df):
    """
    Perform feature selection
    """
    print("\n8. FEATURE SELECTION")
    print("-" * 30)
    
    # Prepare features and target
    X = df.drop(['Outcome', 'BMI_Category', 'Age_Group'], axis=1)
    y = df['Outcome']
    
    # Feature selection using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=8)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    print("Selected features:")
    for i, feature in enumerate(selected_features):
        print(f"{i+1}. {feature}")
    
    return X[selected_features], y, selected_features

# Step 5: Model Training and Evaluation
def train_models(X, y):
    """
    Train multiple machine learning models
    """
    print("\n9. MODEL TRAINING AND EVALUATION")
    print("-" * 30)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for SVM and Logistic Regression
        if name in ['SVM', 'Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
    
    return results, X_test, y_test, scaler

# Step 6: Model Evaluation and Visualization
def evaluate_models(results, X_test, y_test):
    """
    Detailed model evaluation with visualizations
    """
    print("\n10. DETAILED MODEL EVALUATION")
    print("-" * 30)
    
    # Create evaluation plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Evaluation Results', fontsize=16)
    
    # Model comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    auc_scores = [results[name]['auc_score'] for name in model_names]
    
    # Bar plot for accuracies
    axes[0,0].bar(model_names, accuracies, alpha=0.7)
    axes[0,0].set_title('Model Accuracy Comparison')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Bar plot for AUC scores
    axes[0,1].bar(model_names, auc_scores, alpha=0.7, color='orange')
    axes[0,1].set_title('Model AUC Score Comparison')
    axes[0,1].set_ylabel('AUC Score')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # ROC Curves
    for name in model_names:
        fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
        axes[1,0].plot(fpr, tpr, label=f"{name} (AUC: {results[name]['auc_score']:.3f})")
    
    axes[1,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1,0].set_xlabel('False Positive Rate')
    axes[1,0].set_ylabel('True Positive Rate')
    axes[1,0].set_title('ROC Curves')
    axes[1,0].legend()
    
    # Best model confusion matrix
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
    cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
    axes[1,1].set_title(f'Confusion Matrix - {best_model_name}')
    axes[1,1].set_xlabel('Predicted')
    axes[1,1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    # Print best model details
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"AUC Score: {results[best_model_name]['auc_score']:.4f}")
    
    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_test, results[best_model_name]['y_pred']))
    
    return best_model_name

# Step 7: Feature Importance Analysis
def analyze_feature_importance(results, selected_features, best_model_name):
    """
    Analyze feature importance for tree-based models
    """
    print("\n11. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 30)
    
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [selected_features[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("Feature Importance Ranking:")
        for i in range(len(importances)):
            print(f"{i+1}. {selected_features[indices[i]]}: {importances[indices[i]]:.4f}")
    else:
        print(f"Feature importance not available for {best_model_name}")

# Step 8: Model Hyperparameter Tuning
def tune_hyperparameters(X, y, best_model_name):
    """
    Perform hyperparameter tuning for the best model
    """
    print("\n12. HYPERPARAMETER TUNING")
    print("-" * 30)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        model = RandomForestClassifier(random_state=42)
    
    elif best_model_name == 'Logistic Regression':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        model = LogisticRegression(random_state=42)
    
    else:
        print(f"Hyperparameter tuning not implemented for {best_model_name}")
        return None
    
    # Grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate tuned model
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Tuned model accuracy: {accuracy:.4f}")
    
    return grid_search.best_estimator_

# Step 9: Prediction Function
def make_predictions(model, scaler, selected_features, sample_data=None):
    """
    Make predictions on new data
    """
    print("\n13. MAKING PREDICTIONS")
    print("-" * 30)
    
    if sample_data is None:
        # Create sample data for demonstration
        sample_data = pd.DataFrame({
            'Pregnancies': [2],
            'Glucose': [120],
            'BloodPressure': [70],
            'SkinThickness': [25],
            'Insulin': [100],
            'BMI': [28.5],
            'DiabetesPedigreeFunction': [0.5],
            'Age': [35],
            'BMI_Category_Encoded': [2],
            'Age_Group_Encoded': [0]
        })
    
    # Select only the features used in training
    sample_features = sample_data[selected_features]
    
    # Scale if necessary
    if hasattr(model, 'kernel'):  # SVM
        sample_features = scaler.transform(sample_features)
    elif 'Logistic' in str(type(model)):
        sample_features = scaler.transform(sample_features)
    
    # Make prediction
    prediction = model.predict(sample_features)[0]
    probability = model.predict_proba(sample_features)[0]
    
    print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
    print(f"Probability of Diabetes: {probability[1]:.4f}")
    print(f"Probability of Non-Diabetes: {probability[0]:.4f}")
    
    return prediction, probability

# Main execution function
def main():
    """
    Main function to execute the entire pipeline
    """
    print("Starting Healthcare Data Analysis: Diabetes Prediction")
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Explore data
    df = explore_data(df)
    
    # Step 3: Visualize data
    visualize_data(df)
    
    # Step 4: Preprocess data
    df_processed = preprocess_data(df)
    
    # Step 5: Select features
    X, y, selected_features = select_features(df_processed)
    
    # Step 6: Train models
    results, X_test, y_test, scaler = train_models(X, y)
    
    # Step 7: Evaluate models
    best_model_name = evaluate_models(results, X_test, y_test)
    
    # Step 8: Analyze feature importance
    analyze_feature_importance(results, selected_features, best_model_name)
    
    # Step 9: Tune hyperparameters
    tuned_model = tune_hyperparameters(X, y, best_model_name)
    
    # Step 10: Make predictions
    final_model = tuned_model if tuned_model is not None else results[best_model_name]['model']
    make_predictions(final_model, scaler, selected_features)
    
    print("\n" + "="*50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*50)

# Run the complete project
if __name__ == "__main__":
    main()