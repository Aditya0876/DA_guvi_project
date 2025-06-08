import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("DIABETES PREDICTION - MODEL ACCURACY RESULTS")
print("=" * 50)

# Create sample data (same as your original)
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

# Clean data
df['Glucose'] = np.clip(df['Glucose'], 50, 200)
df['BloodPressure'] = np.clip(df['BloodPressure'], 40, 120)
df['BMI'] = np.clip(df['BMI'], 15, 50)

print(f"Dataset created: {df.shape}")

# Prepare features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale data for some models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTRAINING MODELS AND CALCULATING ACCURACY:")
print("-" * 45)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for certain models
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
        'accuracy': accuracy,
        'auc_score': auc_score
    }
    
    print(f"✅ {name}")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   AUC Score: {auc_score:.4f}")

# Summary results
print("\n" + "="*50)
print("FINAL RESULTS SUMMARY:")
print("="*50)

# Sort by accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

print("\nRANKING BY ACCURACY:")
print("-" * 25)
for i, (name, metrics) in enumerate(sorted_results, 1):
    print(f"{i}. {name}: {metrics['accuracy']*100:.2f}% accuracy")

# Best model
best_model = sorted_results[0]
print(f"\n🏆 BEST MODEL: {best_model[0]}")
print(f"   Accuracy: {best_model[1]['accuracy']*100:.2f}%")
print(f"   AUC Score: {best_model[1]['auc_score']:.4f}")

print("\n" + "="*50)
print("COPY THESE RESULTS FOR YOUR PRESENTATION!")
print("="*50)