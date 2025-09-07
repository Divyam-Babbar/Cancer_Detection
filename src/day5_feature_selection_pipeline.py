# 📦 Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 📂 Load dataset
df = pd.read_csv('../data/The_Cancer_data_1500_V2.csv')
df.columns = df.columns.str.strip()

# 🔄 Encode categorical columns
categorical = ['Gender', 'Smoking', 'GeneticRisk', 'CancerHistory']
for col in categorical:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# ✨ Separate features and target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# 🔍 1. Feature Selection - SelectKBest
selector = SelectKBest(score_func=f_classif, k=8)
X_selected = selector.fit_transform(X, y)

# ✅ Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
print("📌 Top 8 Selected Features:", selected_features)

# 🧪 Split data
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

# 📉 2. PCA Transformation (on all features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 🧠 3. Build pipelines with selected features
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# 🧪 Train/Test split on full data with selected features
pipe_lr.fit(X_train_sel, y_train_sel)
pipe_rf.fit(X_train_sel, y_train_sel)

# 📈 Predict and evaluate
y_pred_lr = pipe_lr.predict(X_test_sel)
y_pred_rf = pipe_rf.predict(X_test_sel)

print("\n🔍 Logistic Regression Report:\n", classification_report(y_test_sel, y_pred_lr))
print("\n🔍 Random Forest Report:\n", classification_report(y_test_sel, y_pred_rf))

# 📊 Cross-validation scores
cv_lr = cross_val_score(pipe_lr, X[selected_features], y, cv=5).mean()
cv_rf = cross_val_score(pipe_rf, X[selected_features], y, cv=5).mean()
print(f"\n✅ CV Accuracy (Logistic Regression): {cv_lr:.4f}")
print(f"✅ CV Accuracy (Random Forest): {cv_rf:.4f}")

# 📁 Optional: Export selected features for Tableau
X[selected_features].to_csv('../data/day5_selected_features.csv', index=False)
print("📁 Exported selected features to: day5_selected_features.csv")
