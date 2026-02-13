import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif, RFE, SelectFromModel
import warnings
warnings.filterwarnings('ignore')

path = '/home/darshan/Documents/geekforgeeks(21days21Projects)/Day 7 -  Preventing Customer Churn with Feature Transformation/Telco-Customer-Churn.csv'
print('loading', path)
df = pd.read_csv(path)
# basic cleaning
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
    df.dropna(subset=['Churn'], inplace=True)
# engineer extra features
df_ex = df.copy()
svc_cols = [c for c in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'] if c in df_ex.columns]
if svc_cols:
    df_ex['num_add_services'] = (df_ex[svc_cols]=='Yes').sum(axis=1)
if 'tenure' in df_ex.columns:
    bins = [0,12,24,48,60,72]
    labels = ['0-1','1-2','2-4','4-5','5+']
    df_ex['tenure_group'] = pd.cut(df_ex['tenure'], bins=bins, labels=labels, right=False)
    ord_map = {'0-1':0,'1-2':1,'2-4':2,'4-5':3,'5+':4}
    df_ex['tenure_group_ord'] = df_ex['tenure_group'].map(ord_map)
if 'MonthlyCharges' in df_ex.columns:
    df_ex['avg_charge_per_service'] = df_ex['MonthlyCharges'] / (df_ex.get('num_add_services',0)+1)
if 'SeniorCitizen' in df_ex.columns and 'Partner' in df_ex.columns:
    df_ex['senior_and_alone'] = ((df_ex['SeniorCitizen']==1) & (df_ex['Partner']=='No')).astype(int)
if 'MonthlyCharges' in df_ex.columns:
    df_ex['high_monthly'] = (df_ex['MonthlyCharges']>df_ex['MonthlyCharges'].median()).astype(int)

X = df_ex.drop('Churn', axis=1)
y = df_ex['Churn']
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
pre = ColumnTransformer(transformers=[('num', StandardScaler(), num_cols), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# simple models
models = {
    'Logistic': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
    'GradientBoosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
}
results = {}
for name,m in models.items():
    pipe = Pipeline([('pre', pre), ('clf', m)])
    pipe.fit(X_train, y_train)
    ypred = pipe.predict(X_test)
    rep = classification_report(y_test, ypred, output_dict=True)
    results[name] = {
        'accuracy': accuracy_score(y_test, ypred),
        'precision': precision_score(y_test, ypred, zero_division=0),
        'recall': recall_score(y_test, ypred, zero_division=0),
        'f1': f1_score(y_test, ypred, zero_division=0),
        'report': rep
    }

# small grid search for RF
pipe_rf = Pipeline([('pre', pre), ('clf', RandomForestClassifier(random_state=42))])
param_grid = {'clf__n_estimators':[50,100,150], 'clf__max_depth':[None,6,10]}
gs = GridSearchCV(pipe_rf, param_grid, scoring='f1', cv=3, n_jobs=2)
gs.fit(X_train, y_train)
best_rf = gs.best_estimator_
yp_rf = best_rf.predict(X_test)
results['RandomForest_Grid'] = {'best_params': gs.best_params_, 'accuracy': accuracy_score(y_test, yp_rf), 'precision': precision_score(y_test, yp_rf, zero_division=0), 'recall': recall_score(y_test, yp_rf, zero_division=0), 'f1': f1_score(y_test, yp_rf, zero_division=0), 'report': classification_report(y_test, yp_rf, output_dict=True)}

# feature importances from RF
# fit RF on preprocessed data
rf_full = Pipeline([('pre', pre), ('clf', RandomForestClassifier(random_state=42, n_estimators=150))])
rf_full.fit(X_train, y_train)
try:
    feat_names = pre.get_feature_names_out()
except Exception:
    feat_names = num_cols + cat_cols
X_train_prep = pre.transform(X_train)
importances = rf_full.named_steps['clf'].feature_importances_
imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False).head(15)

out = {'models': results, 'top_features': imp_df.to_dict(orient='records')}
with open('/tmp/day7_full_results.json','w') as f:
    json.dump(out, f)
print('Saved quick results to /tmp/day7_full_results.json')
