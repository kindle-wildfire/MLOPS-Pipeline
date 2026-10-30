import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import re

df = pd.read_csv("data/raw/gym_ghosts_master.csv")
y = df["churned_by_february"]
X = df.drop(columns=["member_id", "churned_by_february"])
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 读 .dvc 指针拿数据版本
with open("data/raw/gym_ghosts_master.csv.dvc") as f:
    content = f.read()
data_md5 = re.search(r"md5:\s*([a-f0-9]+)", content).group(1)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("gym_churn")

with mlflow.start_run() as run:
    mlflow.set_tag("data_version", f"dvc:{data_md5}")

    params = {"n_estimators": 100, "max_depth": 6, "random_state": 42}
    mlflow.log_params(params)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

    print("run_id:", run.info.run_id)
    print("accuracy:", acc)
    print("roc_auc:", auc)
    print("data_version:", f"dvc:{data_md5}")

import os, joblib
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("saved models/model.pkl")