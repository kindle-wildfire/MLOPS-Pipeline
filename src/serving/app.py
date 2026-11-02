import os
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI

# 强制清空代理
for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(k, None)

app = FastAPI()

# 关键：从 Server 获取最新注册的模型（因为刚才清库了，现在只有 v1）
mlflow.set_tracking_uri("http://mlflow:5000")
model = mlflow.pyfunc.load_model("models:/gym_churn_model/latest")

df_train = pd.read_csv("data/raw/gym_ghosts_master.csv")
X_train = df_train.drop(columns=["member_id", "churned_by_february"])
X_train = pd.get_dummies(X_train, drop_first=True)
FEATURE_COLUMNS = X_train.columns.tolist()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    for col in ["member_id", "churned_by_february"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    pred = model.predict(df)
    return {"prediction": int(pred[0])}