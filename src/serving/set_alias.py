# src/serving/set_alias.py
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# 1. 列出当前所有版本
versions = client.search_model_versions("name='gym_churn_model'")
for v in versions:
    print(f"version={v.version} run_id={v.run_id} status={v.status}")

# 2. 选一个版本设 champion（示例用最新版本，你可改成指定版本号）
target_version = max(int(v.version) for v in versions)
client.set_registered_model_alias(
    name="gym_churn_model",
    alias="champion",
    version=target_version,
)
print(f"set champion -> v{target_version}")