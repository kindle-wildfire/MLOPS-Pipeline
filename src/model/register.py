import json
import os
import mlflow
from mlflow.tracking import MlflowClient

# 1. 从 train stage 的产物读 run_id 和数据版本
with open("reports/model_metrics.json") as f:
    metrics = json.load(f)
run_id = metrics["run_id"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. 从 run 注册模型
model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(model_uri, "gym_churn_model")

# 3. 给版本挂数据版本 tag（从 run 的 tag 继承）
client = MlflowClient()
run_info = client.get_run(run_id)
data_version = run_info.data.tags.get("data_version", "unknown")
client.set_model_version_tag(
    name="gym_churn_model",
    version=result.version,
    key="data_version",
    value=data_version,
)

# 4. 写注册结果，供 DVC outs
os.makedirs("reports", exist_ok=True)
with open("reports/registered_version.json", "w") as f:
    json.dump(
        {
            "model_name": result.name,
            "version": result.version,
            "run_id": run_id,
            "data_version": data_version,
        },
        f,
        indent=2,
    )
print(f"registered {result.name} v{result.version} from run {run_id}")