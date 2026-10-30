import mlflow
from mlflow.tracking import MlflowClient

# 用你上一段的 run_id
run_id = "acdd4953a36e4751a9852abc9a5796d4"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 从 run 注册模型
model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(model_uri, "gym_churn_model")

# 给注册的版本打上数据版本 tag，方便审计
client = MlflowClient()
client.set_model_version_tag(
    name="gym_churn_model",
    version=result.version,
    key="data_version",
    value="dvc:de2eb4c701f97e04e404e0039712c1b7",
)

print("registered version:", result.version)
print("model name:", result.name)