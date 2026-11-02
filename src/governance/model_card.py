import json
import os
import re
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "gym_churn_model"
VERSION = 1  # 可改成你想生成的版本

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# 1. 读注册版本与 run
mv = client.get_model_version(MODEL_NAME, VERSION)
run = client.get_run(mv.run_id)
data_version_tag = run.data.tags.get("data_version", "unknown")

# 2. 从 .dvc 指针读 md5（权威数据版本）
with open("data/raw/gym_ghosts_master.csv.dvc") as f:
    dvc_content = f.read()
dvc_md5 = re.search(r"md5:\s*([a-f0-9]+)", dvc_content).group(1)

# 3. 读 reports / alerts
def read_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

metrics = read_json("reports/model_metrics.json", {})
registered = read_json("reports/registered_version.json", {})
serving = read_json("reports/serving_metrics.json", {})
alerts = ""
if os.path.exists("logs/alerts.log"):
    with open("logs/alerts.log") as f:
        alerts = f.read().strip()

# 4. 拼模型卡
os.makedirs("docs", exist_ok=True)
card_path = f"docs/model_card_v{VERSION}.md"
with open(card_path, "w") as f:
    f.write(f"# Model Card: {MODEL_NAME} v{VERSION}\n\n")
    f.write("## 用途\n预测健身房会员是否会在2月底前流失。\n\n")
    f.write("## 已知局限\n- 仅基于单表静态数据训练；\n- 未做公平性/偏差评估；\n- 线上输入分布变化时需重训。\n\n")
    f.write("## 数据版本\n")
    f.write(f"- DVC 指针 md5: `{dvc_md5}`\n")
    f.write(f"- run tag data_version: `{data_version_tag}`\n\n")
    f.write("## 评估指标\n")
    f.write(f"- accuracy: {metrics.get('accuracy')}\n- roc_auc: {metrics.get('roc_auc')}\n- run_id: {metrics.get('run_id')}\n\n")
    f.write("## 注册与审批记录\n")
    f.write(f"- 注册版本: {registered.get('version')}\n- 注册 run_id: {registered.get('run_id')}\n- 状态: {mv.status}\n\n")
    f.write("## 线上监控摘要\n")
    f.write("```json\n")
    f.write(json.dumps(serving.get("metrics", {}), indent=2, ensure_ascii=False))
    f.write("\n```\n\n")
    f.write("## 告警记录\n")
    f.write("```\n")
    f.write(alerts or "无")
    f.write("\n```\n")

print(f"written {card_path}")