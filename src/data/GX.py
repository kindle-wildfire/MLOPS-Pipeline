import pandas as pd
import great_expectations as gx

df = pd.read_csv("/home/kindle/project/MLOPS/data/raw/gym_ghosts_master.csv")
print("shape:", df.shape)
print("columns:", df.columns.tolist())

context = gx.get_context()
data_source = context.data_sources.add_pandas("pandas_source")
data_asset = data_source.add_dataframe_asset(name="gym_data")
batch_def = data_asset.add_batch_definition_whole_dataframe("whole_df")
batch = batch_def.get_batch(batch_parameters={"dataframe": df})

e1 = gx.expectations.ExpectColumnValuesToBeInSet(
   column="churned_by_february", value_set=[0, 1]
)

r1 = batch.validate(e1)
print("target_in_set:", r1.success)

# 校验2：目标列不能缺失
e2 = gx.expectations.ExpectColumnValuesToNotBeNull(
    column="churned_by_february"
)
r2 = batch.validate(e2)
print("target_not_null:", r2.success)

import json
import os
os.makedirs("reports", exist_ok=True)
with open("reports/data_validation.json", "w") as f:
    json.dump(
        {"target_in_set": r1.success, "target_not_null": r2.success},
        f,
        indent=2,
    )
print("written reports/data_validation.json")

# 6. 校验失败则非零退出，让 pipeline 停下（门禁拦截）
if not (r1.success and r2.success):
    raise SystemExit("data validation failed: gate blocked")