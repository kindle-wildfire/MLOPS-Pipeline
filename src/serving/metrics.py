import json
import os
import re
from collections import defaultdict
from datetime import datetime

LOG_PATH = "logs/serving.log"
OUT_PATH = "reports/serving_metrics.json"
ALERT_PATH = "logs/alerts.log"

# 阈值（可按需调）
THRESHOLDS = {
    "error_rate_max": 0.05,     # 错误率 > 5% 告警
    "p95_ms_max": 200.0,        # P95 > 200ms 告警
}

# 匹配日志行：... GET /health status=200 duration_ms=1.08
pattern = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) "
    r"\w+ (?P<method>[A-Z]+) (?P<path>\S+) "
    r"status=(?P<status>\d+) duration_ms=(?P<dur>[\d.]+)"
)

def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

per_path = defaultdict(lambda: {"count": 0, "errors": 0, "durations": []})

with open(LOG_PATH) as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue
        path = m.group("path")
        status = int(m.group("status"))
        dur = float(m.group("dur"))
        per_path[path]["count"] += 1
        per_path[path]["durations"].append(dur)
        if status >= 400:
            per_path[path]["errors"] += 1

metrics = {}
alerts = []

for path, d in per_path.items():
    durations = sorted(d["durations"])
    count = d["count"]
    error_rate = d["errors"] / count if count else 0.0
    metrics[path] = {
        "count": count,
        "error_rate": round(error_rate, 4),
        "p50_ms": round(percentile(durations, 0.50), 2),
        "p95_ms": round(percentile(durations, 0.95), 2),
        "p99_ms": round(percentile(durations, 0.99), 2),
    }
    if error_rate > THRESHOLDS["error_rate_max"]:
        alerts.append(f"{path}: error_rate={error_rate:.2%} > {THRESHOLDS['error_rate_max']:.0%}")
    if metrics[path]["p95_ms"] is not None and metrics[path]["p95_ms"] > THRESHOLDS["p95_ms_max"]:
        alerts.append(f"{path}: p95_ms={metrics[path]['p95_ms']} > {THRESHOLDS['p95_ms_max']}")

os.makedirs("reports", exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump({"generated_at": datetime.now().isoformat(), "metrics": metrics}, f, indent=2)

os.makedirs("logs", exist_ok=True)
if alerts:
    with open(ALERT_PATH, "a") as f:
        for a in alerts:
            f.write(f"{datetime.now().isoformat()} WARN {a}\n")
            print("WARN", a)
else:
    print("no alerts")

print(f"written {OUT_PATH}")