FROM python:3.12-slim
WORKDIR /app

# 关键：使用清华源加速下载，防止超时
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    fastapi==0.162.0 \
    uvicorn==0.54.0 \
    mlflow==3.16.1 \
    pandas \
    scikit-learn \
    pyarrow

COPY src/serving/app.py /app/src/serving/app.py
COPY data/raw/gym_ghosts_master.csv /app/data/raw/gym_ghosts_master.csv
EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]