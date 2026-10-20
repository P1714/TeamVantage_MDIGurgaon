# MadhuTwin REST API + doctor console (static) in one container.
#   docker build -t madhutwin .
#   docker run -p 8000:8000 madhutwin     -> http://localhost:8000/docs  and  http://localhost:8000/console/
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY madhutwin/ madhutwin/
COPY models/ models/
COPY dashboard/ dashboard/
COPY data/processed/twin_params.csv data/processed/api_store.parquet data/processed/
ENV MADHUTWIN_SERVE_CONSOLE=1
EXPOSE 8000
CMD ["uvicorn", "madhutwin.api:app", "--host", "0.0.0.0", "--port", "8000"]
