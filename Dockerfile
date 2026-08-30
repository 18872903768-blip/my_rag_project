# RAG 服务镜像：模型与索引常驻，uvicorn 对外服务
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf_cache

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install uvicorn

COPY api/ api/
COPY rag_modules/ rag_modules/
COPY config.py main.py ./

# 数据/索引/HF缓存全部放 /data 卷，重建容器不重下载模型
ENV RAG_DATA_PATH=/data/cook \
    RAG_INDEX_PATH=/data/vector_index

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
