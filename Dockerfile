FROM python:3.12-slim
WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src
COPY .streamlit/ ./.streamlit/
RUN pip install --no-cache-dir -e ".[ml]"

EXPOSE 8000 8501
