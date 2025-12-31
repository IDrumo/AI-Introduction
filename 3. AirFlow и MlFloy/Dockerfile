# Используем slim-образ для уменьшения размера [citation:3]
FROM apache/airflow:slim-2.10.3

# Переключаемся на root для установки пакетов
USER root

# Опционально: установка системных пакетов, если нужно (например, для сборки psycopg2)
# RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*

# Обратно переключаемся на airflow пользователя
USER airflow

# Устанавливаем Python-зависимости с фиксированными версиями
# Это решает проблему с psycopg2 и конфликтом protobuf
RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.9 \
    mlflow==2.20.2 \
    protobuf==4.25.5