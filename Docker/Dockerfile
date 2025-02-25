FROM nvidia/cuda:12.6.0-runtime-ubuntu20.04

# Отключаем интерактивный режим APT
ENV DEBIAN_FRONTEND=noninteractive

# Устанавливаем Python и необходимые системные зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libglib2.0-0 \
        libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Задаем рабочую директорию
WORKDIR /app

# Копируем только файл зависимостей
COPY requirements.txt .

# Устанавливаем необходимые Python-библиотеки
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-deps ultralytics==8.3.78

# Копируем файлы проекта
COPY . .

# Открываем порт для Gradio
EXPOSE 7860

# Запускаем приложение
CMD ["python3", "main.py"]