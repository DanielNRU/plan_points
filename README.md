# 🏠 Plan Points — Преобразование планов помещений в координаты опорных точек

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![Docker](https://img.shields.io/badge/docker-20.10-blue)](https://www.docker.com/) [![Gradio](https://img.shields.io/badge/gradio-4.44.1-orange)](https://gradio.app/) [![Ultralytics YOLO](https://img.shields.io/badge/YOLO-ultralytics-yellow)](https://github.com/ultralytics/ultralytics) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Автоматизация извлечения опорных точек и границ помещений с помощью ИИ и компьютерного зрения**

---

<table>
<tr>
<td width="120"><img src="https://xrplace.io/images/tild3265-3934-4332-b136-336334653833__logo_white_1.png" width="100" alt="XR Place Logo"></td>
<td>
<b>Заказчик:</b> <a href="https://xrplace.io">XR Place</a> — компания, создающая интерактивные 3D квартиры и дома для сайтов застройщиков. Проект позволяет получать точные координаты опорных точек по плану помещения для последующей 3D-визуализации.
</td>
</tr>
</table>

---

## 📋 Оглавление

* [О проекте](#-о-проекте)
* [Ключевые возможности](#-ключевые-возможности)
* [Архитектура и стек технологий](#-архитектура-и-стек-технологий)
* [Требования](#-требования)
* [Установка](#-установка)
* [Использование](#-использование)
* [Структура проекта](#-структура-проекта)
* [Контакты и поддержка](#-контакты-и-поддержка)

---

## 🔍 О проекте

**Plan Points** — система компьютерного зрения, которая:
- Принимает на вход изображение плана помещения.
- С помощью модели YOLO сегментирует помещения (комнаты).
- Оптимизирует найденные полигоны, оставляя только действительные углы и аппроксимируя дуги.
- Возвращает аннотированное изображение и список координат углов для каждой комнаты.

Система предназначена для автоматизации подготовки данных для 3D-визуализации интерьеров.

---

## 🚀 Ключевые возможности

* 🏠 **Детекция помещений:** автоматическое определение комнат на плане с помощью YOLO.
* 📐 **Оптимизация полигонов:** упрощение контуров, выделение ключевых углов, аппроксимация дуг.
* 🖼 **Визуализация:** аннотированное изображение с выделенными комнатами и опорными точками.
* 📄 **Экспорт координат:** генерация списка координат для каждой комнаты.
* ⚡ **Gradio-интерфейс:** удобная загрузка изображений и просмотр результатов в браузере.
* 🐳 **Контейнеризация:** быстрое развертывание через Docker.
* 🧪 **Ноутбук:** Jupyter Notebook для экспериментов и тестирования.

---

## 🏗 Архитектура и стек технологий

**Стек:** Python 3.8+, Docker, Gradio, Ultralytics YOLO, OpenCV, NumPy

| Компонент         | Технологии                                  |
|-------------------|---------------------------------------------|
| Детекция комнат   | Ultralytics YOLO (model.pt)                 |
| Обработка полигонов | OpenCV, NumPy                             |
| Веб-интерфейс     | Gradio                                      |
| Контейнеризация   | Docker                                      |
| Демо/исследования | Jupyter Notebook                            |

---

## 📋 Требования

* Python 3.8+
* Docker >= 20.10 (для контейнеризации)
* NVIDIA GPU (опционально, для ускорения инференса)
* Файл модели `model.pt` (YOLO, должен быть в папке Docker)
* Зависимости Python (см. [`Docker/requirements.txt`](Docker/requirements.txt)):
  - gradio==4.44.1
  - numpy==1.24.4
  - ultralytics==8.3.78
  - opencv-python-headless==4.10.0.84

---

## ⚙️ Установка

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/DanielNRU/plan_points.git
   cd plan_points/Docker
   ```

2. **Положите файл модели `model.pt` в папку `Docker/`** (если его нет).

3. **Соберите Docker-образ:**
   ```bash
   docker build -t plan_points .
   ```

4. **Запустите контейнер:**
   - **С поддержкой GPU (если доступно):**
     ```bash
     docker run --gpus all -p 7860:7860 plan_points
     ```
   - **Без GPU:**
     ```bash
     docker run -p 7860:7860 plan_points
     ```

5. **Откройте браузер** и перейдите по адресу:  
   [http://localhost:7860](http://localhost:7860)

---

![Gradio интерфейс](screenshot.png)

---

## 💡 Использование

1. Запустите приложение (см. раздел [Установка](#-установка)).
2. Загрузите изображение плана помещения через Gradio-интерфейс.
3. Получите результат:
   - Аннотированное изображение с выделенными комнатами и опорными точками.
   - Список координат углов для каждой обнаруженной комнаты.

---

## 📁 Структура проекта

```
plan_points/
├── Docker/
│   ├── Dockerfile                # Docker-образ для приложения
│   ├── requirements.txt          # Python-зависимости
│   ├── main.py                   # Основной Gradio-приложение
│   ├── app.py                    # Альтернативный запуск для Hugging Face Spaces
│   ├── model.pt                  # Вес модели YOLO (не хранится в репозитории)
│   └── _dockerignore             # Исключения для Docker
├── планы помещений/              # Примеры планов (подпапки: хорошие, средне, плохие)
├── PlanPoints_yolo_11.ipynb      # Jupyter Notebook для экспериментов
├── screenshot.png                # Скриншот интерфейса
├── README.md                     # Документация по проекту
└── .gitignore                    # Исключения для git
```

---

## ✉️ Контакты и поддержка

**Автор:** Мельник Даниил  
* Email: [git@danieln.ru](mailto:git@danieln.ru)  
* GitHub: [DanielNRU](https://github.com/DanielNRU)  
* Hugging Face: [DanielNRU](https://huggingface.co/DanielNRU)

---