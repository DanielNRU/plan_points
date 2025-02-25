import os
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

def fit_circle(points):
    """
    Простая аппроксимация дуги: используется cv2.minEnclosingCircle,
    чтобы получить центр и радиус, охватывающие данный набор точек.

    Параметры:
        points (np.ndarray): массив точек (Nx2)

    Возвращает:
        center (tuple): (x, y) - координаты центра
        radius (float): радиус окружности
    """
    pts = np.array(points, dtype=np.float32)
    center, radius = cv2.minEnclosingCircle(pts)
    return tuple(center), radius

def simplify_polygon(points, rdp_tol=0.02, curvature_threshold=15, arc_resolution=20):
    """
    Оптимизирует набор точек полигона:
      1. Упрощает исходный полигон с помощью алгоритма RDP.
      2. Проходит по упрощённому контуру и определяет "резкие" углы,
         оставляя их как ключевые вершины.
      3. Для участков с плавным изменением направления (почти прямая линия)
         выполняется аппроксимация дугой окружности, которая затем равномерно
         интерполируется заданным числом точек.

    Параметры:
        points (np.ndarray): Исходный набор точек полигона (Nx2).
        rdp_tol (float): Допуск для алгоритма RDP (относительно периметра).
        curvature_threshold (float): Порог отклонения угла от 180° (в градусах).
            Если угол между соседними сегментами близок к 180° (разница ≤ порога),
            участок считается гладким и аппроксимируется дугой.
        arc_resolution (int): Количество точек для интерполяции дуги.

    Возвращает:
        np.ndarray: Оптимизированный массив точек полигона.
    """
    # Шаг 1. Упрощаем полигон с помощью RDP
    epsilon = rdp_tol * cv2.arcLength(points, True)
    simplified = cv2.approxPolyDP(points, epsilon=epsilon, closed=True)
    simplified = simplified.reshape(-1, 2)

    optimized = []
    n = len(simplified)
    i = 0
    while i < n:
        # Определяем предыдущую и следующую точки (с учетом замыкания контура)
        prev = simplified[i - 1]
        curr = simplified[i]
        next = simplified[(i + 1) % n]

        # Вычисляем угол между векторами (curr - prev) и (next - curr)
        v1 = curr - prev
        v2 = next - curr
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            angle_deg = 0
        else:
            cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))

        # Если угол отличается от 180° на более чем порог, считаем вершину "резкой"
        if abs(angle_deg - 180) > curvature_threshold:
            optimized.append(curr.tolist())
            i += 1
        else:
            # Группируем последовательные точки, где угол близок к 180° (гладкий участок)
            arc_points = [curr]
            j = i + 1
            while j < n:
                prev_j = simplified[j - 1]
                curr_j = simplified[j]
                next_j = simplified[(j + 1) % n]
                v1_j = curr_j - prev_j
                v2_j = next_j - curr_j
                norm_v1_j = np.linalg.norm(v1_j)
                norm_v2_j = np.linalg.norm(v2_j)
                if norm_v1_j < 1e-6 or norm_v2_j < 1e-6:
                    angle_j = 0
                else:
                    cos_angle_j = np.clip(np.dot(v1_j, v2_j) / (norm_v1_j * norm_v2_j), -1.0, 1.0)
                    angle_j = np.degrees(np.arccos(cos_angle_j))
                if abs(angle_j - 180) > curvature_threshold:
                    break
                arc_points.append(curr_j)
                j += 1

            # Если группа достаточно длинная, аппроксимируем дугу
            if len(arc_points) >= 3:
                center, radius = fit_circle(arc_points)
                # Определяем начальный и конечный углы для дуги
                start_angle = np.arctan2(arc_points[0][1] - center[1], arc_points[0][0] - center[0])
                end_angle = np.arctan2(arc_points[-1][1] - center[1], arc_points[-1][0] - center[0])
                # Корректируем диапазон углов
                if end_angle < start_angle:
                    end_angle += 2 * np.pi
                # Интерполируем дугу
                interp_angles = np.linspace(start_angle, end_angle, arc_resolution)
                arc_interp = [[int(center[0] + radius * np.cos(a)),
                               int(center[1] + radius * np.sin(a))] for a in interp_angles]
                optimized.extend(arc_interp)
            else:
                # Если группа слишком короткая, просто добавляем точки
                optimized.extend([pt.tolist() for pt in arc_points])
            i = j

    return np.array(optimized, dtype=np.int32)
model_path = os.path.join(os.getcwd(), "model.pt")
model = YOLO(model_path)

def process_image(image):
    """
    Обрабатывает изображение:
    - Выполняет предсказание модели YOLO для детекции комнат.
    - Рисует упрощённые полигоны (с углами) для объектов класса 'room' (id = 2).
    - Формирует текст со списком координат опорных точек.
    
    Параметры:
        image (np.ndarray): Исходное изображение.
    
    Возвращает:
        annotated_img (np.ndarray): Изображение с нанесёнными аннотациями.
        coord_text (str): Текстовое описание координат опорных точек.
    """
    pred_results = list(model.predict(source=image, imgsz=640))
    if len(pred_results) == 0:
        return image, "Ошибка: предсказания не получены."

    pred = pred_results[0]
    if pred.masks is None or pred.boxes is None:
        return image, "На изображении объекты не обнаружены."

    # Получаем оригинальное изображение и конвертируем его в RGB
    orig_img = pred.orig_img
    if orig_img.shape[2] == 3:
        original_rgb = cv2.cvtColor(orig_img.copy(), cv2.COLOR_BGR2RGB)
    else:
        original_rgb = orig_img.copy()

    annotated_img = original_rgb.copy()
    polygons = pred.masks.xy
    class_ids = pred.boxes.cls.cpu().numpy()

    coordinates_list = []
    for idx, polygon in enumerate(polygons):
        class_id = int(class_ids[idx])
        if class_id == 2:
            poly_np = np.array(polygon, dtype=np.float32)
            simplified_poly = simplify_polygon(poly_np, rdp_tol=0.005, curvature_threshold=15, arc_resolution=20)
            coordinates_list.append({
                "object": idx + 1,
                "class": class_id,
                "coordinates": [(float(pt[0]), float(pt[1])) for pt in simplified_poly]
            })
            cv2.polylines(annotated_img, [simplified_poly.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            for point in simplified_poly:
                cv2.circle(annotated_img, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)

    if coordinates_list:
        coord_text = ""
        for item in coordinates_list:
            coord_text += f"Координаты опорных точек в комнате №: {item['object']}\n"
            for coord in item['coordinates']:
                coord_text += f"({coord[0]:.2f}, {coord[1]:.2f})\n"
            coord_text += "-" * 30 + "\n"
    else:
        coord_text = "На изображении не обнаружены комнаты."

    return annotated_img, coord_text

# Создаем интерфейс Gradio (выводим аннотированное изображение и текст с координатами)
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Исходное изображение"),
    outputs=[
        gr.Image(type="numpy", label="Изображение с обнаруженными комнатами"),
        gr.Textbox(label="Координаты опорных точек комнат", lines=10)
    ],
    title="Детекция комнат с использованием YOLO и Gradio",
    description="Загрузите изображение для детекции комнат и отображения опорных точек."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)