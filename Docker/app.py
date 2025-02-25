import os
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

def simplify_polygon(points, tol_ratio=0.02):
    """
    Упрощает полигон с помощью аппроксимации по Перу.
    
    Параметры:
        points (np.ndarray): Исходный набор точек полигона (Nx2).
        tol_ratio (float): Коэффициент допуска для аппроксимации.
        
    Возвращает:
        np.ndarray: Упрощённый набор точек полигона.
    """
    pts = points.reshape((-1, 1, 2)).astype(np.float32)
    perimeter = cv2.arcLength(pts, True)
    epsilon = tol_ratio * perimeter
    approx = cv2.approxPolyDP(pts, epsilon, True)
    return approx.reshape(-1, 2)

# Загружаем модель YOLO (файл model.pt должен находиться в корневой директории Space)
model_path = os.path.join(os.getcwd(), "model.pt")
model12 = YOLO(model_path)

def process_image(image):
    """
    Обрабатывает изображение:
      - Выполняет предсказание модели YOLO для детекции комнат.
      - Рисует упрощённые полигоны (с углами) для объектов класса 'room' (id = 2).
      - Формирует текст со списком координат углов.
      
    Параметры:
        image (np.ndarray): Исходное изображение.
        
    Возвращает:
        annotated_img (np.ndarray): Изображение с нанесёнными аннотациями.
        coord_text (str): Текст с координатами углов.
    """
    pred_results = list(model12.predict(source=image, imgsz=640))
    if len(pred_results) == 0:
        return image, "Ошибка: предсказания не получены."
    
    pred = pred_results[0]
    if pred.masks is None or pred.boxes is None:
        return image, "На изображении объекты не обнаружены."
    
    # Преобразуем исходное изображение в RGB для корректного отображения
    orig_img = pred.orig_img
    if orig_img.shape[2] == 3:
        original_rgb = cv2.cvtColor(orig_img.copy(), cv2.COLOR_BGR2RGB)
    else:
        original_rgb = orig_img.copy()
    annotated_img = original_rgb.copy()
    
    # Извлекаем полигоны и классы
    polygons = pred.masks.xy
    class_ids = pred.boxes.cls.cpu().numpy()
    
    coordinates_list = []
    for idx, polygon in enumerate(polygons):
        class_id = int(class_ids[idx])
        if class_id == 2:  # Обрабатываем только объекты класса "room"
            poly_np = np.array(polygon, dtype=np.float32)
            simplified_poly = simplify_polygon(poly_np, tol_ratio=0.005)
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
            coord_text += f"Координаты углов в комнате №: {item['object']}\n"
            for coord in item['coordinates']:
                coord_text += f"({coord[0]:.2f}, {coord[1]:.2f})\n"
            coord_text += "-" * 30 + "\n"
    else:
        coord_text = "На изображении не обнаружены комнаты."
    
    return annotated_img, coord_text

# Создаем Gradio-интерфейс
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Исходное изображение"),
    outputs=[
        gr.Image(type="numpy", label="Изображение с обнаруженными комнатами"),
        gr.Textbox(label="Координаты углов комнат", lines=10)
    ],
    title="Детекция комнат с использованием YOLO и Gradio",
    description="Загрузите изображение для детекции комнат и отображения углов."
)

iface.launch()