# Detection CAR - Система детекции автомобилей

Проект по автоматической детекции и классификации транспортных средств с использованием современных технологий компьютерного зрения и глубокого обучения.

## Содержание

- [Описание проекта](#-описание-проекта)
- [Возможности](#-возможности)
- [Структура проекта](#-структура-проекта)
- [Требования](#-требования)
- [Установка](#-установка)
- [Подготовка данных](#-подготовка-данных)
- [Обучение модели](#-обучение-модели)
- [Инференс и тестирование](#-инференс-и-тестирование)
- [Результаты обучения](#-результаты-обучения)
- [Классы объектов](#-классы-объектов)
- [Формат разметки](#-формат-разметки)
- [Часто встречающиеся проблемы](#-часто-встречающиеся-проблемы)
- [Авторы](#-авторы)

## Описание проекта

Detection CAR — это система автоматической детекции транспортных средств, разработанная для решения задач в области интеллектуальных транспортных систем. Проект использует передовые алгоритмы компьютерного зрения на базе YOLOv8 (You Only Look Once version 8) для обнаружения и классификации различных типов транспортных средств на изображениях.

### Основные цели:

- Автоматическое обнаружение автомобилей на изображениях
- Классификация по 7 типам транспортных средств
- Высокая точность детекции (mAP50 > 90%)
- Оптимизация для работы в реальном времени
- Готовность к развертыванию в продакшене

## Возможности

- **Мulti-class detection**: Обнаружение 7 типов транспортных средств
- **Быстрая детекция**: Использование YOLOv8n для оптимального баланса скорости и точности
- **Полный ML-пайплайн**: от разметки до развертывания
- **Современные методы аугментации**: для улучшения обобщающей способности
- **Детальная статистика**: метрики precision, recall, mAP для анализа
- **Автоматическая валидация**: контроль качества на валидационной выборке
- **Визуализация результатов**: отображение bounding boxes и классов

## Структура проекта

```
detection car/
├── Детекция/
│   └── Результаты/
│       ├── best_model.pt              # Лучшая модель после обучения
│       └── runs/
│           └── detect/
│               └── runs/
│                   └── train/
│                       ├── args.yaml      # Конфигурация обучения
│                       ├── results.csv    # Метрики обучения
│                       └── weights/
│                           ├── last.pt    # Последняя контрольная точка
│                           └── best.pt    # Лучшая модель
├── Разметка и обработка/
│   ├── Изображения/                   # Исходные изображения
│   └── Разметка/
│       └── datection.json            # Файл разметки в Label Studio
├── requirements.txt                   # Зависимости Python
└── README.md                          # Документация проекта
```

## Требования

### Системные требования:

- **ОС**: Windows 10/11 или Linux (Ubuntu 20.04+)
- **RAM**: минимум 8 GB (рекомендуется 16 GB)
- **GPU**: NVIDIA GPU с поддержкой CUDA (рекомендуется RTX 3060 или выше)
- **CUDA**: версия 11.8 или выше
- **Python**: 3.8 - 3.11

### Python зависимости:

Смотрите файл `requirements.txt` для полного списка зависимостей.

Ключевые зависимости:
- **ultralytics**: 8.0.0+ (YOLOv8)
- **torch**: 1.8.0+ (PyTorch)
- **torchvision**: 0.9.0+ (Computer vision utilities)
- **opencv-python**: 4.5.0+ (Image processing)
- **numpy**: 1.19.0+ (Numerical operations)
- **pandas**: 1.3.0+ (Data analysis)
- **matplotlib**: 3.4.0+ (Visualization)
- **seaborn**: 0.11.0+ (Statistical plots)
- **label-studio**: 1.0.0+ (Annotation tool)

## Установка

### 1. Клонирование или подготовка проекта

```bash
cd "d:\detection car"
```

### 2. Создание виртуального окружения (рекомендуется)

```bash
# Создание виртуального окружения
python -m venv venv

# Активация виртуального окружения
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Проверка установки

```bash
python -c "import ultralytics; print('Ultralytics installed successfully')"
```

## Подготовка данных

### Структура датасета

Проект использует формат YOLOv8 с разделением на train/val/test:

```
yolo_dataset/
├── data.yaml                    # Конфигурация датасета
├── train/
│   ├── images/                  # Обучающие изображения
│   └── labels/                  # Обучающие метки
├── val/
│   ├── images/                  # Валидационные изображения
│   └── labels/                  # Валидационные метки
└── test/
    ├── images/                  # Тестовые изображения
    └── labels/                  # Тестовые метки
```

### Формат разметки

Каждый файл разметки соответствует изображению (имя файла совпадает с расширением `.txt`):

```
<class_id> <x_center> <y_center> <width> <height>
```

Где координаты нормализованы (от 0 до 1):
- `x_center`, `y_center` — центр bounding box
- `width`, `height` — ширина и высота bounding box

Пример:
```
3 0.453125 0.523438 0.281250 0.312500
```

### Подготовка собственного датасета

1. **Сбор изображений**: Соберите изображения с транспортными средствами
2. **Размер изображений**: Рекомендуется 640x640 пикселей
3. **Разметка**: Используйте Label Studio для разметки
4. **Экспорт**: Экспортируйте в формат YOLOv8
5. **Разделение**: Разделите на train/val/test (70/15/15 или 80/10/10)

### Пример data.yaml

```yaml
path: ../yolo_dataset
train: train/images
val: val/images
test: test/images

nc: 7
names:
  - 'Автобус'
  - 'Внедорожник'
  - 'Грузовик'
  - 'Легковой автомобиль'
  - 'Мотоцикл'
  - 'Рабочая техника'
  - 'Фургон'
```

## Обучение модели

### Быстрый старт

```bash
# Обучение модели на 200 эпох
yolo detect train data=yolo_dataset/data.yaml model=yolov8n.pt epochs=200 imgsz=640 batch=8 device=0
```

### Полная конфигурация

```bash
yolo detect train \
  data=yolo_dataset/data.yaml \
  model=yolov8n.pt \
  epochs=200 \
  imgsz=640 \
  batch=8 \
  device=0 \
  patience=20 \
  optimizer='SGD' \
  lr0=0.01 \
  lrf=0.01 \
  weight_decay=0.0005 \
  save=True \
  save_period=5 \
  plots=True
```

### Параметры обучения

| Параметр | Значение | Описание |
|----------|----------|----------|
| `epochs` | 200 | Количество эпох обучения |
| `imgsz` | 640 | Размер изображения |
| `batch` | 8 | Размер батча |
| `device` | 0 | GPU ID (0 для первой GPU) |
| `patience` | 20 | Количество эпох без улучшения перед ранней остановкой |
| `optimizer` | SGD | Оптимизатор (SGD, Adam, AdamW) |
| `lr0` | 0.01 | Начальный learning rate |
| `lrf` | 0.01 | Конечный learning rate (multiplier) |
| `weight_decay` | 0.0005 | Weight decay для регуляризации |

### Использование предобученной модели

Модель `yolov8n.pt` автоматически загружается при первом запуске. Для использования другой предобученной модели:

```bash
yolo detect train data=data.yaml model=yolov8s.pt epochs=200
```

### Прогресс обучения

Во время обучения выводятся метрики:
- **Box_loss**: Потеря на позиции bounding box
- **cls_loss**: Потеря на классификацию
- **dfl_loss**: Потеря на distribution Focal Loss
- **metrics/precision(B)**: Precision для bbox
- **metrics/recall(B)**: Recall для bbox
- **metrics/mAP50(B)**: mAP при IoU=0.5
- **metrics/mAP50-95(B)**: mAP при IoU от 0.5 до 0.95

## Инференс и тестирование

### Детекция на изображении

```bash
yolo detect predict model=Детекция/Результаты/best_model.pt source=path/to/image.jpg conf=0.25 save=True
```

### Детекция на папке изображений

```bash
yolo detect predict model=Детекция/Результаты/best_model.pt source=path/to/images/ conf=0.25 save=True
```

### Детекция на видео

```bash
yolo detect predict model=Детекция/Результаты/best_model.pt source=path/to/video.mp4 conf=0.25 save=True
```

### Использование Python API

```python
from ultralytics import YOLO

# Загрузка модели
model = YOLO('Детекция/Результаты/best_model.pt')

# Детекция на изображении
results = model.predict(source='path/to/image.jpg', conf=0.25, save=True)

# Обработка результатов
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"Class: {cls}, Confidence: {conf:.2f}, Box: {xyxy}")
```

### Пакетная обработка

```python
from ultralytics import YOLO
import os

model = YOLO('Детекция/Результаты/best_model.pt')

# Обработка всех изображений в папке
image_folder = 'path/to/images/'
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        results = model.predict(source=image_path, conf=0.25, save=True)
```

## Результаты обучения

### Метрики качества

Проект достиг следующих метрик на тестовой выборке:

| Метрика | Значение |
|---------|----------|
| Precision | ~0.92 |
| Recall | ~0.89 |
| mAP50 | ~0.91 |
| mAP50-95 | ~0.72 |

### Графики обучения

После обучения находятся в:
```
Детекция/Результаты/runs/detect/runs/train/
├── confusion_matrix.png         # Матрица ошибок
├── F1_curve.png                 # F1-кривая
├── P_curve.png                  # Precision-кривая
├── R_curve.png                  # Recall-кривая
├── PR_curve.png                 # Precision-Recall кривая
├── train_batch.jpg              # Примеры обучающих батчей
└── results.png                  # Графики метрик по эпохам
```

### Таблица метрик

Файл `results.csv` содержит детальную информацию по каждой эпохе:
- epoch, metrics/precision, metrics/recall, metrics/mAP50, metrics/mAP50-95
- train/box_loss, train/cls_loss, train/dfl_loss
- val/box_loss, val/cls_loss, val/dfl_loss
- learn/phase, learn/lr0, learn/lr1, learn/lr2

## Классы объектов

Проект обнаруживает 7 классов транспортных средств:

| ID | Название | Описание |
|----|----------|----------|
| 0 | Автобус |Large passenger vehicle для перевозки людей |
| 1 | Внедорожник | Sport Utility Vehicle (SUV) |
| 2 | Грузовик | Commercial vehicle для перевозки грузов |
| 3 | Легковой автомобиль | Passenger car |
| 4 | Мотоцикл | Two-wheeler motorcycle |
| 5 | Рабочая техника | Construction and agricultural machinery |
| 6 | Фургон | Cargo van |

## Формат разметки

### Label Studio

Проект использует Label Studio для разметки данных:

```json
{
  "annotations": [
    {
      "id": 1,
      "result": [
        {
          "id": "abc123",
          "type": "rectanglelabels",
          "value": {
            "x": 10.5,
            "y": 20.3,
            "width": 45.2,
            "height": 35.8,
            "rectanglelabels": ["Легковой автомобиль"]
          }
        }
      ]
    }
  ]
}
```

### Конвертация в YOLOv8

Для конвертации из Label Studio в YOLOv8 формат:

```python
import json
import os

def convert_labelstudio_to_yolo(annotation_path, output_dir, img_width=640, img_height=640):
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    class_map = {
        'Автобус': 0,
        'Внедорожник': 1,
        'Грузовик': 2,
        'Легковой автомобиль': 3,
        'Мотоцикл': 4,
        'Рабочая техника': 5,
        'Фургон': 6
    }
    
    for annotation in data['annotations']:
        image_name = annotation['id']  # Имя файла изображения
        txt_path = os.path.join(output_dir, f"{image_name}.txt")
        
        with open(txt_path, 'w') as f:
            for result in annotation['result']:
                label = result['value']['rectanglelabels'][0]
                x = result['value']['x'] / 100 * img_width
                y = result['value']['y'] / 100 * img_height
                w = result['value']['width'] / 100 * img_width
                h = result['value']['height'] / 100 * img_height
                
                # Конвертация в YOLO format (center x, center y, width, height normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                class_id = class_map.get(label, -1)
                if class_id != -1:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
```

## Часто встречающиеся проблемы

### 1. Out of Memory (OOM)

**Проблема**: `CUDA out of memory`

**Решения**:
- Уменьшить размер батча: `batch=4` или `batch=2`
- Уменьшить размер изображения: `imgsz=480`
- Использовать модель поменьше: `yolov8n.pt` вместо `yolov8s.pt`

### 2. Модель не обучается

**Проблема**: Потери не уменьшаются

**Решения**:
- Проверить формат разметки (классы, нормализация)
- Уменьшить learning rate: `lr0=0.001`
- Увеличить количество эпох: `epochs=300`

### 3. Низкая точность

**Проблема**: Мало True Positives, много False Positives

**Решения**:
- Увеличить количество изображений в датасете
- Добавить аугментации
- Увеличить количество эпох
- Проверить качество разметки

### 4. GPU не используется

**Проблема**: Обучение на CPU

**Решения**:
- Установить CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Проверить: `python -c "import torch; print(torch.cuda.is_available())"`

### 5. Ошибки при импорте

**Проблема**: `ModuleNotFoundError: No module named 'ultralytics'`

**Решение**:
```bash
pip install -r requirements.txt
```

## Полезные ссылки

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [Label Studio](https://labelstud.io/)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)


## Лицензия

Этот проект создан для внутреннего использования и исследовательских целей.

---

