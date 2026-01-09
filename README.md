# Д.З. №5 — PyTorch


Проект подготовлен с использованием [uv](https://github.com/astral-sh/uv) — быстрого менеджера зависимостей и виртуальных окружений от Astral.

## Быстрый старт

```bash
# Установка uv
curl -Ls https://astral.sh/uv/install.sh | bash

# Клонирование проекта
git clone git clone git@github.com:toxic-crusader/llm_hw5_pytorch.git
cd llm_hw5_pytorch

# Создание виртуального окружения и установка зависимостей
uv venv
uv sync
```

> Если используешь ноутбук: открой `notebooks/animal_faces_classification.ipynb` в Google Colab или локально.



---
# Классификация лиц животных с использованием PyTorch

## Задание

Реализовать систему классификации лиц животных (3 класса).

### 1. Подготовка данных

- Загрузить изображения.
- Привести все изображения к единому размеру.
- Учти, что оригинальное разрешение высокое — слишком простая сеть может плохо справиться.

### 2. Обучение модели

- Построить и обучить нейросеть для распознавания 3 классов.
- Желательно сравнить несколько вариантов модели или параметров.

### 3. Оценка качества

- Вычислить метрики качества отдельно для каждого класса.
- Визуализировать предсказания на тесте — с подписями, в человеко-читаемом виде.

---

## На "отлично"

- Аугментация изображений
- Файнтюнинг предобученной сверточной сети (например, ResNet, EfficientNet и др.)

---

## Подсказки

**Как найти файлы в папках:**  
[https://pythoner.name/walk](https://pythoner.name/walk)

**Как загрузить изображение по пути:**

```python
from PIL import Image
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np

img = np.asarray(Image.open(path_to_image))  
plt.imshow(img_to_array(img))
