# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- Для части A выбран датасет STL10 для простоты
- Для части B выбран трек detection и датасет Pascal VOC
- В части А сравнивались простая CNN без аугментаций, с аугментациями и модель ResNet с дообучением классификатора и fine-tune

## 2. Среда и воспроизводимость

- Python: 3.12
- torch / torchvision: 2.10 / 0.25.0
- Устройство (CPU/GPU): GPU
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: `STL10`
- Разделение train/val/test: 60/20/20
- Базовые transforms: ToTensor, Normalize
- Augmentation transforms: RandomHorizontalFip, RandomResizedCrop
В данных 10 классов, размер изображений 96x96, 3 канала


### 3.2. Часть B: structured vision

- Датасет: `Pascal VOC`
- Трек: `detection`
- Что считается ground truth: исходные bounding boxes из разметки датасета
- Какие предсказания использовались: предсказываем bounding boxes
PascalVOC содержит 20 классов

## 4. Часть A: модели и обучение (C1-C4)

Опишите коротко и сопоставимо:

- C1 (simple-cnn-base): 3 conv blocks, 2 classificator layers
- C2 (simple-cnn-aug): та же модель, что C1, но обучение с аугментациями
- C3 (resnet18-head-only): ResNet18 с дообучением классификационной головы
- C4 (resnet18-finetune): ResNet18 с fine-tune layer 4 + fc

Дополнительно:

- Loss: CrossEntropyLoss
- Optimizer(ы): Adam
- Batch size: 64
- Epochs (макс): 12
- Критерий выбора лучшей модели: accuracy на validation

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран detection track

- Модель: FasterRCNN_ResNet50_FPN_V2
- V1: `score_threshold = 0.3`
- V2: `score_threshold = 0.7`
- Как считался IoU: отношение пересечения и объединения истинных и предсазанных боксов
- Как считались precision / recall: при IoU >= 0.5 на 200 изображениях (для экономии времени)

### Если выбран segmentation track

- Модель:
- Что считается foreground:
- V1: базовая постобработка
- V2: альтернативная постобработка
- Как считался mean IoU:
- Считались ли дополнительные pixel-level метрики:

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: [`./artifacts/runs.csv`](./artifacts/runs.csv)
- Лучшая модель части A: [`./artifacts/best_classifier.pt`](./artifacts/best_classifier.pt)
- Конфиг лучшей модели части A: [`./artifacts/best_classifier_config.json`](./artifacts/best_classifier_config.json)
- Кривые лучшего прогона классификации: [`./artifacts/figures/classification_curves_best.png`](./artifacts/figures/classification_curves_best.png)
- Сравнение C1-C4: [`./artifacts/figures/classification_compare.png`](./artifacts/figures/classification_compare.png)
- Визуализация аугментаций: [`./artifacts/figures/augmentations_preview.png`](./artifacts/figures/augmentations_preview.png)
- Визуализации второй части: 
- Примеры детекции: [`./artifacts/figures/detection_examples.png`](./artifacts/figures/detection_examples.png)
- Метрики детекции: [`./artifracts/figures/detection_metrics.png`](./artifacts/figures/detection_metrics.png)

Короткая сводка (6-10 строк):

- Лучший эксперимент части A: C4
- Лучшая `val_accuracy`: 0.964
- Итоговая `test_accuracy` лучшего классификатора: 0.949
- Что дали аугментации (C2 vs C1): повышение val_accuracy с 0.548 до 0.6
- Что дал transfer learning (C3/C4 vs C1/C2): значительное повышение accuracy с 0.6 до 0.943
- Что оказалось лучше: head-only или partial fine-tuning: partial fine-tuning
- Что показал режим V1 во второй части: низкий (0.442) precision и высокий (0.927) recall
- Что показал режим V2 во второй части: precision увеличился до 0.675, recall уменьшился до 0.865
- Как интерпретируются метрики второй части: При увеличении порога precision значительно увеличился, но при этом recall уменьшился не сильно.

## 7. Анализ

Простая CNN показывает низкую точность из-за не оптимальноц архитектуры самой сети по данную задачу. Аугментации дали небольшое улучшение точности. Pretrained ResNet18 дала знаительный прирост точности. Это произошло потому что данная сеть гораздо больше, чем простая CNN и она гораздо лучше обучена. По результатам эксперимента, head-only показал немного меньшую точность, чем partial fine-tuning. 
Во второй части работы производился инференс детекции изображений на готовой модели. Для оценки качества была использована метрика mean IoU, так как она подходит для оценки результатов детекции. При увеличении порога precision значительно увеличился, но при этом recall уменьшился не сильно. Модель часто дает ложные срабатывания, но практически не пропускает реальные объекты.


## 8. Итоговый вывод

Для классификации лучше всего взять предобученную модель ResNet18 с fine-tune. Transfer learning хорош тем, что не нужно заново переобучать всю модель, а можно просто адаптировать ее для решения новой задачи. Это сильно экономит время и вычислительные ресурсы. В задачах детекции и сегментации мы не просто классифицируем объекты. а находим их местоположение на картинке. Базовая метрика, которая используется для этих задач - это IoU. Она показывает, насколько пердсказанное положение объекта на картинке совпадает с реальным.


## 9. Приложение (опционально)

Если вы делали дополнительные сравнения:

- дополнительные fine-tuning сценарии
- confusion matrix для классификации
- дополнительная постобработка для второй части
- дополнительные графики: `./artifacts/figures/...`
