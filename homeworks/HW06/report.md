# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-02.csv`
- Размер: (18000, 39)
- Целевая переменная:

    target
0    0.737389
1    0.262611
- Признаки:

    0   id       18000 non-null  int64  
 1   f01      18000 non-null  float64
 2   f02      18000 non-null  float64
 3   f03      18000 non-null  float64
 4   f04      18000 non-null  float64
 5   f05      18000 non-null  float64
 6   f06      18000 non-null  float64
 7   f07      18000 non-null  float64
 8   f08      18000 non-null  float64
 9   f09      18000 non-null  float64
 10  f10      18000 non-null  float64
 11  f11      18000 non-null  float64
 12  f12      18000 non-null  float64
 13  f13      18000 non-null  float64
 14  f14      18000 non-null  float64
 15  f15      18000 non-null  float64
 16  f16      18000 non-null  float64
 17  f17      18000 non-null  float64
 18  f18      18000 non-null  float64
 19  f19      18000 non-null  float64
 20  f20      18000 non-null  float64
 21  f21      18000 non-null  float64
 22  f22      18000 non-null  float64
 23  f23      18000 non-null  float64
 24  f24      18000 non-null  float64
 25  f25      18000 non-null  float64
 26  f26      18000 non-null  float64
 27  f27      18000 non-null  float64
 28  f28      18000 non-null  float64
 29  f29      18000 non-null  float64
 30  f30      18000 non-null  float64
 31  f31      18000 non-null  float64
 32  f32      18000 non-null  float64
 33  f33      18000 non-null  float64
 34  f34      18000 non-null  float64
 35  f35      18000 non-null  float64
 36  x_int_1  18000 non-null  float64
 37  x_int_2  18000 non-null  float64

## 2. Protocol

- Разбиение: `test_size` = 0.2, `random_state` = 42
- Подбор: `cv` = 5
- Метрики: accuracy(интуитивно понятная метрика), F1(не зависит от распределения классов), ROC-AUC(показывает уверенность модели)

## 3. Models

Сравнивались следующие модели:

- DummyClassifier
- LogisticRegression (с подбором гиперпараметра `С`)
- DecisionTreeClassifier (контроль сложности: `ccp_alpha`)
- RandomForestClassifier (с подбором гиперпараметра `max_features`)
- boosting AdaBoost

## 4. Results

- Cписок финальных метрик на test по всем моделям:

    {
        "model": "DummyClassifier",
        "accuracy_score": 0.7375,
        "roc_auc_score": 0.5,
        "f1_score": 0.0,
        "confusion_matrix": "[[2655    0]\n [ 945    0]]"
    },
    {
        "model": "LogisticRegression",
        "accuracy_score": 0.8119444444444445,
        "roc_auc_score": 0.7976938789744817,
        "f1_score": 0.5606748864373783,
        "confusion_matrix": "[[2491  164]\n [ 513  432]]"
    },
    {
        "model": "DecisionTreeClassifier",
        "accuracy_score": 0.8405555555555555,
        "roc_auc_score": 0.8399553602566784,
        "f1_score": 0.6631455399061033,
        "confusion_matrix": "[[2461  194]\n [ 380  565]]",
        "node_count": 279
    },
    {
        "model": "RandomForestClassifier",
        "accuracy_score": 0.8905555555555555,
        "roc_auc_score": 0.9261519146264909,
        "f1_score": 0.7555831265508685,
        "confusion_matrix": "[[2597   58]\n [ 336  609]]"
    },
    {
        "model": "AdaBoostClassifier",
        "accuracy_score": 0.8108333333333333,
        "roc_auc_score": 0.8208549307984336,
        "f1_score": 0.5260960334029228,
        "confusion_matrix": "[[2541  114]\n [ 567  378]]"
    }

- Победитель по ROC-AUC или по согласованному критерию и краткое объяснение:
    RandomForestClassifier
    Данная модель оказалась лучшей, тк основана на идее ансамблей, нелинейна, имеет сложные зависимости и устойчива к шумам. Также был произведен подбор гиперпараметра max_features и использовалась кросс-валидация.

## 5. Analysis

- Устойчивость: При изменении `random_state` все модели могут показывать разные метрики, когда `random_forest` будет выдавать стабильный результат. Это связано с тем, что `random_forest` имеет устойчивость к шумам, тк состоит из сотен усредненных моделей, когда остальные зависят от инициализации.

- Ошибки: 
    confusion matrix:

    [[2597   58]
 [ 336  609]]

    среди всех моделей random forest имеет "лучшую" матрицу

- Интерпретация: 
    permutation importance

    f16
f01
f19
f12
f07
f23
f02
f30
f08
f18

    Самым влиятельным признаком оказался f16, когда остальные имеют примерно одинаковые влияние

## 6. Conclusion

Деревья - интуитивно понятный пример нелинейных моделей. Ансабль - на практике полезная идея, помогающая создать модель, показывающую результаты, превосходящие одиночную модель. Один из самых понятных примеров ансамблей - лес. Эта модель обладает многими полезными качествами, описанными выше. 
