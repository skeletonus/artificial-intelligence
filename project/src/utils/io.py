import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    """
    Загружает JSON-файл и возвращает его содержимое.
    Используется для чтения настроек, метаданных и пользовательских данных.
    Параметры:
        path - Путь к JSON-файлу.
    Возвращает:
        Содержимое JSON-файла.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: Path) -> None:
    """
    Сохраняет Python-объект в JSON-файл.
    Используется для сохранения метаданных, метрик и пользовательских данных.
    Параметры:
        data - Объект, который нужно сохранить.
        path - Путь для сохранения JSON-файла.
    Возвращает:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4,
        )