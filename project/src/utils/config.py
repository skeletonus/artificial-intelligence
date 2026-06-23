from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Загружает YAML-конфиг проекта и возвращает его как Python-словарь.
    Если путь к конфигу не передан, используется стандартный файл configs/config.yaml из корня проекта.
    Параметры:
        config_path - Путь к YAML-файлу с конфигурацией.
    Возвращает:
        Словарь с настройками проекта из YAML-файла.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def resolve_project_path(path: str | Path) -> Path:
    """
    Преобразует путь в абсолютный путь относительно корня проекта.
    Если путь уже абсолютный, возвращает его без изменений.
    Параметры:
        path - Путь к файлу или папке.
    Возвращает:
        Абсолютный путь к файлу или папке.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path