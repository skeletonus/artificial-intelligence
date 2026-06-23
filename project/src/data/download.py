import shutil
import urllib.request
import zipfile
from pathlib import Path

from src.utils.config import resolve_project_path


def download_file(url: str, output_path: Path, force: bool = False) -> None:
    """
    Скачивает файл по ссылке и сохраняет его в указанный путь.
    Если файл уже существует и force=False, скачивание пропускается.
    Параметры:
        url - Ссылка на файл для скачивания.
        output_path - Путь, куда нужно сохранить скачанный файл.
        force - Нужно ли перескачать файл, если он уже существует.
    Возвращает:
        None.
    """
    if output_path.exists() and not force:
        print(f"Archive already exists: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved archive to: {output_path}")


def prepare_movielens_dataset(
    config: dict,
    dataset_key: str,
    force: bool = False,
) -> Path:
    """
    Скачивает и распаковывает выбранный датасет MovieLens в папку raw.
    Датасет выбирается по ключу из config["data"]["datasets"].
    Параметры:
        config - Словарь с настройками проекта из config.yaml.
        dataset_key - Ключ датасета в конфиге.
        force - Нужно ли пересоздать датасет, если он уже существует.
    Возвращает:
        Путь к папке raw с CSV-файлами выбранного датасета.
    """
    datasets_config = config["data"]["datasets"]

    if dataset_key not in datasets_config:
        available_datasets = ", ".join(datasets_config.keys())
        raise ValueError(
            f"Unknown dataset_key={dataset_key}. Available datasets: {available_datasets}"
        )

    dataset_config = datasets_config[dataset_key]

    dataset_name = dataset_config["name"]
    dataset_url = dataset_config["url"]
    dataset_dir = resolve_project_path(dataset_config["path"])
    raw_dir = resolve_project_path(dataset_config["raw_dir"])

    data_dir = resolve_project_path(config["paths"]["data_dir"])
    movielens_files = config["data"]["movielens_files"]

    archive_path = data_dir / f"{dataset_name}.zip"
    extract_dir = data_dir / "_tmp_extract"

    if raw_dir.exists() and all((raw_dir / file_name).exists() for file_name in movielens_files) and not force:
        print(f"Raw dataset already exists: {raw_dir}")
        return raw_dir

    if force and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_file(
        url=dataset_url,
        output_path=archive_path,
        force=force,
    )

    print("Extracting archive...")

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    extracted_dataset_dir = extract_dir / dataset_name

    for file_name in movielens_files:
        source_path = extracted_dataset_dir / file_name
        target_path = raw_dir / file_name

        if not source_path.exists():
            raise FileNotFoundError(f"Expected file not found after extraction: {source_path}")

        shutil.move(str(source_path), str(target_path))

    archive_path.unlink(missing_ok=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    print(f"Raw dataset prepared: {raw_dir}")

    return raw_dir