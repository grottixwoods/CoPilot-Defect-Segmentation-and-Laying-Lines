"""
Модуль для переименования файлов в датасете с использованием уникальных идентификаторов.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Set
import os
import random
import string


@dataclass
class RenameConfig:
    """Конфигурация параметров переименования файлов."""
    id_length: int = 12
    prefix: str = "PotholesData2"
    extension: str = ".jpg"


class FileRenamer:
    """
    Класс для переименования файлов с использованием уникальных идентификаторов.
    """

    def __init__(self, directory_path: str):
        """
        Инициализация переименователя файлов.

        Args:
            directory_path (str): Путь к директории с файлами
        """
        self.directory = Path(directory_path)
        self.config = RenameConfig()
        self.existing_ids: Set[str] = set()

    def generate_unique_id(self) -> str:
        """
        Генерирует уникальный идентификатор.

        Returns:
            str: Уникальный идентификатор
        """
        while True:
            unique_id = ''.join(random.choices(string.digits, k=self.config.id_length))
            if unique_id not in self.existing_ids:
                self.existing_ids.add(unique_id)
                return unique_id

    def rename_files(self) -> None:
        """
        Переименовывает все файлы в директории.
        """
        for filename in os.listdir(self.directory):
            file_path = self.directory / filename
            if file_path.is_file():
                unique_id = self.generate_unique_id()
                new_filename = f"{self.config.prefix}{unique_id}{self.config.extension}"
                new_path = self.directory / new_filename
                
                os.rename(file_path, new_path)
                print(f"Renamed '{filename}' to '{new_filename}'")
        
        print("All files have been renamed.")


def main() -> None:
    """
    Основная функция программы.
    """
    directory_path = "C:/Users/Grotti/Desktop/DATA/dataset/"
    renamer = FileRenamer(directory_path)
    renamer.rename_files()


if __name__ == '__main__':
    main()