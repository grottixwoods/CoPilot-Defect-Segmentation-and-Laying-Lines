"""
Модуль для нормализации размеров изображений в датасете.
Вычисляет средние размеры изображений и изменяет их размеры до ближайшего кратного 32.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import os
import math

import cv2
import numpy as np


@dataclass
class ImageConfig:
    """Конфигурация параметров обработки изображений."""
    min_size: int = 640
    multiplier: int = 2
    divisor: int = 32
    supported_formats: Tuple[str, ...] = ('.jpg', '.png')


class ImageProcessor:
    """
    Класс для обработки и нормализации изображений.
    """

    def __init__(self, image_directory: str):
        """
        Инициализация процессора изображений.

        Args:
            image_directory (str): Путь к директории с изображениями
        """
        self.image_directory = Path(image_directory)
        self.config = ImageConfig()
        self.image_files = [
            f for f in os.listdir(self.image_directory)
            if f.lower().endswith(self.config.supported_formats)
        ]
        self.image_sizes: List[Tuple[int, int]] = []

    def get_image_size(self, image_path: Path) -> Optional[Tuple[int, int]]:
        """
        Получает размеры изображения.

        Args:
            image_path (Path): Путь к изображению

        Returns:
            Optional[Tuple[int, int]]: Кортеж (высота, ширина) или None в случае ошибки
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Unable to read image: {image_path}, WH is None Type")
            return None
        height, width, _ = image.shape
        return height, width

    def resize_image(self, image_path: Path, new_height: int, new_width: int) -> Optional[np.ndarray]:
        """
        Изменяет размер изображения.

        Args:
            image_path (Path): Путь к изображению
            new_height (int): Новая высота
            new_width (int): Новая ширина

        Returns:
            Optional[np.ndarray]: Измененное изображение или None в случае ошибки
        """
        image = cv2.imread(str(image_path))
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            print(f"Unable to resize image: {image_path} WH is None Type")
            return None
        return cv2.resize(image, (new_width, new_height))

    def calculate_average_sizes(self) -> Tuple[float, float]:
        """
        Вычисляет средние размеры изображений.

        Returns:
            Tuple[float, float]: Средняя высота и ширина
        """
        for image_file in self.image_files:
            image_path = self.image_directory / image_file
            size = self.get_image_size(image_path)
            if size is not None:
                self.image_sizes.append(size)

        avg_height = np.mean([size[0] for size in self.image_sizes])
        avg_width = np.mean([size[1] for size in self.image_sizes])
        print(f"AVG height = {avg_height}, AVG width = {avg_width}")

        if avg_height <= self.config.min_size or avg_width <= self.config.min_size:
            print(f"AVG height or AVG width <= {self.config.min_size}, HW multiply {self.config.multiplier}")
            avg_height *= self.config.multiplier
            avg_width *= self.config.multiplier

        return avg_height, avg_width

    def normalize_sizes(self, avg_height: float, avg_width: float) -> Tuple[int, int]:
        """
        Нормализует размеры до ближайшего кратного 32.

        Args:
            avg_height (float): Средняя высота
            avg_width (float): Средняя ширина

        Returns:
            Tuple[int, int]: Новые размеры
        """
        new_height = math.ceil((avg_height // self.config.divisor) * self.config.divisor)
        new_width = math.ceil((avg_width // self.config.divisor) * self.config.divisor)
        print(f"NEW height = {new_height}, NEW width = {new_width}")
        return new_height, new_width

    def process_images(self, new_height: int, new_width: int) -> None:
        """
        Обрабатывает все изображения в директории.

        Args:
            new_height (int): Новая высота
            new_width (int): Новая ширина
        """
        print('RESIZEING IN PROCESS')
        for image_file in self.image_files:
            image_path = self.image_directory / image_file
            resized_image = self.resize_image(image_path, new_height, new_width)
            if resized_image is not None:
                cv2.imwrite(str(image_path), resized_image)
        print(f'{len(self.image_files)} image files was resized to H:{new_height};W:{new_width}.')
        print('RESIZEING COMPLETE SUCCESSFULLY')


def main() -> None:
    """
    Основная функция программы.
    """
    image_directory = 'C:/Users/Grotti/Desktop/DATA/dataset/'
    processor = ImageProcessor(image_directory)
    avg_height, avg_width = processor.calculate_average_sizes()
    new_height, new_width = processor.normalize_sizes(avg_height, avg_width)
    processor.process_images(new_height, new_width)


if __name__ == '__main__':
    main()




