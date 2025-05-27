"""
Модуль для обучения модели YOLO на датасете дорожных дефектов.
"""

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO


@dataclass
class TrainingConfig:
    """Конфигурация параметров обучения."""
    model_path: str = 'models_yolo/yolov8m-seg.pt'
    data_path: str = 'dataset/data.yaml'
    image_size: int = 1056
    epochs: int = 40
    batch_size: int = 4
    experiment_name: str = 'yolov8m_seg(1056_40_8)'
    device: int = 0


def train_model(config: TrainingConfig) -> None:
    """
    Обучает модель YOLO на датасете дорожных дефектов.

    Args:
        config (TrainingConfig): Конфигурация параметров обучения
    """
    model = YOLO(config.model_path)
    
    results = model.train(
        data=config.data_path,
        imgsz=config.image_size,
        epochs=config.epochs,
        batch=config.batch_size,
        name=config.experiment_name,
        device=config.device
    )


if __name__ == '__main__':
    config = TrainingConfig()
    train_model(config)
      