"""
Модуль для предсказания дорожных дефектов с использованием обученной модели YOLO.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


@dataclass
class PredictionConfig:
    """Конфигурация параметров предсказания."""
    model_path: str = 'runs/segment/best parameters/yolov8n_seg(1056_batch8_epoch40)best/weights/best.pt'
    video_path: str = 'D:/Projects/NeurealWorkDiplom/123.mp4'
    image_size: int = 1056
    confidence: float = 0.25
    device: int = 0
    stream: bool = True
    show: bool = True
    save: bool = False


def predict_defects(config: PredictionConfig) -> None:
    """
    Выполняет предсказание дорожных дефектов на видео.

    Args:
        config (PredictionConfig): Конфигурация параметров предсказания
    """
    model = YOLO(config.model_path)
    
    results = model(
        source=config.video_path,
        stream=config.stream,
        show=config.show,
        save=config.save,
        imgsz=config.image_size,
        conf=config.confidence,
        device=config.device,
        vid_stride=True,
        retina_masks=True
    )

    for result in results:
        boxes = result.boxes
        masks = result.masks
        probs = result.probs


if __name__ == '__main__':
    config = PredictionConfig()
    predict_defects(config)


