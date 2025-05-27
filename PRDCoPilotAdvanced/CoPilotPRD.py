"""
Модуль для обнаружения дорожных дефектов и разметки с использованием YOLO и калибровки камеры.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from docopt import docopt

from line_processing.CameraCalibration import CameraCalibration
from line_processing.Thresholding import Thresholding
from line_processing.PerspectiveTransformation import PerspectiveTransformation
from line_processing.LaneLines import LaneLines


@dataclass
class ModelConfig:
    """Конфигурация модели YOLO."""
    conf: float = 0.40
    imgsz: int = 1280
    device: int = 0


class InlinePotholeRoad:
    """
    Класс для обработки дорожных дефектов и разметки с учетом калибровки камеры.
    """

    def __init__(self):
        """
        Инициализация компонентов обработки изображения.
        """
        self.calibration = CameraCalibration('camera_calibration', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()
        self.model_config = ModelConfig()

    def forwarded(self, img: np.ndarray) -> np.ndarray:
        """
        Обработка изображения через все этапы обработки.

        Args:
            img (np.ndarray): Входное изображение

        Returns:
            np.ndarray: Обработанное изображение с разметкой
        """
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def prediction(self, model_path: str, video_path: str) -> None:
        """
        Выполняет предсказание дорожных дефектов и разметки на видео.

        Args:
            model_path (str): Путь к предобученной модели YOLO
            video_path (str): Путь к входному видео
        """
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                results = model.predict(
                    frame,
                    vid_stride=True,
                    retina_masks=True,
                    verbose=False,
                    device=self.model_config.device,
                    conf=self.model_config.conf,
                    imgsz=self.model_config.imgsz
                )
                img = cv2.cvtColor(results[0].orig_img, cv2.COLOR_RGB2BGR)
                img_bgr = self.forwarded(img)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                annotated_frame = results[0].plot(img=img_rgb)
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


def get_model_path() -> str:
    """
    Запрашивает у пользователя выбор модели YOLO.

    Returns:
        str: Путь к выбранной модели
    """
    print("Which pre-trained YOLO model would you like to use?")
    print("ENTER the NUMBER of CLASSES in the terminal: 1 or 3?")
    
    while True:
        num_classes = input()
        if num_classes == "1":
            return "pretrained_models/1Classes.pt"
        elif num_classes == "3":
            return "pretrained_models/3Classes.pt"
        print("Invalid input. Please enter '1' or '3'.")


def main() -> None:
    """
    Основная функция программы.
    """
    video_path = 'input_videos/challenge_video.mp4'
    model_path = get_model_path()
    ipr = InlinePotholeRoad()
    ipr.prediction(model_path, video_path)


if __name__ == '__main__':
    main()