"""
Модуль для обнаружения дорожных дефектов и разметки с использованием YOLO и UltrafastLaneDetector.
"""

from typing import Optional

import cv2
from ultralytics import YOLO
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType


def prediction(model_path_yolo: str, video_path: str) -> None:
    """
    Выполняет предсказание дорожных дефектов и разметки на видео.

    Args:
        model_path_yolo (str): Путь к предобученной модели YOLO
        video_path (str): Путь к входному видео
    """
    model = YOLO(model_path_yolo)
    lane_detector = UltrafastLaneDetector(
        model_path="pretrained_models/tusimple_18.pth",
        model_type=ModelType.TUSIMPLE,
        use_gpu=True
    )
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        try:
            success, frame = cap.read()
            if not success:
                break

            results = model.predict(
                frame,
                verbose=False,
                vid_stride=True,
                retina_masks=True,
                device=0,
                conf=0.30,
                imgsz=1280
            )
            new_frame = results[0].orig_img
            output_img = lane_detector.detect_lanes(new_frame)
            annotated_frame = results[0].plot(img=output_img)
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception as e:
            print(f"Ошибка при обработке кадра: {e}")
            continue

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
    video_path = 'input_videos/2.mp4'
    model_path_yolo = get_model_path()
    prediction(model_path_yolo, video_path)


if __name__ == '__main__':
    main()