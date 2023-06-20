import cv2
import numpy as np
from ultralytics import YOLO
from docopt import docopt
from line_processing.CameraCalibration import CameraCalibration
from line_processing.Thresholding import *
from line_processing.PerspectiveTransformation import *
from line_processing.LaneLines import *

class InlinePotholeRoad:

    def __init__(self):
        self.calibration = CameraCalibration('camera_calibration', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forwarded(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def prediction(self, model_path, video_path):
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                results = model.predict(frame,
                                        vid_stride = True,
                                        retina_masks = True,
                                        verbose = False,
                                        device = 0,
                                        conf = 0.40,
                                        imgsz = 1280)
                img = cv2.cvtColor(results[0].orig_img, cv2.COLOR_RGB2BGR)
                img_bgr = self.forwarded(img)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                annotated_frame = results[0].plot(img = img_rgb)
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

IPR = InlinePotholeRoad()

def main():
    video_path = 'input_videos/challenge_video.mp4'
    model_path = ''

    print("Which pre-trained YOLO model would you like to use?")
    print("ENTER the NUMBER of CLASSES in the terminal: 1 or 3?")
    
    while True:
        num_classes = input()

        if num_classes == "1":
            model_path = "pretrained_models/1Classes.pt"
            break
        elif num_classes == "3":
            model_path = "pretrained_models/3Classes.pt"
            break
        else:
            print("Invalid input. Please enter '1' or '3'.")

    IPR.prediction(model_path,video_path)


if __name__ == '__main__': 
    main()