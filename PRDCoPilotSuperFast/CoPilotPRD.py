import cv2
from ultralytics import YOLO
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

def prediction(model_path_yolo, video_path):
    model = YOLO(model_path_yolo)
    lane_detector = UltrafastLaneDetector(model_path = "pretrained_models/tusimple_18.pth",
                                          model_type = ModelType.TUSIMPLE,
                                          use_gpu = True)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()

        try:
            success, frame = cap.read()
        except:
            continue       

        if success:
            results = model.predict(frame,
                                    verbose = False,
                                    vid_stride = True,
                                    retina_masks = True,
                                    device = 0,
                                    conf = 0.30,
                                    imgsz = 1280)
            new_frame = results[0].orig_img
            output_img = lane_detector.detect_lanes(new_frame)
            annotated_frame = results[0].plot(img = output_img)
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    model_path_yolo = ''
    video_path = 'input_videos/2.mp4'

    print("Which pre-trained YOLO model would you like to use?")
    print("ENTER the NUMBER of CLASSES in the terminal: 1 or 3?")
    
    while True:
        num_classes = input()

        if num_classes == "1":
            model_path_yolo = "pretrained_models/1Classes.pt"
            break
        elif num_classes == "3":
            model_path_yolo = "pretrained_models/3Classes.pt"
            break
        else:
            print("Invalid input. Please enter '1' or '3'.")
    
    prediction(model_path_yolo, video_path)

if __name__ == '__main__': 
    main()