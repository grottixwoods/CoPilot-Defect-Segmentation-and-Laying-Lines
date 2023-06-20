from ultralytics import YOLO

model = YOLO('runs/segment/best parameters/yolov8n_seg(1056_batch8_epoch40)best/weights/best.pt')

if __name__ == '__main__': 


    results = model(source= 'D:/Projects/NeurealWorkDiplom/123.mp4',
                    stream = True, show = True, save = False,
                    imgsz=1056,
                    conf= 0.25,
                    device=0,
                    vid_stride= True,
                    retina_masks= True)  

    for result in results:
        boxes = result.boxes
        masks = result.masks  
        probs = result.probs  


