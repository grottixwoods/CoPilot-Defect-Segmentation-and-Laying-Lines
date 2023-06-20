If you want to get the pre-trained weights for the YOLOv8 model and for the parsingNet model:

https://www.dropbox.com/s/q38fun1d2c2a0ig/PretrainedModelsForPRD.zip?dl=0

Next, unzip the archive along the path:

"../CoPilot-Defect-Segmentation-and-Laying-Lines/PRDCoPilotSuperFast/pretrained_models"

And

"../CoPilot-Defect-Segmentation-and-Laying-Lines/PRDCoPilotAdvanced/pretrained_models"

The archive contains three files:
1Classes.pt -> Pretrained weights to segment all holes
2Classes.pt -> Pretrained weights for segmenting holes by class (Small/Medium/Large)
tusimpe_18.pth - Pre-trained weights for parsingNet markup line segmentation (only needed in PRDCoPilotSuperFast)