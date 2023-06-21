##  Привет, меня зовут Сергей!👋🏻

###  Я студент Московского Технического Университета Связи и Информатики!👨‍🎓💻

---

Данный проект является программной реализацией моей **выпускной квалификационной работы**

 на тему: _"Исследование и разработка интеллектуальной системы выявления опасных участков дороги_

 _для повышения безопасности дорожного движения_".

<div align="left">
    <img src="https://github.com/grottixwoods/CoPilot-Defect-Segmentation-and-Laying-Lines/assets/55210700/4cbaa493-bd26-48c7-9c81-54e17b98f142" alt="PRD" width="260px">
    <img src="https://github.com/grottixwoods/CoPilot-Defect-Segmentation-and-Laying-Lines/assets/55210700/c97ce96b-73bc-4e53-9aee-395b456ea5a5" alt="ADV" width="260px">
    <img src="https://github.com/grottixwoods/CoPilot-Defect-Segmentation-and-Laying-Lines/assets/55210700/34f6c9bd-1583-4c25-9d31-128aa0888c94" alt="SF" width="260px">
</div>

---

В рамках данной выпускной квалификационной работы:

*   [x] Был реализован набор данных для сегментации дефектов дорожного покрытия, таких как ямы и выбоины;
*   [x] Осуществлен выбор эффективной модели YOLOv8 на основе анализа и подбора оптимальных параметров обучения;
*   [x] Произведена интеграция конечного алгоритма в системы помощи водителю Advanced Lane Lines и Ultra Fast Lane Detection.

---

Финальный проект имеет в своем составе папку:

1.   «PotholesRoadDetection» с модулями для обучения, анализа и тестирования модели, а также набором данных
2.   «PRDCoPilotAdvanced» в котором была проведена интеграция с проектом Advanced Lane Lines
3.   «PRDCoPilotSuperFast» в котором проведена интеграция с проектом Ultra Fast Lane Detection.

 В папках «PRDCoPilotAdvanced» и «PRDCoPilotSuperFast» хранятся все необхоимые модули для запуска программы,

 а так же исполняемый файл «CoPilotPRD.py»

---

Для работы с проектом необходимо скачать следующие архивы: 

1.  [**Сформированный набор данных для обучения модели YOLOv8**](https://www.dropbox.com/s/m5kcwq3ukvwc9fj/PotholesRoadSegmentationDataset.zip?dl=0) 
2.  [**Предобученные веса модели YOLOv8**](https://www.dropbox.com/s/q38fun1d2c2a0ig/PretrainedModelsForPRD.zip?dl=0) 
3.  [**Модели YOLOv8 с оффициального сайта Ultralytics**](https://www.dropbox.com/s/2c7jz0d7lq7tyan/Models_YOLOv8.zip?dl=0) 

---

Далее необходимо разархивировать их в следующие директории:

*   Датасет: “.../CoPilot-Defect-Segmentation-and-Laying-Lines/PotholesRoadDetection/dataset”
*   Веса: “../CoPilot-Defect-Segmentation-and-Laying-Lines/PRDCoPilotSuperFast/pretrained\_models”
*   Веса: “../CoPilot-Defect-Segmentation-and-Laying-Lines/PRDCoPilotAdvanced/pretrained\_models”
*   Модели: "../CoPilot-Defect-Segmentation-and-Laying-Lines/PotholesRoadDetection/models\_yolo".

---

Следущим шагом необходимо установить необхоимые библиотеки:

    `pip install requirements.txt`

Так же убедитесь что у вас есть поддержка CUDA 12.1, если хотите осуществить запуск программы через GPU,

в противном случае измените переменную `device = cpu` в методе `model.predict()`

---
