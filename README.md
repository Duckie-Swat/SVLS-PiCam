# SSD

## Sample scripts for detect images SSD
python3 TFLite_detection_image.py --modeldir=models --graph=detect_quant_v2.tflite --labels=labelmap.txt --imagedir=samples/images --save_results --results_dir=Results/SSD_MobileNetV2/images/detection --threshold 0.4

## Sample scripts for detect and recognition images SSD
python3 TFLite_detection_and_recognition_image.py --modeldir=models --graph=detect_quant_v2.tflite --labels=labelmap.txt --imagedir=samples/images --save_results --results_dir Results/SSD_MobileNetV2/images/OCR  --threshold 0.4

## Sample scripts for detect a video SSD
python3 TFLite_detection_video.py --modeldir=models --graph=detect_quant_v2.tflite --labels=labelmap.txt --video=samples/videos/IMG_0084.MOV --outputdir=Results/SSD_MobileNetV2/videos/detection --threshold 0.4

## Sample scripts for detect and recognition a video SSD
python3 TFLite_detection_video_and_recognition.py --modeldir=models --graph=detect_quant_v2.tflite --labels=labelmap.txt --video=samples/videos/IMG_0084.MOV --outputdir=Results/SSD_MobileNetV2/videos/OCR --threshold 0.4


# Yolov6
## Sample script for detect a image Yolov6
python3 yolo_image_detection.py --model models/yolov6_opt.onnx --imagedir samples/images --save_results --results_dir Results/Yolov6/images/detection --classesFile models/labelmap.txt

## Sample script for detect a image and recognition Yolov6
python3 yolo_image_detection_and_recognition.py --model models/yolov6_opt.onnx --imagedir samples/images --save_results --results_dir Results/Yolov6/images/OCR --classesFile models/labelmap.txt

## Sample script for detect a video Yolov6
python3 yolo_video_detection.py --model models/yolov6_opt.onnx --source samples/videos/IMG_0084.MOV --output Results/Yolov6/videos/detection/IMG_0084.mp4  --classesFile models/labelmap.txt
## Sample script for detect and recognition a video Yolov6
python3 yolo_video_detection_and_recognition.py --model models/yolov6_opt.onnx --source samples/videos/IMG_0084.MOV --output Results/Yolov6/videos/OCR/IMG_0084.mp4  --classesFile models/labelmap.txt
# Results
## Result with SSD MobileNet V2 for detection and OCR
![frame100](https://user-images.githubusercontent.com/79694464/203388398-b3f106d8-d313-4fa4-be6e-b284c209a301.jpg)