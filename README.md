# Sample scripts for detect and recognition images
python3 TFLite_detection_and_recognition_image.py --modeldir=models --graph=detect_quant_v2.tflite --labels=labelmap.txt --imagedir=samples_img --save_results --threshold 0.4

# Sample scripts for detect a video
python3 TFLite_detection_video.py --modeldir=models --graph=detect_quant_v2.tflite --labels=labelmap.txt --video=samples/IMG_0086.MOV --outputdir=output_videos --threshold 0.4