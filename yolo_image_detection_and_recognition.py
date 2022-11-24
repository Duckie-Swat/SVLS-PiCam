import glob
import cv2
import numpy as np
import os
import argparse
from my_plate_recognition import PlateRecognition

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5			# cls score
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45		# obj confidence

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)
plateRecognition = PlateRecognition()

def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""

    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
    # Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

	# Sets the input to the network.
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)
	# print(outputs[0].shape)

	return outputs


def post_process(input_image, outputs):
	# Lists to hold respective values while unwrapping.
	class_ids = []
	confidences = []
	boxes = []

	# Rows.
	rows = outputs[0].shape[1]

	image_height, image_width = input_image.shape[:2]

	# Resizing factor.
	x_factor = image_width / INPUT_WIDTH
	y_factor =  image_height / INPUT_HEIGHT

	# Iterate through 25200 detections.
	for r in range(rows):
		row = outputs[0][0][r]
		confidence = row[4]

		# Discard bad detections and continue.
		if confidence >= CONFIDENCE_THRESHOLD:
			classes_scores = row[5:]

			# Get the index of max class score.
			class_id = np.argmax(classes_scores)

			#  Continue if the class score is above threshold.
			if (classes_scores[class_id] > SCORE_THRESHOLD):
				confidences.append(confidence)
				class_ids.append(class_id)

				cx, cy, w, h = row[0], row[1], row[2], row[3]

				left = int((cx - w/2) * x_factor)
				top = int((cy - h/2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)

				box = np.array([left, top, width, height])
				boxes.append(box)

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
		crop_img = input_image[top:top+height, left:left+width]
		plate_number = plateRecognition.recognize_ssd(crop_img)
		label = "{}:{:.2f}".format(plate_number, confidences[i])             
		draw_label(input_image, label, left, top)
	return input_image


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default='models/yolov6n.onnx', help="Input your onnx model.")
	parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=None)
	parser.add_argument('--save_results', help='Save labeled images and annotation data to a results folder',
                    action='store_true')
	parser.add_argument('--results_dir', help='Save labeled images and annotation data to a results folder',
                    action='store_true')
	parser.add_argument('--classesFile', default='coco.names', help="Path to your classesFile.")
	args = parser.parse_args()

	# Load class names.
	model_path, imagedir, save_results, results_dir, classesFile = args.model, args.imagedir, args.save_results, args.results_dir, args.classesFile

	if imagedir:
		CWD_PATH = os.getcwd()
		PATH_TO_IMAGES = os.path.join(CWD_PATH,imagedir)
		images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.JPG') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')
		if save_results:
			RESULTS_DIR = results_dir

	# Create results directory if user wants to save results
	if save_results:
		RESULTS_PATH = os.path.join(CWD_PATH,RESULTS_DIR)
		if not os.path.exists(RESULTS_PATH):
			os.makedirs(RESULTS_PATH)

	window_name = os.path.splitext(os.path.basename(model_path))[0]
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	# Load image.
	for img_path in images:

		frame = cv2.imread(img_path)
		input = frame.copy()
		# Give the weight files to the model and load the network using them.
		net = cv2.dnn.readNet(model_path)

		# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the
		# timings for each of the layers(in layersTimes)
		# Process image.
		cycles = 1
		total_time = 0
		for i in range(cycles):
			detections = pre_process(frame.copy(), net)
			img = post_process(frame.copy(), detections)
			t, _ = net.getPerfProfile()
			total_time += t
			# print(f'Cycle [{i + 1}]:\t{t * 1000.0 / cv2.getTickFrequency():.2f}\tms')

		avg_time = total_time / cycles
		label = 'Inference time: %.2f s' % (avg_time  / cv2.getTickFrequency())
		print(f'Model: {window_name}\n{label}')
		cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
		
		if save_results:
			image_fn = os.path.basename(img_path)
			image_savepath = os.path.join(CWD_PATH,RESULTS_DIR,image_fn)
			print(f'Saved {image_savepath}')
			cv2.imwrite(image_savepath, img)