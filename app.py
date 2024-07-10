from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

app = Flask(__name__)

# Load a pre-trained YOLOv10n model
model = YOLO("yolov10n.pt")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read image file as a NumPy array
    npimg = np.fromfile(file, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Perform object detection
    results = model(image)

    # Extract and display the detected objects
    detected_objects = results[0].boxes

    # Create a set to avoid duplicate object names
    object_names = set()
    person_detected = False

    for box in detected_objects:
        class_id = int(box.cls[0])
        object_name = results[0].names[class_id]
        object_names.add(object_name)
        if object_name == 'person':
            person_detected = True

    response = {'objects': list(object_names)}

    # Check if a person is detected in the image
    if person_detected:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=6)
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            ROI = image[y:y+h, x:x+w]
            break

        # Use EasyOCR to read text from the cropped region
        reader = easyocr.Reader(['en'])
        result = reader.readtext(ROI)
        text_results = [{'text': text, 'probability': prob} for (bbox, text, prob) in result]

        response['person_detected'] = True
        response['text_results'] = text_results
    else:
        response['person_detected'] = False

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
