import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_image(image_path):
    """Detect faces in an image with confidence approximation."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        confidence = min(100, (w * h) // 300)  # Approximate confidence
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, f'{confidence}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    cv2.imwrite(result_path, image)
    return 'result.jpg'

def detect_faces_video(video_path):
    """Detect faces in a video and save the output with confidence levels."""
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_video.mp4')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            confidence = min(100, (w * h) // 300)  # Approximate confidence
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f'{confidence}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(result_path, fourcc, 20.0, (width, height))

        out.write(frame)

    cap.release()
    out.release()
    return 'result_video.mp4'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Determine if the file is an image or video
    if filename.lower().endswith(('png', 'jpg', 'jpeg')):
        result_file = detect_faces_image(file_path)
    elif filename.lower().endswith(('mp4', 'avi')):
        result_file = detect_faces_video(file_path)
    else:
        return jsonify({'error': 'Invalid file format'}), 400

    return jsonify({'result_path': result_file})

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

