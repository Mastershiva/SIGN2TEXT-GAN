from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import os
import datetime
import threading
from googletrans import Translator

app = Flask(__name__, static_folder='static')

# Create directories for screenshots and recordings if they don't exist
os.makedirs('static/screenshots', exist_ok=True)
os.makedirs('static/recordings', exist_ok=True)

# Load the YOLOv8 model once at startup
model = YOLO("best.pt")

# Open the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Unable to open camera.")

# Global variables to hold all accumulated detection labels and state
accumulated_labels = []
is_detection_active = True
is_recording = False
video_writer = None

translator = Translator()

def start_recording():
    global is_recording, video_writer
    is_recording = True
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"static/recordings/recording_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

def stop_recording():
    global is_recording, video_writer
    is_recording = False
    if video_writer:
        video_writer.release()
        video_writer = None

def gen_frames():
    """Video streaming generator function."""
    global accumulated_labels, is_detection_active, is_recording, video_writer
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame horizontally for more intuitive user experience
        frame = cv2.flip(frame, 1)

        # Add timestamp to frame
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Run object detection on the frame if detection is active
        if is_detection_active:
            results = model.predict(frame)

            # Dictionary to track current frame detections by their left-most x coordinate
            current_detections = {}
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results[0].names[int(box.cls[0])]
                confidence = round(box.conf[0].item(), 2)
                detection_text = f"{label} {confidence:.2f}"

                # Record the left-most occurrence for each label in the current frame
                if label not in current_detections or x1 < current_detections[label]:
                    current_detections[label] = x1

                # Draw bounding box and detection text on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, detection_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Sort current frame labels left-to-right
            current_labels = [label for label, _ in sorted(current_detections.items(), key=lambda item: item[1])]

            # Update the accumulated list if any new label is found
            for label in current_labels:
                if label not in accumulated_labels:
                    accumulated_labels.append(label)

        # Add status overlay to frame
        status_text = "Active" if is_detection_active else "Paused"
        cv2.putText(frame, f"Status: {status_text}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Write frame to video if recording is active
        if is_recording and video_writer:
            video_writer.write(frame)

        # Encode the processed frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # Yield the frame for MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def landing_page():
    """Landing page."""
    return render_template('landing.html')

@app.route('/detection')
def detection_page():
    """Detection page."""
    return render_template('detection.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Use this as the src for an image tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def detections():
    """Return the accumulated detection labels as JSON."""
    return jsonify(accumulated_labels)

@app.route('/reset_detections', methods=['POST'])
def reset_detections():
    """Reset the accumulated detection labels."""
    global accumulated_labels
    accumulated_labels = []
    return jsonify({"status": "success", "message": "Detections reset successfully"})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle detection on/off."""
    global is_detection_active
    is_detection_active = not is_detection_active
    return jsonify({
        "status": "success",
        "active": is_detection_active,
        "message": f"Detection {'activated' if is_detection_active else 'paused'}"
    })

@app.route('/take_screenshot', methods=['POST'])
def take_screenshot():
    """Take a screenshot of the current frame and save it."""
    ret, frame = cap.read()
    if not ret:
        return jsonify({"status": "error", "message": "Failed to capture frame"})

    # Mirror the frame horizontally for consistency with the display
    frame = cv2.flip(frame, 1)

    # Add timestamp and detected signs to the screenshot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.jpg"
    filepath = os.path.join('static/screenshots', filename)

    # Add labels to the screenshot
    y_pos = 30
    cv2.putText(frame, f"Detected Signs: {len(accumulated_labels)}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for i, label in enumerate(accumulated_labels):
        y_pos += 30
        cv2.putText(frame, f"{i+1}. {label}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Save the screenshot
    cv2.imwrite(filepath, frame)

    return jsonify({
        "status": "success",
        "filename": filename,
        "filepath": url_for('static', filename=f'screenshots/{filename}'),
        "message": "Screenshot saved successfully"
    })

@app.route('/start_recording', methods=['POST'])
def start_recording_route():
    if not is_recording:
        threading.Thread(target=start_recording).start()
        return jsonify({"status": "success", "message": "Recording started"})
    return jsonify({"status": "error", "message": "Recording already in progress"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording_route():
    if is_recording:
        stop_recording()
        return jsonify({"status": "success", "message": "Recording stopped"})
    return jsonify({"status": "error", "message": "No recording in progress"})

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    text = data.get('text', '')
    target_language = data.get('language', 'en')

    if not text:
        return jsonify({"status": "error", "message": "No text to translate"})

    try:
        translation = translator.translate(text, dest=target_language)
        return jsonify({"status": "success", "translation": translation.text})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

@app.route('/help')
def help_page():
    """Help page."""
    return render_template('help.html')

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        cap.release()
