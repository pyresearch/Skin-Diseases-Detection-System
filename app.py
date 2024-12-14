from flask import Flask, render_template, request, url_for
import os
import cv2
from ultralytics import YOLO
import supervision as sv
import pyresearch

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
model = YOLO("last.pt")

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image processing function
def process_image(input_image_path: str, output_image_path: str):
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Resize the image
    resized = cv2.resize(image, (640, 640))

    # Perform detection
    detections = sv.Detections.from_ultralytics(model(resized)[0])

    # Annotate the image
    annotated = sv.BoundingBoxAnnotator().annotate(scene=resized, detections=detections)
    annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections)

    # Save the annotated image
    cv2.imwrite(output_image_path, annotated)
    print(f"Processed and saved: {output_image_path}")

# Video processing function
def process_video(input_video_path: str, output_video_path: str):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Unable to open the video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        resized = cv2.resize(frame, (640, 640))

        # Perform detection
        detections = sv.Detections.from_ultralytics(model(resized)[0])

        # Annotate the frame
        annotated = sv.BoundingBoxAnnotator().annotate(scene=resized, detections=detections)
        annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections)

        # Write the processed frame to the output video
        out.write(annotated)

    cap.release()
    out.release()
    print(f"Processed and saved video: {output_video_path}")

# Route to handle uploads
@app.route('/', methods=['GET', 'POST'])
def upload_files():
    processed_files = []  # List to store URLs of processed files
    if request.method == 'POST':
        files = request.files.getlist('files')  # Get multiple files from the form

        for file in files:
            if file and allowed_file(file.filename):
                # Save the uploaded file
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)

                # Check if the file is an image or video
                file_extension = file.filename.rsplit('.', 1)[1].lower()
                if file_extension in {'png', 'jpg', 'jpeg'}:
                    # Define output image path
                    output_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'annotated_' + file.filename)
                    process_image(filename, output_filename)
                    processed_file_url = url_for('static', filename=f'outputs/annotated_{file.filename}')
                elif file_extension in {'mp4', 'avi', 'mov'}:
                    # Define output video path
                    output_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'annotated_' + file.filename)
                    process_video(filename, output_filename)
                    processed_file_url = url_for('static', filename=f'outputs/annotated_{file.filename}')

                # Add the processed file URL to the list
                processed_files.append(processed_file_url)

    return render_template('index.html', processed_files=processed_files)

if __name__ == "__main__":
    # Create upload and output folders if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    app.run(debug=True)
