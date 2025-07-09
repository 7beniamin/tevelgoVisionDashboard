from flask import Flask, render_template, request, redirect, url_for, flash
import os
import json
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from azure.data.tables import TableServiceClient, UpdateMode
from dotenv import load_dotenv
import requests
import cv2
import io
from datetime import datetime, timedelta
# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key")  # Secret key for session security

# Initialize InsightFace face analysis model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(1024, 1024))

# Azure Table Storage configuration
AZURE_TABLE_CONN_STR = os.getenv('AZURE_TABLE_CONNECTION_STRING')  # Connection string to Azure Table Storage
AZURE_VEHICLE_TABLE_NAME = os.getenv('AZURE_VEHICLE_TABLE', 'AuthorizedVehicles')  # Table for authorized vehicles
AZURE_PEDESTRIAN_TABLE_NAME = os.getenv('AZURE_PEDESTRIAN_TABLE', 'AuthorizedPedestrians')  # Table for authorized pedestrians
AZURE_VEHICLE_LOGS_NAME = os.getenv('AZURE_VEHICLE_TABLE', 'VehicleLogs')  # Table for vehicle verification logs
AZURE_PEDESTRIAN_LOGS_NAME = os.getenv('AZURE_PEDESTRIAN_TABLE', 'PedestrianLogs')  # Table for pedestrian verification logs

# Create Azure Table clients for each table
table_service = TableServiceClient.from_connection_string(AZURE_TABLE_CONN_STR)
vehicle_table_client = table_service.get_table_client(table_name=AZURE_VEHICLE_TABLE_NAME)
pedestrian_table_client = table_service.get_table_client(table_name=AZURE_PEDESTRIAN_TABLE_NAME)
vehicle_table = table_service.get_table_client(table_name=AZURE_VEHICLE_LOGS_NAME)
pedestrian_table = table_service.get_table_client(table_name=AZURE_PEDESTRIAN_LOGS_NAME)

# Base URL for FastAPI backend
#ASTAPI_HOST = os.getenv("FASTAPI_HOST", "http://192.168.10.16:8000")#loacal deployment
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "http://20.51.109.105:4200")#vm wali deployment
VEHICLE_LOGS_API = f"{FASTAPI_HOST}/logs/vehicles"
PEDESTRIAN_LOGS_API = f"{FASTAPI_HOST}/logs/pedestrians"

# Reinitialize InsightFace model with specific settings (redundant setup)
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Decode image bytes into OpenCV BGR format for face detection
def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# Generate face embedding vector from an input image
def get_face_embedding(image_bytes: bytes) -> list:
    img = decode_image_bytes(image_bytes)

    # Apply sharpening filter to enhance image quality
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    img = cv2.filter2D(img, -1, sharpen_kernel)

    # Resize image if it is too small for face detection
    h, w = img.shape[:2]
    if max(h, w) < 800:
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Detect faces in the image
    faces = face_app.get(img)
    if not faces:
        raise ValueError("No face detected in the image. Please upload a clear face picture.")

    # Select the largest face in case of multiple detections
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.embedding.tolist()

# Fetch both vehicle and pedestrian logs from FastAPI backend

def fetch_fastapi_logs():
    try:
        vehicle_logs = requests.get(VEHICLE_LOGS_API).json()
        pedestrian_logs = requests.get(PEDESTRIAN_LOGS_API).json()
        return vehicle_logs, pedestrian_logs
    except Exception as e:
        print("⚠️ Failed to fetch FastAPI logs:", e)
        return [], []

# Process vehicle logs and return aggregated statistics
def process_vehicle_stats(logs):
    stats = {"success": 0, "failed": 0, "mismatch": 0}
    for log in logs:
        make_status = (log.get("car_make_status") or "").lower()
        model_status = (log.get("car_model_status") or "").lower()
        status = (log.get("output") or log.get("status") or "").lower()

        # Final outcome (only Authorized/Unauthorized)
        if status == "authorized":
            stats["success"] += 1
            if make_status == "mismatch" or model_status == "mismatch":
                stats["mismatch"] += 1
        elif status == "unauthorized":
            stats["failed"] += 1
            if make_status == "mismatch" or model_status == "mismatch":
                stats["mismatch"] += 1
    return stats


# Process pedestrian logs and return success/failure counts
def process_pedestrian_stats(logs):
    stats = {"success": 0, "failed": 0, "not_found": 0}
    for log in logs:
        status = (log.get("status") or log.get("output") or "").lower()
        face_status = (log.get("face_status") or log.get("face") or "").lower()
        if status == "authorized":
            stats["success"] += 1
        elif status == "unauthorized":
            stats["failed"] += 1
            if face_status == "not found":
                stats["not_found"] += 1
    return stats




# Utility function to count total records in a given Azure Table
def count_table_entities(table_client):
    return sum(1 for _ in table_client.list_entities())


def get_latest_logs(logs, count=20):
    if not logs:
        return []
    try:
        logs_sorted = sorted(
            logs,
            key=lambda x: datetime.fromisoformat(x.get("timestamp", "").replace("Z", "+00:00")),
            reverse=True
        )
        return logs_sorted[:count]
    except Exception as e:
        print(f"[ERROR] Could not sort logs: {e}")
        return logs[:count]




# Home page route - displays stats, totals, and logs
@app.route('/')
def index():
    vehicle_logs, pedestrian_logs = fetch_fastapi_logs()
    vehicle_stats = process_vehicle_stats(vehicle_logs)
    pedestrian_stats = process_pedestrian_stats(pedestrian_logs)

    total_vehicles = count_table_entities(vehicle_table_client)
    total_pedestrians = count_table_entities(pedestrian_table_client)

    return render_template('index.html',
                           vehicle_stats=vehicle_stats,
                           pedestrian_stats=pedestrian_stats,
                           total_vehicles=total_vehicles,
                           total_pedestrians=total_pedestrians,
                           vehicle_logs=vehicle_logs,
                           pedestrian_logs=pedestrian_logs)

@app.route('/viewPedestrainLogs')
def pedestrainLogs():
    _, pedestrian_logs = fetch_fastapi_logs()
    latest_pedestrian_logs = get_latest_logs(pedestrian_logs, 20)
    return render_template(
        'pedestrian_logs.html',
        pedestrian_logs=latest_pedestrian_logs
    )

@app.route('/viewVehicleLogs')
def VehicleLogs():
    vehicle_logs, _ = fetch_fastapi_logs()
    latest_vehicle_logs = get_latest_logs(vehicle_logs, 20)
    # Pass only the latest, correctly parsed logs
    return render_template(
        'vehicle_logs.html',
        vehicle_logs=latest_vehicle_logs
    )


# Page to render "Add Pedestrian" form
@app.route('/add_pedestrian_page', methods=['GET'])
def add_pedestrian_page():
    return render_template('add_pedestrian.html')

# Page to render "Add Vehicle" form
@app.route('/add_vehicle_page', methods=['GET'])
def add_vehicle_page():
    return render_template('add_vehicle.html')

# View all authorized vehicles from Azure Table
@app.route('/view/vehicles')
def view_vehicles():
    try:
        entities = list(vehicle_table_client.list_entities())
        return render_template('view_vehicles.html', vehicles=entities)
    except Exception as e:
        flash(f"Failed to load vehicles: {e}", "danger")
        return redirect(url_for('dashboard'))

# Delete a vehicle by license plate
@app.route('/delete/vehicle/<license_plate>', methods=['POST'])
def delete_vehicle(license_plate):
    try:
        vehicle_table_client.delete_entity(partition_key='Vehicle', row_key=license_plate)
        flash(f"Vehicle '{license_plate}' deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting vehicle '{license_plate}': {e}", "danger")
    return redirect(url_for('view_vehicles'))

# View all authorized pedestrians from Azure Table
@app.route('/view/pedestrians')
def view_pedestrians():
    try:
        entities = list(pedestrian_table_client.list_entities())
        return render_template('view_pedestrians.html', pedestrians=entities)
    except Exception as e:
        flash(f"Failed to load pedestrians: {e}", "danger")
        return redirect(url_for('dashboard'))

# Delete a pedestrian by pedestrian ID
@app.route('/delete/pedestrian/<pedestrian_id>', methods=['POST'])
def delete_pedestrian(pedestrian_id):
    try:
        pedestrian_table_client.delete_entity(partition_key='Pedestrian', row_key=pedestrian_id)
        flash(f"Pedestrian '{pedestrian_id}' deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting pedestrian '{pedestrian_id}': {e}", "danger")
    return redirect(url_for('view_pedestrians'))

# Add a new authorized vehicle - expects face image + vehicle details
@app.route('/add_vehicle', methods=['POST'])
def add_vehicle():
    face_picture = request.files.get('facePicture')
    license_plate = request.form.get('licensePlate')
    car_make = request.form.get('carMake')
    car_model = request.form.get('carModel')

    # Validate input
    if not face_picture or not license_plate or not car_make or not car_model:
        flash("All fields are required to add a vehicle.", "danger")
        return redirect(url_for('add_vehicle_page'))

    try:
        image_bytes = face_picture.read()
        embedding = get_face_embedding(image_bytes)  # Generate face embedding
        entity = {
            'PartitionKey': 'Vehicle',
            'RowKey': license_plate,
            'CarMake': car_make,
            'CarModel': car_model,
            'Embedding': json.dumps(embedding)
        }
        vehicle_table_client.upsert_entity(entity=entity, mode=UpdateMode.REPLACE)
        flash("Vehicle authorized data stored successfully", "success")
    except ValueError as ve:
        flash(f"Error: {str(ve)}", "danger")
    except Exception as e:
        flash(f"Internal Error: {str(e)}", "danger")

    return redirect(url_for('add_vehicle_page'))

# Add a new authorized pedestrian - expects face image + pedestrian ID
@app.route('/add_pedestrian', methods=['POST'])
def add_pedestrian():
    face_picture = request.files.get('facePicturePedestrian')
    pedestrian_id = request.form.get('pedestrianId')

    # Validate input
    if not face_picture or not pedestrian_id:
        flash("Face picture and ID are required to add a pedestrian.", "danger")
        return redirect(url_for('add_pedestrian_page'))

    try:
        image_bytes = face_picture.read()
        embedding = get_face_embedding(image_bytes)  # Generate face embedding
        entity = {
            'PartitionKey': 'Pedestrian',
            'RowKey': pedestrian_id,
            'Embedding': json.dumps(embedding)
        }
        pedestrian_table_client.upsert_entity(entity=entity, mode=UpdateMode.REPLACE)
        flash("Pedestrian authorized data stored successfully", "success")
    except ValueError as ve:
        flash(f"Error: {str(ve)}", "danger")
    except Exception as e:
        flash(f"Internal Error: {str(e)}", "danger")

    return redirect(url_for('add_pedestrian_page'))


# Run the Flask app on port 5002 in debug mode
if __name__ == '__main__':
    # import waitress
    app.run(debug=True, port=5002)
    # print("App is running.")
    # waitress.serve(app, listen="0.0.0.0:8000")
