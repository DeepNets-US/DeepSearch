# All imports
import os
import json
import pandas as pd
from gradio_client import Client, handle_file
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

# Configuration
db_metadata_path = os.path.join('static', 'db_metadata.json')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit
app.config['QUERY_IMAGE_VECTOR'] = None
app.config['ANNOY_INDEX'] = None

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Get available categories/databases
with open(db_metadata_path, 'r') as file:
    db_metadata = json.load(file)

categories = db_metadata.keys()

# Function to check if file has allowed extension


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']

# Function to clear uploads directory


def clear_uploads():
    """Clear the uploads directory"""
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Route for index page


@app.route('/')
def index():
    clear_uploads()  # Clear previous uploads
    return render_template('index.html', filename=None, categories=categories)

# Route for getting class names


@app.route('/get_class_names', methods=['POST'])
def get_class_names():
    try:
        database = request.get_json().get('database')

        if not database:
            return jsonify({"error": "Database not specified"}), 400

        class_names_list = db_metadata[database]

        return jsonify({"class_names_list": class_names_list})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for uploading image


@app.route('/upload', methods=['POST'])
def upload_file():
    print('Upload file')
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']

    if file.filename == '':
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        clear_uploads()  # Clear previous uploads
        filename = secure_filename(file.filename)

        # Ensure the filename is unique
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            filename = f"{base}_{counter}{ext}"
            counter += 1

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return render_template('index.html',
                               filename=filename,
                               categories=categories)

    return redirect(url_for('index'))

# Route for searching


@app.route('/search', methods=['POST'])
def search():
    print('Searching...')
    try:
        # Get form data
        database = request.form.get('database')
        filename = request.form.get('filename')
        num_images = int(request.form.get('numImages', 20))

        if not database or not filename:
            return render_template('index.html',
                                   error="Missing database or filename",
                                   categories=categories)

        # Verify the uploaded file exists
        if 'static' not in filename:
            query_image_path = os.path.join(
                app.config['UPLOAD_FOLDER'], filename)
        else:
            query_image_path = filename

        app.config['QUERY_IMAGE_PATH'] = query_image_path

        # Query via
        client = Client(f"DeepNets/{database}-DeepSearch")
        closest_class, _, sim_image_paths = client.predict(
            query_image=handle_file(query_image_path),
            num_images=num_images,
            api_name="/similarity_search"
        )

        results = [f'./static/DB{path[1:]}'
                   for path in eval(sim_image_paths)]

        print("Database: ", database)
        print("filename: ", filename)
        print("num_images: ", num_images)
        print("results", results)

        return render_template('results.html',
                               results=results,
                               class_name=closest_class,
                               database=database,
                               filename=filename)

    except Exception as e:
        return render_template('index.html',
                               error=f"Error during search: {str(e)}",
                               categories=categories)

# Route for updating


@app.route('/update', methods=['POST'])
def update():
    # Get Query & Annoy Index
    query_image_path = app.config['QUERY_IMAGE_PATH']
    database = request.form.get('database')
    filename = request.form.get('filename')
    num_images = int(request.form.get('numImages', 20))

    client = Client(f"DeepNets/{database}-DeepSearch")
    closest_class, _, sim_image_paths = client.predict(
        query_image=handle_file(query_image_path),
        num_images=num_images,
        api_name="/similarity_search"
    )

    results = [os.path.join('static', 'DB', path)
               for path in eval(sim_image_paths)]

    return render_template('results.html',
                           results=results,
                           class_name=closest_class,
                           database=database,
                           filename=filename)


# Run the application
if __name__ == '__main__':
    app.run(debug=True)
