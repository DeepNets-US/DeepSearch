import os
import shutil
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename  # Added import for secure_filename
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from annoy import AnnoyIndex

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
DATABASE_FOLDER = os.path.join('static', 'Database')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATABASE_FOLDER'] = DATABASE_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)

# Get available categories/databases
categories = [d for d in os.listdir(DATABASE_FOLDER)
              if os.path.isdir(os.path.join(DATABASE_FOLDER, d))]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']


def clear_uploads():
    """Clear the uploads directory"""
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


@app.route('/')
def index():
    clear_uploads()  # Clear previous uploads
    return render_template('index.html', filename=None, categories=categories)


@app.route('/get_class_names', methods=['POST'])
def get_class_names():
    try:
        data = request.get_json()
        database = data.get('database')
        if not database:
            return jsonify({"error": "Database not specified"}), 400

        class_names_path = os.path.join(
            app.config['DATABASE_FOLDER'],
            database,
            f'{database}-ClassNames.txt'
        )

        if not os.path.exists(class_names_path):
            return jsonify({"error": "Class names file not found"}), 404

        with open(class_names_path, 'r') as f:
            class_names_list = [name.strip()
                                for name in f.read().split(',') if name.strip()]

        return jsonify({"class_names_list": class_names_list})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


def image_to_vector(image_path, image_size=(224, 224)):
    """Convert image to feature vector"""
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.expand_dims(image, 0)
        image = image/255.0  # Normalize image
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


@app.route('/search', methods=['POST'])
def search():
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
        query_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(query_image_path):
            return render_template('index.html',
                                   error="Uploaded image not found",
                                   categories=categories)

        # Path setup
        db_path = os.path.join(app.config['DATABASE_FOLDER'], database)

        # Load feature extractor
        feature_extractor_path = os.path.join(
            db_path, f'{database}-FeatureExtractor.keras')
        if not os.path.exists(feature_extractor_path):
            return render_template('index.html',
                                   error="Feature extractor not found",
                                   categories=categories)

        feature_extractor = keras.models.load_model(
            feature_extractor_path, compile=False)

        # Load Annoy index
        index_path = os.path.join(db_path, f'{database}Subset.ann')
        if not os.path.exists(index_path):
            return render_template('index.html',
                                   error="Index not found",
                                   categories=categories)

        annoy_index = AnnoyIndex(256, 'angular')
        annoy_index.load(index_path)

        # Process query image
        query_image = image_to_vector(query_image_path)
        if query_image is None:
            return render_template('index.html',
                                   error="Error processing image",
                                   categories=categories)

        query_vector = feature_extractor.predict(query_image)[0]
        nearest_neighbors = annoy_index.get_nns_by_vector(
            query_vector, num_images)

        # Load metadata
        csv_name = f'{database}s.csv' if database.lower(
        ) != 'shoes' else f'{database}.csv'
        df_path = os.path.join(db_path, csv_name)

        if not os.path.exists(df_path):
            return render_template('index.html',
                                   error="Metadata file not found",
                                   categories=categories)

        df = pd.read_csv(df_path, index_col=0).iloc[nearest_neighbors]
        closest_class = df.class_name.values[0]

        # Prepare results
        results = []
        for class_name, file_name in zip(df.class_name.values, df.file_name.values):
            img_path = os.path.join(
                db_path,
                f'{database}Subset',
                class_name,
                file_name
            )
            # Convert to web-friendly path
            web_path = os.path.relpath(img_path, 'static').replace('\\', '/')
            results.append(web_path)

        return render_template('results.html',
                               results=results,
                               class_name=closest_class,
                               database=database,
                               filename=filename)

    except Exception as e:
        return render_template('index.html',
                               error=f"Error during search: {str(e)}",
                               categories=categories)


if __name__ == '__main__':
    app.run()
