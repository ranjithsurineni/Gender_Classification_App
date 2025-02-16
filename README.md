# Gender Classification Application

## Overview
This project implements a face recognition and gender classification pipeline using a combination of Haar Cascade for face detection, PCA (Principal Component Analysis) for dimensionality reduction, and an SVM (Support Vector Machine) model for classification. The pipeline identifies faces in an image and predicts gender (male/female) based on trained models. Additionally, a Flask-based web interface is provided for user-friendly interaction.

## Features
- **Face Detection**: Utilizes OpenCV's Haar Cascade Classifier.
- **Feature Extraction**: Applies PCA to optimize facial features.
- **Classification**: Employs an SVM model to predict gender.
- **Web Interface**: Enables easy image upload and real-time predictions.
- **Confidence Score**: Displays the confidence level of predictions.

## Installation
### Prerequisites
Ensure you have Python installed along with the following dependencies:

```bash
pip install numpy opencv-python scikit-learn flask pickle5
```

### Clone the Repository
```bash
git clone https://github.com/your-username/Gender_Classification_App.git
cd Gender_Classification_App
```

### Setup Virtual Environment (Optional)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

## Usage
### Load Models
The following models are required and should be placed in the `models/` directory:
- `haarcascade_frontalface_default.xml`: For face detection
- `model_svm.pickle`: Pre-trained SVM model
- `pca_dict.pickle`: Contains PCA model and mean face array

### Running Face Recognition and Gender Classification
```python
import cv2
from face_recognition import faceRecognitionPipeline

# Path to input image
image_path = "path/to/image.jpg"

# Run the pipeline
output_img, predictions = faceRecognitionPipeline(image_path)

# Display result
cv2.imshow("Result", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Running the Web Application
1. **Start the Flask App**:
   ```bash
   python app.py
   ```
2. **Access the Web Interface**:
   Open a browser and navigate to `http://127.0.0.1:5000/`.
3. **Classify an Image**:
   - Upload an image through the web interface.
   - The application will display the image with predictions.

## Face Recognition Pipeline
### `faceRecognitionPipeline(filename, path=True)`
- Reads the input image.
- Converts to grayscale.
- Detects faces using Haar Cascade.
- Normalizes, resizes, and processes the face region.
- Extracts features using PCA.
- Predicts gender using an SVM classifier.
- Draws bounding boxes with predictions.

### Output
The function returns:
1. The image with predictions visualized.
2. A list of dictionaries containing:
   - Extracted face region (`roi`).
   - Eigen image (`eig_img`).
   - Prediction (`prediction_name`).
   - Confidence score (`score`).

## Project Structure
```
├── models/
│   ├── haarcascade_frontalface_default.xml
│   ├── model_svm.pickle
│   ├── pca_dict.pickle
├── static/
│   └── css/
├── templates/
│   ├── index.html
│   └── result.html
├── app.py
├── face_recognition.py
├── requirements.txt
└── README.md
```

## Model Training
To train the model:
1. **Data Collection**: Gather a dataset of labeled male and female facial images.
2. **Preprocessing**: Convert images to grayscale, detect faces using Haar Cascade, normalize, resize, and flatten the images.
3. **Feature Extraction**: Apply PCA to reduce dimensionality and extract features.
4. **Model Training**: Train an SVM classifier using the extracted features.
5. **Model Evaluation**: Assess the model's performance and fine-tune as necessary.

## Acknowledgements
This project is inspired by various open-source gender classification applications and tutorials.

## License
This project is licensed under the MIT License.

