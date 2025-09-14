# Face Detection Streamlit App

A web application for face detection using OpenCV and Streamlit. This app allows users to detect faces using their webcam or by uploading photos.

## Features

- 📷 **Webcam Face Detection**: Real-time face detection using your device's camera
- 📁 **Photo Upload**: Upload and process multiple images
- 🔍 **Face Counting**: Shows the number of faces detected
- 📥 **Download Results**: Save processed images with face detection boxes
- ⚙️ **Adjustable Settings**: Fine-tune detection parameters

## How to Use

### Webcam Detection
1. Go to the "Webcam" tab
2. Allow camera access when prompted
3. Position your face in view
4. Click to capture the image
5. View the face detection results

### Photo Upload
1. Go to the "Upload Photo" tab
2. Upload one or more image files (PNG, JPG, JPEG)
3. View detection results for each image
4. Download processed images if needed

## Local Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Technology Stack

- **Streamlit**: Web app framework
- **OpenCV**: Computer vision library for face detection
- **PIL/Pillow**: Image processing
- **NumPy**: Numerical computations

## Face Detection Algorithm

This app uses OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) for detecting faces. The algorithm works best with:

- Good lighting conditions
- Frontal face orientation
- Clear, unobstructed faces
- Sufficient image resolution

## Deployment

This app is deployed on Streamlit Cloud and can be accessed at: [Your App URL]

## Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## License

This project is open source and available under the MIT License.