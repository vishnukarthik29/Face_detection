import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Face Detection App",
    page_icon="👤",
    layout="wide"
)

# Load face detection model
@st.cache_resource
def load_face_detector():
    """Load OpenCV face detection classifier"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def detect_faces_in_image(image, face_cascade):
    """Detect faces in an image and return image with bounding boxes"""
    # Convert PIL image to OpenCV format
    if isinstance(image, Image.Image):
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        opencv_image = image
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around faces
    result_image = opencv_image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Convert back to RGB for display
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    return result_image_rgb, len(faces)

def main():
    st.title("👤 Face Detection App")
    st.write("Detect faces in images using webcam or upload photos")
    
    # Load face detector
    face_cascade = load_face_detector()
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📷 Webcam", "📁 Upload Photo"])
    
    with tab1:
        st.header("Webcam Face Detection")
        
        # Webcam controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Camera input
            camera_input = st.camera_input("Take a picture")
            
        with col2:
            st.write("### Instructions:")
            st.write("1. Allow camera access")
            st.write("2. Position your face in view")
            st.write("3. Click to capture")
            st.write("4. See face detection results")
        
        if camera_input is not None:
            # Process the captured image
            image = Image.open(camera_input)
            
            with st.spinner("Detecting faces..."):
                result_image, face_count = detect_faces_in_image(image, face_cascade)
            
            # Display results
            st.success(f"Found {face_count} face(s) in the image!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Face Detection Result")
                st.image(result_image, use_column_width=True)
    
    with tab2:
        st.header("Photo Upload Face Detection")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload one or more image files for face detection"
        )
        
        if uploaded_files:
            # Process each uploaded file
            for i, uploaded_file in enumerate(uploaded_files):
                st.subheader(f"Image {i+1}: {uploaded_file.name}")
                
                # Load and display original image
                image = Image.open(uploaded_file)
                
                with st.spinner(f"Detecting faces in {uploaded_file.name}..."):
                    result_image, face_count = detect_faces_in_image(image, face_cascade)
                
                # Display results
                if face_count > 0:
                    st.success(f"✅ Found {face_count} face(s) in {uploaded_file.name}")
                else:
                    st.warning(f"⚠️ No faces detected in {uploaded_file.name}")
                
                # Show images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original**")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.write("**With Face Detection**")
                    st.image(result_image, use_column_width=True)
                
                # Add download button for result
                result_pil = Image.fromarray(result_image)
                
                # Convert to bytes for download
                import io
                img_buffer = io.BytesIO()
                result_pil.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label=f"📥 Download Result - {uploaded_file.name}",
                    data=img_buffer,
                    file_name=f"face_detected_{uploaded_file.name}",
                    mime="image/png"
                )
                
                st.divider()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses OpenCV's Haar Cascade classifier for face detection.
        
        **Features:**
        - Real-time webcam capture
        - Multiple photo upload
        - Face counting
        - Download results
        
        **Tips for better detection:**
        - Ensure good lighting
        - Face the camera directly
        - Avoid heavy shadows
        - Use clear, high-quality images
        """)
        
        st.header("🛠️ Settings")
        
        # Add some detection parameters (optional)
        with st.expander("Advanced Settings"):
            scale_factor = st.slider(
                "Scale Factor", 
                min_value=1.1, 
                max_value=2.0, 
                value=1.1, 
                step=0.1,
                help="How much the image size is reduced at each scale"
            )
            
            min_neighbors = st.slider(
                "Min Neighbors", 
                min_value=1, 
                max_value=10, 
                value=5,
                help="How many neighbors each candidate rectangle should retain"
            )
            
            min_size = st.slider(
                "Min Face Size", 
                min_value=10, 
                max_value=100, 
                value=30,
                help="Minimum possible face size in pixels"
            )
            
            st.info("These settings affect detection sensitivity. Adjust if needed.")

if __name__ == "__main__":
    main()