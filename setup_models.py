"""
Setup script to download required models for enhanced drowsiness detection
"""
import os
import urllib.request
import bz2
import shutil

def download_dlib_model():
    """Download the dlib facial landmark predictor model"""
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    model_file = "shape_predictor_68_face_landmarks.dat"
    compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
    
    if os.path.exists(model_file):
        print(f"✓ {model_file} already exists")
        return True
    
    print("Downloading dlib facial landmark predictor model...")
    print("This may take a few minutes...")
    
    try:
        # Download the compressed file
        urllib.request.urlretrieve(model_url, compressed_file)
        print("✓ Download completed")
        
        # Extract the compressed file
        print("Extracting model file...")
        with bz2.BZ2File(compressed_file, 'rb') as f_in:
            with open(model_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove the compressed file
        os.remove(compressed_file)
        print(f"✓ Model extracted successfully: {model_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("Please manually download the model from:")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract it and place 'shape_predictor_68_face_landmarks.dat' in this directory")
        return False

def main():
    print("=== Enhanced Drowsiness Detection Setup ===")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("simple_drowsiness_detector.py"):
        print("✗ Please run this script from the drowsiness_detection directory")
        return
    
    # Download dlib model
    success = download_dlib_model()
    
    print()
    if success:
        print("=== Setup completed successfully! ===")
        print("You can now run the enhanced drowsiness detector:")
        print("python simple_drowsiness_detector.py")
    else:
        print("=== Setup partially completed ===")
        print("The script will fall back to basic detection without the dlib model")
    
    print()
    print("Enhanced features include:")
    print("• Eye Aspect Ratio (EAR) calculation for precise drowsiness detection")
    print("• Personalized threshold calibration")
    print("• Yawn detection using Mouth Aspect Ratio (MAR)")
    print("• Head pose estimation")
    print("• Blink frequency monitoring")
    print("• Moving average filtering to reduce false positives")
    print("• Real-time statistics display")

if __name__ == "__main__":
    main()
