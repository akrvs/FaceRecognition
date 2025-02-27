**Face Recognition using LFW Dataset**

This project implements a simple face recognition system using the Labeled Faces in the Wild (LFW) dataset and face_recognition library. 
The system loads images from the LFW dataset, encodes faces, and detects known faces in a video stream.

**Features**

✅ Loads face encodings from the LFW datase
✅ Detects faces in real-time using OpenCV
✅ Matches detected faces with known identities
✅ Uses a threshold to determine similarity

**Installation**

Ensure you have Python 3 installed. Then, install the required dependencies:

pip install opencv-python face-recognition numpy

_Dataset_
Download the LFW dataset:

Visit the LFW website
Download lfw-deepfunneled.tgz
Extract it so that the folder structure looks like this:

lfw-deepfunneled/
    ├── Person_A/
    │   ├── img1.jpg
    │   ├── img2.jpg
    ├── Person_B/
    │   ├── img1.jpg

**Usage**

Run the face recognition script:
python main.py
Press Esc to exit the program.

**File Overview**

simple_facerec.py - Handles face encoding and detection
main.py - Runs the face recognition on a webcam feed

**License**
This project is open-source under the MIT License. Feel free to use and modify it!
