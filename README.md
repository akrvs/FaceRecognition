# Facial Recognition System

A simple and efficient facial recognition system built with Python, OpenCV, and face_recognition. This system can detect and recognize faces from a webcam feed in real-time by comparing them against known faces from the LFW (Labeled Faces in the Wild) dataset.

## Features

- Real-time face detection and recognition from webcam
- Integration with the LFW dataset for pre-trained faces
- Customizable threshold for face matching accuracy
- Simple API for easy integration into other projects

## Requirements

- Python 3.6+
- OpenCV
- face_recognition (which requires dlib and cmake)
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/facial-recognition-system.git
cd facial-recognition-system
```

2. Install the required dependencies:
```bash
pip install opencv-python face_recognition numpy
```

3. Download the LFW dataset:
   - The LFW dataset can be downloaded from [here](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz)
   - Extract it to the project directory

## Usage

1. Run the main script:
```bash
python main.py
```

2. The webcam will activate and start detecting faces
3. Recognized faces will be labeled with their names
4. Press the 'Esc' key to exit the application

## Code Structure

- `main.py`: Main script to run the facial recognition system
- `simple_facerec.py`: Contains the `SimpleFacerec` class that handles face encoding and recognition

## How It Works

1. The system loads face encodings from the LFW dataset
2. Each frame from the webcam is processed to detect faces
3. Detected faces are compared against the known face encodings
4. If a match is found (below the threshold), the face is labeled with the corresponding name
5. If no match is found, the face is labeled as "Unknown"

## Customization

You can adjust several parameters in the code:

- In `simple_facerec.py`, modify the `threshold` value (default: 0.4) to make face matching more or less strict
- Adjust the `frame_resizing` value to balance between performance and accuracy
- Implement additional features such as saving recognized faces to a database

## License

[MIT License](LICENSE)

## Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) library
- [LFW dataset](http://vis-www.cs.umass.edu/lfw/)
- OpenCV community
