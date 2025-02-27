import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def load_lfw_dataset(self, lfw_path):
        """
        Load encoding images from LFW dataset where each person has their own directory
        :param lfw_path: Path to LFW dataset root directory
        :return:
        """
        # Get all directories (one for each person)
        people_dirs = [d for d in os.listdir(lfw_path) if os.path.isdir(os.path.join(lfw_path, d))]

        print(f"Found {len(people_dirs)} people in the dataset.")

        # Store image encoding and names
        for person in people_dirs:
            person_dir = os.path.join(lfw_path, person)
            # Get all images for this person
            image_paths = glob.glob(os.path.join(person_dir, "*.jpg"))

            if not image_paths:
                continue

            # Use the first image for each person for simplicity
            # You could modify this to use all images if needed
            img_path = image_paths[0]

            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read image: {img_path}")
                    continue

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Get face encodings
                face_encodings = face_recognition.face_encodings(rgb_img)

                if not face_encodings:
                    print(f"No face found in {img_path}")
                    continue

                # Use the first face found
                img_encoding = face_encodings[0]

                # Store person name and face encoding
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(person)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print(f"Successfully loaded {len(self.known_face_names)} people with face encodings")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Convert the image from BGR to RGB (face_recognition expects RGB)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces and their encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        threshold = 0.4  # Adjust this if needed

        for face_encoding in face_encodings:
            # Compare the face with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            name = "Unknown"

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < threshold:  # Use threshold to determine if it's a match
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing

        return face_locations.astype(int), face_names
