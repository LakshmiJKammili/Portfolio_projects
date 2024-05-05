import cv2
import numpy as np
import pickle

# Load reference name dictionary from file
with open("ref_name.pkl", "rb") as f:
    ref_dict = pickle.load(f)

# Load reference embeddings dictionary from file
with open("ref_embed.pkl", "rb") as f:
    embed_dict = pickle.load(f)

# Initialize video capture from default camera
video_capture = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not video_capture.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract face region from the grayscale frame
        face_roi = gray[y:y+h, x:x+w]

        # Resize the face region to a fixed size for consistency
        face_resized = cv2.resize(face_roi, (100, 100))

        # Normalize the pixel values of the face region
        face_normalized = face_resized / 255.0

        # Flatten the face region to create a 1D array (feature vector)
        face_flattened = face_normalized.flatten()

        # Compare face encoding with known embeddings
        match_found = False
        for ref_id, embeddings in embed_dict.items():
            for embedding in embeddings:
                # Resize the embedding to match the shape of the flattened face region
                embedding_resized = cv2.resize(embedding, (100, 100)).flatten()
                # Compute Euclidean distance between embeddings
                distance = np.linalg.norm(face_flattened - embedding_resized)
                # If distance is below a threshold, consider it a match
                if distance < 0.6:  # Adjust threshold as needed
                    # Match found, assign name to face
                    name = ref_dict.get(ref_id, 'Unknown')
                    match_found = True
                    break
            if match_found:
                break

        # Draw a rectangle around the detected face and display name
        if match_found:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
