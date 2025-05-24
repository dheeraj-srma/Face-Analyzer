import cv2
import argparse
from deepface import DeepFace
import warnings
warnings.filterwarnings("ignore")

# Suppress DeepFace prints by redirecting stdout temporarily
import contextlib
import os
import sys

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Function to highlight faces
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                 [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, faceBoxes

# Argument parser for image input
parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

# Model paths
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

# Load pre-trained network
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Start webcam or load image
video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

# Main loop
while True:
    hasFrame, frame = video.read()
    if not hasFrame:
        break

    frame = cv2.flip(frame, 1)  # Optional: mirror effect
    resultImg, faceBoxes = highlightFace(faceNet, frame)

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        try:
            # Analyze with DeepFace with output suppressed
            with suppress_stdout():
                analysis = DeepFace.analyze(face, actions=['age', 'gender', 'emotion'], enforce_detection=False)

            age = int(analysis[0]['age'])
            gender = analysis[0]['dominant_gender']
            emotion = analysis[0]['dominant_emotion']

            label = f"[{gender}, {age}, {emotion}]"
        except Exception:
            label = "[Face Detected]"

        # Annotate result
        cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Show result
    cv2.imshow("Detecting age, gender & emotion", resultImg)

    # Exit on ESC or 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    if cv2.getWindowProperty("Detecting age, gender & emotion", cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
