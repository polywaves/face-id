import cv2
import dlib
import numpy
from imutils.face_utils import FaceAligner


class FaceDetector:
    def __init__(self):
        self.min_face_width = 256
        self.confidence = 0.7

        self.detector = cv2.dnn.readNetFromCaffe('dnn/opencv/deploy.prototxt', 'dnn/opencv/res10_300x300_ssd_iter_140000.caffemodel')
        self.predictor = dlib.shape_predictor('dnn/dlib/shape_predictor_68_face_landmarks.dat')
        self.fa = FaceAligner(self.predictor, desiredFaceWidth=self.min_face_width)

    def get_rects(self, frame):
        result = []

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )

        self.detector.setInput(blob)
        detections = self.detector.forward()
        for index in range(0, detections.shape[2]):
            confidence = detections[0, 0, index, 2]

            if confidence > self.confidence:
                box = detections[0, 0, index, 3:7] * numpy.array([frame_width, frame_height, frame_width, frame_height])
                start_x, start_y, end_x, end_y = box.astype("int")

                face = frame[start_y:end_y, start_x:end_x]
                face_height, face_width = face.shape[:2]
                rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                face = self.fa.align(frame, frame_gray, rect)

                result.append({
                    "index": index,
                    "x1": int(start_x),
                    "y1": int(start_y),
                    "x2": int(end_x),
                    "y2": int(end_y),
                    "face": face,
                    "height": int(face_height),
                    "width": int(face_width)
                })

        return result
