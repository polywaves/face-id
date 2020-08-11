import cv2
import grpc_client
import json
from datetime import datetime
from face_recognition import FaceRecognition
# from mtcnn_face_detector import FaceDetector
from face_detector import FaceDetector
from mq import Mq


class App:
    def __init__(self):
        self.camera = grpc_client.get_camera()
        self.check()

        # Defaults
        self.detect_rate = 10
        self.tracking_rate = 5
        self.request_camera_time = 20

        self.mq = Mq()
        self.mq.connect()

        self.face_recognition = FaceRecognition()
        self.face_detector = FaceDetector()

    def check(self):
        self.camera = grpc_client.get_camera()

        if self.camera.recognition_on:
            # check remote commands
            self.face_recognition.classifier.consume()
            print('Service online')
        else:
            print('Service offline')
            exit()

    def capture(self):
        capture = cv2.VideoCapture(self.camera.stream_uri)
        index = 0
        while capture.isOpened():
            index += 1

            if index % self.request_camera_time == 0:
                self.check()

            # Get current frame
            resolve, frame = capture.read()
            if not resolve:
                break

            # Get weights
            height, width = frame.shape[:2]
            if index % self.detect_rate == 0:
                self.face_detector.detect(frame)

            if index % self.tracking_rate == 0:
                rects = []
                for face_id, data in self.face_detector.track(frame).items():
                    rect = self.face_recognition.face_identification(index, data, width, height)

                    if rect:
                        rects.append(rect)

                self.mq.send(json.dumps({
                    "camera_id": self.camera.id,
                    "individual_id": None,
                    "recognition_ts": datetime.utcnow().timestamp(),
                    "data": rects
                }))

        capture.release()
        print('Stream not found, restarting')


app = App()
app.face_recognition.classifier.update()
app.capture()

