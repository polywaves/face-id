import cv2
import grpc_client
from face_recognition import FaceRecognition
from face_detector import FaceDetector
from mq import Mq


class App:
    def __init__(self):
        # Defaults
        self.stream_request_rate = 5
        self.stream_request_rate_send = 4

        self.mq = Mq()
        self.mq.connect()

        self.camera = grpc_client.get_camera()
        self.face_recognition = FaceRecognition()
        self.face_detector = FaceDetector()

    def capture(self):
        capture = cv2.VideoCapture(self.camera.stream_uri)
        index = 0
        while capture.isOpened():
            # check remote commands
            self.face_recognition.classifier.consume()

            index += 1

            # Get current frame
            resolve, frame = capture.read()
            if not resolve:
                break

            # Get weights
            height, width = frame.shape[:2]
            if index % self.stream_request_rate == 0:
                self.face_detector.detect(frame)

            if index % self.stream_request_rate_send == 0:
                for face_id, data in self.face_detector.track(frame).items():
                    self.face_recognition.face_identification(data, width, height)

        capture.release()
        print('Stream not found')


app = App()
app.face_recognition.classifier.update()
app.capture()

