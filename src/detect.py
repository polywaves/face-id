import cv2
import grpc_client
import pickle
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

        self.mq_receive = Mq(
            queue='recognition_records_pass_to_dnn',
            exchange='recognition_records_faces',
            routing_key='new_record'
        )
        self.mq_receive.connect()

        self.face_detector = FaceDetector()

    def check(self):
        self.camera = grpc_client.get_camera()

        if self.camera.recognition_on is not True:
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
                    rects.append(data)

                self.mq_receive.send(pickle.dumps({
                    "detect": True,
                    "data": {
                        "camera_id": self.camera.id,
                        "index": index,
                        "rects": rects,
                        "screen_width": width,
                        "screen_height": height
                    }
                }))

        capture.release()
        print('Stream not found, restarting')


app = App()
app.capture()

