import cv2
import dlib
import numpy
import imutils
from imutils.face_utils import FaceAligner

dlib.DLIB_USE_CUDA = True


class FaceDetector:
    def __init__(self):
        self.size = 300
        self.min_face_width = 256
        self.confidence = 0.7
        self.min_quality = 7

        self.detector = cv2.dnn.readNetFromCaffe('dnn/opencv/deploy.prototxt', 'dnn/opencv/res10_300x300_ssd_iter_140000.caffemodel')
        self.predictor = dlib.shape_predictor('dnn/dlib/shape_predictor_68_face_landmarks.dat')
        self.fa = FaceAligner(self.predictor, desiredFaceWidth=self.min_face_width)

        self.face_trackers = dict()
        self.tracker_current_face_id = 0

    def get_scale(self, frame):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        frame_small = imutils.resize(frame, width=self.size)
        frame_small_height = frame_small.shape[0]
        frame_small_width = frame_small.shape[1]
        scale_x = int(frame_width / frame_small_width)
        scale_y = int(frame_height / frame_small_height)

        return frame_small, scale_x, scale_y, frame_width, frame_height

    def get_faces(self, frame):
        frame_small, scale_x, scale_y, frame_width, frame_height = self.get_scale(frame)

        blob = cv2.dnn.blobFromImage(
            frame_small, 1.0, (self.size, self.size),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )

        self.detector.setInput(blob)
        detections = self.detector.forward()
        rects = []
        for index in range(0, detections.shape[2]):
            confidence = detections[0, 0, index, 2]

            if confidence > self.confidence:
                box = detections[0, 0, index, 3:7] * numpy.array([frame_width, frame_height, frame_width, frame_height])
                start_x, start_y, end_x, end_y = box.astype("int")
                small_start_x = int(start_x / scale_x)
                small_start_y = int(start_y / scale_y)

                face = frame[start_y:end_y, start_x:end_x]
                face_height, face_width = face.shape[:2]
                small_face_width = int(face_width / scale_x)
                small_face_height = int(face_height / scale_y)
                small_end_x = small_start_x + small_face_width
                small_end_y = small_start_y + small_face_height

                rects.append({
                    "x1": int(start_x),
                    "y1": int(start_y),
                    "x2": int(end_x),
                    "y2": int(end_y),
                    "x_center": int(start_x + face_width * 0.5),
                    "y_center": int(start_y + face_height * 0.5),
                    "height": int(face_height),
                    "width": int(face_width),
                    "small_x1": int(small_start_x),
                    "small_y1": int(small_start_y),
                    "small_x2": int(small_end_x),
                    "small_y2": int(small_end_y),
                    "small_x_center": int(small_start_x + small_face_width * 0.5),
                    "small_y_center": int(small_start_y + small_face_height * 0.5),
                    "small_height": int(small_face_width),
                    "small_width": int(small_face_height)
                })

        return rects, frame_small

    def track(self, frame):
        response = dict()
        frame_small, scale_x, scale_y, frame_width, frame_height = self.get_scale(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for face_id, face_tracker in self.face_trackers.items():
            quality = face_tracker.update(frame_small)

            if quality < self.min_quality:
                self.face_trackers.pop(face_id, None)
            else:
                tracked_position = face_tracker.get_position()

                x = int(tracked_position.left() * scale_x)
                y = int(tracked_position.top() * scale_y)
                width = int(tracked_position.width() * scale_x)
                height = int(tracked_position.height() * scale_y)

                if (
                    x > 0 and y > 0 and width > 0 and height > 0 and
                    x + width <= frame_width and y + height <= frame_height
                ):
                    rect = dlib.rectangle(x, y, x + width, y + height)
                    face = self.fa.align(frame, frame_gray, rect)

                    response[face_id] = {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "face": face,
                        "id": face_id
                    }
                else:
                    self.face_trackers.pop(face_id, None)

        return response

    def detect(self, frame):
        rects, frame_small = self.get_faces(frame)

        for rect in rects:
            matched = None

            # Find face match
            for face_id, face_tracker in self.face_trackers.items():
                tracked_position = face_tracker.get_position()

                tracker_x = int(tracked_position.left())
                tracker_y = int(tracked_position.top())
                tracker_width = int(tracked_position.width())
                tracker_height = int(tracked_position.height())

                t_x_center = tracker_x + tracker_width * 0.5
                t_y_center = tracker_y + tracker_height * 0.5

                if (
                    (tracker_x <= rect['small_x_center'] <= (tracker_x + tracker_width)) and
                    (tracker_y <= rect['small_y_center'] <= (tracker_y + tracker_height)) and
                    (rect['small_x1'] <= t_x_center <= (rect['small_x1'] + rect['small_width'])) and
                    (rect['small_y1'] <= t_y_center <= (rect['small_y1'] + rect['small_height']))
                ):
                    matched = face_id

            if matched is None:
                # Create and store the tracker
                tracker = dlib.correlation_tracker()
                tracker.start_track(frame_small, dlib.rectangle(
                    rect['small_x1'],
                    rect['small_y1'],
                    rect['small_x2'],
                    rect['small_y2'],
                ))

                self.tracker_current_face_id += 1
                self.face_trackers[self.tracker_current_face_id] = tracker
