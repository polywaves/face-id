import cv2
import dlib
import imutils
from mtcnn import MTCNN
from imutils.face_utils import FaceAligner

dlib.DLIB_USE_CUDA = True


class FaceDetector:
    def __init__(self):
        self.size = 300
        self.min_face_width = 256
        self.confidence = 0.6
        self.min_quality = 7

        self.detector = MTCNN()
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

        detects = self.detector.detect_faces(frame)
        rects = []
        for detect in detects:
            confidence = detect['confidence']

            if confidence > self.confidence:
                start_x, start_y, face_width, face_height = detect['box']
                end_x = start_x + face_width
                end_y = start_y + face_height

                small_start_x = start_x / scale_x
                small_start_y = start_y / scale_y
                small_face_width = face_width / scale_x
                small_face_height = face_width / scale_y
                small_end_x = end_x / scale_x
                small_end_y = end_y / scale_y

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

    def get_face(self, frame, frame_gray, x, y, width, height):
        rect = dlib.rectangle(x, y, x + width, y + height)
        return self.fa.align(frame, frame_gray, rect)

    def detect_and_get_faces(self, image):
        frame = cv2.imread(image)
        rects, frame_small = self.get_faces(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = []
        for rect in rects:
            face = self.get_face(frame, frame_gray, rect['x1'], rect['y1'], rect['width'], rect['height'])
            faces.append(face)

        return faces

    def track(self, frame):
        response = dict()
        frame_small, scale_x, scale_y, frame_width, frame_height = self.get_scale(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_to_delete = []
        for face_id, face_tracker in self.face_trackers.items():
            quality = face_tracker.update(frame_small)

            if quality < self.min_quality:
                faces_to_delete.append(face_id)
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
                    face = self.get_face(frame, frame_gray, x, y, width, height)

                    response[face_id] = {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "face": face,
                        "id": face_id
                    }
                else:
                    faces_to_delete.append(face_id)

        for face_id in faces_to_delete:
            del self.face_trackers[face_id]

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
