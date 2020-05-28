import cv2
import dlib
import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils.face_utils import shape_to_np

dlib.DLIB_USE_CUDA = True


class FaceDetector:
    def __init__(self):
        self.frame_resize_width = 300
        self.face_width = 300
        self.min_face_width = 128
        self.min_face_height = 128

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('dnn/dlib/shape_predictor_68_face_landmarks.dat')
        self.fa = FaceAligner(self.predictor, desiredFaceWidth=self.face_width)

    def get_rects(self, frame):
        result = []

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        frame_small = imutils.resize(frame, width=self.frame_resize_width)
        frame_small_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        frame_small_height = frame_small.shape[0]
        frame_small_width = frame_small.shape[1]
        frame_scale_x = frame_width / frame_small_width
        frame_scale_y = frame_height / frame_small_height

        index = 0
        rects = self.detector(frame_small_gray, 2)
        for rect in rects:
            (x, y, w, h) = rect_to_bb(rect)
            x *= frame_scale_x
            y *= frame_scale_y
            w *= frame_scale_x
            h *= frame_scale_y
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            try:
                rect = dlib.rectangle(x, y, x + w, y + h)
                face = self.fa.align(frame, frame_small_gray, rect)
                if face.shape[1] >= self.face_width:
                    index += 1

                    result.append({
                        "index": index,
                        "x1": x,
                        "y1": y,
                        "x2": x + w,
                        "y2": y + h,
                        "face": face,
                        "height": h,
                        "width": w
                    })
            except Exception:
                print('Broken face')

        return result

    def check_face(self, face, gray=False, resize=None):
        result = None

        try:
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            rects = self.detector(face_gray, 2)
            for rect in rects:
                shape = self.predictor(face_gray, rect)
                shape = shape_to_np(shape)

                if len(shape) == 68:
                    (x, y, w, h) = rect_to_bb(rect)
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)

                    if gray is True:
                        face = face_gray[y:y + h, x:x + w]
                    else:
                        face = face[y:y + h, x:x + w]

                    face_height = face.shape[0]
                    face_width = face.shape[1]
                    if face_width >= self.min_face_width and face_height >= self.min_face_height:
                        if resize is not None:
                            face = imutils.resize(face, width=resize)

                        result = face
        except Exception:
            print('Broken face')

        return result
