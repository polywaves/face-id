import base64
import dlib
import io
from PIL import Image


class Dataset:
    def __init__(self):
        self.size = 300
        self.confidence = 0.7
        self.detector = dlib.get_frontal_face_detector()

    def check_face(self, face):
        faces = 0
        rects = self.detector(face, 0)
        for rect in rects:
            faces += 1

        return faces

    @staticmethod
    def create_base64_face(face):
        im = Image.fromarray(face.astype("uint8"))
        raw = io.BytesIO()
        im.save(raw, "JPEG")
        raw.seek(0)

        return str(base64.b64encode(raw.read()).decode('utf-8'))
