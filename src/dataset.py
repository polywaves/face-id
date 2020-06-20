import dlib


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
