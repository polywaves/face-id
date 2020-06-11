import pickle
import io
import base64
import db
import numpy
import json
import grpc_client
from PIL import Image
from datetime import datetime
from classifier import Classifier
from mq import Mq


class FaceRecognition:
    def __init__(self):
        self.matches = 1
        self.matches_vgg = 1
        self.confidence = 0

        # Init operations
        self.mq = Mq()
        self.mq.connect()
        self.camera = grpc_client.get_camera()
        self.classifier = Classifier()

        self.temp = dict()
        self.identity = dict()

    def face_identification(self, data, screen_width, screen_height):
        try:
            cube_id = data['id']

            start_time = datetime.now()
            embedding = self.classifier.get_embedding(data['face'])

            send_individual_id = None
            send_object_id = None
            send_confidence = 0
            rejected = None
            face = None
            if self.classifier.training is False:
                if self.classifier.recognizer:
                    predictions = self.classifier.recognizer.predict_proba(embedding)[0]
                    max_value = numpy.argmax(predictions)
                    confidence = int(predictions[max_value] * 100)
                    object_id = int(self.classifier.le.classes_[max_value])

                    if confidence >= self.confidence:
                        print('Predicted', object_id, confidence, '%')

                        embedding_vgg = None
                        if cube_id not in self.identity:
                            if object_id > 0:
                                if object_id in self.classifier.embeddings_vgg:
                                    matches = 0
                                    matches_vgg = 0
                                    for face_id, face_embedding in self.classifier.embeddings[object_id].items():
                                        match, score = self.classifier.is_match(face_embedding, embedding)

                                        if match is True:
                                            matches += 1

                                            if matches >= self.matches:
                                                print('Pre matching for', object_id, score)

                                                if embedding_vgg is None:
                                                    embedding_vgg = self.classifier.get_embedding_vgg(data['face'])

                                                for _face_id, _face_embedding in self.classifier.embeddings_vgg[object_id].items():
                                                    match, score = self.classifier.is_match_vgg(_face_embedding, embedding_vgg)

                                                    if match is True:
                                                        matches_vgg += 1

                                                        if matches_vgg >= self.matches_vgg:
                                                            rejected = False
                                                            self.identity[cube_id] = {
                                                                "object_id": object_id,
                                                                "confidence": confidence
                                                            }
                                                            print('Matching for', object_id, score)
                                                            break

                                                break

                    if cube_id in self.identity:
                        send_confidence = self.identity[cube_id]['confidence']
                        send_object_id = self.identity[cube_id]['object_id']
                        for row in db.Objects.select().where(db.Objects.id == send_object_id).limit(1).execute():
                            send_individual_id = row.individual_id

                            for _row in db.ObjectFaces.select().where(db.ObjectFaces.object_id == send_object_id).limit(1).execute():
                                data = pickle.loads(_row.data)
                                _face = data['rect']['face']

                                im = Image.fromarray(_face.astype("uint8"))
                                raw = io.BytesIO()
                                im.save(raw, "PNG")
                                raw.seek(0)
                                face = base64.b64encode(raw.read())

                            print('Detected face', send_object_id, cube_id)
                    else:
                        send_confidence = confidence
                        rejected = True

            # cv2.imwrite('images/test/1.jpg', face)
            print('Cube id', cube_id)

            self.mq.send(json.dumps({
                "camera_id": self.camera.id,
                "individual_id": send_individual_id,
                "recognition_ts": datetime.utcnow().timestamp(),
                "data": {
                    "x1": data['x'],
                    "y1": data['y'],
                    "width": data['width'],
                    "height": data['height'],
                    "screen_width": screen_width,
                    "screen_height": screen_height,
                    "object_id": send_object_id,
                    "cube_id": cube_id,
                    "confidence": send_confidence,
                    "rejected": rejected,
                    "face": face
                }
            }))

            if self.classifier.training is True:
                if self.classifier.training_object_id:
                    self.classifier.face_store(data['face'], cube_id)

            print(datetime.now() - start_time)
        except Exception:
            print('Broken frame')
