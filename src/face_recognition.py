import db
import numpy
import json
import grpc_client
import operator
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
        cube_id = data['id']

        start_time = datetime.now()
        embedding = self.classifier.get_embedding(data['face'])

        send_individual_id = None
        send_object_id = None
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
                                                        self.identity[cube_id] = object_id
                                                        print('Matching for', object_id, score)
                                                        break

                                            break

                if cube_id in self.identity:
                    send_object_id = self.identity[cube_id]
                    objects = db.Objects.select().where(db.Objects.id == send_object_id).limit(1).execute()

                    for row in objects:
                        send_individual_id = row.individual_id

                        print('Detected face', send_object_id, cube_id)

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
                "cube_id": cube_id
            }
        }))

        if self.classifier.training is True:
            if self.classifier.training_object_id:
                self.classifier.face_store(data['face'], cube_id)

        print(datetime.now() - start_time)
