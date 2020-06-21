import operator
import pickle
import db
import numpy
import json
import grpc_client
from datetime import datetime
from classifier import Classifier
from mq import Mq


class FaceRecognition:
    def __init__(self):
        self.confidence = 30
        self.retries = 10
        self.clear_delay = 200

        # Init operations
        self.mq = Mq()
        self.mq.connect()
        self.camera = grpc_client.get_camera()
        self.classifier = Classifier()

        self.confidences = dict()
        self.identities = dict()
        self.identified = dict()

    def face_identification(self, data, screen_width, screen_height):
        if self.classifier.dataset.check_face(data['face']) == 1:
            cube_id = data['id']

            start_time = datetime.now()
            embedding = self.classifier.get_embedding(data['face'])

            send_individual_id = None
            send_object_id = None
            send_confidence = 0
            face = None
            rejected = None
            if self.classifier.training is False and self.classifier.recognizer:
                predictions = self.classifier.recognizer.predict_proba(embedding)[0]
                max_value = numpy.argmax(predictions)
                confidence = int(predictions[max_value] * 100)
                object_id = int(self.classifier.le.classes_[max_value])

                if object_id > 0:
                    if object_id in self.classifier.embeddings:
                        for face_id, face_embedding in self.classifier.embeddings[object_id].items():
                            match, distance = self.classifier.is_match(face_embedding, embedding)
                            if match is True:
                                if cube_id not in self.identities:
                                    self.confidences[cube_id] = dict()
                                    self.identities[cube_id] = dict()

                                if object_id not in self.identities[cube_id]:
                                    self.confidences[cube_id][object_id] = 0
                                    self.identities[cube_id][object_id] = 0

                                self.identities[cube_id][object_id] += 1

                                if confidence > self.confidences[cube_id][object_id]:
                                    self.confidences[cube_id][object_id] = confidence

                                print('Predicted', object_id, distance, confidence, '%')

                                break

                if cube_id in self.identities:
                    identities = self.identities[cube_id]
                    confidences = self.confidences[cube_id]
                    max_objects = max(identities.items(), key=operator.itemgetter(1))

                    send_object_id = max_objects[0]
                    send_confidence = confidences[send_object_id]
                    rejected = False

                    face = self.classifier.objects[send_object_id]['face']
                    send_individual_id = self.classifier.objects[send_object_id]['individual_id']

                    print('Detected face', send_object_id, cube_id)
                    print(identities)
                else:
                    send_confidence = confidence

                    if cube_id in self.identified and cube_id not in self.identities:
                        if self.identified[cube_id] >= self.retries:
                            if confidence <= self.confidence:
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
                    "face": face,
                    "rejected": rejected,
                    "clear_delay": self.clear_delay
                }
            }))

            if self.classifier.training is True:
                if self.classifier.training_object_id:
                    self.classifier.face_store(data['face'], cube_id)

            print(datetime.now() - start_time)
