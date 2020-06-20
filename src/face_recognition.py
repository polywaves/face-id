import operator
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
        self.confidence = 30
        self.retries = 10
        self.clear_delay = 200

        # Init operations
        self.mq = Mq()
        self.mq.connect()
        self.camera = grpc_client.get_camera()
        self.classifier = Classifier()

        self.temp = dict()
        self.identity = dict()
        self.identified = dict()

    def face_identification(self, data, screen_width, screen_height):
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
                        match, score = self.classifier.is_match(face_embedding, embedding)
                        if match is True:
                            if cube_id not in self.temp:
                                self.temp[cube_id] = dict()

                            if object_id not in self.temp[cube_id]:
                                self.temp[cube_id][object_id] = 0

                            self.temp[cube_id][object_id] += 1
                            self.identity[cube_id] = {
                                "confidence": confidence,
                                "object_id": object_id,
                                "score": score
                            }

                            print('Predicted', object_id, confidence, '%')

                    if cube_id not in self.identified:
                        self.identified[cube_id] = 0

                    self.identified[cube_id] += 1

            if cube_id in self.identity and cube_id in self.temp:
                print(self.temp[cube_id])

                identity = self.identity[cube_id]
                temp = self.temp[cube_id]
                _max = max(temp.items(), key=operator.itemgetter(1))
                max_object_id = _max[0]

                rejected = False
                send_confidence = identity['confidence']
                send_object_id = identity['object_id']
                # send_object_id = max_object_id
                for row in db.Objects.select().where(db.Objects.id == send_object_id).limit(1).execute():
                    send_individual_id = row.individual_id

                    for _row in db.ObjectFaces.select().where(db.ObjectFaces.object_id == send_object_id).limit(1).execute():
                        _data = pickle.loads(_row.data)
                        _face = _data['rect']['face']

                        im = Image.fromarray(_face.astype("uint8"))
                        raw = io.BytesIO()
                        im.save(raw, "JPEG")
                        raw.seek(0)
                        face = str(base64.b64encode(raw.read()).decode('utf-8'))

                    print('Detected face', send_object_id, cube_id)
            else:
                send_confidence = confidence

                if cube_id in self.identified and cube_id not in self.identity:
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
