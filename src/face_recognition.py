import operator
import numpy
from datetime import datetime


class FaceRecognition:
    def __init__(self):
        self.confidence = 30
        self.retries = 5

        # Init operations
        self.classifier = None

        self.confidences = dict()
        self.identities = dict()
        self.identified = dict()

    def identify(self, index, data, screen_width, screen_height):
        if self.classifier.dataset.check_face(data['face']) == 1:
            cube_id = data['id']

            start_time = datetime.now()
            embedding = self.classifier.get_embedding(data['face'])

            send_individual_id = None
            send_object_id = None
            send_confidence = 0
            face = None
            rejected = None
            if self.classifier.recognizer:
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
                    face = self.classifier.dataset.create_base64_face(data['face'])

                    if cube_id in self.identified and cube_id not in self.identities:
                        if self.identified[cube_id] >= self.retries:
                            if confidence <= self.confidence:
                                rejected = True

                print('Cube id', cube_id)

            print(datetime.now() - start_time)

            return {
                "individual_id": send_individual_id,
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
                "index": index
            }
        else:
            return None
