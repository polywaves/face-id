import db
import json
import pickle
from datetime import datetime
from face_recognition import FaceRecognition
from mq import Mq
from classifier import Classifier


class App:
    def __init__(self):
        self.grab_faces = 300
        self.use_faces = 300
        self.stored_faces = []
        self.training = False
        self.last_camera_id = 0
        self.training_individual_id = 0
        self.training_object_id = 0

        self.classifier = Classifier()
        self.face_recognition = FaceRecognition()

        self.mq = Mq()
        self.mq.connect()

        self.mq_receive = Mq(
            queue='recognition_records_pass_to_dnn',
            exchange='recognition_records_faces',
            routing_key='new_record'
        )
        self.mq_receive.connect()

    def face_store(self, face, cube_id):
        # Get new face and store vector to db
        if len(self.stored_faces) < self.grab_faces:
            if face is not None:
                self.stored_faces.append(face)

                print('Store face to object id', self.training_object_id)
                print('amount', len(self.stored_faces))

        # Send learning progress to front
        percent = int(round((100 / self.grab_faces) * len(self.stored_faces)))
        training_individual_id = self.training_individual_id
        training_object_id = self.training_object_id

        if percent > 99:
            percent = 99

        if len(self.stored_faces) == self.grab_faces:
            for face in self.stored_faces:
                # Store to db
                db.ObjectFaces.create(
                    data=pickle.dumps({
                        "rect": {
                            "face": face
                        }
                    }),
                    object_id=self.training_object_id,
                    created_at=datetime.utcnow().date()
                )

            self.stored_faces = []
            self.training = False
            self.training_individual_id = 0
            self.training_object_id = 0

            self.classifier.update_db()
            self.mq.send(json.dumps({
                "type": "recognition_learning_progress",
                "camera_id": self.last_camera_id,
                "individual_id": training_individual_id,
                "data": {
                    "percent": 33,
                    "object_id": training_object_id,
                    "cube_id": cube_id
                }
            }), routing_key='recognition_training')

            self.classifier.update_embeddings()
            self.mq.send(json.dumps({
                "type": "recognition_learning_progress",
                "camera_id": self.last_camera_id,
                "individual_id": training_individual_id,
                "data": {
                    "percent": 66,
                    "object_id": training_object_id,
                    "cube_id": cube_id
                }
            }), routing_key='recognition_training')

            self.classifier.update_classifier()
            self.mq.send(json.dumps({
                "type": "recognition_learning_progress",
                "camera_id": self.last_camera_id,
                "individual_id": training_individual_id,
                "data": {
                    "percent": 100,
                    "object_id": training_object_id,
                    "cube_id": cube_id
                }
            }), routing_key='recognition_training')
        else:
            self.mq.send(json.dumps({
                "type": "recognition_learning_progress",
                "camera_id": self.last_camera_id,
                "individual_id": training_individual_id,
                "data": {
                    "percent": percent,
                    "object_id": training_object_id,
                    "cube_id": cube_id
                }
            }), routing_key='recognition_training')

    def training_start(self, individual_id):
        self.training = True
        self.training_individual_id = individual_id
        self.stored_faces = []

        # Create object id
        if self.training_object_id == 0:
            object_id = db.Objects.insert(
                camera_id=self.last_camera_id,
                individual_id=self.training_individual_id
            ).execute()

            self.training_object_id = object_id

    def training_cancel(self):
        self.stored_faces = []
        self.mq.send(json.dumps({
            "type": "recognition_learning_progress",
            "camera_id": self.last_camera_id,
            "individual_id": self.training_individual_id,
            "data": {
                "percent": 0,
                "object_id": self.training_object_id,
                "cube_id": 0
            }
        }), routing_key='recognition_training')

        self.training = False

    def training_remove(self, object_id, individual_id):
        self.stored_faces = []

        db.Objects.delete().where(
            db.Objects.id == object_id
        ).execute()

        db.ObjectFaces.delete().where(
            db.ObjectFaces.object_id == object_id
        ).execute()

        self.training_individual_id = 0
        self.training_object_id = 0
        self.last_camera_id = 0

        self.mq.send(json.dumps({
            "type": "recognition_learning_progress",
            "camera_id": self.last_camera_id,
            "individual_id": individual_id,
            "data": {
                "percent": 0,
                "object_id": object_id
            }
        }), routing_key='recognition_training')

    def consume(self, channel, method_frame, properties, body):
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)

        if body is not None:
            stream = pickle.loads(body)
            data = stream['data']

            # Get event types
            if 'type' in stream:
                print(stream)

                # Execute other commands
                command = stream['type']
                camera_id = data['camera_id']
                individual_id = data['individual_id']

                objects = db.Objects.select().where(
                    db.Objects.individual_id == individual_id
                ).execute()

                for row in objects:
                    if command == 'training_cancel':
                        self.training_cancel()
                        self.last_camera_id = camera_id
                    elif command == 'training_remove':
                        self.training_remove(row.id, individual_id)
                        self.last_camera_id = camera_id
            elif 'train' in stream:
                print(stream)

                camera_id = data['camera_id']
                individual_id = data['individual_id']

                self.training_start(individual_id)
                self.last_camera_id = camera_id
            elif 'detect' in stream:
                rects = []
                for value in data['rects']:
                    rect = self.face_recognition.identify(data['index'], value, data['screen_width'], data['screen_height'])
                    if rect:
                        if self.training is True:
                            if self.training_object_id:
                                self.face_store(value['face'], value['id'])

                        rects.append(rect)

                self.mq.send(json.dumps({
                    "camera_id": data['camera_id'],
                    "individual_id": None,
                    "recognition_ts": datetime.utcnow().timestamp(),
                    "data": rects
                }))

    def init(self):
        self.classifier.update()
        self.face_recognition.classifier = self.classifier
        self.mq_receive.consume(self.consume)


app = App()
app.init()
