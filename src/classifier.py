import cv2
import os
import db
import json
import pickle
import imutils
import numpy
import grpc_client
from dataset import Dataset
from datetime import datetime
from face_detector import FaceDetector
from mq import Mq
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class Classifier:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.dataset = Dataset()

        self.classifier = 'linear'
        self.grab_faces = 300
        self.use_faces = 300
        self.thresh = 0.4

        self.dnn_picture_size_x = 96
        self.dnn_picture_size_y = 96
        self.dumping_file = 'dump.pickle'
        self.dumping_embeddings_file = 'dump_embeddings.pickle'
        self.dumping_clf_file = 'dump_clf.pickle'
        self.stored_faces = []
        self.dataset_path = 'images/dataset/'

        self.camera = grpc_client.get_camera()
        self.mq = Mq()
        self.mq.connect()
        self.mq_receive = Mq(
            queue='recognition_records_pass_to_dnn',
            exchange='recognition_records_faces',
            routing_key='new_record'
        )
        self.mq_receive.connect()

        self.recognizer = None
        self.le = None
        self.embedder = cv2.dnn.readNetFromTorch('dnn/opencv/openface_nn4.v2.t7')
        self.training = False
        self.training_camera_id = 0
        self.training_individual_id = 0
        self.training_object_id = 0
        self.embeddings = dict()
        self.dnn_picture_size = (self.dnn_picture_size_x, self.dnn_picture_size_y)

        self.objects = dict()
        self.object_faces = dict()
        self.embeddings = dict()
        self.classifier_names = []
        self.classifier_embeddings = []

    def get_embedding(self, face):
        face_blob = cv2.dnn.blobFromImage(face, 1 / 255, self.dnn_picture_size, (0, 0, 0), swapRB=True, crop=False)

        self.embedder.setInput(face_blob)
        embedding = self.embedder.forward()

        return embedding

    def is_match(self, known_embedding, candidate_embedding):
        distance = known_embedding - candidate_embedding
        distance = numpy.sum(numpy.multiply(distance, distance))
        distance = numpy.sqrt(distance)

        match = False
        if distance <= self.thresh:
            match = True

        return match, distance

    def update(self):
        start_time = datetime.now()
        self.update_db()
        self.update_embeddings()
        self.update_classifier()

        print('Done')
        print(datetime.now() - start_time)

    def update_db(self):
        start_time = datetime.now()

        objects = db.Objects.select().order_by(db.Objects.id.asc()).execute()
        self.objects = dict()
        self.object_faces = dict()
        dump_object_faces = dict()

        for row in objects:
            self.objects[row.id] = {
                "individual_id": row.individual_id,
                "face": None
            }
            self.object_faces[row.id] = dict()

        if os.path.exists(self.dumping_file):
            f = open(self.dumping_file, "rb")
            dump_data = pickle.loads(f.read())
            f.close()

            for object_id, faces in dump_data['object_faces'].items():
                dump_object_faces[object_id] = faces
        else:
            os.system('rm -r ' + self.dataset_path)

        for object_id, data in self.object_faces.items():
            if object_id in dump_object_faces:
                self.object_faces[object_id] = dump_object_faces[object_id]

                # print('Object face data restored from dump for', object_id)
            else:
                object_path = os.path.join(self.dataset_path, str(object_id))

                if os.path.exists(object_path):
                    os.system('rm -r ' + object_path)
                os.makedirs(object_path)

                faces = db.ObjectFaces.select().where(
                    db.ObjectFaces.object_id == object_id
                ).limit().execute()

                for _row in faces:
                    data = pickle.loads(_row.data)
                    face = imutils.resize(data['rect']['face'], self.dnn_picture_size_x)
                    if self.dataset.check_face(face) == 1:
                        self.object_faces[object_id][_row.id] = face

                        # Write test images
                        cv2.imwrite(os.path.join(object_path, str(_row.id) + '.jpg'), face)
                        # print('Object face data added for', object_id, _row.id)

            count = 0
            for face_id, face in self.object_faces[object_id].items():
                count += 1

                self.objects[object_id]['face'] = self.dataset.create_base64_face(face)

                if count == 1:
                    break

        f = open(self.dumping_file, "wb")
        f.write(pickle.dumps({
            "object_faces": self.object_faces
        }))
        f.close()

        print('DB data was dumped')
        print(datetime.now() - start_time)

    def update_embeddings(self):
        start_time = datetime.now()
        embeddings = dict()
        if os.path.exists(self.dumping_embeddings_file):
            f = open(self.dumping_embeddings_file, "rb")
            dump_embeddings_data = pickle.loads(f.read())
            f.close()

            for object_id, data in dump_embeddings_data['embeddings'].items():
                embeddings[object_id] = data

        self.embeddings = dict()
        for object_id, faces in self.object_faces.items():
            if object_id in embeddings:
                self.embeddings[object_id] = embeddings[object_id]

                # print('Embeddings was restored from dump', object_id)
            else:
                self.embeddings[object_id] = dict()

                for face_id, face in faces.items():
                    self.embeddings[object_id][face_id] = self.get_embedding(face)

                    # print('Embedding was generated for', object_id, face_id)

        f = open(self.dumping_embeddings_file, "wb")
        f.write(pickle.dumps({
            "embeddings": self.embeddings
        }))
        f.close()

        print(datetime.now() - start_time)

    def update_classifier(self):
        # Encode data for dnn
        start_time = datetime.now()

        self.classifier_names = []
        self.classifier_embeddings = []
        for object_id, faces in self.embeddings.items():
            count = 0
            for face_id, embedding in faces.items():
                count += 1

                self.classifier_names.append(object_id)
                self.classifier_embeddings.append(embedding.flatten())

                if count == self.use_faces:
                    break

        if os.path.exists(self.dumping_clf_file):
            f = open(self.dumping_clf_file, "rb")
            dump_clf_data = pickle.loads(f.read())
            f.close()

            self.recognizer = dump_clf_data['recognizer']
            self.le = dump_clf_data['le']

            print('Classifier loaded from dump')
        else:
            try:
                le = LabelEncoder()
                labels = le.fit_transform(self.classifier_names)

                clf = None
                if self.classifier == 'linear':
                    clf = SVC(C=1, kernel="linear", probability=True)
                elif self.classifier == 'grid':
                    param_grid = [
                        {'C': [1, 10, 100, 1000],
                         'kernel': ['linear']},
                        {'C': [1, 10, 100, 1000],
                         'gamma': [0.001, 0.0001],
                         'kernel': ['rbf']}
                    ]
                    clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)

                clf.fit(self.classifier_embeddings, labels)

                f = open(self.dumping_clf_file, "wb")
                f.write(pickle.dumps({
                    "recognizer": clf,
                    "le": le
                }))
                f.close()

                self.recognizer = clf
                self.le = le

                print('Classifier training complete')
            except Exception:
                print('Classifier training error')

        print(datetime.now() - start_time)

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
            percent = 100

        self.mq.send(json.dumps({
            "type": "recognition_learning_progress",
            "camera_id": self.camera.id,
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
                camera_id=self.camera.id,
                individual_id=self.training_individual_id
            ).execute()

            self.training_object_id = object_id

    def training_cancel(self):
        self.stored_faces = []
        self.mq.send(json.dumps({
            "type": "recognition_learning_progress",
            "camera_id": self.camera.id,
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

        self.mq.send(json.dumps({
            "type": "recognition_learning_progress",
            "camera_id": self.camera.id,
            "individual_id": individual_id,
            "data": {
                "percent": 0,
                "object_id": object_id
            }
        }), routing_key='recognition_training')

    def consume(self):
        body, method_frame = self.mq_receive.get()

        if body is not None:
            data = pickle.loads(body)
            print(data)

            # Get event types
            if 'type' in data:
                _data = data['data']

                if 'type' in data:
                    # Execute other commands
                    _type = data['type']
                    camera_id = _data['camera_id']
                    individual_id = _data['individual_id']

                    if int(camera_id) == int(self.camera.id):
                        self.mq_receive.channel.basic_ack(delivery_tag=method_frame.delivery_tag)

                        objects = db.Objects.select().where(
                            db.Objects.individual_id == individual_id
                        ).execute()

                        for row in objects:
                            if _type == 'training_cancel':
                                self.training_cancel()
                            elif _type == 'training_remove':
                                self.training_remove(row.id, individual_id)
            elif 'train' in data:
                _data = data['data']
                camera_id = _data['camera_id']
                individual_id = _data['individual_id']

                if int(camera_id) == int(self.camera.id):
                    self.mq_receive.channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                    self.training_start(individual_id)
