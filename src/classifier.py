import cv2
import os
import db
import json
import pickle
import numpy
import grpc_client
from datetime import datetime
from face_detector import FaceDetector
from mq import Mq
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import cosine


class Classifier:
    def __init__(self):
        self.face_detector = FaceDetector()

        self.classifier = 'linear'
        self.grab_faces = 300
        self.use_faces = 300
        self.min_faces = 200
        self.thresh = 0.15
        self.thresh_vgg = 0.18

        self.dnn_picture_size_x = 96
        self.dnn_picture_size_y = 96
        self.dnn_vgg_picture_size_x = 224
        self.dnn_vgg_picture_size_y = 224
        self.dumping_file = 'dump.pickle'
        self.dumping_embeddings_file = 'dump_embeddings.pickle'
        self.dumping_embeddings_vgg_file = 'dump_embeddings_vgg.pickle'
        self.dumping_clf_file = 'dump_clf.pickle'
        self.stored_faces = []

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
        self.embedder_vgg = cv2.dnn.readNetFromTorch('dnn/opencv/VGG_FACE.t7')
        self.training = False
        self.training_camera_id = 0
        self.training_individual_id = 0
        self.training_object_id = 0
        self.embeddings = dict()
        self.embeddings_vgg = dict()
        self.dnn_picture_size = (self.dnn_picture_size_x, self.dnn_picture_size_y)
        self.dnn_vgg_picture_size = (self.dnn_vgg_picture_size_x, self.dnn_vgg_picture_size_y)

        self.classifier_names = []
        self.classifier_embeddings = []

        exclude = dict()
        exclude[88] = [13219, 13220, 13221, 13222, 13223, 13224, 13225]
        exclude[91] = [13982, 13983, 13984, 13985, 13986, 13987, 13988, 13989, 13990, 13991]
        exclude[96] = [15391, 15394]
        exclude[97] = [15723, 15724, 15725, 15726, 15727, 15728, 15729, 15730, 15731, 15732, 15733, 15734, 15932, 15925]
        exclude[104] = [17728, 17730, 17736, 17738, 17740, 17742, 17744, 17763]
        exclude[129] = [22403, 22485, 22486, 22487, 22488, 22489, 22490, 22491, 22492, 22493, 22494, 22537, 22538,
                        22539, 22540, 22541, 22542, 22543, 22544]
        exclude[137] = [24774, 24775, 24776, 24777, 24778, 24779, 24780, 24781, 24782, 24783]
        exclude[138] = [25050, 25051, 25052, 25053, 25054, 25055, 25056, 25057, 25058, 25059]
        exclude[167] = [33942, 33943, 33944, 33945, 33946, 33947, 33948, 33949, 33950, 33951]
        exclude[178] = [37097, 37098, 37099, 37100, 37101, 37102, 37103, 37104, 37105]
        exclude[194] = [41882, 41883, 41884, 41885, 41886, 41887, 41888, 41889, 41891]
        exclude[197] = [42772, 42773, 42774, 42775, 42776, 42777, 42778, 42779, 42780, 42781, 43022, 43023, 43024,
                        43025, 43026, 43027, 43028, 43029, 43030]

        self.exclude = exclude

    def get_embedding(self, face):
        face_blob = cv2.dnn.blobFromImage(face, 1 / 255, self.dnn_picture_size, (0, 0, 0), swapRB=True, crop=False)

        self.embedder.setInput(face_blob)
        embedding = self.embedder.forward()

        return embedding

    def get_embedding_vgg(self, face):
        face_blob = cv2.dnn.blobFromImage(face, 1 / 255, self.dnn_vgg_picture_size, (0, 0, 0), swapRB=True, crop=False)

        self.embedder_vgg.setInput(face_blob)
        embedding = self.embedder_vgg.forward()

        return embedding

    def is_match(self, known_embedding, candidate_embedding):
        distance = numpy.sum(numpy.square(known_embedding - candidate_embedding))

        score = 'Distance = ' + str(distance)

        match = False
        if distance <= self.thresh:
            match = True

        return match, score

    def is_match_vgg(self, known_embedding, candidate_embedding):
        distance = known_embedding - candidate_embedding
        distance = numpy.sum(numpy.multiply(distance, distance))
        distance = numpy.sqrt(distance) * 1000

        score = "Distance = " + str(distance)

        match = False
        if distance <= self.thresh_vgg:
            match = True

        return match, score

    def update(self):
        self.update_db()
        self.update_embeddings()
        self.update_vgg_embeddings()
        self.update_classifier()

        print('Done')

    def update_db(self):
        objects = db.Objects.select().execute()
        object_faces = dict()

        # Dump data from db
        start_time = datetime.now()
        if os.path.exists(self.dumping_file) is False:
            os.system('rm -r images/dumps/')

            for row in objects:
                object_faces[row.id] = dict()

                faces = db.ObjectFaces.select().where(
                    db.ObjectFaces.object_id == row.id
                ).limit().execute()

                for face in faces:
                    data = pickle.loads(face.data)
                    _face = data['rect']['face']
                    if _face is not None:
                        object_faces[row.id][face.id] = _face

                        # Write test images
                        _dir = 'images/dumps/' + str(row.id)
                        if not os.path.exists(_dir):
                            os.makedirs(_dir)

                        cv2.imwrite('images/dumps/' + str(row.id) + '/' + str(face.id) + '.jpg', _face)

                        print('Object face data added for', face.object_id, face.id)

            f = open(self.dumping_file, "wb")
            f.write(pickle.dumps({
                "object_faces": object_faces
            }))
            f.close()

            print('DB data was dumped')

        print(datetime.now() - start_time)

    def update_embeddings(self):
        # Dump embeddings
        start_time = datetime.now()
        if os.path.exists(self.dumping_file) and os.path.exists(self.dumping_embeddings_file) is False:
            f = open(self.dumping_file, "rb")
            dump_data = pickle.loads(f.read())
            f.close()

            embeddings = dict()
            for object_id, faces in dump_data['object_faces'].items():
                embeddings[object_id] = dict()

                for face_id, face in faces.items():
                    embeddings[object_id][face_id] = self.get_embedding(face)

                    print('Embedding was generated for', object_id, face_id)

            f = open(self.dumping_embeddings_file, "wb")
            f.write(pickle.dumps({
                "embeddings": embeddings
            }))
            f.close()

            print('Face embeddings was dumped')

        print(datetime.now() - start_time)

        # Load embeddings to memory
        start_time = datetime.now()
        self.classifier_names = []
        self.classifier_embeddings = []
        if os.path.exists(self.dumping_embeddings_file):
            f = open(self.dumping_embeddings_file, "rb")
            dump_embeddings_data = pickle.loads(f.read())
            f.close()

            self.embeddings = dict()
            for object_id, faces in dump_embeddings_data['embeddings'].items():
                self.embeddings[object_id] = dict()
                _count = 0
                if len(faces) >= self.min_faces:
                    for face_id, embedding in faces.items():
                        if object_id in self.exclude:
                            if face_id in self.exclude[object_id]:
                                print('Embedding was excluded from list', object_id, face_id)
                            else:
                                _count += 1

                                self.embeddings[object_id][face_id] = embedding
                                self.classifier_names.append(object_id)
                                self.classifier_embeddings.append(embedding.flatten())
                        else:
                            _count += 1

                            self.embeddings[object_id][face_id] = embedding
                            self.classifier_names.append(object_id)
                            self.classifier_embeddings.append(embedding.flatten())

                        if _count >= self.use_faces:
                            break

        print(datetime.now() - start_time)

    def update_vgg_embeddings(self):
        # Dump embeddings
        start_time = datetime.now()
        if os.path.exists(self.dumping_file) and os.path.exists(self.dumping_embeddings_vgg_file) is False:
            f = open(self.dumping_file, "rb")
            dump_data = pickle.loads(f.read())
            f.close()

            embeddings = dict()
            for object_id, faces in dump_data['object_faces'].items():
                embeddings[object_id] = dict()

                for face_id, face in faces.items():
                    embeddings[object_id][face_id] = self.get_embedding_vgg(face)

                    print('VGG Embedding was generated for', object_id, face_id)

            f = open(self.dumping_embeddings_vgg_file, "wb")
            f.write(pickle.dumps({
                "embeddings": embeddings
            }))
            f.close()

            print('VGG Face embeddings was dumped')

        print(datetime.now() - start_time)

        # Load embeddings to memory
        start_time = datetime.now()
        if os.path.exists(self.dumping_embeddings_vgg_file):
            f = open(self.dumping_embeddings_vgg_file, "rb")
            dump_embeddings_data = pickle.loads(f.read())
            f.close()

            self.embeddings_vgg = dict()
            for object_id, faces in dump_embeddings_data['embeddings'].items():
                self.embeddings_vgg[object_id] = dict()
                _count = 0
                if len(faces) >= self.min_faces:
                    for face_id, embedding in faces.items():
                        if object_id in self.exclude:
                            if face_id in self.exclude[object_id]:
                                print('Embedding was excluded from list', object_id, face_id)
                            else:
                                _count += 1

                                self.embeddings_vgg[object_id][face_id] = embedding
                        else:
                            _count += 1

                            self.embeddings_vgg[object_id][face_id] = embedding

                        if _count >= self.use_faces:
                            break

        print(datetime.now() - start_time)

    def update_classifier(self):
        # Encode data for dnn
        start_time = datetime.now()
        if os.path.exists(self.dumping_clf_file) is False:
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

                print('Classifier training complete')
            except Exception:
                print('Classifier training error')

        if os.path.exists(self.dumping_clf_file):
            f = open(self.dumping_clf_file, "rb")
            dump_clf_data = pickle.loads(f.read())
            f.close()

            self.recognizer = dump_clf_data['recognizer']
            self.le = dump_clf_data['le']

            print('Classifier loaded from dump')

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
        body = self.mq_receive.get()
        if body is not None:
            data = pickle.loads(body)

            # Get event types
            if 'type' in data:
                # Execute service commands
                print('Service command execute was requested by front')
                print(data)

                _data = data['data']

                if 'type' in data:
                    # Execute other commands
                    _type = data['type']
                    camera_id = _data['camera_id']
                    individual_id = _data['individual_id']

                    if int(camera_id) == int(self.camera.id):
                        objects = db.Objects.select().where(
                            db.Objects.individual_id == individual_id
                        ).execute()

                        for row in objects:
                            print('Object Id', row.id)

                            if _type == 'training_cancel':
                                self.training_cancel()
                            elif _type == 'training_remove':
                                self.training_remove(row.id, individual_id)

                        print('Service command execute successful')
            elif 'train' in data:
                # Execute dnn training
                print('DNN Training was requested by front')
                print(data)

                _data = data['data']
                camera_id = _data['camera_id']
                individual_id = _data['individual_id']

                if int(camera_id) == int(self.camera.id):
                    self.training_start(individual_id)
