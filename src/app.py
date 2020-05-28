import os
import cv2
import db
import numpy
import json
import pickle
import threading
import imutils
import grpc_client
import operator
from datetime import datetime
from mq import Mq
from face_detector import FaceDetector
from tracker import CentroidTracker
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class App:
    def __init__(self):
        # Defaults
        self.classifier = 'linear'
        self.grab_faces = 300
        self.use_faces = 300
        self.min_faces = 200
        self.stream_request_rate = 2
        self.thresh = 0.2
        self.matches = 1
        self.max_objects_thresh = 1
        self.confidence = 50
        self.min_confidence = 40
        self.dnn_picture_size_x = 96
        self.dnn_picture_size_y = 96
        self.dumping_file = 'dump.pickle'
        self.dumping_embeddings_file = 'dump_embeddings.pickle'
        self.dumping_clf_file = 'dump_clf.pickle'

        # Init operations
        self.mq = Mq()
        self.mq.connect()
        self.mq_receive = Mq(
            queue='recognition_records_pass_to_dnn',
            exchange='recognition_records_faces',
            routing_key='new_record'
        )
        self.mq_receive.connect()
        self.camera = grpc_client.get_camera()
        self.face_detector = FaceDetector()
        self.ct = CentroidTracker()

        self.recognizer = None
        self.le = None
        self.embedder = cv2.dnn.readNetFromTorch('dnn/opencv/openface_nn4.v2.t7')
        self.training = False
        self.training_camera_id = 0
        self.training_individual_id = 0
        self.training_object_id = 0
        self.embeddings = dict()
        self.dnn_picture_size = (self.dnn_picture_size_x, self.dnn_picture_size_y)
        self.temp = dict()
        self.identity = dict()
        self.stored_faces = []

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

    def update(self):
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
        names = []
        embeddings = []
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
                                names.append(object_id)
                                embeddings.append(embedding.flatten())
                        else:
                            _count += 1

                            self.embeddings[object_id][face_id] = embedding
                            names.append(object_id)
                            embeddings.append(embedding.flatten())

                        if _count >= self.use_faces:
                            break

        print(datetime.now() - start_time)

        # Encode data for dnn
        start_time = datetime.now()
        if os.path.exists(self.dumping_clf_file) is False:
            try:
                le = LabelEncoder()
                labels = le.fit_transform(names)

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

                clf.fit(embeddings, labels)

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
        print('Done')

    def face_identification(self, rect, cube_id, screen_width, screen_height):
        start_time = datetime.now()
        embedding = self.get_embedding(rect['face'])

        send_individual_id = None
        send_object_id = None
        if self.training is False:
            if self.recognizer:
                predictions = self.recognizer.predict_proba(embedding)[0]
                max_value = numpy.argmax(predictions)
                confidence = int(predictions[max_value] * 100)
                object_id = int(self.le.classes_[max_value])

                if confidence >= self.confidence:
                    print(object_id, confidence, '%')

                    if cube_id not in self.temp:
                        self.temp[cube_id] = dict()

                    if object_id not in self.temp[cube_id]:
                        self.temp[cube_id][object_id] = 1

                    self.temp[cube_id][object_id] += 1

                    max_object = max(self.temp[cube_id].items(), key=operator.itemgetter(1))

                    print(self.temp[cube_id])

                    if max_object[1] > 0:
                        if max_object[0] in self.embeddings:
                            matches = 0
                            for face_id, face_embedding in self.embeddings[max_object[0]].items():
                                match, score = self.is_match(face_embedding, embedding)

                                if match is True:
                                    matches += 1

                                    if matches >= self.matches:
                                        self.identity[cube_id] = max_object[0]
                                        print('Matching for', max_object[0], score)
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
                "x1": rect['x1'],
                "y1": rect['y1'],
                "width": rect['width'],
                "height": rect['height'],
                "screen_width": screen_width,
                "screen_height": screen_height,
                "object_id": send_object_id,
                "cube_id": cube_id
            }
        }))

        if self.training is True:
            if self.training_object_id:
                self.rect_store(rect, cube_id)

        print(datetime.now() - start_time)

    def get_embedding(self, face):
        face_blob = cv2.dnn.blobFromImage(face, 1 / 255, self.dnn_picture_size, (0, 0, 0), swapRB=True, crop=False)

        self.embedder.setInput(face_blob)
        embedding = self.embedder.forward()

        return embedding

    def render(self):
        capture = cv2.VideoCapture(self.camera.stream_uri)
        index = 0
        while capture.isOpened():
            index += 1

            # Get current frame
            resolve, frame = capture.read()
            if not resolve:
                break

            # Get weights
            height, width = frame.shape[:2]
            if index % self.stream_request_rate == 0:
                centroids = []
                rects = dict()
                for rect in self.face_detector.get_rects(frame):
                    centroids.append((rect['x1'], rect['y1'], rect['x2'], rect['y2'], rect['index']))
                    rects[rect['index']] = rect

                objects = self.ct.update(centroids)
                for (cube_id, centroid) in objects.items():
                    if centroid[2] in rects:
                        rect = rects[centroid[2]]
                        self.face_identification(rect, cube_id, width, height)

        capture.release()
        print('Stream not found')

    def is_match(self, known_embedding, candidate_embedding):
        distance = numpy.sum(numpy.square(known_embedding - candidate_embedding))

        score = 'Distance = ' + str(distance)

        match = False
        if distance <= self.thresh:
            match = True

        return match, score

    def consume(self, channel, method_frame, properties, body):
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
        data = pickle.loads(body)

        f = open('consume.log', "w")
        f.write(json.dumps(data))
        f.close()

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

    def rect_store(self, rect, cube_id):
        # Get new face and store vector to db
        if len(self.stored_faces) < self.grab_faces:
            if rect['face'] is not None:
                self.stored_faces.append(rect)

                print('Store face to object id', self.training_object_id)
                print('amount', len(self.stored_faces))

        # Send learning progress to front
        percent = int(round((100 / self.grab_faces) * len(self.stored_faces)))
        training_individual_id = self.training_individual_id
        training_object_id = self.training_object_id

        if percent > 99:
            percent = 99

        if len(self.stored_faces) == self.grab_faces:
            for _rect in self.stored_faces:
                # Store to db
                db.ObjectFaces.create(
                    data=pickle.dumps({
                        "rect": _rect
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


app = App()
app.update()

consume = threading.Thread(target=app.mq_receive.consume, args=(app.consume,))
consume.start()

app.render()
