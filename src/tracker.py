import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.nextobject_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid_id):
        self.objects[self.nextobject_id] = centroid_id
        self.disappeared[self.nextobject_id] = 0
        self.nextobject_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        inputcentroid_ids = np.zeros((len(rects), 3), dtype="int")

        for (i, (start_x, start_y, end_x, end_y, index)) in enumerate(rects):
            cX = int((start_x + end_x) / 2.0)
            cY = int((start_y + end_y) / 2.0)
            inputcentroid_ids[i] = (cX, cY, index)

        if len(self.objects) == 0:
            for i in range(0, len(inputcentroid_ids)):
                self.register(inputcentroid_ids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroid_ids = list(self.objects.values())
            distance = dist.cdist(np.array(object_centroid_ids), inputcentroid_ids)
            rows = distance.min(axis=1).argsort()
            cols = distance.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if distance[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = inputcentroid_ids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, distance.shape[0])).difference(used_rows)
            unused_cols = set(range(0, distance.shape[1])).difference(used_cols)
            if distance.shape[0] >= distance.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(inputcentroid_ids[col])

        return self.objects
