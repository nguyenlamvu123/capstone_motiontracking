import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman import KalmanFilter

class Track:
    _id = 0

    def __init__(self, bbox):
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2

        self.kf = KalmanFilter(cx, cy, w, h)
        self.id = Track._id
        Track._id += 1

        self.missed = 0

    def predict(self):
        self.kf.predict()

    def update(self, bbox):
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        z = np.array([cx, cy, 0, 0, w, h])
        self.kf.update(z)
        self.missed = 0

    def get_bbox(self):
        cx, cy, _, _, w, h = self.kf.x
        return int(cx - w/2), int(cy - h/2), int(w), int(h)


class MultiObjectTracker:
    def __init__(self, max_missed=10):
        self.tracks = []
        self.max_missed = max_missed

    def update(self, detections):
        for t in self.tracks:
            t.predict()

        if len(self.tracks) == 0:
            for d in detections:
                self.tracks.append(Track(d))
            return self.tracks

        cost = np.zeros((len(self.tracks), len(detections)))

        for i, t in enumerate(self.tracks):
            tx, ty, tw, th = t.get_bbox()
            for j, d in enumerate(detections):
                dx, dy, dw, dh = d
                cost[i, j] = np.linalg.norm(
                    [(tx - dx), (ty - dy)]
                )

        row, col = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row, col):
            if cost[r, c] < 100:
                self.tracks[r].update(detections[c])
                assigned_tracks.add(r)
                assigned_dets.add(c)

        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.missed += 1

        self.tracks = [t for t in self.tracks if t.missed < self.max_missed]

        for i, d in enumerate(detections):
            if i not in assigned_dets:
                self.tracks.append(Track(d))

        return self.tracks
