import numpy as np

class KalmanFilter:
    def __init__(self, cx, cy, w, h):
        self.dt = 1.0

        self.x = np.array([cx, cy, 0, 0, w, h], dtype=float)

        self.F = np.eye(6)
        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt

        self.H = np.eye(6)

        self.P = np.eye(6) * 10
        self.Q = np.eye(6) * 0.01
        self.R = np.eye(6) * 1

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
