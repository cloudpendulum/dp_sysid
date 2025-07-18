import numpy as np
import pandas as pd

from double_pendulum.controller.abstract_controller import AbstractController


class TrajPIDController(AbstractController):
    def __init__(self, csv_path, dt=0.01, torque_limit=[1.0, 1.0]):
        self.torque_limit = torque_limit
        self.dt = dt

        # load trajectory
        data = pd.read_csv(csv_path)
        time_traj = np.asarray(data["time"])
        pos1_traj = np.asarray(data["pos1"])
        pos2_traj = np.asarray(data["pos2"])
        vel1_traj = np.asarray(data["vel1"])
        vel2_traj = np.asarray(data["vel2"])

        self.T = time_traj.T
        self.X = np.asarray([pos1_traj, pos2_traj, vel1_traj, vel2_traj]).T

        # default weights
        self.Kp = 10.0
        self.Ki = 0.0
        self.Kd = 0.1

        # init pars
        self.errors1 = []
        self.errors2 = []
        self.errorsum1 = 0.0
        self.errorsum2 = 0.0
        self.counter = 0

    def set_parameters(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def init(self):
        self.errors1 = []
        self.errors2 = []
        self.errorsum1 = 0.0
        self.errorsum2 = 0.0
        self.counter = 0

    def get_control_output(self, x, t=None):
        p = self.X[self.counter][:2]
        e1 = p[0] - x[0]
        e2 = p[1] - x[1]
        # e1 = (e1 + np.pi) % (2*np.pi) - np.pi
        # e2 = (e2 + np.pi) % (2*np.pi) - np.pi
        self.errors1.append(e1)
        self.errors2.append(e2)
        self.errorsum1 += e1
        self.errorsum2 += e2

        P1 = self.Kp * e1
        P2 = self.Kp * e2

        # I1 = self.Ki*np.sum(np.asarray(self.errors1))*self.dt
        # I2 = self.Ki*np.sum(np.asarray(self.errors2))*self.dt
        I1 = self.Ki * self.errorsum1 * self.dt
        I2 = self.Ki * self.errorsum2 * self.dt

        if len(self.errors1) > 2:
            D1 = self.Kd * (self.errors1[-1] - self.errors1[-2]) / self.dt
            D2 = self.Kd * (self.errors2[-1] - self.errors2[-2]) / self.dt
        else:
            D1 = 0.0
            D2 = 0.0

        u1 = P1 + I1 + D1
        u2 = P2 + I2 + D2

        u1 = np.clip(u1, -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(u2, -self.torque_limit[1], self.torque_limit[1])
        u = np.asarray([u1, u2])

        if self.counter < np.shape(self.T)[0]-1:
            self.counter += 1

        return u

    def get_init_trajectory(self):
        return self.T, self.X, None
