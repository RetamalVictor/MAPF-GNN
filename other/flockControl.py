import time
import numpy as np

from pprint import pprint



class Flock:
    """
    TO-DO:
    - extract take off and landing to a parent class
    """

    def __init__(self, env, args, logger=None):
        """
        TO-DO:
        [ ] Describe the parameters here
        """
        if logger is not None:
            self.logger = logger
            self.logging = True
        else:
            self.logging = False

        self.env = env
        self.dt = 1 / env.SIM_FREQ
        self.debug = args["debug"]
        self.gui = args["gui"]
        self.idle_time = args["idle_time"]
        self.cntrl_freq = args["cntrl_freq"]
        self.aggr_phy_steps = args["aggr_phy_steps"]
        self.num_drones = args["num_drones"]
        self.sensing_distance = args["sensing_distance"]
        self.epsilon = args["epsilon"]
        self.sigma = args["sigma"]
        self.umax_const = args["umax_cont"]
        self.wmax = args["wmax"]
        self.k1 = args["k1"]
        self.k2 = args["k2"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.positions = np.zeros((2, self.num_drones))
        self.headings = np.array(
            [np.random.uniform(-np.pi, np.pi) for _ in range(self.num_drones)]
        )
        self.action = {str(i): np.array([0, 0, 0, 1]) for i in range(self.num_drones)}

    def plot(self):
        self.logger.plot()

    def init(self):
        START = time.time()
        for i in range(1, int(self.idle_time * self.env.SIM_FREQ), self.aggr_phy_steps):

            _, _, _, _ = self.env.step(self.action)

            if i % self.env.SIM_FREQ == 0:
                self.env.render()

            if self.gui:
                sync(i, START, self.env.TIMESTEP)

    def takeOff(self, target_height, duration):
        self.action = {str(i): np.array([0, 0, 1, 1]) for i in range(self.num_drones)}
        START = time.time()
        for i in range(1, int(duration * self.env.SIM_FREQ), self.aggr_phy_steps):

            _, _, _, _ = self.env.step(self.action)
            if self.env._getDroneStateVector(0)[2] == target_height:
                break

            if i % self.cntrl_freq == 0:
                for j in range(self.num_drones):
                    self.action[str(j)] = np.array([0, 0, 1, 1 / i])

            if i % self.env.SIM_FREQ == 0:
                self.env.render()

            if self.gui:
                sync(i, START, self.env.TIMESTEP)

    def flockLoop(self, duration_sec=None):
        self._restartActions()
        START = time.time()
        for i in range(0, int(duration_sec * self.env.SIM_FREQ), self.aggr_phy_steps):

            obs, _, _, _ = self.env.step(self.action)

            for j in range(self.num_drones):
                self.env._debugHeadings(j, self.headings)

            if i % self.cntrl_freq == 0:
                self._updatePositions()
                self._getForcesFromState()
                self._getVelocityFromForces()
                self._updateHeadings()
                self._computeActions()

            if self.logging:
                self._logSimulation(i, obs, 2)

            if i % self.env.SIM_FREQ == 0:
                self.env.render()

            if self.gui:
                sync(i, START, self.env.TIMESTEP)

    def getActions(self):
        self._computeActions()
        return self.action

    def close(self):
        self.env.close()

    def reset(self):
        obs = self.env.reset()
        return obs

    def _updatePositions(self):
        for j in range(self.num_drones):
            state = self.env._getDroneStateVector(j)
            self.positions[0][j] = state[0]
            self.positions[1][j] = state[1]

    def _getForcesFromState(self):

        # Create Matrices from positions and heading
        X1, XT = np.meshgrid(self.positions[0], self.positions[0])
        Y1, YT = np.meshgrid(self.positions[1], self.positions[1])
        H1, HT = np.meshgrid(self.headings, self.headings)

        # Calculate distance matrix
        D_ij_x = X1 - XT
        D_ij_y = Y1 - YT
        D_ij = np.sqrt(np.multiply(D_ij_x, D_ij_x) + np.multiply(D_ij_y, D_ij_y))
        D_ij[(D_ij >= 3.5) | (D_ij == 0)] = np.inf

        # Calculating bearing angles
        Bearnig_angles = np.arctan2(D_ij_y, D_ij_x)
        Bearnig_angles_local = (
            Bearnig_angles - HT + self.headings * np.identity(self.num_drones)
        )

        # Calculating Force for proximal control
        forces = -(self.epsilon) * (
            (self.sigma**4 / D_ij**5) - (self.sigma**2 / D_ij**3)
        )
        forces[D_ij == np.inf] = 0.0
        forces = np.nan_to_num(forces)

        p_x = self.alpha * np.sum(forces * np.cos(Bearnig_angles_local), axis=1)
        p_y = self.alpha * np.sum(forces * np.sin(Bearnig_angles_local), axis=1)

        # Averaging heading vector
        # now all the headings are always present
        headings_avg = self._getAvgHeadingAngle(self.headings)

        h_x = self.beta * np.cos(headings_avg - self.headings)
        h_y = self.beta * np.sin(headings_avg - self.headings)

        # Computing Control vector
        fx = p_x + h_x
        fy = p_y + h_y

        # Translating force vector to Linear & angular velocity
        U = fx * self.k1
        U[U > self.umax_const] = self.umax_const
        U[U < 0] = 0.005

        W = fy * self.k2
        W[W > self.wmax] = self.wmax
        W[W < -self.wmax] = -self.wmax

        self.linear_velocity = U
        self.angular_velocity = W

    def _getVelocityFromForces(self):
        self.vx = self.linear_velocity * np.cos(self.headings) * self.dt
        self.vy = self.linear_velocity * np.sin(self.headings) * self.dt

    def _updateHeadings(self):
        self.headings = self.wraptopi(self.headings + self.angular_velocity * self.dt)

    def _computeActions(self):
        for j in range(self.num_drones):
            self.action[str(j)] = np.array([self.vx[j], self.vy[j], 0, 1])

    def _logSimulation(self, tick, obs, drones=None):
        if drones is None:
            drones = self.num_drones
        for j in range(drones):
            self.logger.log(
                drone=j,
                timestamp=tick / self.env.SIM_FREQ,
                state=obs[str(j)]["state"],
            )

    def _restartActions(self):
        self.action = {str(i): np.array([0, 0, 0, 1]) for i in range(self.num_drones)}

    def _computeRewards(self):
        rewards = self.env._computeRewardForTest()
        return rewards

    @staticmethod
    def wraptopi(x):
        x = x % (np.pi * 2)
        x = (x + (np.pi * 2)) % (np.pi * 2)
        x[x > np.pi] = x[x > np.pi] - (np.pi * 2)
        return x

    def _getAvgHeadingAngle(self, headings):
        """
        returns the angle from a sum of headings in radians
        """
        headings = headings[headings != 0]
        X = np.sum(np.cos(headings))
        Y = np.sum(np.sin(headings))
        return np.arctan2(Y, X)
