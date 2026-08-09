"""Batched GPU environment for MAPF rollouts.

GraphEnv steps one episode at a time in numpy, so an evaluation run issues one
tiny forward pass per step. The model costs the same at batch 1 as at batch 128
(kernel launch latency dominates a model this small), so the way to use the GPU
is to run many episodes concurrently and pay that latency once for all of them.

This class mirrors GraphEnv's semantics exactly - same action encoding, same
order of obstacle / boundary / collision resolution, same field of view and the
same nearest-4 communication graph - with a leading episode dimension and every
tensor resident on the device. tests/test_env_vectorized.py pins the equivalence.
"""

import numpy as np
import torch

# Index order matches GraphEnv.action_list: 0 idle, 1 right, 2 up, 3 left, 4 down
ACTION_DX = (0, 1, 0, -1, 0)
ACTION_DY = (0, 0, 1, 0, -1)


class VectorizedGraphEnv:
    def __init__(
        self,
        config,
        goals,
        obstacles=None,
        starting_positions=None,
        device="cuda",
        pad=3,
    ):
        """
        Args:
            config: the usual training config (board_size, num_agents, ...)
            goals: (E, N, 2) array of goal cells, x first
            obstacles: (E, M, 2) array of obstacle cells, or None
            starting_positions: (E, N, 2) array, or None to sample uniformly
            device: torch device for every rollout tensor
        """
        self.device = torch.device(device)
        self.config = config
        self.board_size = config["board_size"][0]
        self.nb_agents = config["num_agents"]
        self.sensing_range = config.get("sensing_range", 6)
        self.max_time = config["max_time"]
        self.pad = pad

        self.goal = torch.as_tensor(
            np.asarray(goals), dtype=torch.long, device=self.device
        )
        if self.goal.ndim != 3:
            raise ValueError(f"goals must be (E, N, 2), got {tuple(self.goal.shape)}")
        self.num_envs = self.goal.shape[0]

        self.dx = torch.tensor(ACTION_DX, dtype=torch.long, device=self.device)
        self.dy = torch.tensor(ACTION_DY, dtype=torch.long, device=self.device)

        self.obstacle_mask = torch.zeros(
            (self.num_envs, self.board_size, self.board_size),
            dtype=torch.bool,
            device=self.device,
        )
        if obstacles is not None:
            obs = torch.as_tensor(
                np.asarray(obstacles), dtype=torch.long, device=self.device
            )
            if obs.ndim == 2:  # one shared obstacle layout
                obs = obs.unsqueeze(0).expand(self.num_envs, -1, -1)
            env_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
            self.obstacle_mask[env_idx, obs[..., 1], obs[..., 0]] = True

        self.starting_positions = starting_positions
        self.reset()

    # ── episode lifecycle ────────────────────────────────────────────────

    def reset(self):
        self.time = 0
        if self.starting_positions is not None:
            start = torch.as_tensor(
                np.asarray(self.starting_positions), dtype=torch.long, device=self.device
            )
            self.posX = start[..., 0].clone()
            self.posY = start[..., 1].clone()
        else:
            # GraphEnv samples X and Y independently from the columns and rows
            # that hold no obstacle; free cells are not sampled jointly.
            free = ~self.obstacle_mask.any(dim=1)  # (E, W) columns without obstacles
            self.posX = self._sample_free(free)
            free_rows = ~self.obstacle_mask.any(dim=2)
            self.posY = self._sample_free(free_rows)

        # GraphEnv only stamps agents onto the board inside updateBoard(), which
        # runs during step. Right after a reset the board carries obstacles only,
        # so the first field of view shows no agents at all.
        self.board = torch.zeros(
            (self.num_envs, self.board_size, self.board_size),
            dtype=torch.float32,
            device=self.device,
        )
        self.board[self.obstacle_mask] = 2.0

        self._compute_graph()
        return self.get_observations()

    def _sample_free(self, free):
        """Sample one free coordinate per agent, per episode."""
        weights = free.float()
        weights = torch.where(
            weights.sum(dim=1, keepdim=True) > 0, weights, torch.ones_like(weights)
        )
        return torch.multinomial(weights, self.nb_agents, replacement=True)

    # ── dynamics ─────────────────────────────────────────────────────────

    def step(self, actions):
        """Advance every episode by one step.

        Args:
            actions: (E, N) tensor or array of action indices
        Returns:
            observations dict, done mask (E,)
        """
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        prevX, prevY = self.posX.clone(), self.posY.clone()

        newX = self.posX + self.dx[actions]
        newY = self.posY + self.dy[actions]

        # Obstacles are checked before the boundary clamp, matching GraphEnv:
        # a move off the board is simply not in the obstacle set.
        in_bounds = (
            (newX >= 0)
            & (newX < self.board_size)
            & (newY >= 0)
            & (newY < self.board_size)
        )
        clampedX = newX.clamp(0, self.board_size - 1)
        clampedY = newY.clamp(0, self.board_size - 1)
        env_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        hits_obstacle = self.obstacle_mask[env_idx, clampedY, clampedX] & in_bounds
        newX = torch.where(hits_obstacle, prevX, newX)
        newY = torch.where(hits_obstacle, prevY, newY)

        newX = newX.clamp(0, self.board_size - 1)
        newY = newY.clamp(0, self.board_size - 1)

        # Every agent sharing a cell reverts, including groups of three or more.
        cell = newY * self.board_size + newX
        occupancy = torch.zeros(
            (self.num_envs, self.board_size * self.board_size),
            dtype=torch.long,
            device=self.device,
        )
        occupancy.scatter_add_(1, cell, torch.ones_like(cell))
        collided = occupancy.gather(1, cell) > 1
        self.posX = torch.where(collided, prevX, newX)
        self.posY = torch.where(collided, prevY, newY)

        # updateBoard(): clear every vacated cell first, then stamp the new ones,
        # so an agent moving into a cell another just left survives the clear.
        self.board[env_idx, prevY, prevX] = 0.0
        self.board[env_idx, self.posY, self.posX] = 1.0

        self.time += 1
        self._compute_graph()
        return self.get_observations(), self.all_at_goal()

    def _compute_graph(self):
        """Nearest-4 communication graph within the sensing range."""
        pos = torch.stack([self.posX, self.posY], dim=2).float()  # (E, N, 2)
        dist = torch.cdist(pos, pos)
        dist = torch.where(dist >= self.sensing_range, torch.zeros_like(dist), dist)

        # Keep the four smallest non-zero distances per agent. Padding the sort
        # with +inf reproduces GraphEnv's behaviour when an agent has fewer than
        # four neighbours: nothing is dropped.
        masked = torch.where(dist > 0, dist, torch.full_like(dist, float("inf")))
        threshold = masked.sort(dim=2).values[:, :, 3:4]
        dist = torch.where(dist > threshold, torch.zeros_like(dist), dist)

        self.distance_matrix = dist
        self.adj_matrix = (dist != 0).float()

    # ── observations ─────────────────────────────────────────────────────

    def _fov(self):
        """(E, N, 2, 5, 5): channel 0 the local map, channel 1 the goal marker."""
        side = self.pad * 2 - 1
        board = torch.nn.functional.pad(
            self.board, (self.pad, self.pad, self.pad, self.pad)
        )

        offsets = torch.arange(side, device=self.device)
        rows = (self.posY + 1).unsqueeze(2) + offsets.view(1, 1, side)  # (E, N, 5)
        cols = (self.posX + 1).unsqueeze(2) + offsets.view(1, 1, side)

        env_idx = torch.arange(self.num_envs, device=self.device).view(-1, 1, 1, 1)
        window = board[env_idx, rows.unsqueeze(3), cols.unsqueeze(2)]
        window = window.flip(2)  # GraphEnv flips the rows of each patch

        fov = torch.zeros(
            (self.num_envs, self.nb_agents, 2, side, side),
            dtype=torch.float32,
            device=self.device,
        )
        fov[:, :, 0] = window

        gx, gy = self._goal_marker()
        env_i = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        agent_i = torch.arange(self.nb_agents, device=self.device).unsqueeze(0)
        fov[env_i, agent_i, 1, gx, gy] = 3.0
        return fov

    def _goal_marker(self):
        """Project the goal onto the field of view, as GraphEnv.map_goal does."""
        pad = self.pad
        goal_padded = self.goal + pad
        posx = self.posX + pad
        posy = self.posY + pad
        gpx, gpy = goal_padded[..., 0], goal_padded[..., 1]

        inside_x = (gpx < posx + pad - 1) & (gpx > posx - pad + 1)
        goal_x = torch.where(
            inside_x,
            -(posx - gpx) + pad - 1,
            torch.where(
                gpx <= posx - pad + 1,
                torch.zeros_like(gpx),
                torch.full_like(gpx, 1 + pad),
            ),
        )

        inside_y = (gpy < posy + pad - 1) & (gpy >= posy - pad + 1)
        goal_y = torch.where(
            inside_y,
            (posy - gpy) + pad - 1,
            torch.where(
                gpy <= posy - pad + 1,
                torch.full_like(gpy, 1 + pad),
                torch.zeros_like(gpy),
            ),
        )
        # map_goal returns (y, x) because the FOV is indexed numpy-style
        return goal_y, goal_x

    def get_observations(self):
        return {"fov": self._fov(), "adj_matrix": self.adj_matrix}

    # ── metrics ──────────────────────────────────────────────────────────

    def at_goal(self):
        """(E, N) mask of agents standing on their goal."""
        return (self.posX == self.goal[..., 0]) & (self.posY == self.goal[..., 1])

    def all_at_goal(self):
        """(E,) mask of episodes where every agent is on its goal."""
        return self.at_goal().all(dim=1)

    def success_rate(self):
        """(E,) fraction of agents at their goal."""
        return self.at_goal().float().mean(dim=1)

    def get_positions(self):
        return torch.stack([self.posX, self.posY], dim=2)
