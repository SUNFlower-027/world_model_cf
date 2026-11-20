"""ConfoundedBall: a minimal pixel environment with an action-predictive background cue.

The generative process is deliberately simple so that counterfactual ground truth is
available in closed form:

    obs_t   = render(state_t, cue = a_t)      # the frame leaks the action about to be taken
    state_{t+1}, r_t, done = step(state_t, a_t)

When ``confound=True`` the background tint encodes ``a_t``. A sequence model can then
predict the next frame from the cue without reading the action input. When
``confound=False`` the background is fixed, so the only way to predict the transition is
to use the action.
"""

from __future__ import annotations

import numpy as np

# 0=left, 1=right, 2=up, 3=down
DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float32)


class ConfoundedBallEnv:
    N_ACTIONS = 4
    ACTION_NAMES = ("left", "right", "up", "down")

    def __init__(
        self,
        img_size: int = 64,
        confound: bool = True,
        tint_strength: float = 0.45,
    ):
        self.size = int(img_size)
        self.confound = bool(confound)
        self.tint_strength = float(tint_strength)

        self.radius = max(3, int(round(self.size * 0.10)))
        self.step_px = max(2, int(round(self.size * 0.0625)))
        self.goal = np.array([self.size * 0.80, self.size * 0.50], dtype=np.float32)

        self.base_bg = np.array([0.12, 0.12, 0.12], dtype=np.float32)
        self.ball_color = np.array([1.00, 1.00, 1.00], dtype=np.float32)
        self.goal_color = np.array([0.10, 0.90, 0.90], dtype=np.float32)
        # 四种动作对应四种背景色，训练时与动作强相关
        self.tint_colors = np.array(
            [
                [0.55, 0.05, 0.05],
                [0.05, 0.05, 0.55],
                [0.45, 0.05, 0.45],
                [0.50, 0.42, 0.03],
            ],
            dtype=np.float32,
        )

        yy, xx = np.mgrid[0 : self.size, 0 : self.size].astype(np.float32)
        self._yy = yy
        self._xx = xx
        gy, gx = int(round(self.goal[1])), int(round(self.goal[0]))
        self._goal_slice = (slice(max(0, gy - 2), gy + 3), slice(max(0, gx - 2), gx + 3))

    # ------------------------------------------------------------------ state
    def reset(self, rng: np.random.Generator) -> np.ndarray:
        low = self.size * 0.20
        high = self.size * 0.80
        return rng.uniform(low, high, size=2).astype(np.float32)

    def step(self, pos: np.ndarray, action: int):
        pos = pos + self.step_px * DELTAS[int(action)]
        lo, hi = self.radius, self.size - self.radius
        pos = np.clip(pos, lo, hi).astype(np.float32)
        dist = float(np.linalg.norm(pos - self.goal))
        reward = float(np.exp(-dist / (0.25 * self.size)))
        done = bool(dist < self.radius * 1.5)
        return pos, reward, done

    # ---------------------------------------------------------------- render
    def render(self, pos: np.ndarray, cue: int | None = None) -> np.ndarray:
        """Render a frame. ``cue`` is the action whose colour is painted into the background."""

        bg = self.base_bg
        if self.confound and cue is not None:
            bg = np.clip(bg + self.tint_strength * self.tint_colors[int(cue)], 0.0, 1.0)

        img = np.broadcast_to(bg[:, None, None], (3, self.size, self.size)).copy()
        gy_slice, gx_slice = self._goal_slice
        img[:, gy_slice, gx_slice] = self.goal_color[:, None, None]

        cx, cy = float(pos[0]), float(pos[1])
        mask = (self._xx - cx) ** 2 + (self._yy - cy) ** 2 <= self.radius**2
        img[:, mask] = self.ball_color[:, None]
        return np.clip(img, 0.0, 1.0)

    # ---------------------------------------------------------------- policy
    def behavior_action(
        self,
        pos: np.ndarray,
        noise: float,
        rng: np.random.Generator,
    ) -> int:
        """Mostly goal-directed, sometimes random. Produces temporally correlated actions."""

        if rng.random() < noise:
            return int(rng.integers(0, self.N_ACTIONS))
        dx = self.goal[0] - pos[0]
        dy = self.goal[1] - pos[1]
        if abs(dx) >= abs(dy):
            return 1 if dx > 0 else 0
        return 3 if dy > 0 else 2
