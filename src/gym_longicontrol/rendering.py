"""Optional renderer. Importing the environment never imports matplotlib."""

import numpy as np


class Renderer:
    def __init__(self, mode, fps):
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError as exc:
            raise ImportError(
                "Install gym-longicontrol[render] to render episodes"
            ) from exc
        self.mode, self.fps = mode, fps
        self._plt = None
        self.figure = Figure(figsize=(10, 4.5), dpi=100)
        self.canvas = FigureCanvasAgg(self.figure)
        self._subplot_params = {
            name: getattr(self.figure.subplotpars, name)
            for name in ("left", "right", "bottom", "top", "wspace", "hspace")
        }
        self.history = []

    def reset(self):
        self.history.clear()

    def render(self, state, track, config):
        # Render calls must not duplicate or mutate simulation history.
        if not self.history or self.history[-1][0] != state.elapsed_time_s:
            self.history.append(
                (
                    state.elapsed_time_s,
                    state.position_m,
                    state.velocity_m_s * 3.6,
                    state.acceleration_m_s2,
                )
            )
        self.figure.clear()
        # Before Matplotlib 3.11, clear() preserves the previous tight_layout
        # result. Start from the same margins to avoid frame-to-frame drift.
        self.figure.subplots_adjust(**self._subplot_params)
        axis, acceleration = self.figure.subplots(2, 1, sharex=True)
        positions = np.r_[track.positions_m, config.track_length_m]
        limits = np.r_[track.limits_m_s, track.limits_m_s[-1]] * 3.6
        axis.step(positions, limits, where="post", color="red", label="Speed limit")
        data = np.array(self.history)
        axis.plot(data[:, 1], data[:, 2], color="black", label="Vehicle speed")
        axis.scatter([state.position_m], [state.velocity_m_s * 3.6], color="blue")
        axis.set(ylabel="Speed (km/h)", ylim=(0, 140), xlim=(0, config.track_length_m))
        axis.legend(loc="upper right")
        axis.set_title(
            f"LongiControl | {state.elapsed_time_s:.1f} s | "
            f"{state.total_energy_kwh:.3f} kWh"
        )
        acceleration.plot(data[:, 1], data[:, 3], color="blue")
        acceleration.set(
            xlabel="Position (m)", ylabel="Acceleration (m/s²)", ylim=(-3.2, 3.2)
        )
        self.figure.tight_layout()
        self.canvas.draw()
        frame = np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()
        if self.mode == "rgb_array":
            return frame
        if self._plt is None:
            import matplotlib.pyplot as plt

            self._plt = plt
            self.window, self.window_axis = plt.subplots(figsize=(10, 4.5))
            self.window_axis.axis("off")
            self._image = self.window_axis.imshow(frame)
        else:
            self._image.set_data(frame)
        self._plt.pause(1 / self.fps)
        return None

    def close(self):
        if self._plt is not None:
            self._plt.close(self.window)
        self.figure.clear()
        self.history.clear()
