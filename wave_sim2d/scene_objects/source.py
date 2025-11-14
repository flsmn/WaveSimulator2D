from wave_sim2d.wave_simulation import SceneObject
import torch
import numpy as np
import math


class PointSource(SceneObject):
    """
    Implements a point source scene object. The amplitude can be optionally modulated using a modulator object.
    :param x: source position x.
    :param y: source position y.
    :param frequency: emitting frequency.
    :param amplitude: emitting amplitude, not used when an amplitude modulator is given
    :param phase: emitter phase
    :param amp_modulator: optional amplitude modulator. This can be used to change the amplitude of the source
                          over time.
    """
    def __init__(self, x, y, frequency, amplitude=1.0, phase=0, amp_modulator=None):
        self.x = x
        self.y = y
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.amplitude_modulator = amp_modulator
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def set_amplitude_modulator(self, func):
        self.amplitude_modulator = func

    def render(self, field: torch.Tensor, wave_speed_field: torch.Tensor, dampening_field: torch.Tensor):
        pass

    def update_field(self, field, t):
        if self.amplitude_modulator is not None:
            amplitude = self.amplitude_modulator(t) * self.amplitude
        else:
            amplitude = self.amplitude

        v = torch.sin(torch.tensor(self.phase + self.frequency * t, device=self.device)) * amplitude
        field[self.y, self.x] = v

    def render_visualization(self, image: np.ndarray):
        """ renders a visualization of the scene object to the image """
        pass


class LineSource(SceneObject):
    """
    Implements a line source scene object. The amplitude can be optionally modulated using a modulator object.
    The source emits along a line defined by a start and end point.
    :param start: starting (x, y) coordinates of the line as a tuple.
    :param end: ending (x, y) coordinates of the line as a tuple.
    :param frequency: emitting frequency.
    :param amplitude: emitting amplitude, not used when an amplitude modulator is given
    :param phase: emitter phase
    :param amp_modulator: optional amplitude modulator. This can be used to change the amplitude of the source
                          over time.
    """
    def __init__(self, start, end, frequency, amplitude=1.0, phase=0, amp_modulator=None):
        self.start = start
        self.end = end
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.amplitude_modulator = amp_modulator
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def set_amplitude_modulator(self, func):
        self.amplitude_modulator = func

    def render(self, field: torch.Tensor, wave_speed_field: torch.Tensor, dampening_field: torch.Tensor):
        pass

    def update_field(self, field, t):
        if self.amplitude_modulator is not None:
            amplitude = self.amplitude_modulator(t) * self.amplitude
        else:
            amplitude = self.amplitude

        v = torch.sin(torch.tensor(self.phase + self.frequency * t, device=self.device)) * amplitude

        # Determine the points along the line using NumPy
        x1, y1 = self.start
        x2, y2 = self.end

        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        num_points = int(distance) + 1

        if num_points > 0:
            x_coords = torch.linspace(x1, x2, num_points, device=self.device).round().to(torch.int64)
            y_coords = torch.linspace(y1, y2, num_points, device=self.device).round().to(torch.int64)

            # Create boolean masks for valid indices
            valid_x = (x_coords >= 0) & (x_coords < field.shape[1])
            valid_y = (y_coords >= 0) & (y_coords < field.shape[0])
            valid_indices = valid_x & valid_y

            # Use these valid indices to update the field directly
            valid_y_coords = y_coords[valid_indices]
            valid_x_coords = x_coords[valid_indices]
            field[valid_y_coords, valid_x_coords] = v

    def render_visualization(self, image: np.ndarray):
        """ renders a visualization of the scene object to the image """
        pass

# --- Modulators -------------------------------------------------------------------------------------------------------


class ModulatorSmoothSquare:
    """
    A modulator that creates a smoothed square wave
    """
    def __init__(self, frequency, phase, smoothness=0.5):
        self.frequency = frequency
        self.phase = phase
        self.smoothness = min(max(smoothness, 1e-4), 1.0)

    def __call__(self, t):
        s = math.pow(self.smoothness, 4.0)
        a = (0.5 / math.atan(1.0/s)) * math.atan(math.sin(t * self.frequency + self.phase) / s)+0.5
        return a


class ModulatorDiscreteSignal:
    """
    A modulator that creates a smoothed binary signal
    """
    def __init__(self, signal_array, time_factor, transition_slope=8.0):
        self.signal_array = signal_array
        self.time_factor = time_factor
        self.transition_slope = transition_slope

    def __call__(self, t):
        def smooth_step(t):
            return t * t * (3 - 2 * t)

        # Wrap around the position if it's outside the array range
        sl = len(self.signal_array)
        t = math.fmod(t*self.time_factor, sl)

        # Find the indices of the neighboring values
        index_low = int(t)
        index_high = (index_low + 1) % sl

        # Calculate the interpolation factor
        tf = (t - index_low)
        tf = max(0.0, min(1.0, (tf-0.5)*self.transition_slope+0.5))

        # Use smooth step to interpolate between neighboring values
        l = smooth_step(tf)
        interpolated_value = (1 - l) * self.signal_array[index_low] + l * self.signal_array[index_high]

        return interpolated_value
