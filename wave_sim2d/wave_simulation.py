import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod


# Device management: CUDA → MPS → CPU priority
def _get_device():
    """Get the appropriate device for tensor operations."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class SceneObject(ABC):
    """
    Interface for simulation scene objects. A scene object is anything defining or modifying the simulation scene.
    For example: Light sources, Absorbers or regions with specific refractive index. Scene objects can change the
    simulated field and draw their contribution to the wave speed field and dampening field each frame """

    @abstractmethod
    def render(self, field: torch.Tensor, wave_speed_field: torch.Tensor, dampening_field: torch.Tensor):
        """ renders the scene objects contribution to the wave speed field and dampening field """
        pass

    @abstractmethod
    def update_field(self, field: torch.Tensor, t):
        """ performs updates to the field itself, e.g. for adding sources """
        pass

    @abstractmethod
    def render_visualization(self, image: np.ndarray):
        """ renders a visualization of the scene object to the image """
        pass


class WaveSimulator2D:
    """
    Simulates the 2D wave equation
    The system assumes units, where the wave speed is 1.0 pixel/timestep
    source frequency should be adjusted accordingly
    """
    def __init__(self, w, h, scene_objects, initial_field=None):
        """
        Initialize the 2D wave simulator.
        @param w: Width of the simulation grid.
        @param h: Height of the simulation grid.
        """
        self.device = _get_device()
        self.global_dampening = 1.0
        self.c = torch.ones((h, w), dtype=torch.float32, device=self.device)                      # wave speed field (from refractive indices)
        self.d = torch.ones((h, w), dtype=torch.float32, device=self.device)                      # dampening field
        self.u = torch.zeros((h, w), dtype=torch.float32, device=self.device)                     # field values
        self.u_prev = torch.zeros((h, w), dtype=torch.float32, device=self.device)                # field values of prev frame

        if initial_field is not None:
            assert w == initial_field.shape[1] and h == initial_field.shape[2], 'width/height of initial field invalid'
            self.u[:] = torch.from_numpy(initial_field).to(self.device)
            self.u_prev[:] = torch.from_numpy(initial_field).to(self.device)

        # Define Laplacian kernel as a 2D convolution kernel
        # Shape: (1, 1, 3, 3) for 2D convolution with PyTorch
        laplacian_weights = [[0.066, 0.184, 0.066],
                            [0.184, -1.0, 0.184],
                            [0.066, 0.184, 0.066]]
        self.laplacian_kernel = torch.tensor(laplacian_weights, dtype=torch.float32, device=self.device)
        self.laplacian_kernel = self.laplacian_kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)

        self.t = 0
        self.dt = 1.0

        self.scene_objects = scene_objects if scene_objects is not None else []

    def reset_time(self):
        """
        Reset the simulation time to zero.
        """
        self.t = 0.0

    def update_field(self):
        """
        Update the simulation field based on the wave equation.
        """
        # Calculate laplacian using convolution
        # Pad the field to maintain 'same' mode behavior
        u_padded = F.pad(self.u.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0.0)
        laplacian = F.conv2d(u_padded, self.laplacian_kernel, padding=0).squeeze(0).squeeze(0)

        # Update field
        v = (self.u - self.u_prev) * self.d * self.global_dampening
        r = (self.u + v + laplacian * (self.c * self.dt)**2)

        self.u_prev[:] = self.u.clone()
        self.u[:] = r

        self.t += self.dt

    def update_scene(self):
        # clear wave speed field and dampening field
        self.c.fill_(1.0)
        self.d.fill_(1.0)

        for obj in self.scene_objects:
            obj.render(self.u, self.c, self.d)

        for obj in self.scene_objects:
            obj.update_field(self.u, self.t)

    def get_field(self):
        """
        Get the current state of the simulation field.
        @return: A 2D tensor representing the simulation field.
        """
        return self.u

    def render_visualization(self, image=None):
        # clear wave speed field and dampening field
        if image is None:
            image = np.zeros((self.c.shape[0], self.c.shape[1], 3), dtype=np.uint8)

        for obj in self.scene_objects:
            obj.render_visualization(image)

        return image


