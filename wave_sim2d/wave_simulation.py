"""One scalar wave solver; prepared scenes are the supported Studio API.

The old scene-object interface remains for historical examples only. See docs/MODEL.md.
"""
from abc import ABC, abstractmethod
import math
import numpy as np
import torch
import torch.nn.functional as F
from .config import BackendError, ConfigurationError, NumericalError, Grid, KERNEL_SCALE


def resolve_device(requested="auto"):
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in ("cpu", "mps", "cuda"):
        raise ConfigurationError("Device must be auto, cpu, mps or cuda.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise BackendError("The requested MPS device is unavailable.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise BackendError("The requested CUDA device is unavailable.")
    return torch.device(requested)


_get_device = resolve_device


class SceneObject(ABC):
    """Legacy example interface. Studio uses immutable prepared scenes."""
    def bind(self, device, dt, dx):
        self.device = device
        self.dt, self.dx = dt, dx
        for name, value in vars(self).copy().items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
            elif isinstance(value, SceneObject):
                value.bind(device, dt, dx)
        # Rebuild legacy polygon caches on the requested device.
        if hasattr(self, "_cached_coords"):
            self._cached_coords = None

    @abstractmethod
    def render(self, field, wave_speed_field, dampening_field): ...
    @abstractmethod
    def update_field(self, field, t): ...
    @abstractmethod
    def render_visualization(self, image): ...


class WaveSimulator2D:
    """Calibrated nine-point scalar FDTD solver with float32 state.

    Arrays use [y, x], cell centres are ((x+.5)*dx, (y+.5)*dx).
    get_field() is a legacy borrowed tensor; snapshot() returns an owned CPU copy.
    """
    def __init__(self, w, h, scene_objects=None, initial_field=None, *, device="auto", dx=1.0, dt=None):
        self.grid = Grid(w, h, dx, dt if dt is not None else 0.5*dx, 0, 0)
        self.device = resolve_device(device)
        self.dt, self.dx = self.grid.dt, self.grid.dx
        self.global_dampening = 1.0
        self.c = torch.ones(self.grid.shape, dtype=torch.float32, device=self.device)
        self.d = torch.ones_like(self.c)
        self.gamma = torch.zeros_like(self.c)
        self.walls = torch.zeros(self.grid.shape, dtype=torch.bool, device=self.device)
        self.walls[0, :] = self.walls[-1, :] = True
        self.walls[:, 0] = self.walls[:, -1] = True
        self.u = torch.zeros_like(self.c)
        self.u_prev = torch.zeros_like(self.c)
        self._initial = self._validate_initial(initial_field)
        self.laplacian_kernel = torch.tensor([[.066,.184,.066],[.184,-1,.184],[.066,.184,.066]], dtype=torch.float32, device=self.device)[None,None] / KERNEL_SCALE
        self.scene_objects = list(scene_objects or ())
        for obj in self.scene_objects:
            obj.bind(self.device, self.dt, self.dx)
        self._sources = []
        self._prepared = False
        self.steps = 0
        self.t = 0.0
        self.reset()

    @classmethod
    def from_scene(cls, scene, *, device="auto", initial_field=None):
        grid = scene.grid
        sim = cls(grid.nx, grid.ny, initial_field=initial_field, device=device, dx=grid.dx, dt=grid.dt)
        sim.grid = grid
        sim.c.copy_(torch.tensor(scene.refractive_index.copy(), device=sim.device).reciprocal())
        sim.gamma.copy_(torch.tensor((scene.material_damping+scene.border_damping).copy(), device=sim.device))
        sim.walls |= torch.tensor(scene.walls.copy(), device=sim.device)
        sim._sources = [(torch.tensor(s.indices.copy(), device=sim.device), torch.tensor(s.weights.copy(), device=sim.device), s.omega, s.amplitude, s.phase) for s in scene.sources]
        sim._prepared = True
        sim.reset()
        return sim

    def _validate_initial(self, field):
        if field is None:
            return torch.zeros_like(self.u)
        value = torch.as_tensor(field, dtype=torch.float32, device=self.device).clone()
        if tuple(value.shape) != self.grid.shape or not torch.isfinite(value).all().item():
            raise ConfigurationError(f"Initial field must be finite with shape {self.grid.shape} ([y, x]).")
        return value

    def reset(self):
        """Restore the initial displacement and zero initial velocity, time and count."""
        self.u.copy_(self._initial)
        self.u.masked_fill_(self.walls, 0)
        self.u_prev.copy_(self.u)
        self.steps = 0
        self.t = 0.0

    def reset_time(self):
        """Deprecated compatibility name: now performs a complete reset."""
        self.reset()

    def update_scene(self):
        if self._prepared:
            return
        self.c.fill_(1)
        self.d.fill_(1)
        for obj in self.scene_objects:
            obj.render(self.u, self.c, self.d)
        for obj in self.scene_objects:
            obj.update_field(self.u, self.t)
        self.u.masked_fill_(self.walls, 0)

    @torch.no_grad()
    def update_field(self):
        lap = F.conv2d(self.u[None,None], self.laplacian_kernel, padding=1)[0,0] / self.dx**2
        half_damping = self.gamma * (self.dt/2)
        nxt = (2*self.u - (1-half_damping)*self.u_prev + (self.c*self.dt)**2*lap) / (1+half_damping)
        if not self._prepared:
            # Legacy per-step damping conversion; not part of the calibrated material API.
            nxt += ((self.d*self.global_dampening)-1)*(self.u-self.u_prev)
        for indices, weights, omega, amplitude, phase in self._sources:
            force = amplitude*math.sin(omega*self.t+phase)*self.dt**2
            nxt.view(-1).index_add_(0, indices, weights*force/(1+half_damping.view(-1)[indices]))
        nxt.masked_fill_(self.walls, 0)
        self.u_prev, self.u = self.u, nxt
        self.steps += 1
        self.t = self.steps*self.dt

    def step(self, count=1):
        if type(count) is not int or count < 1 or count > 10000:
            raise ConfigurationError("Step count must be an integer between 1 and 10000.")
        for _ in range(count):
            self.update_scene()
            self.update_field()
        if not torch.isfinite(self.u).all().item():
            raise NumericalError("The wave field contains non-finite values. Reset or change the configuration.")

    def snapshot(self):
        result = self.u.detach().cpu().numpy().copy()
        result.setflags(write=False)
        return result

    def get_field(self):
        return self.u

    def render_visualization(self, image=None):
        if image is None:
            image = np.zeros((*self.grid.shape,3), dtype=np.uint8)
        for obj in self.scene_objects:
            obj.render_visualization(image)
        return image
