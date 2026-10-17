"""Qt-free numerical configuration. Coordinates are fixed project units."""
from dataclasses import dataclass
import math


class ConfigurationError(ValueError):
    """A scene is invalid and must not replace an active scene."""


class BackendError(RuntimeError):
    """The explicitly requested device is unavailable or failed."""


class NumericalError(RuntimeError):
    """The numerical state is no longer finite; the run cannot continue."""


KERNEL_SCALE = 0.316
MAX_EIGENVALUE = 1.472 / KERNEL_SCALE
MAX_COURANT = 2 / math.sqrt(MAX_EIGENVALUE)


def finite(value, name, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(f"{name} must be a finite number.")
    if positive and value <= 0 or minimum is not None and value < minimum:
        raise ConfigurationError(f"{name} is outside the permitted range.")
    return value


@dataclass(frozen=True)
class Grid:
    nx: int = 256
    ny: int = 256
    dx: float = 0.125
    dt: float = 0.0625
    border_width: float = 3.0
    border_rate: float = 2.0

    def __post_init__(self):
        for name in ("nx", "ny"):
            value = getattr(self, name)
            if type(value) is not int or not 8 <= value <= 4096:
                raise ConfigurationError(f"{name} must be an integer between 8 and 4096.")
        finite(self.dx, "Grid spacing", positive=True)
        finite(self.dt, "Time step", positive=True)
        finite(self.border_width, "Border width", minimum=0)
        finite(self.border_rate, "Border damping", minimum=0)
        if self.border_width * 2 >= min(self.width, self.height):
            raise ConfigurationError("The damped border leaves no usable interior.")
        if self.dt / self.dx >= MAX_COURANT:
            raise ConfigurationError(f"Time step exceeds the stability limit ({MAX_COURANT:.5f} × dx).")

    @property
    def width(self):
        return self.nx * self.dx

    @property
    def height(self):
        return self.ny * self.dx

    @property
    def shape(self):
        return self.ny, self.nx

    @property
    def estimated_bytes(self):
        # Fields, coefficients, force/workspace, two CPU image buffers; excludes torch runtime.
        return self.nx * self.ny * 4 * 18


@dataclass(frozen=True)
class Source:
    start: tuple[float, float]
    end: tuple[float, float] | None = None
    wavelength: float = 2.0
    amplitude: float = 1.0
    phase: float = 0.0

    def __post_init__(self):
        for point in (self.start, self.end):
            if point is not None:
                if len(point) != 2:
                    raise ConfigurationError("A source position needs x and y.")
                for v in point:
                    finite(v, "Source position")
        finite(self.wavelength, "Wavelength", positive=True)
        finite(self.amplitude, "Amplitude", minimum=0)
        finite(self.phase, "Phase")
        if self.end == self.start:
            raise ConfigurationError("A line source must have distinct endpoints.")

    @property
    def omega(self):
        return 2 * math.pi / self.wavelength


@dataclass(frozen=True)
class Material:
    vertices: tuple[tuple[float, float], ...]
    refractive_index: float = 1.5
    damping_rate: float = 0.0

    def __post_init__(self):
        finite(self.refractive_index, "Refractive index", minimum=1)
        if self.refractive_index > 4:
            raise ConfigurationError("Refractive index must be between 1 and 4.")
        finite(self.damping_rate, "Material damping", minimum=0)
        validate_polygon(self.vertices)


@dataclass(frozen=True)
class Wall:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self):
        finite(self.x, "Wall x")
        finite(self.y, "Wall y")
        finite(self.width, "Wall width", positive=True)
        finite(self.height, "Wall height", positive=True)


def rectangle(x, y, width, height):
    finite(width, "Rectangle width", positive=True)
    finite(height, "Rectangle height", positive=True)
    return ((x, y), (x + width, y), (x + width, y + height), (x, y + height))


def validate_polygon(vertices):
    if len(vertices) < 3:
        raise ConfigurationError("A material polygon needs at least three vertices.")
    for point in vertices:
        if len(point) != 2:
            raise ConfigurationError("Polygon vertices need x and y.")
        for value in point:
            finite(value, "Polygon coordinate")
    edges = list(zip(vertices, vertices[1:] + vertices[:1]))
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in edges)
    if abs(area) < 1e-10 or any(a == b for a, b in edges):
        raise ConfigurationError("A polygon must have nonzero area and distinct adjacent vertices.")
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def on(a, b, c):
        return abs(cross(a,b,c)) < 1e-12 and min(a[0],b[0]) <= c[0] <= max(a[0],b[0]) and min(a[1],b[1]) <= c[1] <= max(a[1],b[1])
    for i, (a,b) in enumerate(edges):
        for j, (c,d) in enumerate(edges):
            if j <= i + 1 or i == 0 and j == len(edges)-1:
                continue
            if cross(a,b,c)*cross(a,b,d) < 0 and cross(c,d,a)*cross(c,d,b) < 0 or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b):
                raise ConfigurationError("A polygon must not intersect itself.")
