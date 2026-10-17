"""Deterministic cell-centre rasterization, without OpenCV or GUI dependencies."""
from dataclasses import dataclass
import math
import numpy as np
from .config import ConfigurationError, Grid, Material, Source, Wall


@dataclass(frozen=True)
class PreparedSource:
    indices: np.ndarray
    weights: np.ndarray
    omega: float
    amplitude: float
    phase: float


@dataclass(frozen=True)
class PreparedScene:
    grid: Grid
    refractive_index: np.ndarray
    material_damping: np.ndarray
    border_damping: np.ndarray
    walls: np.ndarray
    sources: tuple[PreparedSource, ...]
    warnings: tuple[str, ...]


def polygon_mask(grid, vertices):
    x, y = np.meshgrid((np.arange(grid.nx)+0.5)*grid.dx, (np.arange(grid.ny)+0.5)*grid.dx)
    inside = np.zeros(grid.shape, dtype=bool)
    for (ax, ay), (bx, by) in zip(vertices, vertices[1:] + vertices[:1]):
        if ay != by:
            inside ^= ((ay > y) != (by > y)) & (x < (bx-ax)*(y-ay)/(by-ay)+ax)
    return inside


def _point_weights(grid, point):
    x, y = point
    b = grid.border_width
    if not b <= x < grid.width-b or not b <= y < grid.height-b:
        raise ConfigurationError("Sources must be inside the usable interior, outside the damped border.")
    gx, gy = x/grid.dx-0.5, y/grid.dx-0.5
    ix, iy = math.floor(gx), math.floor(gy)
    result = []
    for dy in (0,1):
        for dx in (0,1):
            weight = (1-abs(gx-ix-dx))*(1-abs(gy-iy-dy))
            if weight > 1e-12:
                if not 1 <= ix+dx < grid.nx-1 or not 1 <= iy+dy < grid.ny-1:
                    raise ConfigurationError("Source support touches the fixed outer boundary.")
                result.append(((iy+dy)*grid.nx+ix+dx, weight))
    return result


def prepare_scene(grid: Grid, materials=(), walls=(), sources=()):
    n = np.ones(grid.shape, dtype=np.float32)
    gamma = np.zeros(grid.shape, dtype=np.float32)
    warnings = []
    for material in materials:
        if not isinstance(material, Material):
            raise ConfigurationError("Invalid material configuration.")
        if any(not 0 <= x <= grid.width or not 0 <= y <= grid.height for x,y in material.vertices):
            raise ConfigurationError("Materials must lie within the simulation domain.")
        mask = polygon_mask(grid, material.vertices)
        if not mask.any():
            warnings.append("A material is smaller than the current grid and is not resolved.")
        n[mask] = material.refractive_index
        gamma[mask] = material.damping_rate  # topmost object replaces both properties
    wall_mask = np.zeros(grid.shape, dtype=bool)
    x, y = np.meshgrid((np.arange(grid.nx)+0.5)*grid.dx, (np.arange(grid.ny)+0.5)*grid.dx)
    for wall in walls:
        if not isinstance(wall, Wall) or wall.x < 0 or wall.y < 0 or wall.x+wall.width > grid.width or wall.y+wall.height > grid.height:
            raise ConfigurationError("Walls must lie within the simulation domain.")
        mask = (x >= wall.x) & (x < wall.x+wall.width) & (y >= wall.y) & (y < wall.y+wall.height)
        if not mask.any():
            raise ConfigurationError("A wall is unresolved. Increase resolution or wall thickness.")
        wall_mask |= mask
    edge = np.minimum.reduce((x, y, grid.width-x, grid.height-y))
    border = grid.border_rate * np.clip(1-edge/grid.border_width, 0, 1)**3 if grid.border_width else np.zeros(grid.shape)
    prepared = []
    for source in sources:
        if not isinstance(source, Source):
            raise ConfigurationError("Invalid source configuration.")
        weights = {}
        if source.end is None:
            points = [(source.start, 1.0)]
        else:
            length = math.dist(source.start, source.end)
            count = max(2, math.ceil(length/grid.dx*2)+1)
            points = [(tuple(a+(b-a)*t for a,b in zip(source.start, source.end)), length/(count-1)*(0.5 if i in (0,count-1) else 1)) for i,t in enumerate(np.linspace(0,1,count))]
        for point, measure in points:
            for index, weight in _point_weights(grid, point):
                weights[index] = weights.get(index, 0.0) + weight*measure/grid.dx**2
        indices = np.array(list(weights), dtype=np.int64)
        if wall_mask.ravel()[indices].any():
            raise ConfigurationError("A source overlaps a wall, including its interpolation support.")
        if source.wavelength / (float(n.max())*grid.dx) < 12:
            warnings.append("Fewer than 12 cells per shortest material wavelength; accuracy is unverified.")
        prepared.append(PreparedSource(indices, np.array(list(weights.values()), dtype=np.float32), source.omega, source.amplitude, source.phase))
    # Freeze CPU inputs. A solver always owns its own device copies.
    border=border.astype(np.float32)
    arrays = [n, gamma, border, wall_mask]
    for s in prepared:
        arrays.extend((s.indices, s.weights))
    for array in arrays:
        array.setflags(write=False)
    return PreparedScene(grid, n, gamma, border, wall_mask, tuple(prepared), tuple(dict.fromkeys(warnings)))


def lens_vertices(x,y,radius,aperture,angle=0.0):
    """Symmetric biconvex contour, zero edge thickness, independent of grid."""
    from .config import finite
    for value,name in ((x,'Lens x'),(y,'Lens y'),(angle,'Lens angle')): finite(value,name)
    finite(radius,'Lens radius',positive=True)
    finite(aperture,'Lens aperture',positive=True)
    if aperture >= 2*radius:
        raise ConfigurationError('Lens aperture must be smaller than twice its radius.')
    offset=math.sqrt(radius**2-(aperture/2)**2)
    yy=np.linspace(-aperture/2,aperture/2,161)
    left=[(offset-math.sqrt(radius**2-v*v),v) for v in yy]
    right=[(-offset+math.sqrt(radius**2-v*v),v) for v in yy[-2:0:-1]]
    c,s=math.cos(angle),math.sin(angle)
    return tuple((x+a*c-b*s,y+a*s+b*c) for a,b in left+right)
