"""Independent frequency-domain reference for compact penetrable inclusions.

Second-order midpoint Nyström volume potential with integrated singular self
cell, area-fraction geometry and GMRES. This is NOT the high-order adaptive
method in Ambikasaran et al.; convergence has to be measured for every case.
The time-domain engine, its stencil, rasterizer and forcing are not used.
"""
import numpy as np
from scipy.fft import fft2, ifft2, next_fast_len
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import hankel2, h2vp, jv, jvp


def green(wavenumber, radius):
    """Outgoing exp(+i*omega*t) fundamental solution of (Delta+k²)G=-delta."""
    return -.25j*hankel2(0, wavenumber*radius)


def self_cell(wavenumber, spacing, order=64):
    """Integrate G over a square using eight triangles and analytic radial integration."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    t = (nodes+1)/2
    a = spacing/2
    b = wavenumber*np.sqrt(1+t*t)
    radial = a*hankel2(1, a*b)/b-2j/(np.pi*b*b)
    return 8*(-.25j)*np.dot(weights/2, radial)


class VolumePotential:
    def __init__(self, shape, spacing, wavenumber):
        self.shape, self.spacing, self.k = shape, spacing, wavenumber
        ny, nx = shape
        yy, xx = np.meshgrid(np.arange(1-ny, ny)*spacing,
                             np.arange(1-nx, nx)*spacing, indexing='ij')
        distance = np.hypot(xx, yy)
        distance[ny-1, nx-1] = 1.
        kernel = green(wavenumber, distance)*spacing**2
        kernel[ny-1, nx-1] = self_cell(wavenumber, spacing)
        self.kernel = kernel
        self.fft_shape = (next_fast_len(3*ny-2), next_fast_len(3*nx-2))
        self.kernel_fft = fft2(kernel, self.fft_shape, workers=1)

    def apply(self, density):
        ny, nx = self.shape
        full = ifft2(fft2(density, self.fft_shape, workers=1)*self.kernel_fft, workers=1)
        return full[ny-1:2*ny-1, nx-1:2*nx-1]

    def solve(self, contrast, incident, rtol=1e-9, restart=40, maxiter=40):
        contrast = np.asarray(contrast)
        incident = np.asarray(incident, dtype=complex)
        if contrast.shape != self.shape or incident.shape != self.shape:
            raise ValueError('Contrast and incident field must match the reference grid.')
        def apply(flat):
            u = flat.reshape(self.shape)
            return (u-self.apply(contrast*u)).ravel()
        operator = LinearOperator((incident.size, incident.size), matvec=apply, dtype=np.complex128)
        residuals = []
        solution, info = gmres(operator, incident.ravel(), rtol=rtol, atol=0.,
                               restart=restart, maxiter=maxiter, callback=residuals.append,
                               callback_type='pr_norm')
        residual = np.linalg.norm(apply(solution)-incident.ravel())/np.linalg.norm(incident)
        if info != 0 or residual > 10*rtol:
            raise RuntimeError(f'Volume equation did not converge: info={info}, residual={residual}')
        return solution.reshape(self.shape), {'gmres_iterations': len(residuals),
                                            'relative_residual': float(residual)}


def grid_cells(bounds, spacing):
    x0, x1, y0, y1 = bounds
    x = (np.arange(np.floor(x0/spacing), np.ceil(x1/spacing))+.5)*spacing
    y = (np.arange(np.floor(y0/spacing), np.ceil(y1/spacing))+.5)*spacing
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def area_fraction(xx, yy, spacing, contains, subcells=8):
    fraction = np.zeros_like(xx)
    offsets = ((np.arange(subcells)+.5)/subcells-.5)*spacing
    for oy in offsets:
        for ox in offsets:
            fraction += contains(xx+ox, yy+oy)
    return fraction/subcells**2


def evaluate_scattered(probes, xx, yy, density, spacing, wavenumber, chunk=16):
    probes = np.asarray(probes)
    result = []
    active = np.abs(density) > 0
    x, y, values = xx[active], yy[active], density[active]
    for start in range(0, len(probes), chunk):
        p = probes[start:start+chunk]
        r = np.hypot(p[:, 0, None]-x, p[:, 1, None]-y)
        if np.any(r == 0):
            raise ValueError('Exterior probes must not coincide with volume cell centres.')
        result.extend(green(wavenumber, r) @ values * spacing**2)
    return np.asarray(result)


def disk_plane_reference(probes, radius, refractive_index, wavenumber, modes=40):
    """Exact exterior penetrable-cylinder series, continuity of u and normal du."""
    probes = np.asarray(probes)
    r = np.linalg.norm(probes, axis=1)
    if np.any(r <= radius):
        raise ValueError('Cylinder series here evaluates exterior points only.')
    angle = np.arctan2(probes[:, 1], probes[:, 0])
    m = np.arange(-modes, modes+1)
    ka, nka = wavenumber*radius, refractive_index*wavenumber*radius
    numerator = refractive_index*jv(m, ka)*jvp(m, nka)-jvp(m, ka)*jv(m, nka)
    denominator = h2vp(m, ka)*jv(m, nka)-refractive_index*hankel2(m, ka)*jvp(m, nka)
    a = (-1j)**m*numerator/denominator
    scattered = np.sum(a*hankel2(m, wavenumber*r[:, None])*np.exp(1j*angle[:, None]*m), axis=1)
    return np.exp(-1j*wavenumber*probes[:, 0])+scattered


def lens_fraction(xx, yy, spacing, *, radius=24., aperture=16., cx=30., cy=24., subcells=8):
    """Same fixed 161-vertex-per-side contour, independently integrated by area.

    Linear interpolation of the continuous polygon edges avoids sharing the
    engine's cell-centre point-in-polygon implementation.
    """
    levels = np.linspace(-aperture/2, aperture/2, 161)
    offset = np.sqrt(radius**2-(aperture/2)**2)
    left = cx+offset-np.sqrt(radius**2-levels**2)
    right = cx-offset+np.sqrt(radius**2-levels**2)
    def contains(x, y):
        local_y = y-cy
        return ((np.abs(local_y) < aperture/2) &
                (x > np.interp(local_y, levels, left)) &
                (x < np.interp(local_y, levels, right)))
    return area_fraction(xx, yy, spacing, contains, subcells)
