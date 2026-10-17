"""Fourier-symbol estimates for the calibrated nine-point scalar stencil.

These homogeneous estimates are a preflight guide, not an interface-error bound.
"""
import math
from .config import KERNEL_SCALE, ConfigurationError


def negative_symbol(kx_dx, ky_dx):
    return (1-.368*(math.cos(kx_dx)+math.cos(ky_dx))-.264*math.cos(kx_dx)*math.cos(ky_dx))/KERNEL_SCALE


def temporal_frequency(kx, ky, dx, dt, refractive_index=1):
    value=(dt/(refractive_index*dx))**2*max(0,negative_symbol(kx*dx,ky*dx))/4
    if value>1:raise ConfigurationError('The Fourier mode exceeds the temporal stability range.')
    return 2*math.asin(math.sqrt(value))/dt


def phase_drift(wavelength, refractive_index, dx, courant, distance):
    """Worst driven-wave phase error (radians) over axial-to-diagonal directions."""
    omega=2*math.pi/wavelength
    dt=courant*dx
    if omega*dt>=math.pi:return math.inf
    required=4*math.sin(omega*dt/2)**2*(refractive_index/courant)**2
    physical=omega*refractive_index
    errors=[]
    for i in range(9):
        angle=i*math.pi/32
        c,s=math.cos(angle),math.sin(angle)
        low,high=0.,math.pi/max(c,s)
        if required>=negative_symbol(high*c,high*s):return math.inf
        for _ in range(52):
            middle=(low+high)/2
            if negative_symbol(middle*c,middle*s)<required:low=middle
            else:high=middle
        errors.append(abs((low+high)/(2*dx)-physical)*distance)
    return max(errors)
