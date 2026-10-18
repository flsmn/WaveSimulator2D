"""Passive windowed intensity and signed live field samples at detector positions."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from .config import ConfigurationError, finite


@dataclass(frozen=True)
class Detector:
    id: str
    start: tuple[float,float]
    end: tuple[float,float] | None = None
    samples: int = 1
    settling: float = 32.0
    window: float = 16.0
    revision: int = 0

    def validate(self,grid):
        if not self.id or type(self.samples) is not int or not 1 <= self.samples <= 2048:
            raise ConfigurationError('A detector requires an ID and 1–2048 spatial samples.')
        if self.end is not None and (self.samples < 2 or self.end == self.start):
            raise ConfigurationError('A detector line requires distinct endpoints and at least two samples.')
        if self.end is None and self.samples != 1:
            raise ConfigurationError('A point detector has exactly one spatial sample.')
        finite(self.settling,'Settling time',minimum=0)
        finite(self.window,'Measurement window',positive=True)
        for point in (self.start,self.end):
            if point is not None:
                if len(point)!=2: raise ConfigurationError('Detector positions require x and y.')
                for value in point: finite(value,'Detector coordinate')
                if not grid.dx <= point[0] <= grid.width-grid.dx or not grid.dx <= point[1] <= grid.height-grid.dx:
                    raise ConfigurationError('Detectors must lie inside the simulation boundary.')
        return self


class Accumulator:
    def __init__(self,spec,grid,device,activated):
        self.spec=spec.validate(grid)
        self.activated=activated
        self.begin=activated+spec.settling
        self.finish=self.begin+spec.window
        self.elapsed=0.0
        self.count=0
        self.sample_interval=grid.dt
        self.last_time=activated
        points=np.linspace(spec.start,spec.end or spec.start,spec.samples)
        self.positions=points
        self.axis=np.linspace(0,math.dist(spec.start,spec.end or spec.start),spec.samples)
        xy=points/grid.dx-.5
        base=np.floor(xy).astype(int)
        frac=xy-base
        self.indices=[]
        self.weights=[]
        for dy in (0,1):
            for dx in (0,1):
                self.indices.append(torch.tensor((base[:,1]+dy)*grid.nx+base[:,0]+dx,device=device))
                self.weights.append(torch.tensor(((1-frac[:,0]) if dx==0 else frac[:,0])*((1-frac[:,1]) if dy==0 else frac[:,1]),device=device,dtype=torch.float32))
        self.total=torch.zeros(spec.samples,device=device)
        self.live=torch.zeros_like(self.total)
        self.live_field=torch.zeros_like(self.total)
        self.field_time=activated

    def sample_field(self,field,t):
        flat=field.view(-1)
        self.live_field=sum(flat[i]*w for i,w in zip(self.indices,self.weights))
        self.field_time=t

    def update(self,field,t,dt):
        self.sample_field(field,t)
        # Interpolate field squared (intensity at cell centres), not field then square.
        flat=field.view(-1)
        self.live=sum(flat[i].square()*w for i,w in zip(self.indices,self.weights))
        overlap=max(0,min(t,self.finish)-max(t-dt,self.begin))
        if overlap > 1e-12 and self.elapsed < self.spec.window-1e-10:
            self.total.add_(self.live,alpha=overlap)
            self.elapsed=min(self.spec.window,self.elapsed+overlap)
            self.count+=1
        self.last_time=t

    def result(self):
        status='complete' if self.elapsed >= self.spec.window-1e-9 else 'settling' if self.last_time < self.begin else 'measuring'
        values=(self.total/self.elapsed if self.elapsed else self.live).detach().cpu().numpy().copy()
        values.setflags(write=False)
        field_values=self.live_field.detach().cpu().tolist()
        return {'detector_id':self.spec.id,'measurement_revision':self.spec.revision,'status':status,'activated':self.activated,'begin':self.begin,'duration':self.spec.window,'elapsed':self.elapsed,'sample_count':self.count,'sample_interval':self.sample_interval,'method':'bilinear field-square, endpoint time quadrature with fractional boundary weights','axis':self.axis.tolist(),'positions':self.positions.tolist(),'values':values.tolist(),'field_values':field_values,'field_time':self.field_time}


class Measurements:
    MAX_DETECTORS=16
    def __init__(self,grid,device):
        self.grid,self.device=grid,device
        self.detectors={}
        self.intensity=torch.zeros(grid.shape,device=device)
        self.intensity_time=0.0

    def configure(self,specs,t,field=None):
        specs=tuple(specs)
        if len(specs)>self.MAX_DETECTORS or len({s.id for s in specs})!=len(specs):
            raise ConfigurationError('At most 16 distinct detectors are supported.')
        # Build replacements before changing any active measurement.
        replacements={}
        for spec in specs:
            old=self.detectors.get(spec.id)
            replacements[spec.id]=old if old and old.spec==spec else Accumulator(spec,self.grid,self.device,t)
        self.detectors=replacements
        if field is not None:
            for detector in self.detectors.values():detector.sample_field(field,t)

    def update(self,field,t,dt):
        # Bounded live intensity preview: exponential decay in physical time.
        alpha=1-math.exp(-dt/4)
        self.intensity.lerp_(field.square(),alpha)
        self.intensity_time=t
        for detector in self.detectors.values(): detector.update(field,t,dt)

    def reset(self,field=None):
        specs=[d.spec for d in self.detectors.values()]
        self.detectors={}
        self.configure(specs,0,field=field)
        self.intensity.zero_()
        self.intensity_time=0

    def results(self):
        return {key:value.result() for key,value in self.detectors.items()}
