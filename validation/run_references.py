"""Reproducible physical probes; returns nonzero while any acceptance is open/failed."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import torch
from wave_sim2d import __version__
from wave_sim2d.config import Grid, Material, Source, Wall, rectangle
from wave_sim2d.geometry import prepare_scene
from wave_sim2d.wave_simulation import WaveSimulator2D

ROOT = Path(__file__).resolve().parent
LIMITS = json.loads((ROOT/'tolerances.json').read_text())


def grid(width,height,dx,border=4,rate=2):
    return Grid(round(width/dx),round(height/dx),dx,.5*dx,border,rate)


def sample(array,g,x,y):
    # Bilinear observation at fixed project coordinates, independent of resolution.
    x,y = np.broadcast_arrays(np.asarray(x)/g.dx-.5,np.asarray(y)/g.dx-.5)
    ix,iy = np.floor(x).astype(int),np.floor(y).astype(int)
    fx,fy = x-ix,y-iy
    return (array[iy,ix]*(1-fx)*(1-fy)+array[iy,ix+1]*fx*(1-fy)+array[iy+1,ix]*(1-fx)*fy+array[iy+1,ix+1]*fx*fy)


def demodulate(scene,settle,window,observe):
    sim = WaveSimulator2D.from_scene(scene,device='cpu')
    omega = scene.sources[0].omega
    sim.step(min(10000,math.ceil(settle/sim.dt)))
    phasor = None
    count = math.ceil(window/sim.dt)
    for _ in range(count):
        sim.step()
        values = observe(sim.snapshot())
        term = values*np.exp(-1j*omega*sim.t)
        phasor = term if phasor is None else phasor+term
    return phasor*2/count, sim


def t01():
    modes=[]
    for dx in (.25,.125):
        g=Grid(round(20/dx)+1,round(20/dx)+1,dx,.5*dx,0,0)
        q=np.arange(g.nx)*dx
        for mx,my in ((8,1),(6,6)):
            u=(np.sin(mx*np.pi*q/20)[None,:]*np.sin(my*np.pi*q/20)[:,None]).astype('float32')
            sim=WaveSimulator2D.from_scene(prepare_scene(g),device='cpu',initial_field=u)
            projections=[]
            for _ in range(180):
                sim.step()
                projections.append(float(np.sum(sim.snapshot()*u,dtype=np.float64)/np.sum(u*u,dtype=np.float64)))
            a=np.asarray(projections)
            cosine=np.dot(a[1:-1],a[:-2]+a[2:])/(2*np.dot(a[1:-1],a[1:-1]))
            speed=math.acos(float(cosine))/g.dt/(np.pi*math.hypot(mx,my)/20)
            modes.append({'dx':dx,'mode':[mx,my],'speed':speed,'relative_error':abs(speed-1)})
    sources=[]
    for dx in (.25,.125):
        g=grid(32,32,dx)
        scene=prepare_scene(g,sources=[Source((16,16),wavelength=4)])
        positions=np.linspace(20,25,65)
        z,_=demodulate(scene,24,16,lambda u: sample(u,g,positions,np.full_like(positions,16)))
        phase=np.unwrap(np.angle(z))
        slope=np.polyfit(positions,phase,1)[0]
        wavelength=2*np.pi/abs(slope)
        sources.append({'dx':dx,'wavelength':float(wavelength),'relative_error':abs(wavelength/4-1),'amplitude_at_x22':float(abs(z[26]))})
    amp_error=abs(sources[0]['amplitude_at_x22']/sources[1]['amplitude_at_x22']-1)
    ok=all(m['relative_error']<=LIMITS['T01']['speed_relative_error'] for m in modes) and all(s['relative_error']<=LIMITS['T01']['source_wavelength_relative_error'] for s in sources) and amp_error<=LIMITS['T01']['source_resolution_amplitude_relative_error']
    return {'passed':ok,'sine_modes':modes,'point_source':sources,'resolution_amplitude_relative_error':amp_error,'additive_phase_incoming_wave':'see tests/test_core.py'}


def t02():
    g=grid(80,64,.25,5)
    angle=math.radians(25)
    start=(16+24*math.sin(angle),32-24*math.cos(angle))
    end=(16-24*math.sin(angle),32+24*math.cos(angle))
    scene=prepare_scene(g,[Material(rectangle(40,0,40,64),1.5)],sources=[Source(start,end,wavelength=4)])
    # Store only the central region used for phase-gradient estimation.
    z,_=demodulate(scene,80,24,lambda u:u[100:156,112:216])
    def phase_angle(xs):
        p=z[16:40,xs]
        kx=np.median(np.angle(p[:,1:]*p[:,:-1].conj()))
        ky=np.median(np.angle(p[1:]*p[:-1].conj()))
        return float(math.degrees(math.atan2(abs(ky),abs(kx))))
    incident=phase_angle(slice(4,24))
    transmitted=phase_angle(slice(64,84))
    expected=math.degrees(math.asin(math.sin(angle)/1.5))
    # Damped sine mode: solve its measured second-order recurrence for gamma.
    decay=[]
    for dx in (.25,.125):
        gg=Grid(round(20/dx)+1,round(20/dx)+1,dx,.5*dx,0,0)
        q=np.arange(gg.nx)*dx
        u=(np.sin(8*np.pi*q/20)[None,:]*np.sin(np.pi*q/20)[:,None]).astype('float32')
        ss=prepare_scene(gg,[Material(rectangle(0,0,gg.width,gg.height),1,.2)])
        sim=WaveSimulator2D.from_scene(ss,device='cpu',initial_field=u)
        history=[]
        for _ in range(180):
            sim.step()
            history.append(float(np.sum(sim.snapshot()*u,dtype=np.float64)))
        a=np.array(history)
        coeff=np.linalg.lstsq(np.column_stack((a[1:-1],a[:-2])),a[2:],rcond=None)[0]
        b=-coeff[1]
        rate=2*(1-b)/(gg.dt*(1+b))
        decay.append({'dx':dx,'measured_gamma':float(rate),'relative_error':float(abs(rate/.2-1))})
    # Reflection: compare a line source's phasor at x=12 in a short domain to a
    # longer domain, whose right-end return cannot yet have reached the probe.
    reflection=[]
    for dx in (.25,.125):
        values=[]
        for width in (40,80):
            gg=grid(width,40,dx,8)
            ss=prepare_scene(gg,sources=[Source((12,9),(12,31),wavelength=3)])
            zz,_=demodulate(ss,70,18,lambda u:sample(u,gg,14.,20.))
            values.append(complex(zz))
        ratio=abs(values[0]-values[1])/abs(values[1])
        reflection.append({'dx':dx,'relative_return_amplitude':float(ratio)})
    ok=abs(incident-25)<=3 and abs(transmitted-expected)<=LIMITS['T02']['angle_error_degrees'] and all(d['relative_error']<=LIMITS['T02']['decay_rate_relative_error'] for d in decay) and all(r['relative_return_amplitude']<=LIMITS['T02']['border_reflected_amplitude_ratio'] for r in reflection)
    return {'passed':ok,'incidence_degrees':incident,'transmission_degrees':transmitted,'snell_reference_degrees':expected,'angle_error_degrees':abs(transmitted-expected),'damping':decay,'border_reflection':reflection}


def t03():
    results=[]
    for dx in (.25,.125):
        g=grid(72,64,dx,6)
        walls=[Wall(30,0,.5,28.5),Wall(30,29.5,.5,5),Wall(30,35.5,.5,28.5)]
        scene=prepare_scene(g,walls=walls,sources=[Source((14,32),wavelength=2)])
        yy=np.linspace(24,40,257)
        z,sim=demodulate(scene,60,24,lambda u:sample(u,g,np.full_like(yy,52),yy))
        intensity=abs(z)**2
        symmetry=float(np.max(abs(intensity-intensity[::-1]))/intensity.max())
        middle=len(yy)//2
        # First local minimum, away from the central maximum.
        minima=[i for i in range(middle+2,len(yy)-1) if intensity[i]<intensity[i-1] and intensity[i]<intensity[i+1]]
        measured=float(yy[minima[0]]-32) if minima else None
        offsets=np.linspace(.01,10,10001)
        difference=np.sqrt(22**2+(offsets+3)**2)-np.sqrt(22**2+(offsets-3)**2)
        reference=float(offsets[np.argmin(abs(difference-1))])
        wall_value=float(abs(sim.snapshot()[scene.walls]).max())
        results.append({'dx':dx,'first_minimum_offset':measured,'reference_offset':reference,'relative_error':abs(measured/reference-1) if measured is not None else 1e9,'symmetry_error':symmetry,'wall_max':wall_value})
    ok=all(r['relative_error']<=LIMITS['T03']['first_minimum_position_relative_error'] and r['symmetry_error']<=LIMITS['T03']['symmetry_relative_error'] and r['wall_max']<=LIMITS['T03']['wall_field_max'] for r in results)
    return {'passed':ok,'measurements':results}


def t04():
    radius, aperture, n, cx, cy = 24.,16.,1.5,30.,24.
    offset=math.sqrt(radius**2-(aperture/2)**2)
    thickness=2*(radius-offset)
    focal=1/((n-1)*(2/radius-(n-1)*thickness/(n*radius**2)))
    back_focal=focal*(1-(n-1)*thickness/(n*radius))
    rear=cx+thickness/2
    yy=np.linspace(-aperture/2,aperture/2,161)
    left=[(cx+offset-math.sqrt(radius**2-y*y),cy+y) for y in yy]
    right=[(cx-offset+math.sqrt(radius**2-y*y),cy+y) for y in yy[-2:0:-1]]
    lens=Material(tuple(left+right),n)
    measurements=[]
    positions=np.linspace(rear+.6*back_focal,rear+1.4*back_focal,321)
    for dx in (.10,.08):
        g=grid(80,48,dx,6)
        source=Source((12,7),(12,41),wavelength=.8)
        scene=prepare_scene(g,[lens],sources=[source])
        z,_=demodulate(scene,80,16,lambda u:sample(u,g,positions,np.full_like(positions,cy)))
        background=prepare_scene(g,sources=[source])
        base,_=demodulate(background,80,16,lambda u:sample(u,g,positions,np.full_like(positions,cy)))
        power=abs(z)**2
        peak=int(np.argmax(power))
        distance=float(positions[peak]-rear)
        measurements.append({'dx':dx,'focus_x':float(positions[peak]),'back_focal_distance':distance,'relative_error':abs(distance/back_focal-1),'focus_gain':float(power[peak]/abs(base[peak])**2)})
    shift=abs(measurements[0]['focus_x']-measurements[1]['focus_x'])/back_focal
    ok=all(m['relative_error']<=LIMITS['T04']['focus_position_relative_error'] and m['focus_gain']>=LIMITS['T04']['minimum_focus_gain'] for m in measurements) and shift<=LIMITS['T04']['grid_focus_shift_relative_error']
    return {'passed':ok,'radius':radius,'aperture':aperture,'index':n,'thickness':thickness,'paraxial_back_focal_distance':back_focal,'grid_shift_relative_error':shift,'measurements':measurements}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('docs/physical-results.json'))
    parser.add_argument('--only',choices=['T01','T02','T03','T04'])
    args=parser.parse_args()
    torch.set_num_threads(1)
    report={'engine':__version__,'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'backend':'cpu','dtype':'float32','tolerances':LIMITS,'results':{}}
    for name,probe in [('T01',t01),('T02',t02),('T03',t03),('T04',t04)]:
        if args.only and args.only!=name: continue
        began=time.monotonic()
        print(f'{name}: measuring',flush=True)
        try:
            result=probe()
        except Exception as error:
            result={'passed':False,'error':str(error)}
        result['elapsed_seconds']=time.monotonic()-began
        report['results'][name]=result
        print(json.dumps({name:result},indent=2),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if all(r['passed'] for r in report['results'].values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
