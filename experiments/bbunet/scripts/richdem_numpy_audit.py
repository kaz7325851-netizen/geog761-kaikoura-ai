"""Audited NumPy translation of RichDEM 0.3.4 Horn terrain formulas.

Source: RichDEM terrain_attributes.hpp (TerrainSetup, Terrain_Aspect,
Terrain_Slope_Degree) and Array2D.hpp cell lengths. Not the RichDEM binary.
Edges/invalid neighbors are replaced by the center elevation. Invalid centers
remain invalid. Preserves source flat-aspect behavior rather than assuming -1.
"""
import math
import numpy as np

def terrain_derivatives(dem, cell_x, cell_y, nodata=-9999):
    dem=np.asarray(dem,dtype='float64')
    assert dem.ndim==2 and cell_x>0 and cell_y>0
    valid=np.isfinite(dem)&(dem!=nodata)
    h,w=dem.shape
    padded=np.pad(dem,1,constant_values=nodata)
    neighbors=[]
    for row in range(3):
        for col in range(3):
            n=padded[row:row+h,col:col+w]
            neighbors.append(np.where(np.isfinite(n)&(n!=nodata),n,dem))
    a,b,c,d,e,f,g,hh,i=neighbors
    dx=((c+2*f+i)-(a+2*d+g))/8/cell_x
    dy=((g+2*hh+i)-(a+2*b+c))/8/cell_y
    slope=np.arctan(np.sqrt(dx*dx+dy*dy))*180/np.pi
    angle=np.arctan2(dy,-dx)*180/np.pi
    aspect=np.where(angle<0,90-angle,np.where(angle>90,450-angle,90-angle))
    return tuple(np.where(valid,x,nodata).astype('float32') for x in [slope,aspect])

def scalar_reference(dem,cell_x,cell_y,nodata=-9999):
    """Separate direct scalar transcription for vectorization/edge checks."""
    h,w=dem.shape
    slope=np.full((h,w),nodata,dtype='float32')
    aspect=slope.copy()
    for r in range(h):
        for c in range(w):
            center=float(dem[r,c])
            if not math.isfinite(center) or center==nodata:
                continue
            p=[]
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    nr,nc=r+dr,c+dc
                    val=float(dem[nr,nc]) if 0<=nr<h and 0<=nc<w else nodata
                    p.append(center if not math.isfinite(val) or val==nodata else val)
            a,b,cc,d,e,f,g,hh,i=p
            dx=((cc+2*f+i)-(a+2*d+g))/8/cell_x
            dy=((g+2*hh+i)-(a+2*b+cc))/8/cell_y
            slope[r,c]=math.atan(math.sqrt(dx*dx+dy*dy))*180/math.pi
            angle=math.atan2(dy,-dx)*180/math.pi
            aspect[r,c]=(90-angle if angle<0 else 450-angle if angle>90 else 90-angle)
    return slope,aspect

def encode_terrain(dem,slope,aspect):
    # Author no-augmentation path includes modulo 360 before half-angle encoding.
    theta=(aspect%360)*np.pi/360
    return np.stack([dem/3000,slope/90,(np.sin(theta)+1)/2,(np.cos(theta)+1)/2]).astype('float32')

def run_tests():
    rng=np.random.default_rng(761)
    a=rng.integers(0,1200,size=(19,17)).astype('float64')
    a[0,0]=-9999;a[5:7,9]=-9999;a[8,3]=np.nan
    v=terrain_derivatives(a,30,40)
    s=scalar_reference(a,30,40)
    errors=[float(np.max(np.abs(x-y))) for x,y in zip(v,s)]
    assert max(errors)<1e-4
    y,x=np.mgrid[:7,:7]
    cases=[]
    for name,dem,expected_slope,expected_aspect in [
        ('flat',np.full((7,7),100.),0,270),
        ('rises_east',x*30.,45,270),
        ('rises_west',-x*30.,45,90),
        ('rises_south',y*30.,45,0),
        ('rises_north',-y*30.,45,180),
        ('rises_southeast',(x+y)*30.,math.degrees(math.atan(math.sqrt(2))),315),
    ]:
        sl,asp=terrain_derivatives(dem,30,30)
        assert abs(float(sl[3,3])-expected_slope)<1e-4
        assert abs(float(asp[3,3])-expected_aspect)<1e-4
        cases.append({'case':name,'center_slope_deg':float(sl[3,3]),'center_aspect_deg':float(asp[3,3])})
    fixture=encode_terrain(np.full((2,2),3000.),np.full((2,2),90.),np.array([[0,90],[180,270]],dtype='float32'))
    assert np.all(fixture[0]==1) and np.all(fixture[1]==1)
    assert np.allclose(fixture[2],[[.5,(math.sqrt(.5)+1)/2],[1,(math.sqrt(.5)+1)/2]])
    assert np.allclose(fixture[3],[[1,(math.sqrt(.5)+1)/2],[.5,(1-math.sqrt(.5))/2]])
    return {'pass':True,'scalar_vs_vector_max_abs_error':errors,'cases':cases,
            'half_angle_encoding_fixture_pass':True,'RichDEM_binary_comparison_performed':False,
            'warning':'Analytic and independently transcribed scalar checks, not native RichDEM execution'}
