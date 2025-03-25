import numpy as np

def _unit(vec):
    d = np.linalg.norm(vec)
    return vec/d
    
def ndenum(arr,y_start,y_end,x_start,x_end):
    h,w = arr.shape
    if y_end is None:
        y_end = h
        
    if x_end is None:
        x_end = w
        
    a = np.ndenumerate(arr[y_start:y_end,x_start:x_end])
    for (y,x),v in a:
        yield np.array([x+x_start,y+y_start,v])


def _oposing_triangle_normals(arr):
    h,w = arr.shape
    y_grid,x_grid = np.meshgrid(
        np.arange(w),
        np.arange(h)
    )
    
    xyz = np.dstack([y_grid,x_grid,arr])
    
    tl = xyz[0:-1,0:-1].reshape((-1,3))
    bl = xyz[1:,0:-1].reshape((-1,3))
    tr = xyz[0:-1,1:].reshape((-1,3))
    br = xyz[1:,1:].reshape((-1,3))

    cr1 = np.cross(tr-tl,bl-tl)
    cr2 = -np.cross(br-tr,br-bl)
    cr = (cr1+cr2)/2
    
    d = np.linalg.norm(cr,axis=1)
    ucr = cr/np.expand_dims(d,axis=1)
    
    return ucr.reshape((h-1,w-1,3))

def calc_normals(depth_map):
    depth_map_padded = np.pad(depth_map,1,mode='edge')
    normal = _oposing_triangle_normals(depth_map_padded)
    return normal[1:-1,1:-1]


def assign_orientation(normals, light_angle,strength = 1):
    shade_direction = np.arctan2(normals[:,:,0],normals[:,:,1])    
    shade_direction = shade_direction + light_angle
    shade_direction = np.mod(shade_direction,2*np.pi) - np.pi
    shade_direction = np.abs(shade_direction) - np.pi/2

    flat_distance =  np.linalg.norm(normals[:,:,:2],axis=2)
    st = flat_distance *strength
    return shade_direction * st
