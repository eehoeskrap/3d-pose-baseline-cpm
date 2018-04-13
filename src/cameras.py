
"""Utilities to deal with the cameras of human3.6m"""

from __future__ import division

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import data_utils
import viz

def project_point_radial( P, R, T, f, c, k, p ):
    
    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    X = R.dot( P.T - T ) # rotate and translate
    XX = X[:2,:] / X[2,:]
    r2 = XX[0,:]**2 + XX[1,:]**2

    radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) );
    tan = p[0]*XX[1,:] + p[1]*XX[0,:]

    XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )

    Proj = (f * XXX) + c
    Proj = Proj.T

    D = X[2,]

    return Proj, D, radial, tan, r2

def world_to_camera_frame(P, R, T):

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.dot( P.T - T ) # rotate and translate

    return X_cam.T

def camera_to_world_frame(P, R, T):

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.T.dot( P.T ) + T # rotate and translate

    return X_cam.T

def load_camera_params( hf, path ):

    R = hf[ path.format('R') ][:]
    R = R.T

    T = hf[ path.format('T') ][:]
    f = hf[ path.format('f') ][:]
    c = hf[ path.format('c') ][:]
    k = hf[ path.format('k') ][:]
    p = hf[ path.format('p') ][:]

    name = hf[ path.format('Name') ][:]
    name = "".join( [chr(item) for item in name] )

    return R, T, f, c, k, p, name

def load_cameras( bpath='cameras.h5', subjects=[1,5,6,7,8,9,11] ):

    rcams = {}

    with h5py.File(bpath,'r') as hf:
        for s in subjects:
            for c in range(4): # There are 4 cameras in human3.6m
                rcams[(s, c+1)] = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1) )

    return rcams
