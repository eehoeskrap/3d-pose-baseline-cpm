
"""Utility functions for dealing with human3.6m data."""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cameras
import viz
import h5py
import glob
import copy

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1,5,6,7,8]
TEST_SUBJECTS  = [9,11]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Spine'
SH_NAMES[8]  = 'Thorax'
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'

def load_data( bpath, subjects, actions, dim=3 ):
    if not dim in [2,3]:
        raise(ValueError, 'dim must be 2 or 3')

    data = {}

    for subj in subjects:
        for action in actions:

            print('Reading subject {0}, action {1}'.format(subj, action))

            dpath = os.path.join( bpath, 'S{0}'.format(subj), 'MyPoses/{0}D_positions'.format(dim), '{0}*.h5'.format(action) )
            print( dpath )

            fnames = glob.glob( dpath )

            loaded_seqs = 0
            for fname in fnames:
                seqname = os.path.basename( fname )

                # This rule makes sure SittingDown is not loaded when Sitting is requested
                if action == "Sitting" and seqname.startswith( "SittingDown" ):
                    continue

        # This rule makes sure that WalkDog and WalkTogeter are not loaded when
        # Walking is requested.
                if seqname.startswith( action ):
                    print( fname )
                    loaded_seqs = loaded_seqs + 1

                    with h5py.File( fname, 'r' ) as h5f:
                        poses = h5f['{0}D_positions'.format(dim)][:]

                    poses = poses.T
                    data[ (subj, action, seqname) ] = poses

            if dim == 2:
                assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead".format( loaded_seqs )
            else:
                assert loaded_seqs == 2, "Expecting 2 sequences, found {0} instead".format( loaded_seqs )

    return data


def load_stacked_hourglass(data_dir, subjects, actions):
    # Permutation that goes from SH detections to H36M ordering.
    SH_TO_GT_PERM = np.array([SH_NAMES.index( h ) for h in H36M_NAMES if h != '' and h in SH_NAMES])
    assert np.all( SH_TO_GT_PERM == np.array([6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10]) )

    data = {}

    for subj in subjects:
        for action in actions:

            print('Reading subject {0}, action {1}'.format(subj, action))

            dpath = os.path.join( data_dir, 'S{0}'.format(subj), 'StackedHourglass/{0}*.h5'.format(action) )
            print( dpath )

            fnames = glob.glob( dpath )

            loaded_seqs = 0
            for fname in fnames:
                seqname = os.path.basename( fname )
                seqname = seqname.replace('_',' ')

                # This rule makes sure SittingDown is not loaded when Sitting is requested
                if action == "Sitting" and seqname.startswith( "SittingDown" ):
                    continue

                # This rule makes sure that WalkDog and WalkTogeter are not loaded when
                # Walking is requested.
                if seqname.startswith( action ):
                    print( fname )
                    loaded_seqs = loaded_seqs + 1

                    # Load the poses from the .h5 file
                    with h5py.File( fname, 'r' ) as h5f:
                        poses = h5f['poses'][:]

                        # Permute the loaded data to make it compatible with H36M
                        poses = poses[:,SH_TO_GT_PERM,:]

                        # Reshape into n x (32*2) matrix
                        poses = np.reshape(poses,[poses.shape[0], -1])
                        poses_final = np.zeros([poses.shape[0], len(H36M_NAMES)*2])

                        dim_to_use_x    = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 2
                        dim_to_use_y    = dim_to_use_x+1

                        dim_to_use = np.zeros(len(SH_NAMES)*2,dtype=np.int32)
                        dim_to_use[0::2] = dim_to_use_x
                        dim_to_use[1::2] = dim_to_use_y
                        poses_final[:,dim_to_use] = poses
                        seqname = seqname+'-sh'
                        data[ (subj, action, seqname) ] = poses_final

      # Make sure we loaded 8 sequences
            if (subj == 11 and action == 'Directions'): # <-- this video is damaged
                assert loaded_seqs == 7, "Expecting 7 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )
            else:
                assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )

    return data


def normalization_stats(complete_data, dim, predict_14=False ):

    if not dim in [2,3]:
        raise(ValueError, 'dim must be 2 or 3')

    data_mean = np.mean(complete_data, axis=0)
    data_std  =  np.std(complete_data, axis=0)

    # Encodes which 17 (or 14) 2d-3d pairs we are predicting
    dimensions_to_ignore = []
    if dim == 2:
        dimensions_to_use    = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
        dimensions_to_use    = np.sort( np.hstack( (dimensions_to_use*2, dimensions_to_use*2+1)))
        dimensions_to_ignore = np.delete( np.arange(len(H36M_NAMES)*2), dimensions_to_use )
    else: # dim == 3
        dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        dimensions_to_use = np.delete( dimensions_to_use, [0,7,9] if predict_14 else 0 )

        dimensions_to_use = np.sort( np.hstack( (dimensions_to_use*3,
                                             dimensions_to_use*3+1,
                                             dimensions_to_use*3+2)))
        dimensions_to_ignore = np.delete( np.arange(len(H36M_NAMES)*3), dimensions_to_use )

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def transform_world_to_camera(poses_set, cams, ncams=4 ):

    t3d_camera = {}
    for t3dk in sorted( poses_set.keys() ):

        subj, action, seqname = t3dk
        t3d_world = poses_set[ t3dk ]

        for c in range( ncams ):
            R, T, f, c, k, p, name = cams[ (subj, c+1) ]
            camera_coord = cameras.world_to_camera_frame( np.reshape(t3d_world, [-1, 3]), R, T)
            camera_coord = np.reshape( camera_coord, [-1, len(H36M_NAMES)*3] )

            sname = seqname[:-3]+"."+name+".h5" # e.g.: Waiting 1.58860488.h5
            t3d_camera[ (subj, action, sname) ] = camera_coord

    return t3d_camera


def normalize_data(data, data_mean, data_std, dim_to_use ):

    data_out = {}

    for key in data.keys():
        data[ key ] = data[ key ][ :, dim_to_use ]
        mu = data_mean[dim_to_use]
        stddev = data_std[dim_to_use]
        data_out[ key ] = np.divide( (data[key] - mu), stddev )

    return data_out


def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_ignore):

    T = normalized_data.shape[0] # Batch size
    D = data_mean.shape[0] # Dimensionality

    orig_data = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = np.array([dim for dim in range(D)
                                if dim not in dimensions_to_ignore])

    orig_data[:, dimensions_to_use] = normalized_data

    # Multiply times stdev and add the mean
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat
    return orig_data


def define_actions( action ):

    actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

    if action == "All" or action == "all":
        return actions

    if not action in actions:
        raise( ValueError, "Unrecognized action: %s" % action )

    return [action]


def project_to_cameras( poses_set, cams, ncams=4 ):

    t2d = {}

    for t3dk in sorted( poses_set.keys() ):
        subj, a, seqname = t3dk
        t3d = poses_set[ t3dk ]

        for cam in range( ncams ):
            R, T, f, c, k, p, name = cams[ (subj, cam+1) ]
            pts2d, _, _, _, _ = cameras.project_point_radial( np.reshape(t3d, [-1, 3]), R, T, f, c, k, p )

            pts2d = np.reshape( pts2d, [-1, len(H36M_NAMES)*2] )
            sname = seqname[:-3]+"."+name+".h5" # e.g.: Waiting 1.58860488.h5
            t2d[ (subj, a, sname) ] = pts2d

    return t2d


def read_2d_predictions( actions, data_dir ):


    train_set = load_stacked_hourglass( data_dir, TRAIN_SUBJECTS, actions)
    test_set  = load_stacked_hourglass( data_dir, TEST_SUBJECTS,  actions)

    complete_train = copy.deepcopy( np.vstack( train_set.values() ))
    data_mean, data_std,  dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )

    train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
    test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def create_2d_data( actions, data_dir, rcams ):


    # Load 3d data
    train_set = load_data( data_dir, TRAIN_SUBJECTS, actions, dim=3 )
    test_set  = load_data( data_dir, TEST_SUBJECTS,  actions, dim=3 )

    train_set = project_to_cameras( train_set, rcams )
    test_set  = project_to_cameras( test_set, rcams )

    # Compute normalization statistics.
    complete_train = copy.deepcopy( np.vstack( train_set.values() ))
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )

    # Divide every dimension independently
    train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
    test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def read_3d_data( actions, data_dir, camera_frame, rcams, predict_14=False ):

    # Load 3d data
    train_set = load_data( data_dir, TRAIN_SUBJECTS, actions, dim=3 )
    test_set  = load_data( data_dir, TEST_SUBJECTS,  actions, dim=3 )

    if camera_frame:
        train_set = transform_world_to_camera( train_set, rcams )
        test_set  = transform_world_to_camera( test_set, rcams )

    # Apply 3d post-processing (centering around root)
    train_set, train_root_positions = postprocess_3d( train_set )
    test_set,  test_root_positions  = postprocess_3d( test_set )

    # Compute normalization statistics
    complete_train = copy.deepcopy( np.vstack( train_set.values() ))
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=3, predict_14=predict_14 )

    # Divide every dimension independently
    train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
    test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions


def postprocess_3d( poses_set ):

    root_positions = {}
    for k in poses_set.keys():
        # Keep track of the global position
        root_positions[k] = copy.deepcopy(poses_set[k][:,:3])

        # Remove the root from the 3d position
        poses = poses_set[k]
        poses = poses - np.tile( poses[:,:3], [1, len(H36M_NAMES)] )
        poses_set[k] = poses

    return poses_set, root_positions
