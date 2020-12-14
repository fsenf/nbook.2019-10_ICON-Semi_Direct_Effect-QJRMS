#!/usr/bin/env python
# coding: utf-8

'''
Prepares and does derivation of domain budgets.
'''

######################################################################
######################################################################


import sys, os
import numpy as np
import xarray as xr
import scipy.ndimage

import tropy.analysis_tools.grid_and_interpolation as gi
import tropy.analysis_tools.segmentation as seg

######################################################################
######################################################################

def update_labeling_based_on_markers( field, thresh, labels, markers, add_prelabels = True, **kws):
    
    '''
    Updates an existing labeling by an additional labeling which
    takes markers for a watershed-based segmentation.
    
    The watershed segmentation gets highest priority for merging the 
    labeling results.
    
    
    Parameters
    ----------
    field : numpy array
        2dim input field
        
    thresh : float
        threshold used for segmentation, field > thresh
        
    labels : numpy array
        existing labels to be modified
        
    markers : numpy array
        markers for watershed where region growing is started
        ( it could be a second labeling result )
    
    
    Returns
    -------
    labels_combined : numpy array
        modified labels
        combination of input labels and watershed-based labeling
    '''
    
    # prepare segmentation parameters
    # ===============================
    kws_seg = dict( # cluster_method = 'watershed', 
                    marker_field = markers, 
                    filter_method = 'gauss',  
                    siggauss = 0  )
    
    kws_seg.update( kws )
    
    
    # segmentation
    # ============
    labels_from_markers = seg.watershed_clustering( field, thresh, **kws_seg)

    
    if add_prelabels:
        
        # shift previous labels
        # =====================
        prelabels = np.where( labels == 0, 0, labels + labels_from_markers.max() )  # add offset


        # combination
        # ===========
        labels_stacked = np.array( [labels_from_markers, prelabels ] )
        labels_combined = combine_object_stack( labels_stacked, sort_cluster_labels = False )
    else:
        labels_combined = labels_from_markers
        
       
    return labels_combined


######################################################################
######################################################################

from tropy.analysis_tools.segmentation import clustering

######################################################################
######################################################################


def sequential_smoothing_segmentation(f, thresh, gauss_sig_list=[0, 1, 2, 3, 5],
                                      **kws):
    '''
    Performs segmentation with sequential smoothing. Keeps the more smooth objects 
    (an other strategy for merging).


    Parameters
    ----------
    f : numpy array, 2dim or 3dim
        input field

    thresh : float
        threshold value for segmentaion

    gauss_sig_list : list or array, optional
        list of gaussian sigma ( ascending )

    **kws : dict
        keywords passed to segmentation routine


    Returns
    -------
    c3d : numpy array, one dimension added
        stack of labeled field, from max. thresh to min. thresh
    '''

    # get shape of field
    s = list(f.shape)

    nsigma = len(gauss_sig_list)

    shape3d = [nsigma, ] + s
    c3d = np.zeros(shape3d)
    noffset = 0

    # now loop over thresholds
    for i in range(nsigma):

        # get labels
        kws_updated = kws.copy()
        kws_updated['siggauss'] = gauss_sig_list[i]
        c = clustering(f, thresh, **kws_updated)

        # store with offset
        c3d[i] = np.where(c == 0, 0, c + noffset)
        noffset = c3d[i].max()

    return c3d

######################################################################
######################################################################


def multi_sigma_clustering(f, thresh, gauss_sig_list=[0, 1, 2, 3, 5],
                           **kws):
    '''
    Performs sequential segmentationwith different smoothing levels 
    and returns the combine result only keeping the predecessor object alive.


    Parameters
    ----------
    f : numpy array, 2dim or 3dim
        input field

    thresh_min : float
        minimum threshold value

    thresh_max : float
        maximum threshold value

    nthresh : int, optional
        number of threshold values

    use_percentile_threshold : {True, False}, optional
        if threshold list is derived from field percentiles
c = skimage.morphology.watershed(-mfield, markers, 
                                     mask = ma_sm,
                                     connectivity = connect)
    **kws : dict
        keywords passed to segmentation routine


    Returns
    -------
    c : numpy array, same shape as f
        combined label field
    '''

    # get the segmented data stack
    c3d = sequential_smoothing_segmentation(f, thresh,
                                            gauss_sig_list=gauss_sig_list,
                                            **kws)

    # we change order here because the predecessor objects are always kept
    c3d_inv = c3d[::-1]
    
    
    # do the sequential combination
    c = combine_object_stack(c3d_inv, sort_cluster_labels = False)

    return c

######################################################################
######################################################################

def combine_object_stack( c3d, sort_cluster_labels = True ):
    
    '''
    Sequentially combines object label fields by searching for new objects 
    that have no predecessor in the higher hierarchy level.
    
    
    Parameters
    ----------
    c3d : numpy array, 3dim or 4dim
        labeled object data stack
        
    
    Returns
    -------
    c : numpy array, 2dim or 3dim
        combined label field
        
    Notes
    -----
    This is a clone of the `tropy`funciton where controll on sorting 
    has been included.
    '''
    
    # get shape and number of stacks
    s = c3d.shape 
    nstack = s[0]
    
    c = c3d[0, :]
    
    for i in range( 1, nstack ):
        
        # extract two labeling levels
        c_prev = c # 3d[i - 1].astype( np.int )
        c_next = c3d[i].astype( np.int )
        
        # get indication if cell is already define for predecessor
        maxlab_prev = scipy.ndimage.measurements.maximum(c_prev, 
                                                         labels = c_next, 
                                                         index = np.arange(0, c_next.max() + 1))
        
        # predecessor mask where a cell already exists
        pred_mask = maxlab_prev[c_next] > 0
        
        # and also get the background
        next_bg_mask = c_next == 0
        
        # combine the two possibilities
        mask = np.logical_or( pred_mask, next_bg_mask)
        
        c = np.where( mask, c, c_next )    
        
    if sort_cluster_labels:
        c = sort_clusters( c.astype(np.int) )
    
    return c

######################################################################
######################################################################

def run_seg_with_marker_xarray(lwp, thresh, 
                               vname = 'clwvi', 
                               template_mode = 'absorbing', 
                               mirror_mode = 'scattering' ):
    
    
    kws_ref = dict(cluster_method = 'connect', filter_method = 'gauss', siggauss = 0, min_size = 0)

    # (1) Make Template Segmentation
    # ===============================
    segmented = []
    
    # placeholder for a loop
    if True:
        c = xr.apply_ufunc( seg.clustering, 
                    lwp, thresh, 
                    input_core_dims=[ ['lon', 'lat'], [] ], 
                    output_core_dims=[['lon', 'lat']],
                    vectorize = True,
                    kwargs = kws_ref)

#        c = xr.apply_ufunc( multi_sigma_clustering, 
#                    lwp, thresh, 
#                    input_core_dims=[ ['lon', 'lat'], [] ], 
#                    output_core_dims=[['lon', 'lat']],
#                    vectorize = True,
#                    kwargs = kws_ref)

        d = c.expand_dims('thresh').rename({vname : 'cluster'})
        d['thresh'] = [thresh, ]
        segmented += [d.copy()]

    segmented = xr.concat( segmented, dim = 'thresh')

    
    # (2) Make "Mirror" Segmentation
    # ===============================

    field = lwp.sel( mode = mirror_mode )[vname]
    labels = segmented.sel( mode = mirror_mode, thresh = thresh )['cluster']
    markers = segmented.sel( mode = template_mode, thresh = thresh )['cluster']

    labels_updated = xr.apply_ufunc( update_labeling_based_on_markers, 
                                     field, thresh, labels, markers,
                                     input_core_dims = [ ['lon', 'lat'], [], ['lon', 'lat'], ['lon', 'lat'] ], 
                                     output_core_dims=[['lon', 'lat']],
                                     vectorize = True,)
   
    mirror = xr.Dataset()
    mirror['cluster'] = labels_updated
    mirror.expand_dims('mode')
    mirror['mode'] = [mirror_mode,]
    
    # (3) Combine two Results
    labels = xr.concat( [segmented.sel(mode = [template_mode,], thresh = thresh), mirror], dim = 'mode')
    
    
    return labels