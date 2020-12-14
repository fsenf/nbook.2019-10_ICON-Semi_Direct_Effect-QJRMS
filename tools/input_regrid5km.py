#!/usr/bin/env python


######################################################################
######################################################################

import os
import numpy as np
import xarray as xr
import pandas as pd
import datetime

import nawdex_analysis.io.tools

######################################################################
######################################################################

def input_fields4exp(main_varset, t, dom = 'DOM01', mode = 'absorbing', filter_time = True):

    '''
    Returns regridded simulation data for a certain output type.

    
    Parameters
    ----------
    main_varset : str
        type of simulation data e.g. `rad`, or `cloud`

    t : datetime object
        time slot

    dom : ('DOM01', 'DOM02'), optional
        select grid spacing via domain argument

    mode : ('absorbing', 'scattering'), optional
        if sim with or w/o absorption is taken

    
    Returns
    -------
    d : xr.Dataset
        regridded simulation read from disk


    Notes
    -----
    Conservative regridding used starting from base data.
    '''
    
    # select dimension
    if main_varset in ['rad', 'cloud', 'surface']:
        dimension = '2d'
    else:
        dimension = '3d'
        
    # select sim experiment
    if mode == 'absorbing':
        subpath = '20130502_CCN_rad'
    elif mode == 'scattering':
        subpath = '20130502_semi_direct_effect'
    
    time_string = t.strftime('%Y%m%dT%H%M%00Z')
    
    # read data
    fdir = '%s/icon-lem/%s/DATA' % (os.environ['LOCAL_DATA_PATH'], subpath )
    
    flist = '%s/%s_%s_*%s*%s*_regrid5km.nc' % (fdir, dimension, main_varset, dom, time_string )
    
    print ( flist )
    d = xr.open_mfdataset( flist, chunks = {'time':1} ).squeeze()
    if 'time' not in d.dims:
        d = d.expand_dims( 'time')

    # patch time vector
    time_vector = nawdex_analysis.io.tools.convert_timevec( d.time.data )
    d['time'] = time_vector
    
    d = d.expand_dims(['mode','dom'])
    d['mode'] = [mode, ]
    d['dom']  = [dom, ]
    
    if filter_time:
        return d.sel( time = t )
    else:
        return d


######################################################################
######################################################################

def input_fields( var, t, filter_time = True ):
    
    '''
    Returns combined & regridded simulation data for a certain output type

    
    Parameters
    ----------
    var : str
        type of simulation data e.g. `rad`, or `cloud`

    t : datetime object
        time slot

    
    Returns
    -------
    d : xr.Dataset
        regridded simulation read from disk


    Notes
    -----
    Conservative regridding used starting from base data.
    '''

    dlist = []
    
    for dom in ['DOM01', 'DOM02']:
        for mode in['absorbing', 'scattering']:
            
            d_input = input_fields4exp(var, t, dom = dom, mode = mode, filter_time = filter_time)
            
            d_resampled = d_input #temporal_resampling( d_input )
            
            dlist += [d_resampled.copy(), ]
            
    return xr.merge( dlist )


######################################################################
######################################################################


def get_difference( dset ):
    
    '''
    Get difference between reference and perturbation run.
    
    
    Parameters
    ----------
    dset : xr.Dataset
        set of fields for which differences will be derived
        
    
    Returns
    --------
    : xr.Dataset
        dataset with differences included

    '''
    
    d = dset.sel( mode = 'absorbing' ) - dset.sel( mode = 'scattering' )
    d.expand_dims('mode')
    d['mode'] = ['difference', ]
    
    return xr.concat([dset, d], dim = 'mode')


######################################################################
######################################################################


def input_2dfields( t, difference = True, filter_time = True ):
    
   
    '''
    Returns combination of all regridded 2d fields from simulation data.

    
    Parameters
    ----------
    t : datetime object
        time slot

    difference : {True, False}, optional
        switch if also differences will be prepared for the dataset


    Returns
    -------
    d : xr.Dataset
        regridded simulation read from disk


    Notes
    -----
    Conservative regridding used starting from base data.
    '''


    dset = xr.merge([ input_fields('cloud',   t, filter_time = filter_time), 
                      input_fields('rad',     t, filter_time = filter_time), 
                      input_fields('surface', t, filter_time = filter_time) ] )


    if difference:
        dset = get_difference( dset )

    return dset


######################################################################
######################################################################


def get_ave_pressure_profile( t ):

    '''
    Read domain-average pressure profile.


    Parameters
    ----------
    t : datetime object or numpy.datetime64 object
       time


    Returns
    -------  
    p_ave : xr.DataArray
        domain-average pressure profile in hPa
    '''
    
    if type(t) != type( datetime.datetime ):
        t = pd.Timestamp(np.str( t )).to_pydatetime()

    dset = input_fields4exp('coarse', t, dom = 'DOM01', mode = 'scattering')
    
    p = dset['pres'].squeeze()
    
    p_ave = 1e-2 * p.mean( ['lon', 'lat'])
    
    attrs = dict( long_name = 'domain-average atmospheric pressure profile', units = 'hPa')
    p_ave.attrs.update( attrs )

    return p_ave

######################################################################
######################################################################

def map_level_and_halflevel_in_icondata( d_in, 
                                         full_lev_name = 'height_2', 
                                         half_lev_name = 'height_2_2', 
                                         do_interpolation2full_level = False):
    '''
    Maps given ICON height coordinates to full and half levels.
    
    
    Parameters
    ----------
    d_in : xr.Dataset
        input ICON dataset
        
    full_lev_name : str, optional
        original name of full level coordinate
        
    half_lev_name : str, optional
        original name of half level coordinate
        
    do_interpolation2full_level : {True, False}, optional
        if data on half level is interpolated to full level
        
    
    Returns
    -------
    d_out : xr.Dataset
        output ICON dataset with standardize level names, 
        i.e. ('full_level', 'half_level')
        
    '''
    
    # first rename the coordinates
    d_out = d_in.rename( {full_lev_name : 'full_level', half_lev_name : 'half_level'} )
    
    # shift the last level such that [0.5 1 1.5 2 ...] is combined levelset
    d_out = d_out.assign_coords( half_level =  d_out.half_level - 0.5 )
    
    if do_interpolation2full_level:
        
        d_out = d_out.interp( half_level = d_out.full_level )
        
    return d_out
