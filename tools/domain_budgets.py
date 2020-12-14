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

import nawdex_analysis.io.tools
import tropy.analysis_tools.grid_and_interpolation as gi

######################################################################
######################################################################


def xy_grid_for_icondata( dset ):

    '''
    Calculates an x-y grid (rotated sinosoidal projection) for 
    simulation data stored on regular lon-lat grid.


    Parameters
    ----------
    dset : xr.Dataset
        dataset containing lon / lat vectors


    Returns
    -------
    d : xr.Dataset
        copy of input data with added x, y grid

    '''

    d = dset.copy()
    
    olon = xr.ones_like( d.lon )
    olat = xr.ones_like( d.lat )


    d['lat2d'] = d.lat * olon
    d['lon2d'] = olat * d.lon


    x,y = gi.ll2xyc( d['lon2d'], d['lat2d'] )
    
    d['x'] = x
    d['y'] = y
    
    return d


######################################################################
######################################################################


def gridbox_area_for_icondata( dset ):

    '''
    Calculates gridbox area field.
    '''
    

    d = dset.copy()

    # add x-y info if needed
    if ('x' not in d) or ('y' not in d):
        d = xy_grid_for_icondata( dset )


    # calculate approximate gridbox area (using rectangles) 
    # * factor 1e6 for km**2 to m**2 conversion
    dA = 1e6 * xr.apply_ufunc( gi.simple_pixel_area, 
                               d['x'], d['y'], kwargs = dict(xy = True))

    dA.attrs.update(  {'standard_name' : 'gridbox area', 'units' : 'm**2'} )
    d['dA'] = dA 

    
    return d


######################################################################
######################################################################


def height_for_icondata( dset, dom = 'DOM01', regrid = 'regrid5km' ):
    
    '''
    Gets height profiles and layer thickness for ICON data.


    Parameters
    ----------
    dset : xr.Dataset
        input data

    dom : {'DOM01', 'DOM02'}, optional
        domain identifier, DOM01 ~ 625 m, DOM02 ~ 312 m
    
    regrid : {'regrid5km', 'coarse'}, optional
        which regridded data set is taken
        
    
    Returns
    -------
    d : xr.Dataset
        input data with height info added
    '''

    d = dset.copy()

    # open georeference
    gdir = '%s/icon-lem/20130502_CCN_rad/GRID/' % os.environ['LOCAL_DATA_PATH']
    gfile =  '%s/GRID_3d_%s_ll_%s_ML_20130502T000000Z.nc' % (gdir, regrid, dom)

    geo = xr.open_dataset(gfile)


    # rewrite coordinates (mitigate rounding errors)
    geo['lon'] = dset.lon
    geo['lat'] = dset.lat


    # get layer thickness
    dz = -geo['z_ifc'].diff( 'height')
    dz = dz.rename({'height' : 'height_2'})
    dz['height_2'] = geo['height_2']

    geo['dz'] = dz

    for vname in ['z_mc', 'dz', 'z_ifc']:
        d[vname] = geo[vname]


    return d


######################################################################
######################################################################


def select_edge_of_icondata( d, offset = 0 ):
    
    '''
    Selects the domain boundary edges for ICON data on lon/lat grid.
    
    
    Parameters
    ----------
    dset : xr.Dataset
        input ICON data on lon/lat grid
        
    
    Returns
    -------
    edge : xr.Dataset
        data at domain edge stored as vector
    '''
    
    dset = d.copy()
    nlon = len( d.lon )
    nlat = len( d.lat )
    dset = dset.isel( lon = slice(offset, nlon - offset), lat = slice( offset, nlat - offset) )

    # left edge

    i_offset = 0

    left_edge = dset.isel( lon = 0 ).reset_coords( 'lon' , drop = True )
    left_edge = left_edge.rename({'lat' : 'idx'})
    left_edge['idx'] = np.arange( len(left_edge.idx) ) + i_offset
    left_edge['edge_type'] = xr.ones_like( left_edge['idx'] )


    # upper edge

    i_offset = left_edge.idx.max()

    upper_edge = dset.isel( lat = -1 ).reset_coords( 'lat', drop = True )
    upper_edge = upper_edge.rename({'lon' : 'idx'})
    upper_edge['idx'] = np.arange( len(upper_edge.idx) ) + i_offset.data
    upper_edge['edge_type'] = 2 * xr.ones_like( upper_edge['idx'] )



    # right edge

    i_offset = upper_edge.idx.max()

    right_edge = dset.isel( lon = -1 ).reset_coords( 'lon' , drop = True )
    right_edge = right_edge.rename({'lat' : 'idx'})
    right_edge['idx'] = np.arange( len(right_edge.idx) ) + i_offset.data
    right_edge['edge_type'] =  3 * xr.ones_like( right_edge['idx'] ) 


    # upper edge

    i_offset = right_edge.idx.max()

    lower_edge = dset.isel( lat = 0 ).reset_coords( 'lat' , drop = True )
    lower_edge = lower_edge.rename({'lon' : 'idx'})

    lower_edge['idx'] = np.arange( len(lower_edge.idx) ) + i_offset.data
    lower_edge['edge_type'] =  4 * xr.ones_like( lower_edge['idx'] ) 



    edge = xr.concat( [left_edge, upper_edge, right_edge, lower_edge], dim = 'idx' , coords = 'minimal', compat = 'override')
    edge['edge_type'].attrs.update( {'left' : 1, 'upper' : 2, 'right' : 3, 'lower':4})
    
    return edge

######################################################################
######################################################################


def velocity_normal_to_edge( edge ):
    
    '''
    Returns velocity normal to edge.
    
    
    Parameters
    ----------
    edge : xr.Dataset
        data at domain edge stored as vector
    
    
    Returns
    -------
    v_normal : xr.Dataset
        normal wind component
        
        
    Notes
    -----
    Assumes that (left, upper, right, lower) edges are flagged with (1, 2, 3, 4).
    '''
    
    u = edge['ua']
    v = edge['va']
    
    
    # initial velocity
    v_normal = xr.zeros_like( u )
    
    v_normal =  u.where( edge['edge_type'] == 1 )
    v_normal = -v.where( edge['edge_type'] == 2, -v_normal.copy() )
    v_normal = -u.where( edge['edge_type'] == 3, -v_normal.copy() )
    v_normal =  v.where( edge['edge_type'] == 4, v_normal )
    
    
    
    # set data attributes
    attrs = dict( name = 'v_norm', 
                  standard_name = 'normal wind', 
                  long_name = 'velocity normal to the domain edges', 
                 units = 'm s**-1')
    v_normal.attrs.update( attrs )
    
    
    return v_normal


######################################################################
######################################################################



def path_length_along_edge( edge ):
    
    dx = edge['x'].differentiate('idx')
    dy = edge['y'].differentiate('idx')
    
    edge['path_length'] = np.sqrt( dx**2   +   dy**2 )
    
    return


######################################################################
######################################################################


def volume_integral( v, d, offset = 0 ):
    
    '''
    Calculates Volume Integral.
    
    
    Parameters
    ----------
    v : xr.Dataset
        input variable, kernel of the integral

    d : xr.Dataset
        input ICON data containing information on grid spacings 
        
    
    Returns
    -------
    Integral : xr.Dataset
        
    '''
    
    dset = d.copy()
    dset['var'] = v
    
    nlon = len( d.lon )
    nlat = len( d.lat )
    dset = dset.isel( lon = slice(offset, nlon - offset), lat = slice( offset, nlat - offset) )
    
    
    Integral = ( dset['var'] * dset['dA'] * dset['dz']).sum(('lon', 'lat', 'height_2'))
    
    return Integral


######################################################################
######################################################################



def flux_integral( v, d, offset = 0 ):
    
    '''
    Calculates Flux Integral.
    
    
    Parameters
    ----------
    v : xr.Dataset
        input variable, kernel of the integral

    d : xr.Dataset
        input ICON data containing information on grid spacings 
        
    
    Returns
    -------
    Flux : xr.Dataset
        
    '''
    dset = d.copy()
    dset['var'] = v
    
    
    edge = select_edge_of_icondata( dset, offset = offset )
    edge['vn'] = velocity_normal_to_edge( edge )
    path_length_along_edge( edge )
    edge['ds'] = edge['path_length'] * 1e3

    Flux = ( edge['var'] * edge['vn'] * edge['ds'] * edge['dz'] ).sum(('idx', 'height_2')) 
    
    return Flux


######################################################################
######################################################################


def flux_divergence( var, dset,
                     method = 'spherical',
                     level_name = 'height_2',
                     do_metric_correction = True ):
    
    '''
    Calculates Flux Divergence.
    
    
    Parameters
    ----------
    var : xr.DataArray
        variable for which flux is calculated
        
    dset : xr.Dataset
        additional data for variable, e.g. winds and geo-ref
        
    method : {'differentiate', 'diff'}, optional
        method how gradient is calculated 
        
        * 'differentiate' : the xr differentiate method
        * 'diff' : the xr diff method and following interpolation to target levels
    
    
    Returns
    -------
    div_F : xr.Dataset o rx
    '''
    
    
    Fx = var * dset['ua']
    Fy = var * dset['va']

    if method == 'differentiate':
        dx = dset['x'].differentiate( 'lon' ) * 1e3   # in meters
        dy = dset['y'].differentiate( 'lat' ) * 1e3   # in meters

        dFx_dx = Fx.differentiate( 'lon' ) / dx
        dFy_dy = Fy.differentiate( 'lat' ) / dy

        div_F = dFx_dx + dFy_dy 
    
    elif method == 'diff':
        dx = dset['x'].diff( 'lon' ) * 1e3   # in meters
        dy = dset['y'].diff( 'lat' ) * 1e3   # in meters

        dFx_dx = Fx.diff( 'lon' ) / dx
        dFy_dy = Fy.diff( 'lat' ) / dy

        div_F = dFx_dx + dFy_dy 

        div_F = div_F.interp({'lon' : Fx.lon} )
   

    elif method == 'spherical':

        # earth radius
        R = 6371e3
        
        phi = np.deg2rad( dset['lat'] )
        lam = np.deg2rad( dset['lon'] )
        
        cosphi = np.cos( phi )
        
        dphi = phi.diff( 'lat' )
        dlam = lam.diff( 'lon' )
        
        dFx_dlam    =  Fx.diff( 'lon' ) / dlam
        dcosFy_dphi =  (cosphi * Fy).diff( 'lat' ) / dphi

        dFx_dlam    = dFx_dlam.interp({'lon' : Fx.lon} )
        dcosFy_dphi = dcosFy_dphi.interp({'lat' : Fy.lat} )

        
        div_F = 1 / (R * cosphi) * ( dFx_dlam + dcosFy_dphi )
        
    # care for terrain-following coordinate
    if do_metric_correction:
        
        # dz = dset['dz']
        z = dset['z_mc']

        grad_z = horizontal_gradient( z, dset )
    
        # vertical gradient of fluxes
        dFx = Fx.differentiate( level_name ) #.diff( 'height_2' ).interp( height_2 = var.height_2 )
        dFy = Fy.differentiate( level_name ) #.diff( 'height_2' ).interp( height_2 = var.height_2 )
        dz  = z.differentiate( level_name )
        
        dFx_dz = dFx / dz
        dFy_dz = dFy / dz
        
        corr = - dFx_dz * grad_z[0]  -  dFy_dz * grad_z[1]
        
        div_F += corr
        
    return div_F


######################################################################
######################################################################


def horizontal_gradient( var, dset ):
    
    '''
    Calculates horizontal gradient.
    
    
    Parameters
    ----------
    var : xr.DataArray
        variable for which flux is calculated
        
    dset : xr.Dataset
        additional data for variable, e.g. winds and geo-ref
        
    
    
    Returns
    -------
    gradient : xr.Dataset o rx
    '''
    
    
 
    # earth radius
    R = 6371e3
     
    phi = np.deg2rad( dset['lat'] )
    lam = np.deg2rad( dset['lon'] )
        
    cosphi = np.cos( phi )
       
    dphi = phi.diff( 'lat' )
    dlam = lam.diff( 'lon' )
        
    dvar_dlam  =  var.diff( 'lon' ) / dlam
    dvar_dphi  =  var.diff( 'lat' ) / dphi

    dvar_dlam  =  dvar_dlam.interp({'lon' : var.lon} )
    dvar_dphi  =  dvar_dphi.interp({'lat' : var.lat} )

        
    gradient = [dvar_dlam / ( R * cosphi), dvar_dphi / R ]
        
    return gradient


######################################################################
######################################################################

def horizontal_advection( var, dset ):
    
    '''
    Calculates horizontal advection.
    
    
    Parameters
    ----------
    var : xr.DataArray
        variable for which flux is calculated
        
    dset : xr.Dataset
        additional data for variable, e.g. winds and geo-ref
        
    
    
    Returns
    -------
    adv : xr.Dataset o rx
    '''
    
    
    # variable shortcuts
    u = dset['ua']
    v = dset['va']
    z = dset['z_mc']
    dz = dset['dz']
    
    
    # horizontal gradient
    grad_var = horizontal_gradient( var, dset )
    
    # care for terrain-following coordinate
    grad_z = horizontal_gradient( z, dset )
    
    # vertical gradient
    dvar = var.diff( 'height_2' ).interp( height_2 = var.height_2 )
    dv_dz = dvar / dz
    
    adv = u * ( 
               grad_var[0] - dv_dz * grad_z[0])   + v * ( 
               grad_var[1] - dv_dz * grad_z[1])  
    
    return adv


######################################################################
######################################################################


def vertical_integral( v, d, offset = 0 ):
    
    '''
    Calculates Vertical Integral and accounts for different grid box areas.
    
    
    Parameters
    ----------
    v : xr.Dataset
        input variable, kernel of the integral

    d : xr.Dataset
        input ICON data containing information on grid spacings 
        
    
    Returns
    -------
    Integral : xr.Dataset
        
    '''
    
    dset = d.copy()
    dset['var'] = v
    
    nlon = len( d.lon )
    nlat = len( d.lat )
    dset = dset.isel( lon = slice(offset, nlon - offset), lat = slice( offset, nlat - offset) )
    
    # area-based weights
    w = dset['dA'] / dset['dA'].mean( ['lon', 'lat'] )
    
    Integral = ( dset['var'] * w * dset['dz']).sum('height_2')
    
    
    
    return Integral

######################################################################
######################################################################



def horizontal_integral( v, d, offset = 0 ):
    
    '''
    Calculates Horizontal Integral and accounts for different grid box areas.
    
    
    Parameters
    ----------
    v : xr.Dataset
        input variable, kernel of the integral

    d : xr.Dataset
        input ICON data containing information on grid spacings 
        
    
    Returns
    -------
    Integral : xr.Dataset
        
    '''
    
    dset = d.copy()
    dset['var'] = v
    
    nlon = len( d.lon )
    nlat = len( d.lat )
    dset = dset.isel( lon = slice(offset, nlon - offset), lat = slice( offset, nlat - offset) )
    
    # layer-thickness weights
    w = dset['dz'] / dset['dz'].mean( ['lon', 'lat'] )
    
    Integral = ( dset['var'] * w * dset['dA']).sum( ['lon', 'lat'] )
    
    
    return Integral

######################################################################
######################################################################

