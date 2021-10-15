##This module contains functions that calculate blocking indices from iris cubes of geopotential##

import iris
import numpy as np
from regime import reg_lens

##1D Tibaldi Molteni Index############

#get geopotential gradient in South region
def _GHGS(Z,lat_n=80,lat_0=60,lat_s=40,delta=0,MGI=False):
    
    Z_0=Z.extract(iris.Constraint(latitude=lat_0+delta))
    Z_s=Z.extract(iris.Constraint(latitude=lat_s+delta))
    ghgs=(Z_0-Z_s)/(lat_0-lat_s)
    return ghgs

#get geopotential gradient in North region
def _GHGN(Z,lat_n=80,lat_0=60,lat_s=40,delta=0):
    
    Z_0=Z.extract(iris.Constraint(latitude=lat_0+delta))
    Z_n=Z.extract(iris.Constraint(latitude=lat_n+delta))
    ghgn=(Z_n-Z_0)/(lat_n-lat_0)
    return ghgn

#Calculate the 1D Tibaldi Molteni blocking index:
def TM_index(Z,lat_n=80,lat_0=60,lat_s=40,deltas=[-5,0,5],thresh=-10):
    
    #Set up memory storage
    TM=Z.collapsed("latitude",iris.analysis.MEAN)
    s_test=np.zeros([len(deltas),*TM.shape])
    n_test=np.zeros([len(deltas),*TM.shape])

    #for each delta check if conditions are true
    for i,d in enumerate(deltas):
        
        #1.) is the Southern gradient positive?
        ghgs=_GHGS(Z,lat_n,lat_0,lat_s,d)
        s_test[i]=ghgs.data>0
        
        #2.) is the Northern gradient less than -10m/deg lat?
        ghgn=_GHGN(Z,lat_n,lat_0,lat_s,d)
        n_test[i]=ghgn.data<thresh
        
    #If both conditions are true than the test is passed:
    test=np.array(s_test)*np.array(n_test)
    
    #If the test is passed for any delta we are happy:
    test=np.any(test,axis=0)
    
    #Pack the result in an iris cube and return
    TM.data=test
    TM.long_name="TM index"
    return TM

## 2D indices like Davini 2012##################

#Used to filter out low latitude blocks (LLB)
def _GHGS2(Z,lat_n=80,lat_0=60,lat_s=40,delta=0):
    
    Z_ss=Z.extract(iris.Constraint(latitude=lat_s-15))
    Z_s=Z.extract(iris.Constraint(latitude=lat_s))
    ghgs2=(Z_s-Z_ss)/(15)
    return ghgs2


#Calculate the 2D instantaneous blocking (IB) index:
#If LLB filter is true a third test criteria is applied that removes
#low latitude events that aren't true blocking because they don't block the flow.
#If MGI is True, return also the meridional gradient intensity, a metric of blocking strength
def IB_index(Z,lat_delta=15,lat0_min=30,lat0_max=75,thresh=-10,LLB_filter=False,LLB_thresh=-5,MGI=False):
    
    
    #Set up memory storage
    IB=Z.intersection(latitude=[lat0_min,lat0_max]).copy()
    s_test=np.zeros([*IB.shape])
    n_test=np.zeros([*IB.shape])
    
    s2_test=np.zeros([*IB.shape])

    lons=IB.coord("longitude").points
    lats=IB.coord("latitude").points
    
    if MGI:
        mgi=np.zeros_like(s_test)
    #for each latitude and longitude check if conditions are true
    for j,lon in enumerate(lons):
        Zlon=Z.extract(iris.Constraint(longitude=lon))
        for i,lat in enumerate(lats):
        
            #1.) is the Southern gradient positive?
            ghgs=_GHGS(Zlon,lat+lat_delta,lat,lat-lat_delta,0)
            s_test[...,i,j]=ghgs.data>0
            
            #optionally reutn MGI
            if MGI:
                mgi[...,i,j]=ghgs.data
                
            #2.) is the Northern gradient less than -10m/deg lat?
            ghgn=_GHGN(Zlon,lat+lat_delta,lat,lat-lat_delta,0)
            n_test[...,i,j]=ghgn.data<thresh
            
            if LLB_filter:
                #3.) is the Far southern gradient negative?
                ghgs2=_GHGS2(Zlon,lat+lat_delta,lat,lat-lat_delta,0)
                s2_test[...,i,j]=ghgs2.data<LLB_thresh


        
    #If both conditions are true than the test is passed:
    test=np.array(s_test)*np.array(n_test)
    
    if LLB_filter:
        test=test*np.array(s2_test)
        
    #Pack the result in an iris cube and return
    IB.data=test
    IB.long_name="IB index"
    
    if MGI:
        
        mgi[mgi<0]=0
        mgi_ix=IB.copy()
        mgi_ix.long_name="meridional gradient index"
        mgi_ix.data=mgi
        return IB,mgi_ix
    return IB

#Compute Large Scale Blockings from instantaneous blockings
#based on whether they extend over at least +/-lon_thresh deg lat:
#(Assumes longitude is third coord)
def LSB_index(IB_ix,lon_thresh=7.5):
    
    lons=IB_ix.coord("longitude").points
    LSB_ix=IB_ix.copy()
    
    dL=lons[1]-lons[0]
    Nlats=np.ceil(lon_thresh/dL)
    lon_dim=LSB_ix.coord_dims("longitude")[0]
    #Loop over longitudes
    for i,lon in enumerate(lons):
        LSBs=[]
        #Loop over offsets:
        for n in range(int(2*Nlats+1)):
            #Extract the slice along longitude
            lon_slice=IB_ix.intersection(longitude=[lon-dL*n,lon+dL*((2*Nlats)-n)])
    
            #check if all points are blocked
            LSB=np.all(lon_slice.data,axis=lon_dim)
            LSBs.append(LSB)
        
        #If for any offset we find the whole slice contiguously blocked
        #then the point is part of a large scale block.
        LSB_ix.data[:,:,i]=np.any(np.array(LSBs),axis=0)
        
        
    LSB_ix.long_name="LSB index"
    return LSB_ix

#An auxilliary function used by blocking_event_index
#Takes an index, and checks if any points in a lat lon box of +/-lat_thresh
# +/- lon thresh around each point is true. If so make that point true.
def _box_ix(ix,lat_thresh=2.5,lon_thresh=5):
    
    box_ix=ix.copy()
    box_ix.long_name="box index"
    
    lons=ix.coord("longitude").points
    lats=ix.coord("latitude").points
    
    lat_dim=ix.coord_dims("latitude")[0]
    lon_dim=ix.coord_dims("longitude")[0]

    #loop over lat and lon
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            #extract the desired box
            box=ix.intersection(latitude=[lat-lat_thresh,lat+lat_thresh]\
                              ,longitude=[lon-lon_thresh,lon+lon_thresh])
            #check if any value in the box is true
            box_ix.data[...,i,j]=np.any(box.data,axis=(lat_dim,lon_dim)).astype(np.int32)
    
    return box_ix

#An auxilliary function used by blocking_event_index.
# Takes in an index and only returns true when the index
#is true for pers_thresh consecutive timesteps or more.
#Assumes TxLatxLon cube
def _check_persistence(ix,pers_thresh=5):
    pers_ix=ix.copy()
    lons=ix.coord("longitude").points
    lats=ix.coord("latitude").points
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            L,S=reg_lens(ix[:,i,j].data)
            keepS=S==1
            keepL=L>pers_thresh
            keep_points=(np.repeat(keepL,L)*np.repeat(keepS,L))
            pers_ix.data[:,i,j]=keep_points
    return pers_ix
    
#Defines a blocking event by requiring spatial and temporal persistence:
def blocking_event_index(LSB_ix,lat_thresh=2.5,lon_thresh=5,pers_thresh=5):
    
    #Find all points where there is LSB within
    #a certain lat lon region
    box_index=_box_ix(LSB_ix,lat_thresh,lon_thresh)
    
    blocking_event=_check_persistence(box_index,pers_thresh)
    
    return blocking_event

#This is a metric of intensity of blocking using only Z500 differences
def BI_index(Z,lon_thresh=60):
    lons=Z.coord("longitude").points
    lats=Z.coord("latitude").points
    BI_ix=Z.copy()
    BI_ix.long_name="blocking intensity index"
    #looping over gridpoints:
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            
            Z0=Z[:,i,j].data
            
            #get minimum of lon slice upstream of Z0:
            Z_upstream=Z[:,i].intersection(longitude=[lon-lon_thresh,lon])
            Z_u=Z_upstream.data.min(axis=1)
            
            #and minimum of downstream slice:
            Z_downstream=Z[:,i].intersection(longitude=[lon,lon+lon_thresh])
            Z_d=Z_downstream.data.min(axis=1)
            
            RC=(Z0/2) +(Z_u+Z_d)/4
            BI=100*(-1+(Z0/RC))
            BI_ix.data[:,i,j]=BI
    return BI_ix

#The wavebreaking index will be positive during anticyclonic blocks, and negative during cyclonic blocks
def wave_breaking_index(Z,lat0_min=30,lat0_max=75,lat_thresh=7.5,lon_thresh=7.5):
    
        WBI=Z.intersection(latitude=[lat0_min,lat0_max]).copy()
        lats=WBI.coord("latitude").points
        lons=WBI.coord("longitude").points
        
        for i,lat in enumerate(lats):
            lat_s=lat-lat_thresh
            Zlat=Z.extract(iris.Constraint(latitude=lat_s))
            for j,lon in enumerate(lons):
                lon_w=(lon-lon_thresh)%360
                lon_e=(lon+lon_thresh)%360
                
                Zw=Zlat.extract(iris.Constraint(longitude=lon_w)).data
                Ze=Zlat.extract(iris.Constraint(longitude=lon_e)).data
                wbi=(Zw-Ze)/(2*lon_thresh)
                WBI.data[:,i,j]=wbi
        WBI.long_name="wave breaking index"
        return WBI

                

    