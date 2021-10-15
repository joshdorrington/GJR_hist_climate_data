import cartopy.crs as ccrs
import iris.plot as iplt
import cmocean.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def quick_contourf(cube,cmap=cm.balance,clevs=None,ax=None,cbar_labs=None,cbar=True,proj_lat=60,proj_lon=-20,zoom=1):
    
    sat_height=35785831 #(in metres)
    proj=ccrs.NearsidePerspective(central_latitude=proj_lat,central_longitude=proj_lon,satellite_height =zoom*sat_height)
    
    if ax is None:
        fig=plt.figure()
        ax=plt.subplot(1,1,1,projection=proj)
        
    if clevs is None:
        clevs=np.linspace(-abs(cube.data).max(),abs(cube.data).max(),21)
        
    ax.coastlines()
    ax.set_global()

    plot=iplt.contourf(cube, levels=clevs, cmap=cmap,extend="both",axes=ax)
    if cbar:
        cbar=plt.colorbar(orientation="horizontal",mappable=plot,ax=ax)

        if cbar_labs is not None:
            cbar.set_ticks(cbar_labs)
            cbar.ax.set_xticklabels(cbar_labs)
        
    return plot,ax

def quick_reg_plot(cube,figdims=(15,10),clevs=None,cbar_labs=None,cmap=cm.balance,axes=None,cbar=True):
    
    proj=ccrs.NearsidePerspective(central_latitude=60,central_longitude=-20)

    K=np.shape(cube)[0]
    #Set up figure if needed:
    if axes is None:
        
        fig=plt.figure()
        fig.set_figwidth(figdims[0])
        fig.set_figheight(figdims[1])
        axes=[]
        for i in range(K):
            ax=plt.subplot(1,K,i+1,projection=proj)
            axes.append(ax)
            
    #Plot:
    for i in range(K):
        
        quick_contourf(cube[i],ax=axes[i],clevs=clevs,cbar_labs=cbar_labs,cmap=cmap,cbar=cbar)
        axes[i].set_title(f"Cluster {i+1}")
        
    fig=plt.gcf()
    return fig,axes

def add_rectangle(ax,lats,lons,**kwargs):
    
    lat1,lat2=lats
    lon1,lon2=lons
    
    height=lat2-lat1
    width=lon2-lon1
    ax.add_patch(mpatches.Rectangle(xy=[lon1,lat1], width=width, height=height,transform=ccrs.PlateCarree(),**kwargs))
    return

def blank_carto_plot(x=1,y=1,proj=ccrs.NearsidePerspective(central_latitude=60,central_longitude=-20)):
    
    fig=plt.figure()
    axis_set=[]
    for i in range(x):
        axes=[]
        for j in range(y):
            ax=plt.subplot(y,x,j*x+i+1,projection=proj)
            axes.append(ax)
        axis_set.append(axes)
    axis_set=np.array(axis_set)
    return fig,np.squeeze(axis_set)