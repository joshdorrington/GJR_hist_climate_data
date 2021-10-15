import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
import iris
from iris.analysis.stats import pearsonr as ipearsonr
from iris.analysis.cartography import cosine_latitude_weights
from scipy.spatial import distance
import cartopy.crs as ccrs
import itertools
import warnings
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",DeprecationWarning)

@dataclass
class Cluster_params:
    '''Class for keeping track of cluster parameters.'''
    K: int
    states: list
    inertia: float
    centroids: np.ndarray 
    
    def get_variance_ratio(self,data) -> float:
        #The mean position of points in a state
        centers=[]
        #The variance of points in a state
        intras=[]
        
        #looping over states:
        for s in np.unique(self.states):
            
            centers.append(data[self.states==s].mean(axis=0))
            
            intra=np.sum(data[self.states==s].var(axis=0))
            intras.append(intra)
        #We add the 
        inter=np.sum(np.var(np.array(centers),axis=1))
        self.var_ratio=np.sqrt(np.mean(intras)/inter)
        return self.var_ratio
    
    def get_transmat(self,state_combinations=None,exclude_diag=False):
        if state_combinations is None:
            state_combinations=np.unique(self.states)
        trans=np.zeros([self.K,self.K])
        
        for i,state1 in enumerate(state_combinations):
            for j,state2 in enumerate(state_combinations):
                trans[i,j]=sum((np.isin(self.states[1:],state2))&\
                     (np.isin(self.states[:-1],state1)))\
                     /sum(np.isin(self.states,state1))
        
        if exclude_diag:
            trans -= np.diag(trans)*np.eye(self.K)
            trans /= trans.sum(axis=1)[:,None]
        self.transmat=trans
        return trans    
    
    def reorder(self,new_order):
        
        new_states=-np.ones_like(self.states)
        for old,new in new_order:
            new_states[self.states==old]=new
        self.states=new_states
        self.get_transmat()
        
        mapping=np.array([m[1] for m in new_order])
        self.centroids=self.centroids[mapping]
        


def Kmeans_cluster(data,K):
    
    kmeans=KMeans(n_clusters=K,n_init=100)
    states=kmeans.fit_predict(data)
    inertia=kmeans.inertia_
    centroids=kmeans.cluster_centers_
    
    params=Cluster_params(K,states,inertia,centroids)
    params.get_variance_ratio(data)
    params.get_transmat()
    return params

#Assumes time is first axis
def get_cluster_cube(input_cube,states,as_anomaly=True):
    Ks=np.unique(states)
    if len(states)!=len(input_cube.coord('time').points):
        print(f"length of state vector was {len(states)}, was expecting {len(input_cube.coord('time').points)}")
        return None
    
    #Create array to store data
    clusters=np.zeros([len(Ks),*input_cube.shape[1:]])
    
    #Loop over regimes and extract time means
    for i,K in enumerate(Ks):
        
        clusters[i] = input_cube.data[states==K].mean(axis=0)
        
    #Subtract anomaly of time axis
    if as_anomaly:
        mean=input_cube.data.mean(axis=0)
        clusters-=mean[None,...]
        
    cluster_cube=iris.cube.Cube(data=clusters)
    cluster_cube.add_dim_coord(iris.coords.DimCoord(Ks,long_name="mean cluster composites"),0)
    
    #If time is a dim coord, we assume its the first dim coord
    #Sometimes time is not monotonic and gets demoted to an aux
    #coord, hence this if-else logic.
    if type(input_cube.coord("time"))==iris.coords.DimCoord:
        coords=input_cube.dim_coords[1:]
    else:
        coords=input_cube.dim_coords
        
    for i,coord in enumerate(coords):
        cluster_cube.add_dim_coord(coord,input_cube.coord_dims(coord)[0])
        
    return cluster_cube


def correlate_clusters(c1,c2,and_mapping=False,mean_only=True):
    from scipy.optimize import linear_sum_assignment as lsa
    K=c1.shape[0]
     
     #get the correlation between all the clusters
     #in a matrix
    corrs=np.zeros([K,K])
    weights=cosine_latitude_weights(c1[0])
    for i,j in itertools.product(np.arange(K),repeat=2):
        corrs[i,j]=ipearsonr(c1[i],c2[j],weights=weights).data
    order=lsa(-corrs)
    mapping=[(a,b) for a,b in zip(order[0],order[1])]
    
    pattern_corrs=np.array([corrs[map_] for map_ in mapping])
    mean_corrs=pattern_corrs.mean()
    if mean_only:
        corrs=mean_corrs
    else:
        corrs=(mean_corrs,pattern_corrs)
        
    if and_mapping==True:
        return corrs,mapping
    else:
        return corrs

def correlate_cubes(c1,c2):
    
    weights=cosine_latitude_weights(c1[0])
    c2dummy=c1.copy()
    c2dummy.data=c2.data
    corr=ipearsonr(c1,c2,weights=weights).data
    return corr