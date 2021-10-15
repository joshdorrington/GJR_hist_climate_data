# What is this for? We want to take in a PC sequence, a jet speed sequence and a Z500 sequence.
#These must match in terms of iris time coord metadata. Once that is checked these funcs should perform all
#aspects of the clustering problem.

#Needed imports and additional functions:
import sys
import numpy as np
import iris
import pickle
import os
import cf_units
from cluster import Kmeans_cluster, get_cluster_cube, correlate_clusters
import iris.util

def combine_time_coords(arr,t_coord="time"):
    
    C1=arr[0].coord(t_coord)
    
    arr=[a.coord(t_coord).copy() for a in arr]
    for a in arr:
        a.convert_units(C1.units)
        
    attrs={m: getattr(C1,m) for m in ["standard_name","long_name","var_name","units","attributes","coord_system"]}
    
    points=[c for C in arr for c in C.points]
    bounds=[b for C in arr for b in C.bounds]
    
    time_coord=iris.coords.DimCoord(points,bounds=bounds,**attrs)
    return time_coord

#regresses series x from y (i.e. jet speed from pcs)
def regress(x,y, deg=1, prnt=True):
    import numpy as np
    import numpy.ma as ma
    from scipy.stats import pearsonr
    #Check for mask
    if ma.is_masked(x) or ma.is_masked(y):
        fitter = ma.polyfit
    else:
        fitter = np.polyfit
    model = fitter(x,y,deg)
    prediction = np.polyval(model, x)
    residuals = y - prediction
    corr, pval = pearsonr(prediction, y)
    if prnt:
        print("Fitted model: y = %.3f*x + %.3f" % (model[0], model[1]))
        print("Correlation = %.3f (p=%.3f)" % (corr, pval))
        print("Explained variance = %.3f" % (corr**2))
        print("Returning (coefficients, prediction, residuals)")
    return corr, residuals

#Like np.atleast_2d, but puts extra dimension on the end.
#Needed for combining different cubes
def atleast_2d_at_end(a):
    a=np.array(a)
    if a.ndim>1:
        return a
    elif a.ndim==1:
        return a.reshape([len(a),1])
    else:
        return np.array([[a]])


#Wrapper around regress
def regress_pc(pc,regressor):
    npcs=pc.shape[1]
    output=[regress(regressor,pc[:,i].data,prnt=False) for i in range(npcs)]
    
    corrs=np.array([o[0] for o in output])
    r_data=np.array([o[1] for o in output])
    
    regressed_pc=pc.copy()
    regressed_pc.data=r_data.T
    return corrs,regressed_pc

def make_cube_with_different_1st_axis(input_cube,new_t,t_ax=None):
    S=input_cube.shape
    
    if t_ax is None:
        t_ax=iris.coords.DimCoord(np.arange(0,new_t),"time",units=cf_units.Unit(f"days since {cf_units.EPOCH}"))
        
    new_cube=iris.cube.Cube(data=np.zeros([new_t,*S[1:]]))
    
    new_cube.add_dim_coord(t_ax,0)

    #If time is a dim coord, we assume its the first dim coord
    #Sometimes time is not monotonic and gets demoted to an aux
    #coord, hence this if-else logic.
    if type(input_cube.coord("time"))==iris.coords.DimCoord:
        coords=input_cube.dim_coords[1:]
    else:
        coords=input_cube.dim_coords
        
    for i,coord in enumerate(coords):
        new_cube.add_dim_coord(coord,input_cube.coord_dims(coord)[0])

        
    new_cube.standard_name=input_cube.standard_name
    new_cube.long_name=input_cube.long_name
    new_cube.var_name=input_cube.var_name
    new_cube.units=input_cube.units
    new_cube.metadata=input_cube.metadata
    new_cube.attributes=input_cube.attributes
    
    try:
        new_cube.attributes["history"]=\
        new_cube.attributes["history"]+" Remade into a different shape by function make_cube_with_different_1st_axis."
    except:
        new_cube.attributes["history"]=" Remade into a different shape by function make_cube_with_different_1st_axis."
        
    return new_cube

#The Main Class Object:

class ClusteringExperiment:
    
    def __init__(self,exp_id="Unnamed",pc_cube=None,regressor=None,field_data=None,auto_squeeze=False):
        
        self.auto_squeeze=auto_squeeze
        self.id=exp_id
        self.pcs=self._squeeze_if_requested(pc_cube)
        self.regressor=self._squeeze_if_requested(regressor)
        self.field_data=self._squeeze_if_requested(field_data)
        
        self._confirm_time_coords_match()
        self.regression_correlations=None
        self.regressed_pcs=None
        self.windowed_pcs=None
        self.windowed_regressed_pcs=None
        self.windowed_field_data=None
        self.regressed_field_data=None
        self.windowed_regressed_field_data=None
        self.windowed_regressor=None
        self.clusters={}
        self.cluster_cubes={}
        self.cluster_correlations={}
        self.pc_list=["pcs"]
        self.random_samples=None
        
    def _squeeze_if_requested(self,C):
        if not self.auto_squeeze:
            return C
        else:
            return iris.util.squeeze(C)
        
    #Not currently very rigorous.
    #Makes sure the time axes align.
    def _confirm_time_coords_match(self):
        
        t_axes=[]
        if self.pcs is not None:
            t_axes.append(self.pcs.coord("time"))
        if self.regressor is not None:
            t_axes.append(self.regressor.coord("time"))
        if self.field_data is not None:
            t_axes.append(self.field_data.coord("time"))
            
        for t_ax in t_axes:
            try:
                assert np.all(t_ax.points==t_axes[0].points)
            except:
                print(t_axes)
                raise(ValueError())
    #regresses self.regressor against self.pcs
    def regress_pcs(self):
            
        if self.regressor is None:
            raise(ValueError("No regressor attribute defined."))
            
        
        corr,regressed_pc=regress_pc(self.pcs,self.regressor.data)
        
        self.regression_correlations=corr
        self.regressed_pcs=regressed_pc
        self.pc_list.append("regressed_pcs")
    
    #Called by combine_with, and used to stick different
    #PC sequences together. 
    def _combine_attribute(self,attr,cluster_array,time_coord=None):
                    
        array=[getattr(self,attr)]
        
        for C in cluster_array:
            array.append(getattr(C,attr))

        if time_coord =="keep":
            time_coord=combine_time_coords(array)

        array=[a.data for a in array]
        
        #The atleast_2d_at here helps make sure 1D attributes
        #(like the regressor) get treated the same way as 2D attributes
        #Unlike an earlier implementation, this handles A of different lengths.
        array=np.vstack([atleast_2d_at_end(A) for A in array])
        
        T=array.shape[0]
        

        new_attr=make_cube_with_different_1st_axis(getattr(self,attr),T,t_ax=time_coord)
        try:
            #We want to get rid of length 1 dimensions, hence the squeeze here
            new_attr.data=np.squeeze(array)
        except:
            print(new_attr)
            print(array.shape)
            raise(ValueError("A problem occurred during squeezing"))
        return new_attr
    
    #Combines the current ClusteringExperiment PCs with some others,
    #appending state sequences together. 
    #If time_coord is "keep", will try to combine existing time coords
    def combine_with(self,cluster_array,time_coord=None,new_id=None):
        
        New_ClusterExperiment=ClusteringExperiment(exp_id=new_id)
        
        for attribute in ["pcs","regressed_pcs","regressor","field_data"]:
            if getattr(self,attribute) is not None:
                combined_attribute=self._combine_attribute(attribute,cluster_array,time_coord)
                setattr(New_ClusterExperiment,attribute,combined_attribute)
                
        return New_ClusterExperiment
    
    def _window_attribute(self,attribute,width,overlap):
        
        data=getattr(self,attribute)
        
        windowed_array=[]
        
        T=data.shape[0]
        window_num=np.floor((T-width)/overlap).astype(int)
        windows=[slice(i*overlap,(i*overlap)+width) for i in range(window_num+1)]

        for window in windows:
            windowed_array.append(data[window])
            
        return windowed_array
    
    #if detailed_name is True, then the window params go into the
    #attribute name, making it easier to do multiple windowing experiments
    #at once.
    def window_data(self,width,overlap,detailed_name=False):
        
        if detailed_name:
            prefix=f"w{width}_o{overlap}_"
        else:
            prefix="windowed_"
        for attribute in ["pcs","regressed_pcs"]:
            if getattr(self,attribute) is not None:
                windowed_attribute=self._window_attribute(attribute,width,overlap)
                setattr(self,prefix+attribute,windowed_attribute)
                self.pc_list.append(prefix+attribute)
                
        for attribute in ["regressor","field_data","regressed_field_data"]:
            if getattr(self,attribute) is not None:
                windowed_attribute=self._window_attribute(attribute,width,overlap)
                setattr(self,prefix+attribute,windowed_attribute)
                
    def randomise_data(self,length,number,detailed_name=False):
        
        if detailed_name:
            prefix=f"r{length}_n{number}_"
        else:
            prefix="randomised_"
        
        for attribute in ["pcs","regressed_pcs"]:
            if getattr(self,attribute) is not None:
                randomised_attribute=self._random_attribute(attribute,length,number)
                setattr(self,prefix+attribute,randomised_attribute)
                self.pc_list.append(prefix+attribute)
                
        for attribute in ["regressor","field_data","regressed_field_data"]:
            if getattr(self,attribute) is not None:
                randomised_attribute=self._random_attribute(attribute,length,number)
                setattr(self,prefix+attribute,randomised_attribute)
    
    def _get_random_samples(self,attribute,length,number):
        
        if self.random_samples is not None:
            return self.random_samples
        
        else:
            data=getattr(self,attribute)
            L=data.shape[0]
            assert length<=L
            
            self.random_samples=[np.random.choice(L,length,replace=False) for n in range(number)]
            return self.random_samples
        
    def _random_attribute(self,attribute,length,number):
        
        samples=self._get_random_samples(attribute,length,number)
        data=getattr(self,attribute)
        
        random_array=[]
        
        for sample in samples:
            random_array.append(data[sample])
            
        return random_array
    
    def _handle_null_partition(self,in_partition,coord,partition):
        #raise(ValueError(f"coord {coord} had no values in the given partition: {partition}"))
        pass #We're trying out just doing nothing.
    
    def _handle_missing_coord(self,data,coord,err,iccat_func):
        if iccat_func is None:
            raise(err)
        else:
            iccat_func(data,"time")
        
    def _partition_attribute(self,attribute,coord, coord_slices,iccat_func=None,min_len=1):
        
        data=getattr(self,attribute)
        try:
            data_coord=data.coord(coord)
            
        except iris.exceptions.CoordinateNotFoundError as e:
            self._handle_missing_coord(data,coord,e,iccat_func)
            data_coord=data.coord(coord)
            
        data_coord_points=data_coord.points
        
        partitioned_array=[]
        
        for partition in coord_slices:
            
            in_partition=np.isin(data_coord_points,partition)
            
            if sum(in_partition)<min_len:
                self._handle_null_partition(in_partition,coord,partition)
            else:
                partitioned_data=data[in_partition]
                partitioned_array.append(partitioned_data)
            
        return partitioned_array

    #similar to window_data but based on a cube coordinate falling within certain slices.
    #iccat func is needed if you want to create coords dynamically as needed.
    def partition_data(self,coord, coord_slices,detailed_name=False,iccat_func=None,min_len=1):
        
        if detailed_name:
            prefix=f"{coord}_sliced_"
        else:
            prefix="sliced_"
            
        for attribute in ["pcs","regressed_pcs"]:
            if getattr(self,attribute) is not None:
                partitioned_attribute=self._partition_attribute(attribute,coord, coord_slices,iccat_func,min_len=min_len)
                setattr(self,prefix+attribute,partitioned_attribute)
                self.pc_list.append(prefix+attribute)
                
        for attribute in ["regressor","field_data","regressed_field_data"]:
            if getattr(self,attribute) is not None:
                partitioned_attribute=self._partition_attribute(attribute,coord, coord_slices,iccat_func,min_len=min_len)
                setattr(self,prefix+attribute,partitioned_attribute)

    def cluster_pcs(self,Ks,pc_list=None):
        
        self.Ks=Ks
        
        if pc_list is None:
            pc_list=self.pc_list
            
        for pcs in pc_list:
            if getattr(self,pcs) is not None:
                clusters=self._cluster_attribute(pcs,Ks)
                self.clusters[pcs]=clusters
            
    def _cluster_attribute(self,pcs,Ks):
        
        data=getattr(self,pcs)
        #data will either be an iris cube or a list of cubes. 
        #We try and iterate and if that fails then its a cube
        
        try:
            clusters=[{K:Kmeans_cluster(cube.data,K) for K in Ks} for cube in data]
        except:
            clusters={K:Kmeans_cluster(data.data,K) for K in Ks}
            
        return clusters
    
    def get_cluster_cubes(self,pc_list=None):
        
        if pc_list is None:
            pc_list=self.pc_list
                    
        for pc in pc_list:
                            
            if getattr(self,pc) is not None:
                cl_data=self.clusters[pc]
                #This isnt a great fix for the introduction of detailed window names, but it works for now
                try:
                    field_name=pc[:-3]+"field_data"
                    try:
                        wf_data=getattr(self,field_name)
                    except:
                        field_name=pc[:-13]+"field_data"
                        wf_data=getattr(self,field_name)
                    
                    ccs=[{K:get_cluster_cube(F,cl[K].states) for K in self.Ks} for F,cl in zip(wf_data,cl_data)]
                
                except:
                
                    #Assume its iterable (i.e. windowed):
                    try:
                        ccs=[{K:get_cluster_cube(F,cl[K].states) for K in self.Ks} for F,cl in zip(self.windowed_field_data,cl_data)]

                        
                    #If not, then it might be full data:
                    except(TypeError,AttributeError):
                        
                        ccs={K:get_cluster_cube(self.field_data,cl_data[K].states) for K in self.Ks}
                    
                
                
                self.cluster_cubes[pc]=ccs


    def _reorder_clusters(self,pc,K,mapping,window=None):
        
        order=np.array([m[1] for m in mapping])
        
        if window is None:
            #Reorder cluster cubes
            self.cluster_cubes[pc][K]=self.cluster_cubes[pc][K][order]
            #Reorder cluster data
            self.clusters[pc][K].reorder(mapping)
        else:
            #Reorder cluster cubes
            self.cluster_cubes[pc][window][K]=self.cluster_cubes[pc][window][K][order]
            #Reorder cluster data
            self.clusters[pc][window][K].reorder(mapping)

    #regrid_ref will use linear interpolation to match reference to target grid. Assumes all target grid cubes have the same lat lon coord, but they really should do
    #or you're doing something more complicated than I can currently think of.
    def correlate_clusters(self,reference_clusters,reorder_clusters=False,pc_list=None,reference_id="None",squeeze_all=False,proc_func=False,regrid_ref=False):
        
        
        #Allows us to collapse all length one coords if needed
        if proc_func is False:
            if squeeze_all:
                proc_func=iris.util.squeeze
            else:
                proc_func=lambda x: x
            
        
        if pc_list is None:
            pc_list=self.pc_list

        correlation_dict={}
        
        for pc in pc_list:
            
            cubes1=self.cluster_cubes[pc]
            
            
            #Non windowed clusters:
            if type(cubes1) is dict:
                
                mean_corr_dict={}
                reg_corr_dict={}
                for K in self.Ks:
                    
                    ref_clusters=reference_clusters[K] 
                    if regrid_ref:
                        ref_clusters=ref_clusters.regrid(cubes1[K],iris.analysis.Linear())
                        
                    (mean_corr,reg_corrs),mapping=correlate_clusters(proc_func(ref_clusters),proc_func(cubes1[K]),and_mapping=True,mean_only=False)
                    
                    mean_corr_dict[K]=mean_corr
                    reg_corr_dict[K]=reg_corrs
                    
                    if reorder_clusters:
                        self._reorder_clusters(pc,K,mapping,window=None)
                        
                correlation_dict[pc]=[mean_corr_dict,reg_corr_dict]
                
            #Windowed clusters
            elif type(cubes1) is list:

                mean_corr_arr=[]
                reg_corr_arr=[]
                
                for w,cc1 in enumerate(cubes1):
                    
                    mean_corr_dict={}
                    reg_corr_dict={}
                    for K in self.Ks:
                        
                        ref_clusters=reference_clusters[K] 
                        if regrid_ref:
                            ref_clusters=ref_clusters.regrid(cubes1[0][K],iris.analysis.Linear())

                        (mean_corr,reg_corrs),mapping=correlate_clusters(proc_func(ref_clusters),proc_func(cc1[K]),and_mapping=True,mean_only=False)

                        mean_corr_dict[K]=mean_corr
                        reg_corr_dict[K]=reg_corrs

                        if reorder_clusters:
                            self._reorder_clusters(pc,K,mapping,window=w)
                            
                    mean_corr_arr.append(mean_corr_dict)
                    reg_corr_arr.append(reg_corr_dict)
                    
                correlation_dict[pc]=[mean_corr_arr,reg_corr_arr]

            else:
                raise(ValueError("cubes should be stored either as a dict or list"))
            
        self.cluster_correlations[reference_id]=correlation_dict
        
    #Just a little syntactic sugar to make data retrieval more convenient
    
    #THIS ONE IS BROKEN
    def get_cl_data(self,pc_data,cl_attr,K,window=None):
        if window is None:
            return getattr(self.clusters[pc_data][K],cl_attr)
        else:
            return getattr(self.clusters[pc_data][window][K],cl_attr)
        
    def get_cc_data(self,pc_data,K,window=None):
        if window is None:
            return self.cluster_cubes[pc_data][K]
        else:
            return self.cluster_cubes[pc_data][window][K]
        
    def get_correlation_data(self,exp_id,pc_data,K,window=None,mean=True):
        if mean:
            mode=0
        else:
            mode=1
        if window is None:
            return self.cluster_correlations[exp_id][pc_data][mode][K]
        else:
            return self.cluster_correlations[exp_id][pc_data][mode][window][K]

    def pickle_experiment(self,filepath):
        with open(filepath, 'wb') as handle: 
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return 1

    def experiment_to_folder(self,filepath,existok=False):
        
        directory=filepath+self.id+"/"
        os.makedirs(directory,existok=existok)
        
        #This could and probably should be expanded
        #to cover all attributes but for now its just a stub
        #as we don't mind pickles
        
        pass
    
#Workflow:
run=False
if run:
    test_pcs1=iris.load_cube("../data/derived_data/primavera_data/pcs/EC-Earth3P-HR_r1i1p2f1_pcs.nc")
    test_js1=iris.load_cube("../data/derived_data/primavera_data/jet_speeds/EC-Earth3P-HR_r1i1p2f1_jet_speed.nc")
    test_z5001=load_pickle("../data/raw_data/primavera/u_full_Z500.pkl")["EC-Earth3P-HR_r1i1p2f1"]
    
    test_pcs2=iris.load_cube("../data/derived_data/primavera_data/pcs/EC-Earth3P-HR_r2i1p2f1_pcs.nc")
    test_js2=iris.load_cube("../data/derived_data/primavera_data/jet_speeds/EC-Earth3P-HR_r2i1p2f1_jet_speed.nc")
    test_z5002=load_pickle("../data/raw_data/primavera/u_full_Z500.pkl")["EC-Earth3P-HR_r2i1p2f1"]
    
    T=30*90
    t=10*90
    
    C=ClusteringExperiment(exp_id="test1",pc_cube=test_pcs1,regressor=test_js1,field_data=test_z5001)
    C.regress_pcs()

    C2=ClusteringExperiment(exp_id="test2",pc_cube=test_pcs2,regressor=test_js2,field_data=test_z5002)
    C2.regress_pcs()

    Ccomb=C.combine_with([C2],time_coord=None,new_id="joined_test")

    Ccomb.window_data(width=T,overlap=t)

    Ccomb.cluster_pcs(Ks=np.arange(2,4))

    Ccomb.get_cluster_cubes()

    Ccomb.correlate_clusters(Ccomb.cluster_cubes["pcs"],reorder_clusters=True,pc_list=["pcs","windowed_pcs"])

    print("done.")