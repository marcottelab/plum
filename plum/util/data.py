import os
import copy
import dendropy
import numpy as np
import pandas as pd
import plum.util.ctree
import plum.models.ErrorModels
import plum.models.MarkovModels
from collections import defaultdict
from plum.models.modelABC import Model


class plumData(object):
    '''
    Base class for all optimization methods. Performs data read-in and collation of the two
    model components - the Markov Model and the Error Model - which are supplied by the user
    taking starting values along with them.
    '''
    
    def __init__(self,markov_model,error_model,tree,data,as_sorted=False):
        '''
        - `markov_model`: An instance of plum.models.MarkovModels. Parameters used as starting values in optimization
        - `error_model`: An instance of plum.models.ErrorModels. Parameters used as starting values in optimization
        - `tree`: A rooted newick tree with labeled internal nodes and root. Either a dendropy.Tree object or a string path to a newick file.
        - `data`: A datafile in tidy csv format with columns: ID1, ID2, species, data, state
        '''
        
        
        ## Should add checking that data and tree taxa match
        
        ## Add threshold for amount of acceptable missing data per pair? ##
        
        ## List of hardcoded crap:
        ##      - tree reading
        ##      - data reading
        

        assert isinstance(markov_model,Model), "`markov_model` must be instance deriving from modelABC.Model: {}".format(markov_model)
        assert isinstance(error_model,Model), "`error_model` must be instance deriving from modelABC.Model"
        assert np.isclose( sum(markov_model.stationaryFrequencies[1]), 1. ),"Stationary frequencies from Markov model don't sum to 1"
        # Should check that all nodes of tree are labeled

        self._error_model = copy.deepcopy(error_model)
        self._em_param_names = self._error_model.freeParams # list
        self._markov_model = copy.deepcopy(markov_model)
        self._mm_param_names = self._markov_model.freeParams # list
        self._equil_freqs = dict(list(zip(markov_model.stationaryFrequencies[0],markov_model.stationaryFrequencies[1])))
        self._states = self._error_model.states # asserted above that err and markov states are the same

        self._free_params = copy.deepcopy(self._error_model.freeParams + self._markov_model.freeParams) # list
        self._param_boundD = copy.deepcopy(self._error_model.paramBounds)
        self._param_boundD.update(copy.deepcopy(self._markov_model.paramBounds))
        self._param_boundVec = tuple([self._param_boundD[p] for p in self._free_params])
        
        # properties to be reset during optimization
        ## first setting of _paramD current uses error_model._params, which is a bit weird because
        ## this will include dependent parameters as well as free. This is overridden when update is
        ## called and _paramD gets re-written with only the free parameters. Works, but a bit inelegant.
        self._paramD = self._error_model._params.copy() # main holder of all parameters - to be updated during optim
        self._paramD.update(self._markov_model._params)
        self._param_vec = [self._paramD[p] for p in self._free_params] # serve as starting params, will be updated during optim.
        
        self._start_params = dict(list(zip(self._free_params,self._param_vec))) # will stay the same
        self._all_params = None

        if type(tree) is str: # tree should be a path to a file if it's a string
            assert os.path.isfile(tree), "Can't find treefile {}".format(tree)
            #self.tree = dendropy.Tree.get_from_path(tree,'newick',preserve_underscores=True,suppress_internal_node_taxa=False,rooting='default-rooted')            
            dtree = dendropy.Tree.get_from_path(tree,'newick',preserve_underscores=True,suppress_internal_node_taxa=False,rooting='default-rooted')
            self.tree = plum.util.ctree.Tree(dtree)
        else:
            assert type(tree) is dendropy.datamodel.treemodel.Tree, "`tree` must be either a dendropy.Tree object or a path to a newick treefile"
            #self.tree = tree
            self.tree = plum.util.ctree.Tree(tree)
        
        if type(data) is str: # should be a path
            assert os.path.isfile(data), "Can't find datafile {}".format(data)
            
            if as_sorted == False:
                print("Loading dataframe")
                self._dataframe = pd.read_csv(data)
                self._features = self._dataframe.columns[3:-1]
                if len(self._features) == 1:
                    self._features = self._features[0]
                    self._is_multivariate = False
                else:
                    self._is_multivariate = True
                self._load_data()
            else:
                self._dataframe = None
                self._load_data(as_sorted=True,infile=data)
        else: # Must be dataframe
            assert type(data) is pd.core.frame.DataFrame, "`data` must be either a pandas DataFrame object, a path to a csv file, or None"
            self._dataframe = data
            self._features = self._dataframe.columns[3:-1]
            if len(self._features) == 1:
                self._features = self._features[0]
                self._is_multivariate = False
            else:
                self._is_multivariate = True
            self._load_data()
                
        assert self.is_multivariate == self._error_model.is_multivariate, \
        "Dimension mismatch: data.is_multivariate is {}, error_model.is_multivariate is {}".format(self.is_multivariate,self._error_model.is_multivariate)
                
        
    @property
    def is_multivariate(self):
        '''Whether the dataset is multivariate, i.e. considers more than one feature'''
        return self._is_multivariate
    
    @property
    def startingParams(self):
        '''The starting parameters before optimization'''
        return self._start_params
    
    @property
    def freeParamDict(self):
        '''The current free parameters of the model. Dictionary mapping
        strings (param names) to float values'''
        return self._paramD
        
    @property
    def params(self):
        ## all_params should be an attribute that is updated and returned here
        all_params = copy.deepcopy(self._error_model.params)
        all_params.update(self._markov_model.params)
        return all_params
    
    @property
    def markovModelParams(self):
        '''Current Markov model parameters'''
        return {p:self._paramD[p] for p in self._mm_param_names}
        
    @property
    def errorModelParams(self):
        '''Current Error model parameters'''
        return {p:self._paramD[p] for p in self._em_param_names}
        
    @property
    def freeParams(self):
        '''Get or set the parameters that will be optimized.
            - A list of strings'''
        return self._free_params
        
    @property
    def features(self):
        '''The feature type in the supplied data file. Extracted
        from the 4th column of the data'''
        return list(self._features)

    def _load_data(self, as_sorted=False, infile=None):
        '''Load in data file.''' 
        if as_sorted == False:
            print("Parsing unsorted dataframe")
            grouped = self._dataframe.groupby(["ID1","ID2"])
            
            group_parser = lambda group: ( dict(list(zip(group.species,group[self._features].values))), dict(list(zip(group.species,group.state))) )
            parsed = grouped.apply(group_parser)
            cols = np.asarray(list(parsed.values)) 
            
            self._featureDs = cols[:,0] # dictionaries mapping taxon names to feature values
            self._knownLabelDs = cols[:,1] # dictionaries mapping taxon names to known labels (states)
            self._pairs = np.asarray(parsed.index) # tuples of ids
            
            assert len(self._featureDs) == len(self._knownLabelDs), "Error parsing datafile: data and known labels don't match"
            assert len(self._featureDs) > 0, "Error parsing datafile: no data were read in"
        else:
            print("Parsing sorted data")
            self._featureDs, self._knownLabelDs, self._pairs = [],[],[]
            with open(infile) as f:
                header = f.readline().strip().split(",")
                assert len(header) > 4, "Infile should have at least 5 columns: ID1, ID2, species, feature1,...featureN, state"
                self._features = header[3:-1]
                if len(self._features) == 1:
                    self._features = self._features[0]
                    self._is_multivariate = False
                else:
                    self._is_multivariate = True
                pair = None
                dataD = {}
                knownD = {}
                for line in f:
                    line = line.strip().split(",")
                    line_pair = tuple(sorted(line[:2]))
                    species = line[2]
                    known_state = line[-1]
                    if line_pair != pair:
                        if pair == None:
                            pair = line_pair

                            try:
                                if self._is_multivariate:
                                    dataD[species] = list(map(float,line[3:-1]))  # index on species
                                else:
                                    dataD[species] = float(line[3])
                                if known_state == '':
                                    knownD[species] = np.nan
                                else:
                                    knownD[species] = float(known_state) # index on species
                            except ValueError as e:
                                raise Exception("Improper type in feature or state column: {}".format(e))
                        else:
                            assert pair not in self._pairs, "{} found twice, data is not sorted!".format(pair)
                            self._featureDs.append(dataD)
                            self._knownLabelDs.append(knownD)
                            self._pairs.append(pair)
                            dataD = {}
                            knownD = {}
                            pair = line_pair
                            
                            assert species not in dataD and species not in knownD, "{} {} found twice".format(pair, species)
                            try:
                                if self._is_multivariate:
                                    dataD[species] = list(map(float,line[3:-1]))  # index on species
                                else:
                                    dataD[species] = float(line[3])
                                if known_state == '':
                                    knownD[species] = np.nan
                                else:
                                    knownD[species] = float(known_state) # index on species
                            except ValueError as e:
                                raise Exception("Improper type in feature or state column: {}".format(e))
                    else:
                        
                        assert species not in dataD and species not in knownD, "{} {} found twice".format(pair, species)
                        try:
                            if self._is_multivariate:
                                dataD[species] = list(map(float,line[3:-1]))  # index on species
                            else:
                                dataD[species] = float(line[3])
                            if known_state == '':
                                knownD[species] = np.nan
                            else:
                                knownD[species] = float(known_state) # index on species
                        except ValueError as e:
                            raise Exception("Improper type in feature or state column: {}".format(e))
                            
                # Load final pair
                self._featureDs.append(dataD)
                self._knownLabelDs.append(knownD)
                self._pairs.append(pair)
                
            self._featureDs = np.array(self._featureDs)
            self._knownLabelDs = np.array(self._knownLabelDs)
        print("Finished loading data")
        #if self.is_multivariate == False:
        del self._dataframe
          
          
def parse_param_file(infile):
    '''Parse a parameter file. Returns a dictionary of dictionaries holding the parsed fields
    
    An example parameter file
    -------------------------------------------
    
    # Error Model
    Name: TwoStateNormalPositiveMixture
    Params: mean0=0 sd0=1 lambda1_1=.5 mean1_1=0 sd1_1=1 mean1_2=.5 sd1_2=1
    Bounds: mean0=(-.5,.5) sd0=(0.001,2) lambda1_1=(0,1) mean1_1=(0,1) sd1_1=(0.001,2) mean1_2=(0,1) sd1_2=(0.001,2)
    
    # Markov Model
    Name: TwoState
    Params: alpha=1 beta=3
    Bounds: alpha=(0.0001,10) beta=(0.0001,10)
    
    -------------------------------------------
    
    "Bounds" is an optional fields used to constrain the search space.
    "Name" and "Params" are non-optional
    Must delineate model types with "#" headers
    '''
    
    models = sorted(["error model", "markov model"])
    fields = ["name","params","bounds"]
    required_fields = sorted(["name","params"])
    found_models = []
    found_fields = []
    func = lambda x: [x[0],eval(x[1])]
    parse_func = lambda x: dict([func(i.strip().split("=")) for i in x.split(";")])
    model_specs = defaultdict(dict)
    with open(infile) as f:
        is_first = True
        for line in f:  
            if line.strip() == '': 
                continue
            elif line.startswith("//"): # comment lines
                continue
            elif line.startswith("#"): # start of new model
                if is_first:
                    is_first = False
                else:
                    diff = set(required_fields) - set(found_fields)
                    assert len(diff) == 0, "Didn't find all fields:{} {}".format(model_type + ", ", ",".join(diff))
                    found_fields = []
                line = line.lstrip("#")
                model_type = line.strip().lower()
                assert model_type in models, "{}".format(model_type)
                found_models.append(model_type)
            else: # is model field line
                line = line.strip().split(":")
                field = line[0].strip().lower()
                assert field in fields,"{}".format(field)
                if field == "name":
                    parsed = line[1].strip().split(".")[-1]
                else:
                    string = line[1]
                    parsed = parse_func(string)
                found_fields.append(field)
                model_specs[model_type][field] = parsed
        diff = set(required_fields) - set(found_fields)
        assert len(diff) == 0, "Didn't find all fields:{} {}".format(model_type + ", ", ",".join(diff))
        assert sorted(found_models) == models, "Didn't find all models: {}".format(",".join( set(models) - set(found_models) ))
    return model_specs
    
    
def get_models_from_file(infile):
    '''Parse a parameter file and return ErrorModel, MarkovModel'''
    parsed = parse_param_file(infile)
    
    # Create Error Model
    try:
        em = getattr(plum.models.ErrorModels,parsed["error model"]["name"])(**parsed["error model"]["params"])
    except AttributeError:
        raise Exception("{} not a valid error model name".format(parsed["error model"]["name"]))
    except TypeError:
        raise Exception("{} invalid keyword arguments".format(parsed["error model"]["params"]))
    
    if "bounds" in parsed["error model"]:
        em.paramBounds = parsed["error model"]["bounds"]
    
    # Create Markov Model
    try:
        mm = getattr(plum.models.MarkovModels,parsed["markov model"]["name"])(**parsed["markov model"]["params"])
    except AttributeError:
        raise Exception("{} not a valid error model name".format(parsed["markov model"]["name"]))
    except TypeError:
        raise Exception("{} not valid keyword arguments".format(parsed["markov model"]["params"]))

    if "bounds" in parsed["markov model"]:
        mm.paramBounds = parsed["markov model"]["bounds"]
        
    return em, mm

def _bound_string(bounds):
    bound_string = []
    for i,j in bounds.items():
        jnew = []
        for x in j:
            if x == np.inf:
                jnew.append("np.inf")
            elif x == -np.inf:
                jnew.append("-np.inf")
            else:
                jnew.append(str(x))
        bound_string.append("=".join([i,str(tuple(jnew)).replace("'","")]))
    return bound_string
    
def write_parameter_file(outfile,error_model,markov_model):
    '''Write a parameter file specifying the supplied error and 
    Markov models
    
     - `outfile`: <str> filename
     - `error_model: plum.models.ErrorModels object
     - `markov_model: plum.models.MarkovModels object'''
    assert isinstance(markov_model,plum.models.modelABC.Model), "`markov_model` must be instance deriving from plum.models.MarkovModels: {}".format(markov_model)
    assert isinstance(error_model,plum.models.modelABC.Model), "`error_model` must be instance deriving from plum.models.ErrorModels: {}".format(error_model)
    with open(outfile, 'w') as out:
        out.write("# Error Model\n")
        out.write("Name: {}\n".format(str(error_model).split(".")[-1]))
        if error_model.is_multivariate:
            out.write( "Params: {}\n".format( ';'.join( ["=".join([i,str(j.tolist()).replace(" ","")]) for i,j in error_model.freeParamDict.items()] ) ) )
        else:
            out.write( "Params: {}\n".format( ';'.join( ["=".join([i,str(j)]) for i,j in error_model.freeParamDict.items()] ) ) )
        if error_model.paramBounds != error_model._outer_bounds:
            bound_string = _bound_string(error_model.paramBounds)
            out.write("Bounds: {}\n\n".format( ';'.join(bound_string) ) )
        else:
            out.write("\n")
            
        out.write("# Markov Model\n")
        out.write("Name: {}\n".format(str(markov_model).split(".")[-1]))
        out.write( "Params: {}\n".format( ';'.join( ["=".join([i,str(j)]) for i,j in markov_model.freeParamDict.items()] ) ) )
        if markov_model.paramBounds != markov_model._outer_bounds:
            bound_string = _bound_string(markov_model.paramBounds)
            out.write("Bounds: {}\n".format( ';'.join(bound_string) ) )
