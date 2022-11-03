####################################
# build a graph with VIPERS data
####################################

import spektral
import numpy as np
from spektral.data import Dataset, Graph
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import haversine_distances
import scipy.sparse as sp

class NearestNeigh(Dataset):
    """ 
    This class is just to create the nearest neighbours, to plot them and see the correlations
    """
    def __init__(self, ids, *norms, n_nbrs=5, 
                 filedata = '/content/drive/MyDrive/GraphoVerse/W1_PHOT-SPEC_MATCH_PDR.txt',
                  z_min=0.5, z_max=1.2, **kwargs):
        self.ids = ids
        self.filedata = filedata
        self.n_nbrs = n_nbrs
        self.z_min = z_min
        self.z_max = z_max

        # upload dataset with quality flag
        self.W = np.loadtxt(filedata,usecols=(1,2,4,5,6,7,9,11,13,15,17))
        sel_flg = ((self.W[:,3]>2.)&(self.W[:,3]<10.))|((self.W[:,3]>12.)&(self.W[:,3]<20.))| \
                   ((self.W[:,3]>22.)&(self.W[:,3]<30.))|((self.W[:,3]>212.)&(self.W[:,3]<220.))
        self.W = self.W[sel_flg]   # high quality flag
        sel_z = (self.W[:,2]>z_min) & (self.W[:,2]<z_max)
        self.W = self.W[sel_z]

        #find renormalizations to have [0.,1.] range for color magnitudes
        self.norm5  = norms[0]
        self.norm6  = norms[1]
        self.norm7  = norms[2]
        self.norm8  = norms[3]
        self.norm9  = norms[4]
        self.norm10 = norms[5]
        self.min5   = norms[6]
        self.min6   = norms[7]
        self.min7   = norms[8]
        self.min8   = norms[9]
        self.min9   = norms[10]
        self.min10  = norms[11]

        self.W[:,0] *= np.pi/180.0  # right ascension/longitude
        self.W[:,1] *= np.pi/180.0  # declination/latitute
        features = np.stack( ( self.W[:,1], self.W[:,0] ),axis=-1 ) # order according to haversine metric

        # n_nbrs closest angular neighbours
        self.nbrs = NearestNeighbors(n_neighbors=self.n_nbrs, algorithm='ball_tree', metric='haversine').fit(features)
        _, self.indices = self.nbrs.kneighbors(features)
        super().__init__(**kwargs)

    def read(self):
        def make_graph(id):
            ''' id refers to the id of the galaxy whose z-spec should not be considered '''
            
            assert id < len(self.W)
            
            indices = self.indices[id]   # take all the NearNeigh

            # fill an adjency matrix
            A  = sp.csr_matrix((self.n_nbrs, self.n_nbrs), dtype = np.int8)

            features = np.stack((self.W[indices,2],self.W[indices,4],),axis=-1)       # z-spec, z-photo

            features[0,0] = 0.0       # set z-spec=0 on node 0, missing feature
            label = self.W[id,2]

            return Graph( x=features, a=A, y=label )

        return [make_graph(ids) for ids in self.ids]


class VIPERSOriginal(Dataset):
    """ first implementation, not real angular neighbours """
    def __init__(self, ids, *norms, n_nbrs=31, Dz=0.08,
                 filedata = '/content/drive/MyDrive/GraphoVerse/W1_PHOT-SPEC_MATCH_PDR.txt',
                  z_min=0.0, z_max=6.0, **kwargs):
        self.ids = ids
        self.filedata = filedata
        self.n_nbrs = n_nbrs
        self.Dz = Dz
        self.z_min = z_min
        self.z_max = z_max

        # upload dataset with quality flag
        self.W = np.loadtxt(filedata,usecols=(1,2,4,5,6,7,9,11,13,15,17))
        sel_flg = ((self.W[:,3]>2.)&(self.W[:,3]<10.))|((self.W[:,3]>12.)&(self.W[:,3]<20.))| \
                   ((self.W[:,3]>22.)&(self.W[:,3]<30.))|((self.W[:,3]>212.)&(self.W[:,3]<220.))
        self.W = self.W[sel_flg]   # high quality flag
        sel_z = (self.W[:,2]>z_min) & (self.W[:,2]<z_max)
        self.W = self.W[sel_z]

        #find renormalizations to have [0.,1.] range for color magnitudes
        self.norm5  = norms[0]
        self.norm6  = norms[1]
        self.norm7  = norms[2]
        self.norm8  = norms[3]
        self.norm9  = norms[4]
        self.norm10 = norms[5]
        self.min5   = norms[6]
        self.min6   = norms[7]
        self.min7   = norms[8]
        self.min8   = norms[9]
        self.min9   = norms[10]
        self.min10  = norms[11]

        self.W[:,0] *= np.cos(self.W[:,1]*np.pi/180.0)                                # fix coordinates on celestial sphere
        angle  = np.stack( ( self.W[:,0], self.W[:,1], ),axis=-1 )                    # angular neighbours

        # n_nbrs closest angular neighbours
        self.nbrs = NearestNeighbors(n_neighbors=self.n_nbrs, algorithm='ball_tree').fit(angle)
        _, self.indices = self.nbrs.kneighbors(angle)
        super().__init__(**kwargs)

    def read(self):
        def make_graph(id):
            ''' id refers to the id of the galaxy whose z-spec should not be considered '''
            
            assert id < len(self.W)
            
            indices = self.indices[id]             # take all the NearNeigh
            assert indices[0] == id                # the first neighbour is itself

            # pairwise adjacency matrix, all ones
            row  = np.array([0, 0, 1, 1])
            col  = np.array([0, 1, 0, 1])
            data = np.array([0, 1, 1, 0])
            A = sp.csr_matrix((data, (row, col)), shape=(2, 2))

            # pairwise graphs
            graphs = []
            for n in range(1,self.n_nbrs):
                pid = np.array([indices[0], indices[n]])
                features = np.stack((
                                   self.W[pid,2],
                                   (self.W[pid,5]-self.min5)/ self.norm5,                     # u
                                   (self.W[pid,6]-self.min6)/ self.norm6,                     # g
                                   (self.W[pid,7]-self.min7)/ self.norm7,                     # r
                                   (self.W[pid,8]-self.min8)/ self.norm8,                     # i
                                   (self.W[pid,9]-self.min9)/ self.norm9,                     # z
                                   (self.W[pid,10]-self.min10)/ self.norm10,                  # Ks
                                   abs(self.W[pid,0]-self.W[id,0]),
                                   abs(self.W[pid,1]-self.W[id,1]),)
                                   ,axis=-1)

                #check if it is a real NN
                features[0,0] = 0.0         # set central galaxy to zspec=0.0
                label = np.squeeze([1 if abs(self.W[pid[0],2]-self.W[pid[j],2])<self.Dz*(1+self.W[pid[j],2]) else 0 for j in range(2)])
                labelz = self.W[pid,2]
                labels = np.stack((label,labelz),axis=0)

                g = Graph( x=features, a=A, y=labels )
                graphs.append(g)
            return graphs

        graphs = [make_graph(ids) for ids in self.ids]
        return [gr for gr_sublist in graphs for gr in gr_sublist]


class VIPERSGraphPhoto(Dataset):
    ''' Graph, features do not include angles, only photometry 
    and z spec of neighbour.
    filedata: string with address of VIPERS catalogue
    norms: list of renormalization of colors
    n_nbrs: the angular nearest neighbours are n_nbrs-1
    Dz: redshift separation to define a real neighbour
    ids: id of photometric galaxy in the catalogue
    z_min: remove spectroscopic galaxies with zspec less than z_min
    z_max: remove spectroscopic galaxies with zspec greater than z_max
    '''
    def __init__(self, ids, *norms, n_nbrs=31, Dz=0.08,
                 filedata = '/content/drive/MyDrive/GraphoVerse/W1_PHOT-SPEC_MATCH_PDR.txt',
                  z_min=0.0, z_max=6.0, **kwargs):
        self.ids = ids
        self.filedata = filedata
        self.n_nbrs = n_nbrs
        self.Dz = Dz
        self.z_min = z_min
        self.z_max = z_max

        # upload dataset with quality flag
        self.W = np.loadtxt(filedata,usecols=(1,2,4,5,6,7,9,11,13,15,17))
        sel_flg = ((self.W[:,3]>2.)&(self.W[:,3]<10.))|((self.W[:,3]>12.)&(self.W[:,3]<20.))| \
                   ((self.W[:,3]>22.)&(self.W[:,3]<30.))|((self.W[:,3]>212.)&(self.W[:,3]<220.))
        self.W = self.W[sel_flg]   # high quality flag
        sel_z = (self.W[:,2]>z_min) & (self.W[:,2]<z_max)
        self.W = self.W[sel_z]

        #renormalizations to have [0.,1.] range for color magnitudes
        self.norm5  = norms[0]
        self.norm6  = norms[1]
        self.norm7  = norms[2]
        self.norm8  = norms[3]
        self.norm9  = norms[4]
        self.norm10 = norms[5]
        self.min5   = norms[6]
        self.min6   = norms[7]
        self.min7   = norms[8]
        self.min8   = norms[9]
        self.min9   = norms[10]
        self.min10  = norms[11]

        self.W[:,0] *= np.pi/180.0  # right ascension/longitude in rad
        self.W[:,1] *= np.pi/180.0  # declination/latitute in rad
        angle = np.stack( ( self.W[:,1], self.W[:,0] ),axis=-1 ) # haversine metric order
        # n_nbrs closest angular neighbours
        self.nbrs = NearestNeighbors(n_neighbors=self.n_nbrs, algorithm='ball_tree', metric='haversine').fit(angle)
        _, self.indices = self.nbrs.kneighbors(angle)
        super().__init__(**kwargs)

    def read(self):
        def make_graph(id):
            ''' id refers to the id of the galaxy whose z-spec should not be considered '''
            
            assert id < len(self.W)
            
            indices = self.indices[id]             # take all the NearNeigh
            assert indices[0] == id                # the first neighbour is itself

            # pairwise adjacency matrix
            row  = np.array([0, 0, 1, 1])
            col  = np.array([0, 1, 0, 1])
            data = np.array([0, 1, 1, 0])
            A = sp.csr_matrix((data, (row, col)), shape=(2, 2))

            # pairwise graphs
            graphs = []
            for n in range(1,self.n_nbrs):
                pid = np.array([indices[0], indices[n]])
                features = np.stack((
                                   self.W[pid,2],
                                   (self.W[pid,5]-self.min5)/ self.norm5,                     # u
                                   (self.W[pid,6]-self.min6)/ self.norm6,                     # g
                                   (self.W[pid,7]-self.min7)/ self.norm7,                     # r
                                   (self.W[pid,8]-self.min8)/ self.norm8,                     # i
                                   (self.W[pid,9]-self.min9)/ self.norm9,                     # z
                                   (self.W[pid,10]-self.min10)/ self.norm10,)                 # Ks
                                   ,axis=-1)


                #check if it is a real NN
                features[0,0] = 0.0         # set central galaxy to zspec=0.0
                label = np.squeeze([1 if abs(self.W[pid[0],2]-self.W[pid[j],2])<self.Dz*(1+self.W[pid[j],2]) else 0 for j in range(2)])
                labelz = self.W[pid,2]
                labels = np.stack((label,labelz),axis=0)

                g = Graph( x=features, a=A, y=labels )
                graphs.append(g)
            return graphs

        graphs = [make_graph(ids) for ids in self.ids]
        return [gr for gr_sublist in graphs for gr in gr_sublist]



class VIPERSGraph(Dataset):
    '''
    features include alpha and delta * cos alfa
    filedata: string with address of VIPERS catalogue
    norms: list of renormalization of colors
    n_nbrs: the angular nearest neighbours are n_nbrs-1
    Dz: redshift separation to define a real neighbour
    ids: id of photometric galaxy in the catalogue
    z_min: remove spectroscopic galaxies with zspec less than z_min
    z_max: remove spectroscopic galaxies with zspec greater than z_max
    '''
    def __init__(self, ids, *norms, n_nbrs=31, Dz=0.08,
                 filedata = '/content/drive/MyDrive/GraphoVerse/W1_PHOT-SPEC_MATCH_PDR.txt',
                  z_min=0.0, z_max=6.0, **kwargs):
        self.ids = ids
        self.filedata = filedata
        self.n_nbrs = n_nbrs
        self.Dz = Dz
        self.z_min = z_min
        self.z_max = z_max

        # upload dataset with quality flag
        self.W = np.loadtxt(filedata,usecols=(1,2,4,5,6,7,9,11,13,15,17))
        sel_flg = ((self.W[:,3]>2.)&(self.W[:,3]<10.))|((self.W[:,3]>12.)&(self.W[:,3]<20.))| \
                   ((self.W[:,3]>22.)&(self.W[:,3]<30.))|((self.W[:,3]>212.)&(self.W[:,3]<220.))
        self.W = self.W[sel_flg]   # high quality flag
        sel_z = (self.W[:,2]>z_min) & (self.W[:,2]<z_max)
        self.W = self.W[sel_z]

        #renormalizations to have [0.,1.] range for color magnitudes
        self.norm5  = norms[0]
        self.norm6  = norms[1]
        self.norm7  = norms[2]
        self.norm8  = norms[3]
        self.norm9  = norms[4]
        self.norm10 = norms[5]
        self.min5   = norms[6]
        self.min6   = norms[7]
        self.min7   = norms[8]
        self.min8   = norms[9]
        self.min9   = norms[10]
        self.min10  = norms[11]

        self.W[:,0] *= np.pi/180.0  # right ascension/longitude in rad
        self.W[:,1] *= np.pi/180.0  # declination/latitute in rad
        angle = np.stack( ( self.W[:,1], self.W[:,0] ),axis=-1 ) # haversine metric order
        self.W[:,0] *= np.cos(self.W[:,1])                       # fix coordinates on celestial sphere
        self.nbrs = NearestNeighbors(n_neighbors=self.n_nbrs, algorithm='ball_tree', metric='haversine').fit(angle)
        _, self.indices = self.nbrs.kneighbors(angle)
        super().__init__(**kwargs)

    def read(self):
        def make_graph(id):
            ''' id refers to the id of the galaxy whose z-spec should not be considered '''
            
            assert id < len(self.W)
            
            indices = self.indices[id]             # take all the NearNeigh
            assert indices[0] == id                # the first neighbour is itself

            # pairwise adjacency matrix
            row  = np.array([0, 0, 1, 1])
            col  = np.array([0, 1, 0, 1])
            data = np.array([0, 1, 1, 0])
            A = sp.csr_matrix((data, (row, col)), shape=(2, 2))

            # pairwise graphs
            graphs = []
            for n in range(1,self.n_nbrs):
                pid = np.array([indices[0], indices[n]])
                features = np.stack((
                                   self.W[pid,2],
                                   (self.W[pid,5]-self.min5)/ self.norm5,                     # u
                                   (self.W[pid,6]-self.min6)/ self.norm6,                     # g
                                   (self.W[pid,7]-self.min7)/ self.norm7,                     # r
                                   (self.W[pid,8]-self.min8)/ self.norm8,                     # i
                                   (self.W[pid,9]-self.min9)/ self.norm9,                     # z
                                   (self.W[pid,10]-self.min10)/ self.norm10,                  # Ks
                                   abs(self.W[pid,0]-self.W[id,0]),
                                   abs(self.W[pid,1]-self.W[id,1]),)
                                   ,axis=-1)


                #check if it is a real NN
                features[0,0] = 0.0         # set central galaxy to zspec=0.0
                label = np.squeeze([1 if abs(self.W[pid[0],2]-self.W[pid[j],2])<self.Dz*(1+self.W[pid[j],2]) else 0 for j in range(2)])
                labelz = self.W[pid,2]
                labels = np.stack((label,labelz),axis=0)

                g = Graph( x=features, a=A, y=labels )
                graphs.append(g)
            return graphs

        graphs = [make_graph(ids) for ids in self.ids]
        return [gr for gr_sublist in graphs for gr in gr_sublist]


class VIPERSGraphHav(Dataset):
    '''
    Features include the haverstine distance.
    filedata: string with address of VIPERS catalogue
    norms: list of renormalization of colors
    n_nbrs: the angular nearest neighbours are n_nbrs-1
    Dz: redshift separation to define a real neighbour
    ids: id of photometric galaxy in the catalogue
    z_min: remove spectroscopic galaxies with zspec less than z_min
    z_max: remove spectroscopic galaxies with zspec greater than z_max
    '''
    def __init__(self, ids, *norms, n_nbrs=31, Dz=0.08,
                 filedata = '/content/drive/MyDrive/GraphoVerse/W1_PHOT-SPEC_MATCH_PDR.txt',
                  z_min=0.0, z_max=6.0, **kwargs):
        self.ids = ids
        self.filedata = filedata
        self.n_nbrs = n_nbrs
        self.Dz = Dz
        self.z_min = z_min
        self.z_max = z_max

        # upload dataset with quality flag
        self.W = np.loadtxt(filedata,usecols=(1,2,4,5,6,7,9,11,13,15,17))
        sel_flg = ((self.W[:,3]>2.)&(self.W[:,3]<10.))|((self.W[:,3]>12.)&(self.W[:,3]<20.))| \
                   ((self.W[:,3]>22.)&(self.W[:,3]<30.))|((self.W[:,3]>212.)&(self.W[:,3]<220.))
        self.W = self.W[sel_flg]   # high quality flag
        sel_z = (self.W[:,2]>z_min) & (self.W[:,2]<z_max)
        self.W = self.W[sel_z]

        #renormalizations to have [0.,1.] range for color magnitudes
        self.norm5  = norms[0]
        self.norm6  = norms[1]
        self.norm7  = norms[2]
        self.norm8  = norms[3]
        self.norm9  = norms[4]
        self.norm10 = norms[5]
        self.min5   = norms[6]
        self.min6   = norms[7]
        self.min7   = norms[8]
        self.min8   = norms[9]
        self.min9   = norms[10]
        self.min10  = norms[11]

        self.W[:,0] *= np.pi/180.0  # right ascension/longitude in rad
        self.W[:,1] *= np.pi/180.0  # declination/latitute in rad
        angle = np.stack( ( self.W[:,1], self.W[:,0] ),axis=-1 ) # haversine metric order

        # n_nbrs closest angular neighbours
        self.nbrs = NearestNeighbors(n_neighbors=self.n_nbrs, algorithm='ball_tree', metric='haversine').fit(angle)
        _, self.indices = self.nbrs.kneighbors(angle)
        super().__init__(**kwargs)

    def read(self):
        def make_graph(id):
            ''' id refers to the id of the galaxy whose z-spec should not be considered '''
            
            assert id < len(self.W)
            
            indices = self.indices[id]             # take all the NearNeigh
            assert indices[0] == id                # the first neighbour is itself

            # pairwise adjacency matrix
            row  = np.array([0, 0, 1, 1])
            col  = np.array([0, 1, 0, 1])
            data = np.array([0, 1, 1, 0])
            A = sp.csr_matrix((data, (row, col)), shape=(2, 2))

            # pairwise graphs
            graphs = []
            for n in range(1,self.n_nbrs):
                pid = np.array([indices[0], indices[n]])
                features = np.stack((
                                   self.W[pid,2],
                                   (self.W[pid,5]-self.min5)/ self.norm5,                     # u
                                   (self.W[pid,6]-self.min6)/ self.norm6,                     # g
                                   (self.W[pid,7]-self.min7)/ self.norm7,                     # r
                                   (self.W[pid,8]-self.min8)/ self.norm8,                     # i
                                   (self.W[pid,9]-self.min9)/ self.norm9,                     # z
                                   (self.W[pid,10]-self.min10)/ self.norm10,                  # Ks
                                   haversine_distances( [self.W[pid,1],self.W[pid,0]])[0]/np.pi ,)  # relative ang distance, first is always 0.0
                                   ,axis=-1)

                #check if it is a real NN
                features[0,0] = 0.0         # set central galaxy to zspec=0.0
                label = np.squeeze([1 if abs(self.W[pid[0],2]-self.W[pid[j],2])<self.Dz*(1+self.W[pid[j],2]) else 0 for j in range(2)])
                labelz = self.W[pid,2]
                labels = np.stack((label,labelz),axis=0)

                g = Graph( x=features, a=A, y=labels )
                graphs.append(g)
            return graphs

        graphs = [make_graph(ids) for ids in self.ids]
        return [gr for gr_sublist in graphs for gr in gr_sublist]

