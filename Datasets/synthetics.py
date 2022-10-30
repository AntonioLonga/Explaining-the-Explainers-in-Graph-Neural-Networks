from typing import Optional, Callable

import torch
from networkx.generators import random_graphs, lattice, small, classic
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import barabasi_albert_graph
import networkx as nx
import pickle as pkl
import random
import numpy as np
import torch_geometric.transforms as T
from networkx.algorithms.operators.binary import compose, union
from torch_geometric.utils import from_networkx
from Datasets.utils_infection import CreateInfection
from sklearn.model_selection import train_test_split


class BA_houses_color(InMemoryDataset):
    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/BA-houses_color.pkl','rb') as fin:
            (adjs, feas, labels,_) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index
            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            data = Data(x=torch.tensor(fea,dtype=torch.float), edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)



class ER_nb_stars(InMemoryDataset):
    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/ER-nb_stars.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            data = Data(x=torch.tensor(fea), edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)


class ER_nb_stars2(InMemoryDataset):
    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/ER-nb_stars2.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            data = Data(x=torch.tensor(fea), edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)





class BA_2grid_house(InMemoryDataset):

    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/BA-2grid-house.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            
            
            data = Data(x=torch.tensor(fea), edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)


class BA_2grid(InMemoryDataset):
    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/BA-2grid.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            data = Data(x=torch.tensor(fea), edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)



class BA_2motfs(InMemoryDataset):

    def __init__(self,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/BA-2motif.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            
            
            data = Data(x=torch.tensor(fea), edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)





class BA_multipleShapes2(InMemoryDataset):

    def __init__(self,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/BA-multipleShapes2.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        num_nodes = len(adjs[0][0])
        self.num_nodes = num_nodes
        data_list = []
        for i in range(len(adjs)):
            adj = adjs[i]

            if labels[i]  == 0.0:
                label = 0
            else:
                label = 1
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)
            data = from_networkx(nx.from_numpy_matrix(adj)) #to make edge_index undirected
            data = Data(x=torch.tensor(feas[i]), edge_index=data.edge_index, y=label,expl_mask=expl_mask)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)
        
        
class Infection(InMemoryDataset):
    def __init__(self, num_nodes=1000,
                 transform: Optional[Callable] = None):
        super().__init__('.', transform)

        # Create an Infection graph with num_nodes nodes
        seed = 42
        data = CreateInfection(sample_count=1)
        data.run(num_nodes=num_nodes, seed=seed)
        
        # Extract the info that we use (we create only a training set,
        #   while the test set is empty)
        data = data.train_dataset[0][0]
        
        # Generate the Data object
        data = Data(edge_index=data.edge_index,
                    # num_nodes=data.num_nodes,
                    x=data.x,
                    y=data.y)
        
        # Collapse the labels from 0-5 to 1-3: new classes 0-(1-2)-3
        # First 1 and 2 go to new 1
        data.y[(data.y==1) | (data.y==2)] = 1
        # Now all those with label >1 are 3, 4, 5
        data.y[data.y>1] = 2
        
        # Define the train and test mask
        train_idx, test_idx = train_test_split(np.arange(num_nodes), train_size=0.8, stratify=data.y, random_state=10)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.test_mask[test_idx] = True
        
        # Define the expl mask
        data.expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.expl_mask[torch.arange(int(num_nodes / 2), num_nodes, 5)] = True

        # Run collate to get data and slices
        self.data, self.slices = self.collate([data])