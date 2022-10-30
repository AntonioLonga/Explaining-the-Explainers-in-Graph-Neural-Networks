import random
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx


class CreateInfection(object):
    NUM_GRAPHS = 1
    TEST_RATIO = 0
    
    def __init__(self, sample_count):
        self.sample_count = sample_count
    
    
    def run(self, num_nodes=1000, seed=42): 
        self.train_dataset = []
        self.test_dataset = []
        for experiment_i in range(self.sample_count):
            dataset = [self.create_dataset(num_nodes) for i in range(self.NUM_GRAPHS)]
            split_point = int(len(dataset) * self.TEST_RATIO)
            self.test_dataset.append(dataset[:split_point])
            self.train_dataset.append(dataset[split_point:])
            
   
    def create_dataset(self, num_nodes=1000, seed=42): 
        max_dist = 3  # anything larger than max_dist has a far away label
 
        random_state = np.random.RandomState(42)
        g = nx.erdos_renyi_graph(num_nodes, 0.004, directed=True, seed=random_state)
        N = len(g.nodes())
        
        random.seed(seed)
        infected_nodes = random.sample(g.nodes(), 50)
        g.add_node('X')  # dummy node for easier computation, will be removed in the end
        for u in infected_nodes:
            g.add_edge('X', u)
        shortest_path_length = nx.single_source_shortest_path_length(g, 'X')
        labels = []
        features = np.zeros((N, 2))
        for i in range(N):
            if i == 'X':
                continue
            length = shortest_path_length.get(i, 100) - 1  # 100 is inf distance
            labels.append(min(max_dist + 1, length))
            col = 0 if i in infected_nodes else 1
            features[i, col] = 1
            if 0 < length <= max_dist:
                path_iterator = iter(nx.all_shortest_paths(g, 'X', i))
                unique_shortest_path = next(path_iterator)
                if next(path_iterator, 0) != 0:
                    continue
                unique_shortest_path.pop(0)  # pop 'X' node
                if len(unique_shortest_path) == 0:
                    continue

        g.remove_node('X')
        data = from_networkx(g)
        data.x = torch.tensor(features, dtype=torch.float)
        data.y = torch.tensor(labels)
        return data


            