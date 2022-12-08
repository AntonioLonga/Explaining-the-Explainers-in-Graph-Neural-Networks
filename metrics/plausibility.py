from sklearn.metrics import roc_auc_score
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism


# grid
def get_plausibility(graphs,GT_len=9):
    res = []
    for gid,g in graphs[1].items():
        nb_node_GT = GT_len
        node_attributes = list(nx.get_node_attributes(g,"node_imp_norm").values())

        node_attributes

        GT = list(np.zeros(len(node_attributes[nb_node_GT:])))
        GT.extend(list(np.ones(len(node_attributes[:nb_node_GT]))))
        node_attributes = np.array(node_attributes)
        GT = np.array(GT,dtype=int)


        r = roc_auc_score(GT, node_attributes)
        res.append(r)
    return np.mean(res)



# grid house 
from sklearn.metrics import roc_auc_score
def my_house():
    g = nx.Graph()
    g.add_edge(0,1)
    g.add_edge(1,2)
    g.add_edge(2,3)
    g.add_edge(0,3)
    g.add_edge(3,4)
    g.add_edge(2,4)
    return(g)

def my_grid():
    g = nx.Graph()
    g.add_edge(0,1)
    g.add_edge(1,2)
    g.add_edge(0,3)
    g.add_edge(1,4)
    g.add_edge(2,5)
    g.add_edge(3,4)
    g.add_edge(4,5)
    g.add_edge(3,6)
    g.add_edge(4,7)
    g.add_edge(5,8)
    g.add_edge(6,7)
    g.add_edge(7,8)
    
    return g
def split_class0(graphs):
    HOUSE = my_house()
    GRID = my_grid()
    len_house = 5
    len_grid = 9
    split_house_grid = {"house":[],"grid":[]}

    for gid,g in graphs.items():
        flag = False
        sub_g = nx.subgraph(g,list(g.nodes())[-len_grid:])
        if nx.is_isomorphic(sub_g,GRID):
            split_house_grid["grid"].append(g)

        sub_g = nx.subgraph(g,list(g.nodes())[-len_house:])
        if  nx.is_isomorphic(sub_g,HOUSE):
            split_house_grid["house"].append(g)
            
    return split_house_grid

def get_plausibility_ba_grid_house(graphs,GT_len = 9):
    res = []
    for g in graphs:

        nb_node_GT = GT_len
        node_attributes = list(nx.get_node_attributes(g,"node_imp_norm").values())
        
        GT = list(np.zeros(len(node_attributes[nb_node_GT:])))
        GT.extend(list(np.ones(len(node_attributes[:nb_node_GT]))))
        node_attributes = np.array(node_attributes)
        GT = np.array(GT,dtype=int)


        r = roc_auc_score(GT, node_attributes)
        res.append(r)
    return np.mean(res)



# nb stars2

def get_GT(graph,k=1):
    deg = dict(nx.degree(graph))
    deg = dict(sorted(deg.items(), key=lambda item: item[1],reverse=True))

    start_GT = np.min(list(deg.keys())[:k])
    if 0 == start_GT:
        tmp = list(deg.keys())[:k+1]
        tmp.remove(0)
        start_GT = np.min(tmp)
    return start_GT

def get_plausibility_nb_stars(graphs,k):
    res = []
    c = 0
    for g in graphs:
        start_GT = get_GT(g,k)
        node_attributes = list(nx.get_node_attributes(g,"node_imp_norm").values())

        GT = list(np.zeros(len(node_attributes[start_GT:])))
        GT.extend(list(np.ones(len(node_attributes[:start_GT]))))
        node_attributes = np.array(node_attributes)
        GT = np.array(GT,dtype=int)

        c = c+1
        r = roc_auc_score(GT, node_attributes)
        
        res.append(r)
    return np.mean(res)



# BA houses color

def get_plausibility_houses_color(graphs,y):
    res = []
    for _,g in graphs[y].items():

        node_att = list(dict(nx.get_node_attributes(g,"node_imp_norm")).values())
        gt = list(dict(nx.get_node_attributes(g,"GT")).values())
        
        r = roc_auc_score(gt, node_att)
        res.append(r)
    return np.mean(res)



import pickle as pkl
import torch
from sklearn.model_selection import train_test_split

def load_gt_BA_houses_color(graphs,train):

    with open('Datasets/BA-houses_color.pkl','rb') as fin:
        (_, _, y,GT) = pkl.load(fin)
    GT = np.array(GT)
    idx = torch.arange(len(GT))
    train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=y,random_state=10)
    if train:
        idx = train_idx.numpy()
    else:
        idx = test_idx.numpy()
        
    GT = GT[idx]
    if not graphs[0] == None:
        for k,g in graphs[0].items():
            for node in g.nodes():
                if node in GT[k][0]:
                    g.nodes()[node]["GT"]=1
                else:
                    g.nodes()[node]["GT"]=0
    
    if not graphs[1] == None:
        for k,g in graphs[1].items():
            for node in g.nodes():
                if node in GT[k][0]:
                    g.nodes()[node]["GT"]=1
                else:
                    g.nodes()[node]["GT"]=0
                
    return graphs





##
# Node classification
##

def get_shapes_plausibility(graphs, num_layers):
    class_rocs = []
    for c in range(len(graphs)):
        if c == 0 or graphs[c] is None: #class w/o motifs
            roc = float("nan")    
        else:
            roc = gt_roc_ba_shapes(graphs[c], num_layers)
        class_rocs.append(roc)
        
    return class_rocs

def gt_roc_ba_shapes(graphs, gnn_num_layers):
    res = []
    for n_id , g in graphs.items():
        node_attributes_all = np.array(list(nx.get_node_attributes(g,"node_imp_norm").values()))
        g = nx.ego_graph(g, n=n_id, radius=gnn_num_layers)        
        
        GM = isomorphism.GraphMatcher(g, my_house())
        match = list(GM.subgraph_isomorphisms_iter())
        
        # nodes of the motif
        match = match[0] #just take the first possible match
        nodes_match = list(match.keys())
        trues = [1] * len(nodes_match)
        preds = node_attributes_all[nodes_match].tolist()
        
        # nodes outside the motif
        trues.extend([0]*(len(g.nodes()) - len(nodes_match)))
        preds.extend(node_attributes_all[[i for i in g.nodes() if i not in nodes_match]])

        if len(np.unique(trues)) == 1:
            trues.append(0)
            preds.append(0)
        
        r = roc_auc_score(trues, preds)
        res.append(r)
    return np.mean(res)