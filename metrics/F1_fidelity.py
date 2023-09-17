# fidelity 
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx
from metrics.load_expl import load_graphs
import networkx as nx



# expl --> only the subgraph


def build_expl(DATASET,MODEL,dataset_fun,framework,expl,verbose=False,lamb=0.001,nomralize=True,MODE="train"):
    
    path = "models/"+DATASET+"_"+MODEL
    print(path)
    dataset = dataset_fun()
    if MODEL == "MinCutPooling":
        gcn = framework(dataset,max_nodes=70,device="cpu")
    else:
        gcn = framework(dataset,device="cpu")
    gcn.load_model(path)
    gcn.evaluate()

             
    graphs = load_graphs(DATASET=DATASET,
                         MODEL=MODEL,
                         EXPL=expl,
                         MODE=MODE,
                         verbose=verbose,
                         lamb=lamb,
                         normalize=nomralize)
    
    if MODE == "train":
        c = 0
        for i in gcn.train_loader.dataset:
            y = i.y.item()
            g = nx.Graph(to_networkx(i,node_attrs=["x"]))
            for n in g.nodes():
                if not graphs[y] == None:
                    if c in graphs[y]:
                        assert len(graphs[y][c].nodes()) == len(g.nodes())
                        assert len(graphs[y][c].edges()) == len(g.edges())
                        graphs[y][c].nodes()[n]["x"] = g.nodes()[n]["x"]
            c = c + 1
    if MODE == "test":
        c = 0
        for i in gcn.test_loader.dataset:
            y = i.y.item()
            g = nx.Graph(to_networkx(i,node_attrs=["x"]))
            for n in g.nodes():
                if not graphs[y] == None:
                    if c in graphs[y]:
                        assert len(graphs[y][c].nodes()) == len(g.nodes())
                        assert len(graphs[y][c].edges()) == len(g.edges())
                        graphs[y][c].nodes()[n]["x"] = g.nodes()[n]["x"]
            c = c + 1
    return gcn,graphs


import torch
def convert_to_torch_graphs(g,g_no_exp,g_exp):
    
    data_g = from_networkx(g)
    data_g.node_imp = None
    data_g.node_imp_norm = None
    
    data_g_exp = from_networkx(g_exp)
    data_g_exp.node_imp = None
    data_g_exp.node_imp_norm = None
    
    data_g_no_exp = from_networkx(g_no_exp)
    data_g_no_exp.node_imp = None
    data_g_no_exp.node_imp_norm = None
    
    return data_g,data_g_no_exp,data_g_exp

def evaluate_expls(gcn,data_g,data_g_no_exp,data_g_exp,y=1,color=False):
    
    if not color:

        fg = torch.exp(gcn.model(data_g.x.double(),data_g.edge_index,torch.zeros(data_g.x.shape[0],dtype=torch.int64))[0][y])
        if  data_g_no_exp.x == None:
            fg_no_e = 0.5
        else:
            fg_no_e = torch.exp(gcn.model(data_g_no_exp.x.double(),data_g_no_exp.edge_index,torch.zeros(data_g_no_exp.x.shape[0],dtype=torch.int64))[0][y])
                
        if  data_g_exp.x == None:
            fg_e = 0.5
        else:
            fg_e = torch.exp(gcn.model(data_g_exp.x.double(),data_g_exp.edge_index,torch.zeros(data_g_exp.x.shape[0],dtype=torch.int64))[0][y])
    else:

        fg = torch.exp(gcn.model(data_g.x,data_g.edge_index,torch.zeros(data_g.x.shape[0],dtype=torch.int64))[0][y])
        if  data_g_no_exp.x == None:
            fg_no_e = 0.5
        else:
            fg_no_e = torch.exp(gcn.model(data_g_no_exp.x,data_g_no_exp.edge_index,torch.zeros(data_g_no_exp.x.shape[0],dtype=torch.int64))[0][y])
                
        if  data_g_exp.x == None:
            fg_e = 0.5
        else:
            fg_e = torch.exp(gcn.model(data_g_exp.x,data_g_exp.edge_index,torch.zeros(data_g_exp.x.shape[0],dtype=torch.int64))[0][y])
    
    return fg,fg_no_e,fg_e

def get_threshold(g):
    node_att = nx.get_node_attributes(g,"node_imp_norm")
    thresholds = []
    for i in sorted(list(node_att.values())):
        thresholds.append(float("%.2f" % i))

    thresholds = np.unique(thresholds)
    
    return list(thresholds)

# returns v t.c v >= threshold
def get_hard_mask(g_in,threshold):
    node_att = nx.get_node_attributes(g_in,"node_imp_norm")
    
    node_g = []
    node_exp = []
    
    for k,v in node_att.items():
        if v >= threshold:
            node_exp.append(k)
        else:
            node_g.append(k)
    g = nx.subgraph(g_in,node_g)
    g_exp = nx.subgraph(g_in,node_exp)
    
    return g,g_exp


def fidelity_sufficiency(fg,fg_e):
    return (fg - fg_e).item()

def fidelity_comprehensiveness(fg,fg_no_e):
    return (fg - fg_no_e).item()
import numpy as np


def compute_fidelity(gcn,graphs,y=1,color=False):
    res_suf = []
    res_comp = []
    
    cc = 0
    for gid,g in graphs[y].items():
        thresholds = get_threshold(g)
        suf = []
        comp = []
        for threshold in thresholds:
            threshold = threshold
            g_no_exp,g_exp = get_hard_mask(g,threshold)
            data_g,data_g_no_exp,data_g_exp = convert_to_torch_graphs(g,g_no_exp,g_exp)
            fg, fg_no_e, fg_e = evaluate_expls(gcn,data_g,data_g_no_exp,data_g_exp,y,color)
            
            suf.append(fidelity_sufficiency(fg,fg_e))
            comp.append(fidelity_comprehensiveness(fg,fg_no_e))
        
        res_suf.append(np.mean(suf))
        res_comp.append(np.mean(comp))
        cc = cc +1
        if cc % 200 == 0:
            print(cc)
        #if cc == 20:
        #    break    


    return np.mean(res_suf),np.mean(res_comp)
