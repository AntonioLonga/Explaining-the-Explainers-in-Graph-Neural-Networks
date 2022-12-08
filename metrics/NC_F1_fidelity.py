##
# Node classification
##

import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx
from metrics.load_expl import nc_load_graphs
import networkx as nx
import random
import numpy as np
import torch



def build_expl(DATASET, MODEL, dataset_fun, framework, expl,verbose=False,lamb=0.001,nomralize=True):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    path = "models/"+DATASET+"_"+MODEL    
    dataset = dataset_fun()
    gcn = framework(dataset,device="cpu")
    gcn.load_model(path)
    #gcn.evaluate()
             
    graphs = nc_load_graphs(DATASET=DATASET,
                         MODEL=MODEL,
                         EXPL=expl,
                         MODE="train",
                         verbose=verbose,
                         lamb=lamb,
                         normalize=nomralize)    
    
    for c in range(len(graphs)):
        if graphs[c] is None: continue
        for i , g in graphs[c].items():
            for n in g.nodes():
                g.nodes()[n]["x"] = [1.] *10
    return gcn , graphs


def convert_to_torch_graphs(g,g_no_exp,g_exp):    
#     data_g = from_networkx(g)
#     data_g.node_imp = None
#     data_g.node_imp_norm = None
    
    data_g_exp = from_networkx(g_exp)
    data_g_exp.node_imp = None
    data_g_exp.node_imp_norm = None
    
    data_g_no_exp = from_networkx(g_no_exp)
    data_g_no_exp.node_imp = None
    data_g_no_exp.node_imp_norm = None 
    return None , data_g_no_exp , data_g_exp

def evaluate_expls(gcn, data_g, data_g_no_exp, data_g_exp, node_idx, y, ori_g):        
    fg = torch.exp(gcn.model.forward_single(ori_g.x, ori_g.edge_index, node_idx)[y])
    
    if  data_g_no_exp.x == None:
        fg_no_e = 0.5
    else:
        fg_no_e = torch.exp(gcn.model.forward_single(data_g_no_exp.x,
                                                      data_g_no_exp.edge_index,
                                                      node_idx)[y])
        
    if  data_g_exp.x == None:
        fg_e = 0.5
    else:
        fg_e = torch.exp(gcn.model.forward_single(data_g_exp.x,
                                                   data_g_exp.edge_index,
                                                   node_idx)[y])
    return fg , fg_no_e , fg_e



def get_threshold(g):
    node_att = nx.get_node_attributes(g,"node_imp_norm")
    thresholds = []
    for i in sorted(list(node_att.values())):
        thresholds.append(float("%.2f" % i))
    thresholds = np.unique(thresholds)    
    return list(thresholds)


# keep features only of nodes with importance >= threshold
def get_hard_mask(g_in, threshold):    
    node_att = nx.get_node_attributes(g_in, "node_imp_norm")
    
    node_g = []
    node_exp = []    
    for k,v in node_att.items():
        if v >= threshold:
            node_exp.append(k)
        else:
            node_g.append(k)
    #g = nx.subgraph(g_in,node_g)
    #g_exp = nx.subgraph(g_in,node_exp)   
    
    g = g_in.copy()
    g_exp = g_in.copy()
    for n in g.nodes():
        if n not in node_g:
            g.nodes()[n]["x"] = [0.] *10
    for n in g.nodes():
        if n not in node_exp:
            g_exp.nodes()[n]["x"] = [0.] *10
    return g , g_exp


def fidelity_sufficiency(fg,fg_e):
    return (fg - fg_e).item()

#def fidelity_comprehensiveness(fg, fg_no_e):
#    return (fg - fg_no_e).item()

def compute_fidelity(gcn,graphs,y=1):
    res_suf = []
    res_comp = []
    
    cc = 0
    for gid , g in graphs[y].items():
        thresholds = get_threshold(g)
        suf = []
        #comp = []
        
        node_idx = np.argmax(np.array(g.nodes()) == gid) # because if we keep only the k-hop neigh. the index of the node does not correspond to the index in the node feature matrix            
        for threshold in thresholds:
            threshold = threshold
            
            global g_no_exp
            global g_exp
            g_no_exp,g_exp = get_hard_mask(g,threshold)
            
            global data_g
            global data_g_no_exp 
            global data_g_exp
            data_g,data_g_no_exp,data_g_exp = convert_to_torch_graphs(g,g_no_exp,g_exp)            
            
            fg, fg_no_e, fg_e = evaluate_expls(gcn, data_g, data_g_no_exp, data_g_exp, node_idx=node_idx, y=y, ori_g=gcn.dataset.data)
            
            suf.append(fidelity_sufficiency(fg, fg_e))
            #comp.append(fidelity_comprehensiveness(fg, fg_no_e))
        
        
        res_suf.append(np.mean(suf))
        res_comp.append(np.nan)
        cc = cc + 1
    return np.mean(res_suf) #,np.mean(res_comp)