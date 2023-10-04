from turtle import st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os


def load_graphs(DATASET,MODEL,EXPL,MODE,verbose=True,lamb=0.001,normalize=True):
    if EXPL in ["sal_edge","ig_edge","gnnexpl","pgexpl","rgexpl","subgraphX"]:
        FOLDER = "edge_imp"
    else:
        FOLDER = "node_imp"

    path = "Explanations/GraphClassification/"+DATASET+"/"+MODEL+"/"+FOLDER+"/"+EXPL+"/"+MODE+"/"
    graphs = dict()

    labels = os.listdir(path)
    for lab in labels:
        if "ipynb" in lab: continue
        graphs[int(lab)] = dict()
        for i in os.listdir(path+"/"+lab):
            if i[-7:] == "gpickle":
                gid,y = i.split(".")[0].split("_")
                if y == lab:
                    graphs[int(lab)][int(gid)] = g = nx.read_gpickle(path+"/"+lab+"/"+i)

    # get node/edge attrs
    g = graphs[0][list(graphs[0].keys())[0]]
    node_impo = nx.get_node_attributes(g,"node_imp")
    edge_impo = nx.get_edge_attributes(g,"edge_imp")

    # convert to node_impo
    if not edge_impo == {}:
        for lab,graph_dict in graphs.items():
            for gid,g in graph_dict.items():
                graphs[lab][gid] = from_edge_to_nodeExpl(g)
    
    graphs = get_cleaned_graphs(graphs,verbose =verbose,lamb=lamb)


    # normalize 
    if normalize:
        all_node_imp = []
        for lab,graph_dict in graphs.items():
            if not graph_dict == None:
                for gid,g in graph_dict.items():
                    all_node_imp.extend(list(dict(nx.get_node_attributes(g,"node_imp")).values()))
        if not all_node_imp == []:
            min_val = np.min(all_node_imp)
            max_val = np.max(all_node_imp)

            for lab,graph_dict in graphs.items():
                if not graph_dict == None:
                    for gid,g in graph_dict.items():
                        for node in g.nodes():
                            orig = g.nodes()[node]["node_imp"]
                            g.nodes()[node]["node_imp_norm"] = (orig - min_val)/ (max_val-min_val)

    return graphs


def nc_load_graphs(DATASET,MODEL,EXPL,MODE,verbose=True,lamb=0.001,normalize=True, raw=False):
    if EXPL in ["sal_edge", "ig_edge", "gnnexpl", "rgexpl"]:
        FOLDER = "edge_imp"
    else:
        FOLDER = "node_imp"

    path = "Explanations/NodeClassification/"+DATASET+"/"+MODEL+"/"+FOLDER+"/"+EXPL+"/"+MODE+"/"
    print(path)
    graphs = dict()

    labels = os.listdir(path)
    for lab in labels:
        graphs[int(lab)] = dict()
        for i in os.listdir(path+"/"+lab):
            if i[-7:] == "gpickle":
                gid,y = i.split(".")[0].split("_")
                if y == lab:
                    graphs[int(lab)][int(gid)] = g = nx.read_gpickle(path+"/"+lab+"/"+i)

    
    # get node/edge attrs
    g = graphs[0][list(graphs[0].keys())[0]]
    node_impo = nx.get_node_attributes(g,"node_imp")
    edge_impo = nx.get_edge_attributes(g,"edge_imp")

    # convert to node_impo
    if node_impo == {}:
        for lab , graph_dict in graphs.items():
            for gid,g in graph_dict.items():
                graphs[lab][gid] = from_edge_to_nodeExpl(g)    
    if raw:
        return graphs

    # clean_expl
    graphs = nc_get_cleaned_graphs(graphs, explainer_name=EXPL, verbose=verbose, lamb=lamb)

    # normalize 
    if normalize:
        all_node_imp = []
        for lab,graph_dict in graphs.items():
            if not graph_dict == None:
                for gid,g in graph_dict.items():
                    all_node_imp.extend(list(dict(nx.get_node_attributes(g,"node_imp")).values()))
        if not all_node_imp == []:
            min_val = min(all_node_imp)
            max_val = max(all_node_imp)

            for lab,graph_dict in graphs.items():
                if not graph_dict == None:
                    for gid,g in graph_dict.items():
                        for node in g.nodes():
                            orig = g.nodes()[node]["node_imp"]
                            g.nodes()[node]["node_imp_norm"] = (orig - min_val) / (max_val-min_val)

    return graphs


def from_edge_to_nodeExpl(g):
    for n in g.nodes():
        nei = list(nx.neighbors(g,n))
        scores2 = []
        for u in nei:
            scores2.append(g.edges()[(u,n)]["edge_imp"])
        g.nodes()[n]["node_imp"] = np.mean(scores2)
    return g

def clean(graphs,lab,lamb=0.001):
    graphs_to_keep = dict()
    for gid,g in graphs[lab].items():
        a = list(nx.get_node_attributes(g,"node_imp").values())
        un_a = np.unique(a)
        if len(un_a)>1:
            #if np.isnan(un_a[-1]) or np.isnan(un_a[0]):
            #    print("IS NAN")
            diff = max(un_a) - min(un_a)
            if diff > lamb:
                graphs_to_keep[gid] = g
    return graphs_to_keep

def get_cleaned_graphs(graphs,verbose =True,lamb=0.001):
    orig_0 = len(graphs[0])
    orig_1 = len(graphs[1])
    
    if 2 in graphs:
        orig_2 = len(graphs[2])
    
    res0 = clean(graphs,0,lamb)
    res1 = clean(graphs,1,lamb)
    if 2 in graphs:
        res2 = clean(graphs,2,lamb)
    
    if verbose:
        if 2 in graphs:
            print("{:.3f}".format(len(res0)/orig_0),",","{:.3f}".format(len(res1)/orig_1),",","{:.3f}".format(len(res2)/orig_2))
        else:
            print("{:.3f}".format(len(res0)/orig_0),",","{:.3f}".format(len(res1)/orig_1))

    if len(res0)/orig_0 < 0.5:
        res0 = None
    if len(res1)/orig_1 < 0.5:
        res1 = None
    if 2 in graphs:
        if len(res2)/orig_2 < 0.5:
            res2 = None 
        return {0:res0, 1:res1,2:res2}
            
    return {0:res0, 1:res1}


def nc_get_cleaned_graphs(graphs, explainer_name, verbose=True, lamb=0.001):
    originals , kepts = [] , []
    for c in sorted(graphs.keys()):
        original = len(graphs[c])
        kept = clean(graphs, c, lamb)
        originals.append(original)
        kepts.append(kept)
    
    if verbose:
        print(",".join(["{:.3f} ".format(len(kepts[i]) / originals[i]) for i in range(len(originals))]))  #f"{explainer_name:12}: \t", 
    
    return {i: kepts[i] if len(kepts[i])/originals[i] >= 0.5 else None for i in range(len(originals))}
