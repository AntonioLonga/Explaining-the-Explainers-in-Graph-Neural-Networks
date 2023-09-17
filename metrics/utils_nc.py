import os
import os.path as osp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from networkx.algorithms import isomorphism
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score


#%% Read and preprocess
def load_graphs(DATASET,
                MODEL,
                EXPL,
                MODE,
                verbose=True,
                lamb=0.001,
                normalize=True,
                raw=False):
    
    # Define the path
    if EXPL in ['gnnexpl', 'pgexplainer']:
        FOLDER = 'edge_imp'
    else:
        FOLDER = 'node_imp'
    path = 'Explanations/NodeClassification/' + DATASET + '/' + MODEL + '/' + FOLDER + '/' + EXPL + '/' + MODE + '/'
    
    # Read the explanations
    graphs = {}
    # Get the labels
    labels = os.listdir(path)
    # Loop over the labels
    for label in labels:
        # Initialize a dictionary to collect all the nodes with this label
        graphs[int(label)] = {}
        # Loop over the single nodes
        for node_name in os.listdir(osp.join(path, label)):
            # Check if the format is correct
            if node_name[-7:] == 'gpickle':
                # Split the node name to get the node number and its class
                node_number, node_class = node_name.split('.')[0].split('_')
                # Double check if the class is correct
                if node_class == label:
                    # Read and store the graph
                    graphs[int(label)][int(node_number)] = nx.read_gpickle(osp.join(path, osp.join(label, node_name)))
                    # print(nx.is_directed(nx.read_gpickle(osp.join(path, osp.join(label, node_name)))))


    # Convert to node_importance if there are edge importances
    # Get a random graph (the first one that is found) from the dataset
    for label in graphs:
        if graphs[label]:
            g = graphs[label][list(graphs[label].keys())[0]]
            break
    # else:
    #     # No valid graph is found
    #     raise
        
    # Check if it has node importances
    has_edge_impo = nx.get_edge_attributes(g, 'edge_imp') != {}
    # If so, convert them to node importances
    if has_edge_impo:
        for label, graph_dict in graphs.items():
            for node_number, graph in graph_dict.items():
                graphs[label][node_number] = from_edge_to_nodeExpl(graph)
    
    # If required, process the graphs
    if not raw:
        # Clean the explanations
        graphs = get_cleaned_graphs(graphs, explainer_name=EXPL, verbose=verbose, tol=lamb)

        # Normalize the explanations 
        if normalize:
            all_node_imp = []
            for lab, graph_dict in graphs.items():
                if not graph_dict == None:
                    for gid, g in graph_dict.items():
                        all_node_imp.extend(list(dict(nx.get_node_attributes(g, 'node_imp')).values()))
            if not all_node_imp == []:
                min_val = np.min(all_node_imp)
                max_val = np.max(all_node_imp)

                for lab, graph_dict in graphs.items():
                    if not graph_dict == None:
                        for gid, g in graph_dict.items():
                            for node in g.nodes():
                                orig = g.nodes()[node]['node_imp']
                                g.nodes()[node]['node_imp_norm'] = (orig - min_val) / (max_val - min_val)
    
    # Return the graphs                        
    return graphs


def from_edge_to_nodeExpl(graph):
    for node in graph.nodes():
        scores2 = []
        neighbors = list(nx.neighbors(graph, node))
        for neighbor in neighbors:
            scores2 += [graph.edges()[(node, neighbor)]['edge_imp']]
        graph.nodes()[node]['node_imp'] = np.mean(scores2)
    return graph


def clean(graphs, label, tolerance=0.001):
    graphs_to_keep = {}
    
    # Loop over the graphs with the given label
    for node_number, graph in graphs[label].items():
        # Extract the values of all the node importances
        node_imp = list(nx.get_node_attributes(graph, 'node_imp').values())
        # Filter and sort unique values
        unique_node_imp = np.unique(np.array(node_imp)[~np.isnan(node_imp)])
        # If there is more than one value
        if len(unique_node_imp) > 1:
            # Compute the difference between the max and min node importance 
            # (unique returns sorted values)
            diff = unique_node_imp[-1] - unique_node_imp[0]
            # If the difference is large enough, keep the graph
            if diff > tolerance:
                graphs_to_keep[node_number] = graph
                
    return graphs_to_keep


def get_cleaned_graphs(graphs, explainer_name, verbose=True, tol=0.001):
    kepts, fraction_kept = [], []
    # Loop over the class labels
    for label in sorted(graphs.keys()):
        # Filter the graphs and store the one to keep
        kepts += [clean(graphs, label, tol)]
        # Count the fraction of graphs which are kept
        fraction_kept += [len(kepts[-1]) / len(graphs[label])]
        
    # Print the stats 
    if verbose:
        print(' '.join([' {:.3f}'.format(x) for x in fraction_kept]) + '\t' + explainer_name) 
       
    # Filter the explanations where at least the 50% of the graphs are kept
    val = {idx: kepts[idx] if val >= 0.5 else None for idx, val in enumerate(fraction_kept)}
    return val


#%% Plot
def plot_expl(g, only_relevant_nodes=False, big_plot=False, with_labels=False):
    if big_plot:
        plt.figure(figsize=(20,20))
    if only_relevant_nodes:
        node_attr = np.array(list(nx.get_node_attributes(g, "node_imp").values()))
        node_list = np.nonzero(node_attr > 0)[0]
        node_color = node_attr[node_list]
        edge_list = [(u,v) for u,v in g.edges() if u in node_list and v in node_list]
    else:
        node_list = list(g.nodes())
        edge_list = list(g.edges())
        node_color = list(nx.get_node_attributes(g, "node_imp").values())
    nx.draw(g, node_size=50, nodelist=node_list, edgelist=edge_list, node_color=node_color, with_labels=with_labels)
    plt.show()
    

#%%
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx
import random
import torch


def compute_f1(s, c):
    f1 = 2 * (((1 - s) * c) / (1 - s + c))
    return f1

def set_seeds():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
#%%
def build_expl(DATASET, MODEL, EXPL, GNN_NUM_LAYERS, num_features=2, verbose=False, 
               lamb=0.001, 
               normalize=True, 
               cut_ego=False):
    set_seeds()
    
    graphs = load_graphs(DATASET=DATASET,
                         MODEL=MODEL,
                         EXPL=EXPL,
                         MODE='train',
                         verbose=verbose,
                         lamb=lamb,
                         normalize=normalize)    
    
    # print(graphs)
    for c in range(len(graphs)):
        if graphs[c] is None: 
            # print('Here')
            continue
        # print('There')
        for i, g in graphs[c].items():
            for n in g.nodes():
                g.nodes()[n]["x"] = [1.] * num_features
            
            if cut_ego:
                graphs[c][i] = nx.ego_graph(graphs[c][i], n=i, radius=GNN_NUM_LAYERS[MODEL])
    return graphs


def convert_to_torch_graphs(g_no_exp, g_exp):    
    # data_g = from_networkx(g)
    # data_g.node_imp = None
    # data_g.node_imp_norm = None
    
    data_g_exp = from_networkx(g_exp)
    data_g_exp.node_imp = None
    data_g_exp.node_imp_norm = None
    
    data_g_no_exp = from_networkx(g_no_exp)
    data_g_no_exp.node_imp = None
    data_g_no_exp.node_imp_norm = None
    
    return data_g_no_exp, data_g_exp


#%%
def evaluate_expls(gcn, data_g, data_g_no_exp, data_g_exp, node_idx, y):        
    val = gcn.model.forward_single(data_g.x, data_g.edge_index, node_idx)
    # print('Val:', val)
    
    fg = torch.exp(val[y])
    
    if  data_g_no_exp.x == None:
        fg_no_e = 0.5
    else:
        z = gcn.model.forward_single(data_g_no_exp.x, data_g_no_exp.edge_index,
                                                      node_idx)
        # print('NO:', z)
        fg_no_e = torch.exp(z[y])
        # print('NO:', fg_no_e)
        # print('NO:', torch.exp(z))
        
    if  data_g_exp.x == None:
        fg_e = 0.5
    else:
        z = gcn.model.forward_single(data_g_exp.x,
                                                   data_g_exp.edge_index,
                                                   node_idx)
        # print(z)
        fg_e = torch.exp(z[y])
        # print(fg_e)
        # print(torch.exp(z))

    return fg, fg_no_e, fg_e


#%%
def get_threshold(g):
    node_att = nx.get_node_attributes(g,"node_imp_norm")
    # print(node_att)
    thresholds = []
    for i in sorted(list(node_att.values())):
        thresholds.append(float("%.2f" % i))
    thresholds = np.unique(thresholds)    
    return list(thresholds)


# keep features only of nodes with importance >= threshold
def get_hard_mask(g_in, threshold, num_features):    
    node_att = nx.get_node_attributes(g_in, 'node_imp_norm')
    
    node_g = []
    node_exp = []    
    for k, v in node_att.items():
        if v >= threshold:
            node_exp.append(k)
        else:
            node_g.append(k)
    
    g = g_in.copy()
    g_exp = g_in.copy()
    for n in g.nodes():
        if n not in node_g:
            g.nodes()[n]["x"] = [0.] * num_features
    for n in g.nodes():
        if n not in node_exp:
            g_exp.nodes()[n]["x"] = [0.] * num_features
    
    return g, g_exp


#%%
def fidelity_sufficiency(fg, fg_e):
    return (fg - fg_e).item()


def fidelity_comprehensiveness(fg, fg_no_e):
    return (fg - fg_no_e).item()


#%%
def compute_fidelity(gcn, original_data, graphs, num_features, y=1):
    res_suf = []
    
    cc = 0
    for gid, g in graphs[y].items():
        thresholds = get_threshold(g)

        suf = []
        node_idx = np.argmax(np.array(g.nodes()) == gid) # because if we keep only the k-hop neigh. the index of the node does not correspond to the index in the node feature matrix
            
        for threshold in thresholds:
            global g_no_exp
            global g_exp
            g_no_exp, g_exp = get_hard_mask(g, threshold, num_features)
            
            # global data_g
            global data_g_no_exp 
            global data_g_exp
            data_g_no_exp, data_g_exp = convert_to_torch_graphs(g_no_exp, g_exp)            
            
            fg, fg_no_e, fg_e = evaluate_expls(gcn, original_data, data_g_no_exp, data_g_exp, 
                                               node_idx=node_idx, y=y)
            
            suf.append(fidelity_sufficiency(fg, fg_e))
        
        
        res_suf.append(np.mean(suf))
        cc += 1

    return np.mean(res_suf)


#%% Plausibility
def get_expl_graph(graph, threshold, node_number, radius):    
    # Get the node importances
    node_att = nx.get_node_attributes(graph, 'node_imp_norm')
    
    # Filter the nodes based on the threshold
    idx_expl_nodes = [idx for idx in node_att if node_att[idx] > threshold]

    # Extract the corresponding subgraph
    graph_expl = graph.subgraph(idx_expl_nodes).nodes()
    
    return graph_expl


def compute_plausibility(motifs, graph_expl):
    plausibility = []
    
    for motif in motifs:

        plausibility += [len(set(motif) & set(graph_expl)) / len(set(motif) | set(graph_expl))]
        # plausibility += [len(set(motif) & set(graph_expl)) / len(set(motif))]
        
    val = np.mean(plausibility)
    
    return val


def get_motifs(node_number, graph, label, node_labels, radius):
    # Get the list of infected nodes
    infected_nodes = np.argwhere(node_labels == 0)[:, 0]

    if label == 0:
        # The node is infected, so the ground truth is the node itself
        motif =[[node_number]]
    else:
        if node_number in graph: 
            # In this case we need the node's ego graph
            ego_graph = nx.ego_graph(graph, n=node_number, radius=radius, undirected=True)
            # print(nx.is_directed(graph), nx.is_directed(ego_graph))
            
            # Get the infected nodes in the motif
            infected_nodes = list(set(ego_graph) & set(infected_nodes))
            tmp = []
            for node in infected_nodes:
                directed_ego_graph = nx.ego_graph(graph, n=node, radius=radius)
                if node_number in directed_ego_graph and nx.has_path(directed_ego_graph, node, node_number):
                    tmp += [node]
            infected_nodes = tmp
            
            if label == 2:     
                # Double check
                if infected_nodes:
                    print('Something is wrong with class 2')
                    # print(node_number, infected_nodes)
                    motif = []
                else:
                    # The minimal path from the node to an infected one has length larger 
                    # than 2, so the ground truth is the set of nodes which have distance up to radius 
                    # from node_number
                    motif = []
                    for node in ego_graph.nodes():
                        directed_ego_graph = nx.ego_graph(graph, n=node, radius=radius)
                        if node_number in directed_ego_graph and nx.has_path(directed_ego_graph, node, node_number):
                            motif += [nx.shortest_path(directed_ego_graph, node, node_number)]
                    # motif = [list(ego_graph.nodes())]
                        
            else: # label == 1
                # Double check
                if not infected_nodes:
                    print('Something is wrong with class 1')
                    motif = []
                else:
                    # Initialize the best distance so far
                    min_path_len = radius + 1
                    # Initialize the list of motifs
                    motif = []
                    # Loop over all the infected nodes in the ego graph
                    for neigh_number in infected_nodes:
                        path = nx.shortest_path(graph, neigh_number, node_number)
                        if len(path) <= min_path_len:
                            min_path_len = len(path)
                            motif.append(path)
                        # Check and remove the long paths
                        min_path_len = np.min([len(path) for path in motif])
                        motif = [path for path in motif if len(path) == min_path_len]
            
        else:
            # If the node is not in the graph, return an empty motif
            motif = []

    return motif


def get_roc_infection(graphs, label, gnn_num_layers, node_labels):
    # Save the ROC for each graph
    rocs = []
    
    # Loop over the graphs
    for node_number, graph in graphs[label].items():

        # Get all the possible ground truths
        motifs = get_motifs(node_number, graph, label, node_labels, 
                            radius=gnn_num_layers)        
        
        if motifs: # else motifs = []: isolated node
            # Get the soft explanation mask
            preds = np.array(list(nx.get_node_attributes(graph, 'node_imp_norm').values()))
            preds[np.isnan(preds)] = 0
            
            # Compute the ROC AUC score for each motif
            tmp_roc = []
            for motif in motifs:
                trues = [1 if node in motif else 0 for node in graph.nodes()]
                tmp_roc += [roc_auc_score(trues, preds)]
    
            # Keep the best motif
            rocs += [np.max(tmp_roc)]
        
    # Compute the average ROC over the dataset
    roc = np.mean(rocs)
    
    return roc


# def get_roc_infection(graphs, label, gnn_num_layers, node_labels):
#     # Save the ROC for each graph
#     rocs = []
    
#     # Loop over the graphs
#     for node_number, graph in graphs[label].items():

#         # Get all the possible ground truths
#         motifs = get_motifs(node_number, graph, label, node_labels, 
#                             radius=gnn_num_layers)        

#         # Get the thresholds associated to the current graph
#         thresholds = get_threshold(graph)

#         # Accumulate the values corresponding to the different thresholds
#         plausibilities = []
        
#         # Loop over the thresholds
#         for threshold in thresholds:
#             # Get the hard explanation mask
#             graph_expl = get_expl_graph(graph, threshold)
            
#             # Compute the current plausibility
#             plausibility = compute_plausibility(motifs, graph_expl)
            
#             # Save the result
#             plausibilities += [plausibility]        
        
#         # Compute the ROC as the mean
#         rocs += [np.mean(plausibilities)]
        
#     # Compute the average ROC over the dataset
#     roc = np.mean(rocs)
    
#     return roc

    
