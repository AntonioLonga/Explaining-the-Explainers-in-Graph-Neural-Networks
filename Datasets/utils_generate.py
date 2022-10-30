import pickle as pkl
import random
import networkx as nx
import numpy as np

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


def generate_grid_networks_BA(nb_house,node_ba=30):
    g = nx.barabasi_albert_graph(node_ba,1)
    if nb_house == 0:
         return g
    for i in range(nb_house):
        house = my_grid()
        tmp = len(g.nodes)
        g = nx.union(g,house,rename=("a","b"))
        tmp2 = len(g.nodes)
        # connect house BA:
        node_a = np.random.randint(tmp)
        node_b = np.random.randint(tmp,tmp2)
        g.add_edge(list(g.nodes())[node_a],list(g.nodes())[node_b])

    mapping = dict()
    c = 0
    for n in g.nodes():
        mapping[n] = c
        c = c + 1
    g = nx.relabel_nodes(g,mapping)
    
    return g
    

def generate_BA_grid_house(target,node_ba=30):
    
    if target == 0:
        if np.random.rand() >0.5:
            g = nx.barabasi_albert_graph(node_ba-5,1)
            g1 = my_house()
        else:
            g = nx.barabasi_albert_graph(node_ba-9,1)
            g1 = my_grid()
            
        tmp = len(g.nodes)
        g = nx.union(g,g1,rename=("a","b"))
        tmp2 = len(g.nodes)
        
        # connect g to BA:
        node_a = np.random.randint(tmp)
        node_b = np.random.randint(tmp,tmp2)
        g.add_edge(list(g.nodes())[node_a],list(g.nodes())[node_b])
            
    if target == 1:
        g = nx.barabasi_albert_graph(node_ba-9-5,1)
        g1 = my_house()  
        tmp = len(g.nodes)
        g = nx.union(g,g1,rename=("a","b"))
        tmp2 = len(g.nodes)
        # connect g to BA:
        node_a = np.random.randint(tmp)
        node_b = np.random.randint(tmp,tmp2)
        g.add_edge(list(g.nodes())[node_a],list(g.nodes())[node_b])

        
        g1 = my_grid()  
        tmp = len(g.nodes)
        g = nx.union(g,g1,rename=("a","b"))
        tmp2 = len(g.nodes)
        # connect g to BA:
        node_a = np.random.randint(tmp)
        node_b = np.random.randint(tmp,tmp2)
        g.add_edge(list(g.nodes())[node_a],list(g.nodes())[node_b])
        
        
    mapping = dict()
    c = 0
    for n in g.nodes():
        mapping[n] = c
        c = c + 1
    g = nx.relabel_nodes(g,mapping)
    
    return g



def generate_star_network_ER(nb_nodesBA,nb_stars,nb_rewire=10,star_size=16):
    is_connected = False
    while not is_connected:
        g = nx.fast_gnp_random_graph(nb_nodesBA,0.1)
        is_connected = nx.is_connected(g)

    for i in range(nb_stars):
        star = nx.star_graph(star_size)
        g = nx.union(g,star,rename=("a","b"))

    mapping = dict()
    c = 0
    for n in g.nodes():
        mapping[n] = c
        c = c + 1
    g = nx.relabel_nodes(g,mapping)
    
    for i in range(nb_rewire):
        a,b = np.random.choice(g.nodes(),2,replace=False)
        if not g.has_edge(a,b):
            g.add_edge(a,b)
            
    Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(Gcc[0])
    return g


def gen_stars_ER_3class():
    graphs = []
    labels = []
    gt = []
    for i in range(500):
        nb_nodes_BA1 = np.random.randint(30,50)
        g1 = generate_star_network_ER(nb_nodesBA = nb_nodes_BA1,
                                      nb_stars = 1,
                                      nb_rewire=10,star_size=16)
        nb_nodes_BA2 = np.random.randint(30-13,50)
        g2 = generate_star_network_ER(nb_nodesBA = nb_nodes_BA2,
                                      nb_stars = 2,
                                      nb_rewire=10,star_size=16)
        
        nb_nodes_BA3 = np.random.randint(30-26,50-26)
        if 0.5> np.random.randn():
            g3 = generate_star_network_ER(nb_nodesBA = nb_nodes_BA3,
                                          nb_stars = 3,
                                          nb_rewire=10,star_size=16)
        else:
            g3 = generate_star_network_ER(nb_nodesBA = nb_nodes_BA3,
                                          nb_stars = 4,
                                          nb_rewire=10,star_size=16)
        
        if nx.is_connected(g1):
            if nx.is_connected(g2):
                if nx.is_connected(g3):
                    labels.append(0)
                    labels.append(1)
                    labels.append(2)
                    graphs.append(g1)
                    graphs.append(g2)
                    graphs.append(g3)
                    gt.append(nb_nodes_BA1)
                    gt.append(nb_nodes_BA2)
                    gt.append(nb_nodes_BA3)
    
    return graphs,labels,gt

def generate_house_color_networks_BA(nb_house,node_ba=30,house_color="blue"):
    GT = []
    g = nx.barabasi_albert_graph(node_ba,1)
    g = color_ba(g)
    
    nb_nodes_curr = len(g.nodes())
    first = True
    for i in range(nb_house):
        house = my_house()
        if first:
            GT.append(list(np.arange(nb_nodes_curr,nb_nodes_curr+5)))
            house = color_house(house,gt=True,base_color=house_color)
            first = False
        else:
            house = color_house(house,gt=False,base_color=house_color)
        g = nx.union(g,house,rename=("a","b"))
        # connect house BA:
        node_a = np.random.randint(nb_nodes_curr)
        node_b = np.random.choice(5)+nb_nodes_curr
        g.add_edge(list(g.nodes())[node_a],list(g.nodes())[node_b])
        nb_nodes_curr = len(g.nodes())
        
    mapping = dict()
    c = 0
    for n in g.nodes():
        mapping[n] = c
        c = c + 1
    g = nx.relabel_nodes(g,mapping)
    
    return g,GT

def color_ba(g,colors=["red","blue","green"]):
    for n in g.nodes():
        g.nodes()[n]["color"]=np.random.choice(colors)
        
    return g
def color_house(h,gt=True,base_color="blue"):
    if gt == True:
        for n in h.nodes():
            h.nodes()[n]["color"]=base_color
    else:
        if base_color == "blue":
            nodes_h = list(h.nodes())
            h.nodes()[nodes_h[0]]["color"] = "blue"
            h.nodes()[nodes_h[1]]["color"] = "red"
            for n in nodes_h[2:]:
                h.nodes()[n]["color"]=np.random.choice(["red","blue","green"])
        
        if base_color == "green":
            nodes_h = list(h.nodes())
            h.nodes()[nodes_h[0]]["color"] = "green"
            h.nodes()[nodes_h[1]]["color"] = "red"
            for n in nodes_h[2:]:
                h.nodes()[n]["color"]=np.random.choice(["red","blue","green"])
    return h

def get_feat(g):
    emb = []
    for n,c in nx.get_node_attributes(g,"color").items():
        if c == "blue":
            emb.append([1,0,0])
        elif c == "green":
            emb.append([0,1,0])
        else:
            emb.append([0,0,1])
    return emb