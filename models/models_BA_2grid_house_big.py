
from torch_geometric.nn import GCN

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from torch_geometric.nn import GCNConv,global_max_pool,global_add_pool,global_mean_pool, GATConv
from torch_geometric.loader import DataLoader


from torch.nn import Linear

class GCN_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self,num_features,num_classes):
                super().__init__()
                self.conv1 = GCNConv(num_features, 60)
                self.conv2 = GCNConv(60, 60)
                self.conv3 = GCNConv(60, 60)
                self.conv4 = GCNConv(60, 60)
                self.lin1 = Linear(60,60)
                self.lin2 = Linear(60,10)
                self.lin3 = Linear(10,num_classes)

            def forward(self,x,edge_index,batch,edge_mask=None):

                x = F.relu(self.conv1(x, edge_index,edge_mask))
                x = F.relu(self.conv2(x, edge_index,edge_mask))
                x = F.relu(self.conv3(x, edge_index,edge_mask))
                x = F.relu(self.conv4(x, edge_index,edge_mask))
                #x = F.relu(self.conv3(x, edge_index,edge_mask))
                #x = self.conv4(x, edge_index,edge_mask)

                x = global_max_pool(x,batch)
                
                x = F.relu(self.lin1(x))
                x = F.relu(self.lin2(x))
                x = self.lin3(x)


                return F.log_softmax(x, dim=-1)


            

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=256)
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self,loader):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset),total_loss / len(self.train_loader.dataset)

    def iterate(self):

        for epoch in range(1, 1001):
            loss = self.train()
            train_acc,train_loss = self.test(self.train_loader)
            test_acc,test_loss = self.test(self.test_loader)
            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):
        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):

        train_acc,train_loss = self.test(self.train_loader)
        test_acc,test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            


from torch_geometric.nn import SAGEConv

class GraphSAGE_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self,num_features,num_classes):
                super().__init__()
                self.conv1 = SAGEConv(num_features, 30,normalize=True,project=True)
                self.conv2 = SAGEConv(30, 30,normalize=True,project=True)
                self.conv3 = SAGEConv(30, num_classes,normalize=True,project=True)
                #self.lin1 = Linear(30, 30)
                #self.lin2 = Linear(30, num_classes)

            def forward(self,x,edge_index,batch,edge_mask=None):
                x = F.relu(self.conv1(x, edge_index,edge_mask))
                x = F.relu(self.conv2(x, edge_index,edge_mask))
                x = self.conv3(x, edge_index,edge_mask)

                #x1 = global_mean_pool(x,batch)
                x = global_mean_pool(x,batch)
                #x3 = global_add_pool(x,batch)

                #x = torch.cat((x1,x2,x3),dim = 1)

                #x = F.relu(self.lin1(x))
                #x = self.lin2(x)

                return F.log_softmax(x, dim=-1)

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.1) #0.0001

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=256)
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self,loader):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset),total_loss / len(self.train_loader.dataset)

    def iterate(self):

        for epoch in range(1, 501):
            loss = self.train()
            train_acc,train_loss = self.test(self.train_loader)
            test_acc,test_loss = self.test(self.test_loader)
            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):
        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):

        train_acc,train_loss = self.test(self.train_loader)
        test_acc,test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            
############################ myGIN conv layer
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


from typing import Any
import torch
from torch import Tensor


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)



class GINConv_my(MessagePassing):

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None,edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)
  
    
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

from torch_geometric.nn import MLP, GINConv
class GINmy_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.mlp1 = torch.nn.Linear(in_channels, 30)
                self.conv1 = GINConv_my(self.mlp1)
                self.mlp2 = torch.nn.Linear(30, 30)
                self.conv2 = GINConv_my(self.mlp2)


                self.lin1 = Linear(30, 30)
                self.lin2 = Linear(30, out_channels)



            def forward(self, x, edge_index, batch,edge_weight=None):
                x = self.conv1(x, edge_index,edge_weight)
                x = x.relu()
                x = self.conv2(x, edge_index,edge_weight)
                x = x.relu()

                x = global_mean_pool(x, batch)

                x = F.relu(self.lin1(x))
                x = self.lin2(x)

                return F.log_softmax(x, dim=-1)
     

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=256)
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self,loader):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset),total_loss / len(self.train_loader.dataset)

    def iterate(self):

        for epoch in range(1, 501):
            loss = self.train()
            train_acc,train_loss = self.test(self.train_loader)
            test_acc,test_loss = self.test(self.test_loader)
            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):
        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):

        train_acc,train_loss = self.test(self.train_loader)
        test_acc,test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            
############# end my GIN

from torch_geometric.nn import MLP, GINConv
class GIN_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.mlp1 = torch.nn.Linear(in_channels, 30)
                self.conv1 = GINConv(self.mlp1)
                self.mlp2 = torch.nn.Linear(30, 30)
                self.conv2 = GINConv(self.mlp2)

                self.lin1 = Linear(30,30)
                self.lin2 = Linear(30,out_channels)

                

            def forward(self, x, edge_index, batch):
                x = self.conv1(x, edge_index)
                x = x.relu()
                x = self.conv2(x, edge_index)
                x = x.relu()

                x = global_mean_pool(x, batch)

                x = F.relu(self.lin1(x))
                x = self.lin2(x)

                return F.log_softmax(x, dim=-1)
     

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=256)
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self,loader):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset),total_loss / len(self.train_loader.dataset)

    def iterate(self):

        for epoch in range(1, 1001):
            loss = self.train()
            train_acc,train_loss = self.test(self.train_loader)
            test_acc,test_loss = self.test(self.test_loader)
            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):
        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):

        train_acc,train_loss = self.test(self.train_loader)
        test_acc,test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            




from torch_geometric.nn import ChebConv

class Cheb_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset


        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.conv1 = ChebConv(in_channels,30,K=5)
                self.conv2 = ChebConv(30,30,K=5)
                self.conv3 = ChebConv(30,30,K=5)
                self.lin1 = Linear(30,30)
                self.lin2 = Linear(30,out_channels)


            def forward(self, x, edge_index, batch,edge_weight=None):
                x = self.conv1(x, edge_index,edge_weight)
                x = x.relu()
                x = self.conv2(x, edge_index,edge_weight)
                x = x.relu()
                x = self.conv3(x, edge_index,edge_weight)

                x = global_mean_pool(x, batch)

                x = F.relu(self.lin1(x))
                x = self.lin2(x)

                return F.log_softmax(x, dim=-1)


     

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=256)
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self,loader):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset),total_loss / len(self.train_loader.dataset)

    def iterate(self):

        for epoch in range(1, 501):
            loss = self.train()
            train_acc,train_loss = self.test(self.train_loader)
            test_acc,test_loss = self.test(self.test_loader)
            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):
        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):

        train_acc,train_loss = self.test(self.train_loader)
        test_acc,test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            





              
        
##### DIFFPOOL
from math import ceil
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
from torch_geometric.utils import to_dense_adj
import numpy as np




class Diffpool_framework:
    def __init__(self,dataset,max_nodes,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.max_nodes = max_nodes
        self.dataset = dataset
        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DenseDataLoader(self.dataset[self.train_idx],batch_size=64)
        self.test_loader = DenseDataLoader(self.dataset[self.test_idx],batch_size=64)

        class GNN(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels,lin=True):
                super().__init__()

                self.conv1 = DenseGCNConv(in_channels, hidden_channels)
                self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
                self.conv2 = DenseGCNConv(hidden_channels, hidden_channels)
                self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
                self.conv3 = DenseGCNConv(hidden_channels, out_channels)
                self.bn3 = torch.nn.BatchNorm1d(out_channels)
                if lin is True:
                    self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                            out_channels)
                else:
                    self.lin = None

            def bn(self, i, x):
                batch_size, num_nodes, num_channels = x.size()

                x = x.view(-1, num_channels)
                x = getattr(self, f'bn{i}')(x)
                x = x.view(batch_size, num_nodes, num_channels)
                return x

            def forward(self, x, adj, mask=None):
                batch_size, num_nodes, in_channels = x.size()

                x0 = x
                x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
                x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
                x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

                x = torch.cat([x1, x2, x3], dim=-1)

                if self.lin is not None:
                    x = F.relu(self.lin(x))

                return x


        class Net(torch.nn.Module):
            def __init__(self,max_nodes,dataset):
                super().__init__()
                self.max_nodes = max_nodes
                num_nodes = ceil(0.25 * max_nodes)
                self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
                self.gnn1_embed = GNN(dataset.num_features, 64, 64, lin=False)

                num_nodes = ceil(0.25 * num_nodes)
                self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
                self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

                self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

                self.lin1 = torch.nn.Linear(3 * 64, 64)
                self.lin2 = torch.nn.Linear(64, dataset.num_classes)

            def forward(self, x, edge_index, mask=None,batch=None):
                adj = from_eds_to_adjs(edge_index,self.max_nodes)
                s = self.gnn1_pool(x, adj, mask)
                x = self.gnn1_embed(x, adj, mask)

                x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

                s = self.gnn2_pool(x, adj)
                x = self.gnn2_embed(x, adj)

                x, adj, l2, e2 = dense_diff_pool(x, adj, s)

                x = self.gnn3_embed(x, adj)

                x = x.mean(dim=1)
                x = F.relu(self.lin1(x))
                x = self.lin2(x)
                return F.log_softmax(x, dim=-1)


            # used for PGExplainer
            def get_emb(self,x,edge_index, mask=None,batch=None):
                adj = from_eds_to_adjs(edge_index,self.max_nodes)
                s = self.gnn1_pool(x, adj, mask)
                x = self.gnn1_embed(x, adj, mask)

                x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

                s = self.gnn2_pool(x, adj)
                x = self.gnn2_embed(x, adj)

                x, adj, l2, e2 = dense_diff_pool(x, adj, s)

                x = self.gnn3_embed(x, adj) 

  
                return x

            
        self.model = Net(self.max_nodes,self.dataset).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)


    def train(self,epoch):
        self.model.train()
        loss_all = 0
        for data in self.train_loader:
            self.data = data.to(self.device)
            self.optimizer.zero_grad()
            eds = from_adjs_to_eds(data.adj)
            output = self.model(data.x, eds, data.mask)
            
            loss = F.nll_loss(output, data.y.view(-1))
            loss.backward()
            loss_all += data.y.size(0) * loss.item()
            self.optimizer.step()
        return loss_all / len(self.train_loader.dataset)


    @torch.no_grad()
    def test(self,loader):
        self.model.eval()
        correct = 0
        for data in loader:
            data = data.to(self.device)
            eds = from_adjs_to_eds(data.adj)

            out = self.model(data.x, eds, data.mask)
            loss = F.nll_loss(out,data.y.view(-1))
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
        return loss,correct / len(loader.dataset)
    def iterate(self):
        for epoch in range(1, 51):
            train_loss = self.train(epoch)
            test_loss,test_acc = self.test(self.test_loader)
            _,train_acc = self.test(self.train_loader)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f},Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
   


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):
        train_loss,train_acc = self.test(self.train_loader)
        test_loss,test_acc = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            
# from adj to tensor
def from_adjs_to_eds(adjs):
    eds = []
    for i in adjs:
        eds.append(torch.nonzero(i).T)
    return eds

def from_eds_to_adjs(eds,max_nodes):
    adjs = []
    for ed in eds:
        adjs.append(to_dense_adj(ed,max_num_nodes=max_nodes))
    return torch.cat(adjs)