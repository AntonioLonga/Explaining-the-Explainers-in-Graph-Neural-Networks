# Node classification dataset
# Infection
from torch_geometric.nn import GCN, GATConv, GATv2Conv, GINConv

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_add_pool, global_mean_pool, GATConv, ChebConv
from torch_geometric.loader import DataLoader


#%%
class GCN_framework:
    def __init__(self, dataset, device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super().__init__()
                self.conv1 = GCNConv(num_features, 30)
                self.conv2 = GCNConv(30, 30)
                self.lin2 = Linear(30, num_classes)

            def forward(self, x, edge_index, edge_mask=None):
                x = self.conv1(x.float(), edge_index)
                x = F.relu(self.conv2(x, edge_index, edge_mask))
                x = self.lin2(x)
                return F.log_softmax(x, dim=-1)            
            
            def forward_single(self, x, edge_index, node_idx, edge_mask=None):
                x = self.conv1(x.float(), edge_index)
                x = F.relu(self.conv2(x, edge_index, edge_mask))
                x = self.lin2(x)
                return F.log_softmax(x, dim=-1)[node_idx]

        self.model = Net(self.dataset.num_features, self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.loader = DataLoader(self.dataset, batch_size=128)
        
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x.float(), data.edge_index)
            loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach() * data.num_graphs
        return total_loss.item() / sum(self.dataset.data.train_mask)
    

    @torch.no_grad()
    def test(self, loader, mask):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)
            total_correct += int((out.argmax(-1)[mask] == data.y[mask]).sum())
            
            loss = F.nll_loss(out[mask], data.y[mask])
            total_loss += loss.detach() * data.num_graphs            
        return total_correct / sum(mask), total_loss.item() / sum(mask)
    

    def iterate(self, num_epochs=100):
        for epoch in range(1, num_epochs+1):
            loss = self.train()
            train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
            test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
            if epoch % 50 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self, path):        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):
        train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
        test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            


#%%
from torch_geometric.nn import SAGEConv
class GraphSAGE_framework:
    def __init__(self, dataset, device=None):   
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        class Net(torch.nn.Module):
            def __init__(self,num_features,num_classes):
                super().__init__()
                self.conv1 = SAGEConv(num_features, 30, aggr='mean')
                self.conv2 = SAGEConv(30, 30, aggr='mean')
                self.lin1 = Linear(30, num_classes)

            def forward(self,x,edge_index,edge_mask=None):
                x = F.relu(self.conv1(x, edge_index, edge_mask))
                x = F.relu(self.conv2(x, edge_index, edge_mask))
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)

            def forward_single(self, x, edge_index, node_idx, edge_mask=None):
                x = F.relu(self.conv1(x, edge_index, edge_mask))
                x = F.relu(self.conv2(x, edge_index, edge_mask))
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)[node_idx]
            
        self.dataset = dataset
        self.model = Net(self.dataset.num_features,self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        
        self.loader = DataLoader(self.dataset, batch_size=128)
        
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x.float(), data.edge_index)
            loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach() * data.num_graphs
        return total_loss.item() / sum(self.dataset.data.train_mask)
    

    @torch.no_grad()
    def test(self, loader, mask):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)
            total_correct += int((out.argmax(-1)[mask] == data.y[mask]).sum())
            
            loss = F.nll_loss(out[mask], data.y[mask])
            total_loss += loss.detach() * data.num_graphs            
        return total_correct / sum(mask), total_loss.item() / sum(mask)
    

    def iterate(self, num_epochs=100):
        for epoch in range(1, num_epochs+1):
            loss = self.train()
            train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
            test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
            if epoch % 50 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):
        train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
        test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            

#%%
class GATV2_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self,num_features, num_classes):
                super().__init__()
                self.conv1 = GATv2Conv(num_features, 10)
                self.conv2 = GATv2Conv(10, 10)
                self.lin1 = Linear(10, num_classes)

            def forward(self,x,edge_index,edge_mask=None):
                x = F.relu(self.conv1(x.float(), edge_index, edge_mask))
                x = F.relu(self.conv2(x, edge_index,edge_mask))
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)

            def forward_single(self, x, edge_index, node_idx, edge_mask=None):
                x = F.relu(self.conv1(x, edge_index, edge_mask))
                x = F.relu(self.conv2(x, edge_index, edge_mask))
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)[node_idx]
            

        self.model = Net(self.dataset.num_features,self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.005)
        
        self.loader = DataLoader(self.dataset, batch_size=128)
        
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x.float(), data.edge_index)
            loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach() * data.num_graphs
        return total_loss.item() / sum(self.dataset.data.train_mask)
    

    @torch.no_grad()
    def test(self, loader, mask):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)
            total_correct += int((out.argmax(-1)[mask] == data.y[mask]).sum())
            
            loss = F.nll_loss(out[mask], data.y[mask])
            total_loss += loss.detach() * data.num_graphs            
        return total_correct / sum(mask), total_loss.item() / sum(mask)
    

    def iterate(self, num_epochs=100):
        for epoch in range(1, num_epochs+1):
            loss = self.train()
            train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
            test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
            if epoch % 50 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):
        train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
        test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')


#%%
class GAT_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self,num_features, num_classes):
                super().__init__()
                self.conv1 = GATConv(num_features, 30)
                self.conv2 = GATConv(30, 30)
                self.lin1 = Linear(30, num_classes)

            def forward(self,x,edge_index,edge_mask=None):
                x = F.relu(self.conv1(x.float(), edge_index, edge_mask))
                x = F.relu(self.conv2(x, edge_index,edge_mask))
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)
            
            def forward_single(self, x, edge_index, node_idx, edge_mask=None):
                x = F.relu(self.conv1(x, edge_index, edge_mask))
                x = F.relu(self.conv2(x, edge_index, edge_mask))
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)[node_idx]
            

        self.model = Net(self.dataset.num_features,self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.005)
        
        self.loader = DataLoader(self.dataset, batch_size=128)
        
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x.float(), data.edge_index)
            loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach() * data.num_graphs
        return total_loss.item() / sum(self.dataset.data.train_mask)
    

    @torch.no_grad()
    def test(self, loader, mask):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)
            total_correct += int((out.argmax(-1)[mask] == data.y[mask]).sum())
            
            loss = F.nll_loss(out[mask], data.y[mask])
            total_loss += loss.detach() * data.num_graphs            
        return total_correct / sum(mask), total_loss.item() / sum(mask)
    

    def iterate(self, num_epochs=100):
        for epoch in range(1, num_epochs+1):
            loss = self.train()
            train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
            test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
            if epoch % 50 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):
        train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
        test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')


#%%
class GIN_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self,num_features, num_classes):
                super().__init__()
                self.mlp1 = torch.nn.Linear(num_features, 30)
                self.conv1 = GINConv(self.mlp1)
                self.mlp2 = torch.nn.Linear(30, 30)
                self.conv2 = GINConv(self.mlp2)
                self.lin1 = torch.nn.Linear(30, num_classes)

            def forward(self,x,edge_index,edge_mask=None):
                x = F.relu(self.conv1(x.float(), edge_index, edge_mask))
                x = F.relu(self.conv2(x, edge_index, edge_mask))
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)

            def forward_single(self, x, edge_index, node_idx, edge_mask=None):
                x = F.relu(self.conv1(x.float(), edge_index, edge_mask))
                x = F.relu(self.conv2(x, edge_index, edge_mask))
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)[node_idx]
            

        self.model = Net(self.dataset.num_features,self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        
        self.loader = DataLoader(self.dataset, batch_size=128)
        
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x.float(), data.edge_index)
            loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach() * data.num_graphs
        return total_loss.item() / sum(self.dataset.data.train_mask)
    

    @torch.no_grad()
    def test(self, loader, mask):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)
            total_correct += int((out.argmax(-1)[mask] == data.y[mask]).sum())
            
            loss = F.nll_loss(out[mask], data.y[mask])
            total_loss += loss.detach() * data.num_graphs            
        return total_correct / sum(mask), total_loss.item() / sum(mask)
    

    def iterate(self, num_epochs=100):
        for epoch in range(1, num_epochs+1):
            loss = self.train()
            train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
            test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
            if epoch % 50 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):
        train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
        test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')


#%%
class CHEB_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes, hidden=30):
                super().__init__()
                self.conv1 = ChebConv(num_features, hidden, K=5)
                self.conv2 = ChebConv(hidden, hidden, K=5)
                #self.conv3 = GINConv(self.mlp3)
                self.lin1 = torch.nn.Linear(hidden, num_classes)

            def forward(self, x, edge_index, edge_mask=None):
                x = F.relu(self.conv1(x.float(), edge_index, edge_mask))
                x = F.relu(self.conv2(x.float(), edge_index, edge_mask))
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)


            def forward_single(self, x, edge_index, node_idx, edge_mask=None):
                x = F.relu(self.conv1(x.float(), edge_index, edge_mask))
                x = F.relu(self.conv2(x.float(), edge_index, edge_mask))
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)[node_idx]


        self.model = Net(self.dataset.num_features,self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0005)
        
        self.loader = DataLoader(self.dataset, batch_size=128)
        
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x.float(), data.edge_index)
            loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach() * data.num_graphs
        return total_loss.item() / sum(self.dataset.data.train_mask)
    

    @torch.no_grad()
    def test(self, loader, mask):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)
            total_correct += int((out.argmax(-1)[mask] == data.y[mask]).sum())
            
            loss = F.nll_loss(out[mask], data.y[mask])
            total_loss += loss.detach() * data.num_graphs            
        return total_correct / sum(mask), total_loss.item() / sum(mask)
    

    def iterate(self, num_epochs=100):
        for epoch in range(1, num_epochs+1):
            loss = self.train()
            train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
            test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
            if epoch % 50 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):
        train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
        test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')