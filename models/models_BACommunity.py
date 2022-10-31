# Node classification dataset
# BA + house
from torch_geometric.nn import GCN, GATConv, GATv2Conv, GINConv

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from sklearn.model_selection import train_test_split

from torch_geometric.nn import DenseGCNConv, GCNConv,global_max_pool,global_add_pool,global_mean_pool, GATConv, ChebConv
from torch_geometric.loader import DataLoader


# find . -type f -name '*.gpickle' -delete
# find . -type f | wc -l



class GCN_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self,num_features, num_classes):
                super().__init__()
                self.conv1 = GCNConv(num_features, 20)
                self.conv2 = GCNConv(20, 20)
                self.conv3 = GCNConv(20, 20)
                self.conv4 = GCNConv(30, 30)
                self.lin1 = Linear(20*3,num_classes)
                self.lin2 = Linear(10,num_classes)
                self.bn1 = BatchNorm1d(20)
                self.bn2 = BatchNorm1d(20)
                self.bn3 = BatchNorm1d(20)

            def forward(self,x,edge_index,edge_mask=None):
                x_all = []
                x = F.relu(self.conv1(x.float(), edge_index))
                x = self.bn1(x)
                x_all.append(x)
                x = F.relu(self.conv2(x, edge_index,edge_mask))
                x = self.bn2(x)
                x_all.append(x)
                x = F.relu(self.conv3(x, edge_index,edge_mask))
                #x = self.bn3(x)
                x_all.append(x)
                #x = F.relu(self.conv4(x, edge_index,edge_mask))
                #x = F.relu(self.conv3(x, edge_index,edge_mask))
                #x = self.conv4(x, edge_index,edge_mask)
                
                x = torch.cat(x_all, axis=1)
                x = self.lin1(x)
                #x = self.lin2(x)
                return F.log_softmax(x, dim=-1)


        class BA_Community_GCN(torch.nn.Module):
            def __init__(self, num_in_features, num_hidden_features, num_classes):
                super(BA_Community_GCN, self).__init__()
                self.conv0 = GCNConv(num_in_features, num_hidden_features)
                self.conv1 = GCNConv(num_hidden_features, num_hidden_features)
                self.conv2 = GCNConv(num_hidden_features, num_hidden_features)
                # self.conv3 = DenseGCNConv(num_hidden_features, num_hidden_features)
                # self.conv4 = DenseGCNConv(num_hidden_features, num_hidden_features)
                # self.conv5 = DenseGCNConv(num_hidden_features, num_hidden_features)
                # linear layers
                self.linear = torch.nn.Linear(num_hidden_features*3+10, num_classes)

            def forward(self, x, edge_index):
                #edge_index = to_dense_adj(edge_index)
                x_all = []
                x_all.append(x)
                x = self.conv0(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, 0.2)
                x_all.append(x)
                #x = F.dropout(x, 0.2)
                #x = torch.nn.functional.normalize(x, p=2, dim=1)
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, 0.2)
                x_all.append(x)
                
                x = self.conv2(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, 0.2)
                x_all.append(x)
                #x = F.dropout(x, 0.2)
                #x = torch.nn.functional.normalize(x, p=2, dim=1)
                #x = self.conv2(x, edge_index)
                #x = F.relu(x)
                #x = F.dropout(x, 0.2)
                #x = torch.nn.functional.normalize(x, p=2, dim=1)
                # x = self.conv3(x, edge_index)
                # x = F.relu(x)
                # x = F.dropout(x, 0.2)
                # x = self.conv4(x, edge_index)
                # x = F.relu(x)
                # x = F.dropout(x, 0.2)
                # x = self.conv5(x, edge_index)
                # x = F.relu(x)
                # x = F.dropout(x, 0.2)

                x = torch.cat(x_all, dim=1)
                x = self.linear(x)
                return F.log_softmax(x, dim=-1).squeeze()


            

        #self.model = Net(10, self.dataset.num_classes).to(self.device).float()
        self.model = BA_Community_GCN(10, 40, self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001,weight_decay=0.005)

        self.train_idx, self.test_idx = train_test_split(torch.arange(self.dataset.data.num_nodes), train_size=0.6, stratify=self.dataset.data.y,random_state=10)

        self.dataset.data.train_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.test_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.train_mask[self.train_idx] = True
        self.dataset.data.test_mask[self.test_idx] = True
        
        self.loader = DataLoader(self.dataset)
        
            
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
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 2.0)
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
        train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
        test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            





from torch_geometric.nn import SAGEConv
class GraphSAGE_framework:
    def __init__(self,dataset,device=None):   
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        class Net(torch.nn.Module):
            def __init__(self,num_features,num_classes):
                super().__init__()
                self.conv1 = SAGEConv(num_features, 30, aggr="sum")
                self.conv2 = SAGEConv(30, 30, aggr="sum")
                self.lin1 = Linear(30, num_classes)

            def forward(self,x,edge_index,edge_mask=None):
                x = F.relu(self.conv1(x, edge_index,edge_mask))
                x = F.relu(self.conv2(x, edge_index,edge_mask))

                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)
            
        self.dataset = dataset
        self.model = Net(10, self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)

        self.train_idx, self.test_idx = train_test_split(torch.arange(self.dataset.data.num_nodes), train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.dataset.data.train_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.test_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.train_mask[self.train_idx] = True
        self.dataset.data.test_mask[self.test_idx] = True
        
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
            if epoch % 20 == 0:
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
                self.conv2 = GATv2Conv(10, 20)
                self.conv3 = GATv2Conv(20, 10)
                self.lin1 = Linear(10, num_classes)
                self.lin2 = Linear(10, num_classes)

            def forward(self,x,edge_index,edge_mask=None):
                x = F.relu(self.conv1(x.float(), edge_index, edge_mask))
                x = F.relu(self.conv2(x, edge_index,edge_mask))
                x = F.relu(self.conv3(x, edge_index,edge_mask))
                #x = F.relu(self.conv3(x, edge_index,edge_mask))
                #x = self.conv4(x, edge_index,edge_mask)                
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)


            

        self.model = Net(10, self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0005)

        self.train_idx, self.test_idx = train_test_split(torch.arange(self.dataset.data.num_nodes), train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.dataset.data.train_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.test_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.train_mask[self.train_idx] = True
        self.dataset.data.test_mask[self.test_idx] = True
        
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
        train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
        test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')



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
                self.conv1 = GATv2Conv(num_features, 30)
                self.conv2 = GATv2Conv(30, 30)
                self.conv3 = GATv2Conv(30, 30)
                self.lin1 = Linear(30, 10)
                self.lin2 = Linear(10, num_classes)

            def forward(self,x,edge_index,edge_mask=None):
                x = F.relu(self.conv1(x.float(), edge_index, edge_mask))
                #x = F.relu(self.conv2(x, edge_index,edge_mask))
                #x = F.relu(self.conv3(x, edge_index,edge_mask))
                #x = F.relu(self.conv3(x, edge_index,edge_mask))
                #x = self.conv4(x, edge_index,edge_mask)                
                x = F.relu(self.lin1(x))
                x = self.lin2(x)
                return F.log_softmax(x, dim=-1)


            

        self.model = Net(10, self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0005)

        self.train_idx, self.test_idx = train_test_split(torch.arange(self.dataset.data.num_nodes), train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.dataset.data.train_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.test_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.train_mask[self.train_idx] = True
        self.dataset.data.test_mask[self.test_idx] = True
        
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
        train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
        test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')



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
                self.mlp1 = torch.nn.Linear(num_features, 70)
                self.conv1 = GINConv(self.mlp1)
                self.mlp2 = torch.nn.Linear(70, 70)
                self.conv2 = GINConv(self.mlp2)
                self.mlp3 = torch.nn.Linear(70, 70)
                self.conv3 = GINConv(self.mlp3)
                
                self.lin1 = torch.nn.Linear(70, num_classes)

            def forward(self,x,edge_index,edge_mask=None):
                x = F.relu(self.conv1(x.float(), edge_index, edge_mask))
                x = F.relu(self.conv2(x, edge_index,edge_mask))
                x = F.relu(self.conv3(x, edge_index,edge_mask))
                #x = F.relu(self.conv3(x, edge_index,edge_mask))
                #x = self.conv4(x, edge_index,edge_mask)                
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)


            

        self.model = Net(10, self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0005)

        self.train_idx, self.test_idx = train_test_split(torch.arange(self.dataset.data.num_nodes), train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.dataset.data.train_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.test_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.train_mask[self.train_idx] = True
        self.dataset.data.test_mask[self.test_idx] = True
        
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
        train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
        test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')



        



class CHEB_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self,num_features, num_classes, hidden=30):
                super().__init__()
                self.conv1 = ChebConv(num_features, hidden, K=5)
                self.conv2 = ChebConv(hidden, hidden, K=5)
                #self.conv3 = GINConv(self.mlp3)
                
                self.lin1 = torch.nn.Linear(hidden, num_classes)

            def forward(self,x,edge_index,edge_mask=None):
                print(edge_mask)
                x = F.relu(self.conv1(x.float(), edge_index, edge_mask))
                x = F.relu(self.conv2(x, edge_index,edge_mask))
                #x = F.relu(self.conv3(x, edge_index,edge_mask))
                #x = F.relu(self.conv3(x, edge_index,edge_mask))
                #x = self.conv4(x, edge_index,edge_mask)                
                x = self.lin1(x)
                return F.log_softmax(x, dim=-1)


            

        self.model = Net(10, self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0005)

        self.train_idx, self.test_idx = train_test_split(torch.arange(self.dataset.data.num_nodes), train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.dataset.data.train_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.test_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.train_mask[self.train_idx] = True
        self.dataset.data.test_mask[self.test_idx] = True
        
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
        train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
        test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
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
        self.train_idx, self.test_idx = train_test_split(torch.arange(self.dataset.data.num_nodes), train_size=0.8, stratify=self.dataset.data.y, random_state=10)

        self.dataset.data.train_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.test_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.train_mask[self.train_idx] = True
        self.dataset.data.test_mask[self.test_idx] = True
        
        self.loader = DenseDataLoader(self.dataset, batch_size=128)

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

            def forward(self, x, adj, mask=None,batch=None):

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

            
        self.model = Net(self.max_nodes,self.dataset).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)


    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x.float(), data.adj, data.mask)
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
        train_acc,train_loss = self.test(self.loader, mask=self.dataset.data.train_mask)
        test_acc,test_loss = self.test(self.loader, mask=self.dataset.data.test_mask)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            
