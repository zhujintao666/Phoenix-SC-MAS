## write a GCN based model for graph representation learning
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sc_graph import create_agent_profiles
from config import env_configs, get_env_configs
import os

gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(8, 4)
        self.layer2 = GCNLayer(4, 1)
        # self.layer1 = GCNLayer(1433, 16)
        # self.layer2 = GCNLayer(16, 7)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x
    
net = Net()


def load_sc_graph_data(agent_profiles, num_stage: int, num_agent_per_stage: int):
    
    node_feats = []
    edge_feats = []
    g = dgl.DGLGraph()
    node_labels = []
    edge_u, edge_v = [], []
    ndata = len(agent_profiles)
    for ap in agent_profiles:
        node_labels.append(ap.name)
        node_features = ap.get_node_features()
        node_feats.append(th.tensor(node_features, dtype=th.float32))  # Ensure features are float type
        
        stage = ap.stage
        agent = ap.agent
        if stage < num_stage - 1:            
            for dem, label in enumerate(ap.suppliers):
                if label == 1:
                    edge_u.append(stage*num_agent_per_stage+agent)
                    edge_v.append((stage-1)*num_agent_per_stage+dem)

    g.add_nodes(ndata)
    g.add_edges(edge_u, edge_v)
    g.ndata['feat'] = th.tensor(node_feats)
    return g, g.ndata['feat']



class LinkPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(LinkPredictor, self).__init__()
        self.layer1 = GCNLayer(in_feats, hidden_feats)
        self.layer2 = GCNLayer(hidden_feats, out_feats)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x

def compute_loss(pos_score, neg_score):
    scores = th.cat([pos_score, neg_score])
    labels = th.cat([th.ones(pos_score.shape[0]), th.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_scores(g, model, edges):
    with th.no_grad():
        g.ndata['h'] = model(g, g.ndata['feat'])
        g.apply_edges(fn.u_dot_v('h', 'h', 'score'), edges)
        return g.edata['score']

def train_link_predictor(g, features, pos_edges, neg_edges, epochs=50, lr=1e-2):
    model = LinkPredictor(features.shape[1], 16, 1)
    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        pos_score = compute_scores(g, model, pos_edges)
        neg_score = compute_scores(g, model, neg_edges)
        loss = compute_loss(pos_score, neg_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:05d} | Loss {loss.item():.4f}")


if __name__ == "__main__":

    env_config_name = "large_graph_test"
    os.makedirs(f"env/{env_config_name}", exist_ok=True)
    env_config = get_env_configs(env_configs=env_configs[env_config_name])
    print(env_config.keys())
    agent_profiles = create_agent_profiles(env_config)

    # Example usage with Cora dataset
    g, features = load_sc_graph_data(agent_profiles=agent_profiles, num_stage=env_config['num_stages'], num_agent_per_stage=env_config['num_agents_per_stage'])
    print("edges", g.edges())
    u, v = g.edges()
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    train_pos_u, train_pos_v = u[eids[:train_size]], v[eids[:train_size]]
    test_pos_u, test_pos_v = u[eids[train_size:]], v[eids[train_size:]]

    train_neg_u = np.random.choice(g.number_of_nodes(), train_size)
    train_neg_v = np.random.choice(g.number_of_nodes(), train_size)
    test_neg_u = np.random.choice(g.number_of_nodes(), test_size)
    test_neg_v = np.random.choice(g.number_of_nodes(), test_size)

    train_pos_edges = (train_pos_u, train_pos_v)
    train_neg_edges = (train_neg_u, train_neg_v)
    test_pos_edges = (test_pos_u, test_pos_v)
    test_neg_edges = (test_neg_u, test_neg_v)

    train_link_predictor(g, features, train_pos_edges, train_neg_edges)