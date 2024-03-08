from math import *

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import RGCNConv
from torch.nn import Sequential,Linear,ReLU
import torch.nn.functional as F

from utilis import get_factor

# -------------------------------------------------------------------


class gbf_expansion(nn.Module):
    def __init__(self, gbf):
        super().__init__()
        self.min = gbf["dmin"]
        self.max = gbf["dmax"]
        self.steps = gbf["steps"]
        self.gamma = (self.max - self.min) / self.steps
        self.register_buffer(
            "filters", torch.linspace(self.min, self.max, self.steps)
        )

    def forward(self, data: torch.Tensor, bond=True) -> torch.Tensor:
        if bond:
            return torch.exp(
                -((data.unsqueeze(2) - self.filters) ** 2) / self.gamma**2
            )
        else:
            return torch.exp(
                -((data.unsqueeze(3) - self.filters) ** 2) / self.gamma**2
            )


# -------------------------------------------------------------------


class legendre_expansion(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.l = l

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        P0 = torch.clone(data)
        P0[:, :, :] = 1
        P1 = data
        if self.l == 0:
            return P0.unsqueeze(3) * get_factor(0)
        if self.l == 1:
            return torch.stack([P0, P1 * get_factor(1)], dim=3)
        else:
            factors = [get_factor(0), get_factor(1)]
            retvars = [P0, P1]
            for i in range(2, self.l):
                P = (1 / (i + 1)) * (
                    (2 * i + 1) * data * retvars[i - 1] - i * retvars[i - 2]
                )
                retvars.append(P)
                factors.append(get_factor(i))

        retvars = [var * factors[i] for i, var in enumerate(retvars)]
        return torch.stack(retvars, dim=3)


# -------------------------------------------------------------------


def Concat(atom_feature, bond_feature, nbr_idx):

    N, M = nbr_idx.shape
    _, O = atom_feature.shape
    _, _, P = bond_feature.shape

    index = nbr_idx.unsqueeze(1).expand(N, M, M)
    xk = atom_feature[index, :]
    xj = atom_feature[nbr_idx, :]
    xi = atom_feature.unsqueeze(1).expand(N, M, O)
    xij = torch.cat([xi, xj], dim=2)
    xij = xij.unsqueeze(2).expand(N, M, M, 2 * O)
    xijk = torch.cat([xij, xk], dim=3)

    eij = bond_feature.unsqueeze(2).expand(N, M, M, P)
    eik = bond_feature[nbr_idx, :]
    eijk = torch.cat([eij, eik], dim=3)

    return torch.cat([xijk, eijk], dim=3)


# -------------------------------------------------------------------


class ConvAngle(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(
        self,
        edge_fea_len,
        angle_fea_len,
    ):

        super(ConvAngle, self).__init__()

        self.angle_fea_len = angle_fea_len
        self.edge_fea_len = edge_fea_len

        # -------------------Angle----------------------

        self.lin_angle = nn.Linear(
            self.angle_fea_len + 2 * self.edge_fea_len, self.angle_fea_len
        )

        # ---------------Angle 3body attention------------------------

        self.attention_1 = nn.Linear(
            self.angle_fea_len + 2 * self.edge_fea_len, 1
        )
        self.leakyrelu_1 = nn.LeakyReLU(negative_slope=0.01)
        self.bn_ijkl = nn.LayerNorm(self.angle_fea_len + 2 * self.edge_fea_len)
        self.bn_ijkl = nn.LayerNorm(self.angle_fea_len)
        self.softplus_1 = nn.Softplus()

        self.bn_2 = nn.LayerNorm(self.angle_fea_len)
        self.softplus_2 = nn.Softplus()

    def forward(self, angle_fea, edge_fea, nbr_idx):

        N, M, O, P = angle_fea.shape

        # ---------------Edge update--------------------------

        eij = edge_fea.unsqueeze(2).expand(N, M, M, P)
        eik = edge_fea[nbr_idx, :]
        eijk = torch.cat([eij, eik], dim=3)

        angle_fea_cat = torch.cat([eijk, angle_fea], dim=3)

        attention_1 = self.attention_1(angle_fea_cat)
        alpha_1 = self.leakyrelu_1(attention_1)
        angle_fea_cat = alpha_1 * self.lin_angle(angle_fea_cat)

        angle_fea_summed = angle_fea_cat
        angle_fea_summed = angle_fea + angle_fea_summed
        angle_fea_summed = self.bn_2(angle_fea_summed)
        angle_fea_summed = self.softplus_2(angle_fea_summed)

        return angle_fea_summed


# -------------------------------------------------------------------


class ConvEdge(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, edge_fea_len, angle_fea_len):

        super(ConvEdge, self).__init__()

        self.edge_fea_len = edge_fea_len
        self.angle_fea_len = angle_fea_len

        # -------------------Angle----------------------
        self.lin_edge = nn.Linear(
            2 * self.edge_fea_len + self.angle_fea_len, self.edge_fea_len
        )

        # ---------------edege attention------------------------

        self.attention_1 = nn.Linear(
            2 * self.edge_fea_len + self.angle_fea_len, 1
        )
        self.leakyrelu_1 = nn.LeakyReLU(negative_slope=0.01)
        self.softmax_1 = nn.Softmax(dim=2)
        self.bn_1 = nn.LayerNorm(self.edge_fea_len)
        self.softplus_1 = nn.Softplus()

        self.bn_2 = nn.LayerNorm(self.edge_fea_len)
        self.softplus_2 = nn.Softplus()

    def forward(self, edge_fea, angle_fea, nbr_idx):

        N, M = nbr_idx.shape

        eij = edge_fea.unsqueeze(2).expand(N, M, M, self.edge_fea_len)

        eik = edge_fea[nbr_idx, :]

        edge_fea_cat = torch.cat([eij, eik, angle_fea], dim=3)

        attention_1 = self.attention_1(edge_fea_cat)
        alpha_1 = self.softmax_1(self.leakyrelu_1(attention_1))

        edge_fea_cat = alpha_1 * self.lin_edge(edge_fea_cat)

        edge_fea_cat = self.bn_1(edge_fea_cat)
        edge_fea_cat = self.softplus_1(edge_fea_cat)

        edge_fea_summed = edge_fea + torch.sum(edge_fea_cat, dim=2)

        edge_fea_summed = self.bn_2(edge_fea_summed)
        edge_fea_summed = self.softplus_2(edge_fea_summed)

        return edge_fea_summed


# -------------------------------------------------------------------


class CEGAN(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total material properties.
    """

    def __init__(
        self,
        gbf_bond,
        gbf_angle,
        n_conv_edge=3,
        h_fea_edge=128,
        h_fea_angle=128,
        n_classification=2,
        pooling=False,
        embedding=False,
    ):

        super(CEGAN, self).__init__()

        self.bond_fea_len = gbf_bond["steps"]
        self.angle_fea_len = gbf_angle["steps"]
        self.gbf_bond = gbf_expansion(gbf_bond)
        self.gbf_angle = gbf_expansion(gbf_angle)
        self.pooling = pooling
        self.embedding = embedding

        self.EdgeConv = nn.ModuleList(
            [
                ConvEdge(self.bond_fea_len, self.angle_fea_len)
                for _ in range(n_conv_edge)
            ]
        )
        self.AngConv = nn.ModuleList(
            [
                ConvAngle(self.bond_fea_len, self.angle_fea_len)
                for _ in range(n_conv_edge - 1)
            ]
        )

        self.expandEdge = nn.Linear(self.bond_fea_len, h_fea_edge)
        self.expandAngle = nn.Linear(self.angle_fea_len, h_fea_angle)

        self.bn = nn.LayerNorm(h_fea_edge + h_fea_angle)
        self.conv_to_fc_softplus = nn.Softplus()

        self.out = nn.Linear(h_fea_edge + h_fea_angle, n_classification)

        self.dropout = nn.Dropout()

    def forward(self, data):

        # print(f'data: {data}')

        bond_fea, angle_fea, species, nbr_idx, crys_idx = data

        edge_fea = self.gbf_bond(bond_fea)
        angle_fea = self.gbf_angle(angle_fea, bond=False)

        # print(angle_fea.shape)

        edge_fea = self.EdgeConv[0](edge_fea, angle_fea, nbr_idx)

        for conv_edge, conv_angle in zip(self.EdgeConv[1:], self.AngConv):
            angle_fea = conv_angle(angle_fea, edge_fea, nbr_idx)
            edge_fea = conv_edge(edge_fea, angle_fea, nbr_idx)

        edge_fea = self.expandEdge(self.dropout(edge_fea))
        angle_fea = self.expandAngle(self.dropout(angle_fea))

        edge_fea = torch.sum(self.conv_to_fc_softplus(edge_fea), dim=1)

        # print(edge_fea.shape)
        # print(f'CEGAN_edge_fea.size(): {edge_fea.size()}')

        angle_fea = torch.sum(
            self.conv_to_fc_softplus(
                torch.sum(self.conv_to_fc_softplus(angle_fea), dim=2)
            ),
            dim=1,
        )
        # print(f'CEGAN_angle_fea.size(): {angle_fea.size()}')

        # print(angle_fea.shape)

        crys_fea = torch.cat([edge_fea, angle_fea], dim=1)
        # print(crys_fea.shape)
        # print(f'CEGAN_crys_fea.size(): {crys_fea.size()}')

        if self.pooling:
            # print(f'self.pooling is True, entered if block')
            crys_fea = self.pool(crys_fea, crys_idx)
            # print("pooled",crys_fea.shape)
        # print(f'after self.pooling CEGAN_crys_fea.size(): {crys_fea.size()}')

        crys_fea = self.conv_to_fc_softplus(self.bn(crys_fea))
        # print(f'after self.conv_to_fc_softplus CEGAN_crys_fea.size() {crys_fea.size()}')
        if self.embedding:
            embed = crys_fea
        # print(f'after if self.embedding CEGAN_crys_fea.size(): {crys_fea.size()}')

        crys_fea = self.dropout(crys_fea)
        out = self.out(crys_fea)
        # print(out.shape)
        # print(f'CEGAN_out.size(): {out.size()}')

        if self.embedding:
            return out, embed

        else:
            return out

    def pool(self, atom_fea, crys_idx):

        # print(crystal_atom_idx)

        summed_fea = [
            torch.mean(
                atom_fea[idx_map[0] : idx_map[1], :], dim=0, keepdim=True
            )
            for idx_map in crys_idx
        ]
        return torch.cat(summed_fea, dim=0)

class GIN(nn.Module):
    def __init__(
        self,
        gbf_bond,
        gbf_angle,
        n_classification,
        neigh,
        pooling,
        embedding,
        ):
        super(GIN, self).__init__()
        # keep the CEGAN bond and angle features
        self.pooling=pooling
        self.embedding=embedding
        self.bond_fea_len = gbf_bond["steps"]
        self.angle_fea_len = gbf_angle["steps"]
        # print(type(gbf_bond))
        # print(type(gbf_angle))
        self.gbf_bond = gbf_expansion(gbf_bond)
        self.gbf_angle = gbf_expansion(gbf_angle)
        # Define the neural network layer
        # edge_angle_fea_len = self.bond_fea_len + self.angle_fea_len
        # print(f'self.bond_fea_len: {self.bond_fea_len}')
        self.nn1_bond = nn.Sequential(
            nn.Linear(self.bond_fea_len*neigh, 32),
            nn.ReLU(),
            nn.Linear(32,32)
        )
        self.nn1_angle = nn.Sequential(
            nn.Linear(self.angle_fea_len*neigh*neigh, 32),
            nn.ReLU(),
            nn.Linear(32,32)
        )
        self.conv1_bond = GINConv(self.nn1_bond)
        self.conv1_angle = GINConv(self.nn1_angle)
        self.bn1_bond = nn.BatchNorm1d(32)
        self.bn1_angle = nn.BatchNorm1d(32)
        self.nn2_bond = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.nn2_angle = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.conv2_bond = GINConv(self.nn2_bond)
        self.conv2_angle = GINConv(self.nn2_angle)
        self.bn2_bond = nn.BatchNorm1d(32)
        self.bn2_angle = nn.BatchNorm1d(32)
        # self.bn = nn.LayerNorm(64)
        # self.conv_to_fc_softplus = nn.Softplus()

        self.fc1_bond = Linear(32, 16)
        self.fc1_angle = Linear(32, 16)
        self.fc2_bond = Linear(16, 8)
        self.fc2_angle = Linear(16, 8)
        dropout_prob = 0.5
        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(16, n_classification)
        # stop here for now...
    def forward(self, data):
        bond_fea, angle_fea, species, nbr_idx, crys_idx = data
        # print(f'model.py bond_fea: {bond_fea}')
        # print(f'model.py angle_fea: {angle_fea}')
        # print(f'model.py nbr_idx: {nbr_idx}')
        # print(f'model.py crys_idx: {crys_idx}')
        edge_index = get_edge_index(nbr_idx)
        # print(f'edge_index: {edge_index}')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        edge_index = edge_index.to(device)

        # try printing the nbr_idx to see what we can learn
        # print(f'nbr_idx: {nbr_idx}')
        edge_fea = self.gbf_bond(bond_fea)
        angle_fea = self.gbf_angle(angle_fea, bond = False)
        # print(f'angle_fea.size(): {angle_fea.size()}')
        angle_fea_2d = angle_fea.view(angle_fea.size()[0], -1)

        edge_fea_2d = edge_fea.view(edge_fea.size()[0], -1)
        # print(f'edge_fea_2d.size(): {edge_fea_2d.size()}')
        # print(f'edge_index.size(): {edge_index.size()}')
        bond_fea_out = F.relu(self.conv1_bond(edge_fea_2d, edge_index))

        angle_fea_out = F.relu(self.conv1_angle(angle_fea_2d, edge_index))
        # Concatenate the output of nn1 for bond_fea and angle_fea along the feature dimension
        # combined_fea = torch.cat([bond_fea_out, angle_fea_out], dim=1)

        bond_fea_out = self.bn1_bond(bond_fea_out)
        angle_fea_out = self.bn1_angle(angle_fea_out)
        bond_fea_out = F.relu(self.conv2_bond(bond_fea_out, edge_index))
        angle_fea_out = F.relu(self.conv2_angle(angle_fea_out, edge_index))
        bond_fea_out = self.bn2_bond(bond_fea_out)
        angle_fea_out = self.bn2_angle(angle_fea_out)

        bond_fea_out = self.fc1_bond(bond_fea_out)
        angle_fea_out = self.fc1_angle(angle_fea_out)
        bond_fea_out = self.fc2_bond(bond_fea_out)
        angle_fea_out = self.fc2_angle(angle_fea_out)

        crys_fea = torch.cat([bond_fea_out, angle_fea_out], dim=1)
        crys_fea = self.dropout_layer(crys_fea)

        if self.pooling:
            crys_fea = self.pool(crys_fea, crys_idx)

        if self.embedding:
            embed = crys_fea

        out = self.fc(crys_fea)

        if self.embedding:
            return out, embed
        else:
            return out
        # return out

    def pool(self, atom_fea, crys_idx):

        summed_fea = [
            torch.mean(
                atom_fea[idx_map[0] : idx_map[1], :], dim=0, keepdim=True
            )
            for idx_map in crys_idx
        ]
        return torch.cat(summed_fea, dim=0)

class mySAGE(nn.Module):
    def __init__(
        self,
        gbf_bond,
        gbf_angle,
        n_classification,
        neigh,
        pool,
        embedding,
        ):
        super(mySAGE, self).__init__()
        self.pool = pool
        self.embedding=embedding
        self.bond_fea_len = gbf_bond["steps"]
        self.angle_fea_len = gbf_angle["steps"]
        self.gbf_bond = gbf_expansion(gbf_bond)
        self.gbf_angle = gbf_expansion(gbf_angle)
        hidden_size = 128
        # add one SAGEConv layer
        self.conv1_bond = SAGEConv(
            self.bond_fea_len*neigh, hidden_size
        )
        self.conv1_angle = SAGEConv(
            self.angle_fea_len*neigh*neigh, hidden_size
        )
        # end add one SAGEConv layer

        # add a second SAGEConv layer
        self.conv2_bond = SAGEConv(
            hidden_size, hidden_size
        )
        self.conv2_angle = SAGEConv(
            hidden_size, hidden_size
        )

        # end add a second SAGEConv layer

        self.fc = nn.Linear(2*hidden_size, n_classification)

    def forward(self, data):
        bond_fea, angle_fea, species, nbr_idx, crys_idx = data
        edge_index = get_edge_index(nbr_idx)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        edge_index = edge_index.to(device)
        # Guassian expansion of bond and angle features
        edge_fea = self.gbf_bond(bond_fea)
        angle_fea = self.gbf_angle(angle_fea, bond = False)
        # re-shape to 2D for input to Graph SAGE layers
        edge_fea_2d = edge_fea.view(edge_fea.size()[0], -1)
        angle_fea_2d = angle_fea.view(angle_fea.size()[0], -1)
        # start 1st Graph SAGE layer on bond and angles
        bond_fea_out = self.conv1_bond(edge_fea_2d, edge_index)
        angle_fea_out = self.conv1_angle(angle_fea_2d, edge_index)
        bond_fea_out = F.relu(bond_fea_out)
        angle_fea_out = F.relu(angle_fea_out)
        bond_fea_out = F.dropout(bond_fea_out, p=0.2)
        angle_fea_out = F.dropout(angle_fea_out, p=0.2)
        # end 1st Graph SAGE layer on bond and angles
        # 2nd SAGE layer
        bond_fea_out = self.conv2_bond(bond_fea_out, edge_index)
        angle_fea_out = self.conv2_angle(angle_fea_out, edge_index)
        bond_fea_out = F.relu(bond_fea_out)
        angle_fea_out = F.relu(angle_fea_out)
        bond_fea_out = F.dropout(bond_fea_out, p=0.2)
        angle_fea_out = F.dropout(angle_fea_out, p=0.2)
        # end 2nd SAGE layer

        crys_fea = torch.cat([bond_fea_out, angle_fea_out], dim=1)

        if self.pool:
            crys_fea = self.pool_function(crys_fea, crys_idx)

        if self.embedding:
            embed = crys_fea

        out = self.fc(crys_fea)

        if self.embedding:
            return out, embed
        else:
            return out


    def pool_function(self, atom_fea, crys_idx):

        summed_fea = [
            torch.mean(
                atom_fea[idx_map[0] : idx_map[1], :], dim=0, keepdim=True
            )
            for idx_map in crys_idx
        ]
        return torch.cat(summed_fea, dim=0)

class myRGCN(nn.Module):
    def __init__(
        self,
        gbf_bond,
        gbf_angle,
        n_classification,
        neigh,
        pool,
        embedding,
    ):
        super(myRGCN, self).__init__()
        self.pool = pool
        self.bond_fea_len = gbf_bond["steps"]
        self.angle_fea_len = gbf_angle["steps"]
        self.gbf_bond = gbf_expansion(gbf_bond)
        self.gbf_angle = gbf_expansion(gbf_angle)
        self.embedding = embedding

        hidden_size = 64

        self.conv1_bond = RGCNConv(self.bond_fea_len*neigh, hidden_size, 3)
        self.conv1_angle = RGCNConv(self.angle_fea_len*neigh*neigh, hidden_size, 3)

        self.conv2_bond = RGCNConv(hidden_size, hidden_size, 3)
        self.conv2_angle = RGCNConv(hidden_size, hidden_size, 3)

        self.fc = nn.Linear(2*hidden_size, n_classification)

    def forward(self, data):
        bond_fea, angle_fea, species, nbr_idx, crys_idx = data
        edge_index = get_edge_index(nbr_idx)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        edge_index = edge_index.to(device)
        edge_fea = self.gbf_bond(bond_fea)
        angle_fea = self.gbf_angle(angle_fea, bond = False)
        # num1, num2 = edge_index.shape
        # print(f'num1 = {num1}, num2 = {num2}')
        edge_type = get_edge_type(edge_index, species)
        edge_type = edge_type.to(device)

        # print(f'edge_fea.shape: {edge_fea.shape}, edge_index.shape: {edge_index.shape}, edge_type.shape {edge_type.shape}')
        edge_fea_2d = edge_fea.view(edge_fea.size()[0], -1)
        angle_fea_2d = angle_fea.view(angle_fea.size()[0], -1)

        bond_fea_out = self.conv1_bond(edge_fea_2d, edge_index, edge_type)
        angle_fea_out = self.conv1_angle(angle_fea_2d, edge_index, edge_type)
        bond_fea_out = F.relu(bond_fea_out)
        angle_fea_out = F.relu(angle_fea_out)
        bond_fea_out = F.dropout(bond_fea_out, p=0.2)
        angle_fea_out = F.dropout(angle_fea_out, p=0.2)
        
        bond_fea_out = self.conv2_bond(bond_fea_out, edge_index, edge_type)
        angle_fea_out = self.conv2_angle(angle_fea_out, edge_index, edge_type)
        bond_fea_out = F.relu(bond_fea_out)
        angle_fea_out = F.relu(angle_fea_out)
        bond_fea_out = F.dropout(bond_fea_out, p=0.2)
        angle_fea_out = F.dropout(angle_fea_out, p=0.2)


        crys_fea = torch.cat([bond_fea_out, angle_fea_out], dim=1)

        if self.pool:
            crys_fea = self.pool_function(crys_fea, crys_idx)

        if self.embedding:
            embed = crys_fea

        out = self.fc(crys_fea)

        if self.embedding:
            return out, embed
        else:
            return out

        # return crys_fea

    def pool_function(self, atom_fea, crys_idx):

        summed_fea = [
            torch.mean(
                atom_fea[idx_map[0] : idx_map[1], :], dim=0, keepdim=True
            )
            for idx_map in crys_idx
        ]
        return torch.cat(summed_fea, dim=0)


def get_edge_type(local_edge_index, species):
    # num1, num2 = local_edge_index.shape
    # print(f'num1 = {num1}, num2 = {num2}')
    num1, num_edges = local_edge_index.shape
    edge_type = torch.zeros([num_edges], dtype=torch.int64)
    for i in range(num_edges):
        atom1 = local_edge_index[0][i]
        atom2 = local_edge_index[1][i]
        # Cu-Cu: 0-0: edge type 0
        # Cu-Zr: 0-1 or 1-0: edge type 1
        # Zr-Zr: 1-1: edge type 2
        if species[atom1] == 0.0 and species[atom2] == 0.0:
            edge_type[i] = 0
        if species[atom1] == 0.0 and species[atom2] == 1.0:
            edge_type[i] = 1
        if species[atom1] == 1.0 and species[atom2] == 0.0:
            edge_type[i] = 1
        if species[atom1] == 1.0 and species[atom2] == 1.0:
            edge_type[i] = 2
    return edge_type

def get_edge_index(neighbor_list):
    # determine if self.nbr has scope inside this subroutine
    # print(f'neighbor_list: {neighbor_list}')
    # edge_index = torch.tensor()
    length, width = neighbor_list.shape
    # print(f'length = {length}')
    # print(f'width = {width}')
    edge_index = torch.zeros([2, length*width], dtype=torch.int64)
    # print(f'edge_index.shape: {edge_index.shape}')
    pointer = 0
    for i in range(length):
        source = i
        for j in range(width):
            dest = neighbor_list[i][j]
            # edge_index[pointer][0] = source
            # edge_index[pointer][1] = dest
            edge_index[0][pointer] = source
            edge_index[1][pointer] = dest
            pointer += 1
    # print(f'edge_index: {edge_index}')
    return edge_index # dummy return statement until we can get it working