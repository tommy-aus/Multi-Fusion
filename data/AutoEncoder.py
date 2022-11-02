import scipy.io
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
import pandas as pd


data_set = scipy.io.loadmat('Liu_dataset.mat')
side_effect = data_set['side_effect']
chemical = data_set['chemical']
Targets = data_set['Targets']
Transporters = data_set['Transporters']
Enzymes = data_set['Enzymes']
Pathways = data_set['Pathways']
Treatment = data_set['Treatment']
Other_side_effects = data_set['Other_side_effects']

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)  #881->2048
        self.enc_2 = Linear(n_enc_1, n_enc_2)  #2048->1024
        self.z_layer = Linear(n_enc_2, n_z)    #1024->65

        self.dec_1 = Linear(n_z,n_enc_2)  #65->1024
        self.dec_2 = Linear(n_enc_2, n_enc_1) #1024->2048
        self.x_bar_layer = Linear(n_enc_1, n_input)  # 2048->881

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x)) #2048
        enc_h2 = F.relu(self.enc_2(enc_h1))
        # enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h2)
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        # dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h2)
        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print("pretrain_ae_model:",model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(30):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            # x = x.cuda()
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))

        if epoch == 29:
            X_Z= np.array(z.detach().numpy())
            print (X_Z.shape)
            np.savetxt('z_class572.csv',X_Z,delimiter=',')

        torch.save(model.state_dict(), 'pre_ae_1716_2000_256_65.pkl')

def cal_similarity(data):
    data = data.transpose()
    import pandas as pd
    df = pd.DataFrame(data)
    from sklearn.metrics.pairwise import pairwise_distances
    similarity_matrix = 1-pairwise_distances(df.T, metric = "hamming")
    similarity_matrix_df = pd.DataFrame(similarity_matrix)
    normalized_similarity_matrix_df=(similarity_matrix_df-similarity_matrix_df.min())/(similarity_matrix_df.max()-similarity_matrix_df.min())
    return normalized_similarity_matrix_df

chemical_similarity = cal_similarity(chemical)
chemical_similarity.to_csv("chemical_similarity.csv")
chemical_similarity_numpy = chemical_similarity.to_numpy()

Targets_similarity = cal_similarity(Targets)
Targets_similarity.to_csv("Targets_similarity.csv")
Targets_similarity_numpy = Targets_similarity.to_numpy()

Transporters_similarity = cal_similarity(Transporters)
Transporters_similarity.to_csv("Transporters_similarity.csv")
Transporters_similarity_numpy = Transporters_similarity.to_numpy()

Enzymes_similarity = cal_similarity(Enzymes)
Enzymes_similarity.to_csv("Enzymes_similarity.csv")
Enzymes_similarity_numpy = Enzymes_similarity.to_numpy()

Pathways_similarity = cal_similarity(Pathways)
Pathways_similarity.to_csv("Pathways_similarity.csv")
Pathways_similarity_numpy = Pathways_similarity.to_numpy()

Treatment_similarity = cal_similarity(Treatment)
Treatment_similarity.to_csv("Treatment_similarity.csv")
Treatment_similarity_numpy = Treatment_similarity.to_numpy()

Other_side_effects_similarity = cal_similarity(Other_side_effects)
Other_side_effects_similarity.to_csv("Other_side_effects_similarity.csv")
Other_side_effects_similarity_numpy = Other_side_effects_similarity.to_numpy()

con = pd.concat([chemical_similarity, Targets_similarity, Transporters_similarity, 
Enzymes_similarity, Pathways_similarity, Treatment_similarity, Other_side_effects_similarity], axis=1)


con.to_csv('con.csv')
con = con.to_numpy()

model = AE(
        n_enc_1=2048,
        n_enc_2=1024,
        n_input=5824,
        n_z=65
         )

x = con
print(x.shape)

dataset = LoadDataset(x)
pretrain_ae(model, dataset)