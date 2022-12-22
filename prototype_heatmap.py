import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import euclidean_distances


class Prototype_Heatmap(Dataset):
    def __init__(self, id_data, ood_data, encoder, classifier, device, num_classes, num_ood, fw_layers=1):
        self.encoder = encoder
        self.classifier = classifier
        self.device = device
        self.num_classes = num_classes
        self.num_ood = num_ood
        self.encoder.eval()
        self.classifier.eval()
        self.id_data = id_data
        self.ood_data = ood_data
        self.num_id = len(self.id_data)
        self.len = self.num_id + self.num_ood
        self.fw_layers = fw_layers

        id_zs, id_ys = self.get_features_and_labels(id_data)
        ood_zs, ood_ys = self.get_features_and_labels(ood_data)
        self.ood_idx = np.arange(len(ood_data))
        self.id_idx = np.array(len(id_data))
        self.selected_idx = sorted(np.random.choice(self.ood_idx, num_ood, replace=False))
        self.prototype_idx = self.find_id_prototype(id_zs, ood_zs[self.selected_idx], id_ys, ood_ys[self.selected_idx])

    def get_features_and_labels(self, data):
        features = []
        predictions = []
        dataloader = DataLoader(data, batch_size=200, shuffle=False)
        for i, (images, _) in enumerate(dataloader):
            images = images.to(self.device)
            with torch.no_grad():
                zs = self.encoder(images)
                if self.fw_layers > 1:
                    zs = zs[0]
                y_hat = self.classifier(zs)
                y_hat = torch.argmax(y_hat, dim=1)
                features.append(zs.cpu().numpy())
                predictions.append(y_hat.cpu().numpy())
        return np.concatenate(features), np.concatenate(predictions)

    def find_id_prototype(self, zs_id, zs_ood, ys_id, ys_ood):
        proto_idx = []
        ood_idx = np.arange(self.num_ood)
        id_idx = np.arange(zs_id.shape[0])
        for cls_nr in range(self.num_classes):
            id_mask = ys_id == cls_nr
            ood_mask = ys_ood == cls_nr
            zs_id_masked = zs_id[id_mask]
            id_idx_masked = id_idx[id_mask]
            idx = ood_idx[ood_mask]

            for j in list(idx):
                item = zs_ood[j]
                dists = euclidean_distances(item.reshape(1, -1), zs_id_masked)
                t_idx = np.argsort(dists).squeeze()[0]
                proto_idx.append((id_idx_masked[t_idx], self.selected_idx[j]))
        return proto_idx

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if item < self.num_id:
            img = self.id_data[item][0]
            proto = self.id_data[item][0]
        else:
            ood_idx = self.selected_idx[item - self.num_id]
            img = self.ood_data[ood_idx][0]
            proto = self.id_data[self.prototype_idx[item - self.num_id][0]][0]
        return img, proto
