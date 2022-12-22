import numpy as np
import torch
from torch.utils.data import DataLoader


def get_classifier_features(encoder, classifier, images, fw_layers=1):
    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        if fw_layers == 1:
            zs = encoder(images)
        elif fw_layers == 2:
            zs, zs2 = encoder(images)
        elif fw_layers == 3:
            zs, zs2, zs3 = encoder(images)
        elif fw_layers == 4:
            zs, zs2, zs3, zs4 = encoder(images)
        elif fw_layers == 5:
            zs, zs2, zs3, zs4, zs5 = encoder(images)
        else:
            raise NotImplementedError
        y_hat = classifier(zs)
        y_hat = torch.argmax(y_hat, dim=1)

    if fw_layers == 1:
        return zs, y_hat
    elif fw_layers == 2:
        return [zs, zs2], y_hat
    elif fw_layers == 3:
        return [zs, zs2, zs3], y_hat
    elif fw_layers == 4:
        return [zs, zs2, zs3, zs4], y_hat
    elif fw_layers == 5:
        return [zs, zs2, zs3, zs4, zs5], y_hat


def get_probability_and_accuracy(encoder, classifier, dataset, batch_size, device, fw_layers=1, n_samples=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoder.eval()
    classifier.eval()

    if not n_samples:
        n_samples = len(dataset)

    accuracy = []
    softmax_p = []

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            if fw_layers == 2:
                zs, _ = encoder(images)
            elif fw_layers == 3:
                zs, _, _ = encoder(images)
            elif fw_layers == 4:
                zs, _, _, _ = encoder(images)
            elif fw_layers == 5:
                zs, _, _, _, _ = encoder(images)
            else:
                zs = encoder(images)
            outputs = classifier(zs)
            correct = (torch.argmax(outputs, dim=1) == labels).float()
            probability = torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1).values
            accuracy.append(correct.cpu().numpy())
            softmax_p.append(probability.cpu().numpy())

        if (i * batch_size + batch_size) >= n_samples:
            break

    if len(accuracy) > 1:
        accuracy = np.concatenate(accuracy, axis=0)
        softmax_p = np.concatenate(softmax_p, axis=0)
    else:
        accuracy = np.array(accuracy).squeeze()
        softmax_p = np.array(softmax_p).squeeze()

    return accuracy, softmax_p


def get_predictions(encoder, classifier, dataset, batch_size, device, fw_layers=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoder.eval()
    classifier.eval()

    prediction = []

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)

        _, ys = get_classifier_features(encoder, classifier, images, fw_layers)
        prediction.append(ys.cpu().numpy())

    if len(prediction) > 1:
        prediction = np.concatenate(prediction, axis=0)
    else:
        prediction = np.array(prediction).squeeze()

    return prediction


def get_latent_representations(encoder, classifier, dataset, batch_size, device, fw_layers=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoder.eval()
    classifier.eval()

    features = []

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)

        zs, _ = get_classifier_features(encoder, classifier, images, fw_layers)
        if fw_layers > 1:
            features.append(zs[0].cpu().numpy())
        elif fw_layers == 1:
            features.append(zs.cpu().numpy())

    if len(features) > 1:
        features = np.concatenate(features, axis=0)
    else:
        features = np.array(features).squeeze()

    return features


def get_features(encoder, classifier, dataset, batch_size, device, fw_layers=1, t_layer=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoder.eval()
    classifier.eval()

    if t_layer > fw_layers:
        print('Target layer can not be larger than number of forwarded layers.')
        raise NotImplementedError

    features = []
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)

        zs, _ = get_classifier_features(encoder, classifier, images, fw_layers)
        features.append(zs[t_layer-1].cpu().numpy().reshape(zs[t_layer-1].shape[0], -1))
    if len(features) > 1:
        features = np.concatenate(features, axis=0)
    else:
        features = np.array(features).squeeze()

    return features


def get_norm_values(encoder, classifier, dataset, batch_size, device, fw_layers=1):
    min_val = 0
    max_val = 0
    encoder.eval()
    classifier.eval()
    min_vals = np.zeros(fw_layers)
    max_vals = np.zeros(fw_layers)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)

        zs, ys = get_classifier_features(encoder, classifier, images, fw_layers)

        min_vals[0], max_vals[0] = zs[0].cpu().numpy().min(), zs[0].cpu().numpy().max()
        if fw_layers >= 2:
            min_vals[1], max_vals[1] = zs[1].cpu().numpy().min(), zs[1].cpu().numpy().max()
        if fw_layers >= 3:
            min_vals[2], max_vals[2] = zs[2].cpu().numpy().min(), zs[2].cpu().numpy().max()
        if fw_layers >= 4:
            min_vals[3], max_vals[3] = zs[3].cpu().numpy().min(), zs[3].cpu().numpy().max()
        if fw_layers == 5:
            min_vals[4], max_vals[4] = zs[4].cpu().numpy().min(), zs[4].cpu().numpy().max()

        curr_min = min_vals.min()
        curr_max = max_vals.max()
        min_val = min(min_val, curr_min)
        max_val = max(max_val, curr_max)

    return min_val, max_val