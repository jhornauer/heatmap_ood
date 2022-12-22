import numpy as np
import torch
from torch.utils.data import DataLoader
from classifier_utils import get_classifier_features
from sklearn.metrics import roc_auc_score, average_precision_score


def test_classifier(net, cls, testloader, device):
    net.eval()

    running_acc = 0
    for i, (images, labels) in enumerate(testloader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = cls(net(images))

            running_acc += (torch.argmax(outputs, dim=1) == labels).float().sum().item()

    running_acc = running_acc / len(testloader.dataset)
    return running_acc


def test_generator(encoder, classifier, generator, testset, batch_size, min_val, max_val, device,
                   fw_layers=1, n_samples=None):
    encoder.eval()
    classifier.eval()
    generator.eval()

    if not n_samples:
        n_samples = len(testset)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    scores = []
    for i, (images, _) in enumerate(testloader):
        images = images.to(device)

        zs, ys = get_classifier_features(encoder, classifier, images, fw_layers)

        for l_idx in range(fw_layers):
            zs[l_idx] = (zs[l_idx] - min_val) / (max_val - min_val)

        with torch.no_grad():
            x_hat = generator(zs, ys)

        x_hat = torch.abs(x_hat)
        # if we use the image skip connection we do not need to calculate any of the metrics
        max_ch = torch.max(x_hat, dim=1)[0]
        losses = torch.mean(max_ch, dim=(tuple(range(1, max_ch.dim())))).cpu().numpy()
        scores.append(losses)
        if (i * batch_size + batch_size) >= n_samples:
            break
    if len(scores) > 1:
        scores = np.concatenate(scores, axis=0)
    else:
        scores = np.array(scores).squeeze()
    return scores


def results2file_ood(path, r_dict, ood_dataset):
    f = open(path, 'a')
    f.writelines("\n {} \n".format(ood_dataset))
    f.writelines('ROCAUC: {:.2%} \n'.format(r_dict["rocauc"]))
    f.writelines('AUPR Success: {:.2%} \n'.format(r_dict["aupr_success"]))
    f.writelines('AUPR Error: {:.2%} \n'.format(r_dict["aupr_error"]))
    f.writelines('FPR @ 95% TPR: {:.2%} \n'.format(r_dict["fpr"]))
    f.close()


def get_metrics_ood(label, score, invert_score=False):
    results_dict = {}
    if invert_score:
        score = score - score.max()
        score = np.abs(score)

    error = 1 - label
    rocauc = roc_auc_score(label, score)

    aupr_success = average_precision_score(label, score)
    aupr_errors = average_precision_score(error, (1 - score))

    # calculate fpr @ 95% tpr
    fpr = 0
    eval_range = np.arange(score.min(), score.max(), (score.max() - score.min()) / 10000)
    for i, delta in enumerate(eval_range):
        tpr = len(score[(label == 1) & (score >= delta)]) / len(score[(label == 1)])
        if 0.9505 >= tpr >= 0.9495:
            fpr = len(score[(error == 1) & (score >= delta)]) / len(score[(error == 1)])
            break

    results_dict["rocauc"] = rocauc
    results_dict["aupr_success"] = aupr_success
    results_dict["aupr_error"] = aupr_errors
    results_dict["fpr"] = fpr
    return results_dict
