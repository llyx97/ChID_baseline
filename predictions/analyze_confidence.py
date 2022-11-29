import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_metrics(predictions, label_ids):
    preds = np.argmax(predictions, axis=1)
    correct_indices = (preds == label_ids).nonzero()[0]
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

def plot_bat(dist, n_bins=10):
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    # We can set the number of bins with the *bins* keyword argument.
    axs.hist(dist, bins=n_bins)
    plt.show()

threshold = 0.5

split = 'valid'

gts = np.load('gt_ids_%s.npy'%split)

base_logits = np.load('logits_base_%s_zero.npy'%split)
base_logits = torch.tensor(base_logits)
print(compute_metrics(base_logits.tolist(), gts))
base_probs = F.softmax(base_logits, dim=1)
confident_base_ids = (base_probs.max(dim=1)[0]>threshold).nonzero().reshape(-1)
unconfident_base_ids = (base_probs.max(dim=1)[0]<=threshold).nonzero().reshape(-1)
correct_base_ids = np.load('correct_indices_base.npy')
print(confident_base_ids.shape, unconfident_base_ids.shape, correct_base_ids.shape)
conf_corect_base = set(confident_base_ids.numpy()).intersection(set(correct_base_ids))
unconf_correc_base = set(unconfident_base_ids.numpy()).intersection(set(correct_base_ids))
unconf_incorrec_base = set(unconfident_base_ids.numpy()).difference(unconf_correc_base)
print(len(conf_corect_base))

para_logits = np.load('logits_para_%s_zero.npy'%split)
para_logits = torch.tensor(para_logits)
print(compute_metrics(para_logits.tolist(), gts))
para_probs = F.softmax(para_logits, dim=1)
correct_para_ids = np.load('correct_indices_para.npy')
incorrect_para_ids = set(np.arange(gts.shape[0])).difference(correct_para_ids)

# predicted correctly by para by incorrectly by base
#differ = np.array(list(set(correct_para_ids).difference(correct_base_ids)))
#print(differ)
#print(len(set(incorrect_para_ids).intersection(unconf_correc_base)), len(set(correct_para_ids).intersection(unconf_incorrec_base)))
#base_probs_diff = base_probs[differ].max(dim=1)[0]
#plot_bat(base_probs_diff.numpy())

base_logits[unconfident_base_ids] = para_logits[unconfident_base_ids]   # ensembling
print(compute_metrics(base_logits.tolist(), gts))
