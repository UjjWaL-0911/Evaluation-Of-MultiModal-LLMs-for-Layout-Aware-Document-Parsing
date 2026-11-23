"""
Steps 2-3 of SAVs (Mitra et al., arXiv 2412.00142, Sec. 3.2).

Step 2 - for every head (l, m): class centroids mu_c = mean of the support examples' head vectors;
         score(l, m) = accuracy of a nearest-centroid (cosine) classifier on the support set.
         Keep the top-k heads (k = 20 in the paper).
Step 3 - classify a query by majority vote of the k selected heads' nearest-centroid predictions.
         Ties are broken by the summed cosine similarity (not specified in the paper).
"""
import numpy as np
import torch
import torch.nn.functional as F


class SAVClassifier:
    def __init__(self, k=20):
        self.k = k

    def fit(self, acts, labels):
        """acts: [N, L, H, D] tensor; labels: length-N sequence of hashable class labels."""
        X = F.normalize(acts.float(), dim=-1)
        self.classes = sorted(set(labels))
        y = torch.tensor([self.classes.index(l) for l in labels])
        cent = torch.stack([X[y == c].mean(0) for c in range(len(self.classes))])   # [C, L, H, D]
        self.centroids = F.normalize(cent, dim=-1)
        sims = torch.einsum("nlhd,clhd->nlhc", X, self.centroids)
        head_acc = (sims.argmax(-1) == y[:, None, None]).float().mean(0)           # [L, H]
        self.head_acc = head_acc
        L, H = head_acc.shape
        k = min(self.k, L * H)
        top = torch.topk(head_acc.flatten(), k).indices
        self.heads = [(int(i) // H, int(i) % H) for i in top]                       # (layer, head)
        return self

    def predict(self, acts):
        X = F.normalize(acts.float(), dim=-1)
        li = torch.tensor([l for l, _ in self.heads])
        hi = torch.tensor([h for _, h in self.heads])
        sims = torch.einsum("nkd,ckd->nkc", X[:, li, hi], self.centroids[:, li, hi])  # [N, k, C]
        votes = F.one_hot(sims.argmax(-1), len(self.classes)).sum(1).float()          # [N, C]
        votes += 1e-3 * sims.sum(1) / sims.shape[1]                                     # tie-break
        return [self.classes[i] for i in votes.argmax(-1).tolist()]


def stratified_support(labels, n_shot, rng):
    """Pick n_shot indices per class for the support set; the rest is the test set."""
    labels = np.asarray(labels)
    sup = []
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        sup.extend(rng.choice(idx, size=min(n_shot, len(idx)), replace=False).tolist())
    sup = np.array(sorted(sup))
    test = np.setdiff1d(np.arange(len(labels)), sup)
    return sup, test
