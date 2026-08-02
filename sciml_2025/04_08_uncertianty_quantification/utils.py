
import torch 
import inspect
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import math, itertools
import numpy as np
import numpy as np
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassConfusionMatrix
import math
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image  
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle


def plot_training_curves(train_metrics, test_metrics):
    """
    Plot Accuracy, F1 (macro), and Loss per epoch for train/test in a single horizontal row.
    Expects each item in train_metrics/test_metrics to have keys:
      - 'accuracy' (float in [0,1])
      - 'f1_macro' (float in [0,1])
      - 'loss' (float)
    """
    assert len(train_metrics) == len(test_metrics), "train/test metrics must have same length"
    epochs = np.arange(1, len(train_metrics) + 1)

    # Extract series
    train_acc = [m["accuracy"] for m in train_metrics]
    test_acc  = [m["accuracy"] for m in test_metrics]
    train_f1  = [m["f1_macro"] for m in train_metrics]
    test_f1   = [m["f1_macro"] for m in test_metrics]
    train_loss = [m["loss"] for m in train_metrics]
    test_loss  = [m["loss"] for m in test_metrics]

    # Helper: auto y-limits with a small margin
    def _auto_ylim(series_list, margin=0.05):
        vals = np.concatenate([np.asarray(s, dtype=float) for s in series_list])
        vals = vals[np.isfinite(vals)]
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmin == vmax:
            pad = max(abs(vmin) * 0.01, 1e-6)
            vmin, vmax = vmin - pad, vmax + pad
        rng = vmax - vmin
        return (vmin - rng * margin, vmax + rng * margin)

    # Create row of subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Accuracy
    ax = axes[0]
    ax.plot(epochs, train_acc, label="Train")
    ax.plot(epochs, test_acc,  label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.set_ylim(*_auto_ylim([train_acc, test_acc]))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # F1 (macro)
    ax = axes[1]
    ax.plot(epochs, train_f1, label="Train")
    ax.plot(epochs, test_f1,  label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 (macro)")
    ax.set_title("F1 (Macro)")
    ax.set_ylim(*_auto_ylim([train_f1, test_f1]))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Loss
    ax = axes[2]
    ax.plot(epochs, train_loss, label="Train")
    ax.plot(epochs, test_loss,  label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.set_ylim(*_auto_ylim([train_loss, test_loss]))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plt.show()
    
# we'd like to evaluate after each epoch to make sure the model is training well
def evaluate(split, model, train_loader, test_loader, num_classes = 10, scaler = None):
    """
    Evaluate after an epoch on 'train' or 'test'.
    Returns dict with accuracy, f1_macro, confusion_matrix (torch.Tensor).
    """
    loader = train_loader if split == "train" else test_loader

    acc = MulticlassAccuracy(num_classes=num_classes)
    f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
    cm = MulticlassConfusionMatrix(num_classes=num_classes)
    lossi = []
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for imgs, labels in loader:
            logits, loss = model(imgs, labels)   # [B, C]; 
            if scaler:
                logits = scaler(logits)
            acc.update(logits, labels)
            f1.update(logits, labels)
            cm.update(logits, labels)
            lossi.append(loss.item())

    out = {
        "accuracy": float(acc.compute().item()),
        "f1_macro": float(f1.compute().item()),
        "confusion_matrix": cm.compute(), # torch.Tensor [C, C]
        "loss" : sum(lossi)/len(lossi) 
    }

    if was_training:
        model.train()

    return out


@torch.no_grad()
def collect_misclassified(model, loader,*, device=None, scaler=None):
    """Return a list of dicts: {'img','true','pred','conf','probs'} for misclassified samples."""
    device = device or next(model.parameters()).device
    model.eval().to(device)
    recs = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if scaler:
            logits = scaler(logits)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        mis = pred.ne(yb)
        for i in mis.nonzero(as_tuple=False).squeeze(1):
            recs.append({
                "img": xb[i],
                "true": int(yb[i]),
                "pred": int(pred[i]),
                "conf": float(conf[i]),
                "probs": probs[i],
            })
    return recs

def show_misclassified(records, n=20, cols=10, sort="confidence", largest=True, denorm=None):
    """Visualize misclassified samples with optional uncertainties."""
    import itertools, math
    if len(records) == 0:
        print("No misclassifications found."); return
    def margin(r):
        p = r["probs"].numpy()
        top2 = np.partition(p, -2)[-2:]
        return float(top2[-1] - top2[-2])
    key = (lambda r: r["conf"]) if sort == "confidence" else margin
    recs = sorted(records, key=key, reverse=largest)[:n]

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.6*cols, 2.2*rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in itertools.zip_longest(axes, recs):
        if r is None: ax.axis("off"); continue
        img = r["img"]
        if denorm is not None: img = denorm(img)
        ax.imshow(img.squeeze().numpy(), cmap="gray")
        title = f"pred {r['pred']} ({r['conf']:.2f})\ntrue {r['true']}"
        if "H_pred" in r:
            title += f"\nH={r['H_pred']:.2f} Ae={r['H_exp']:.2f} Ep={r['MI']:.2f}"
        ax.set_title(title, fontsize=8.5, pad=2)
        ax.axis("off")
    for ax in axes[len(recs):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# def show_misclassified(records, n=20, cols=10, sort="confidence", largest=True, denorm=None):
#     """Visualize misclassified samples with title 'pred (conf) | true'."""
#     import itertools, math
#     if len(records) == 0:
#         print("No misclassifications found."); return
#     def margin(r):
#         p = r["probs"].numpy()
#         top2 = np.partition(p, -2)[-2:]
#         return float(top2[-1] - top2[-2])
#     key = (lambda r: r["conf"]) if sort == "confidence" else margin
#     recs = sorted(records, key=key, reverse=largest)[:n]

#     rows = math.ceil(n / cols)
#     fig, axes = plt.subplots(rows, cols, figsize=(1.6*cols, 1.9*rows))
#     axes = np.atleast_1d(axes).ravel()
#     for ax, r in itertools.zip_longest(axes, recs):
#         if r is None: ax.axis("off"); continue
#         img = r["img"]
#         if denorm is not None: img = denorm(img)
#         ax.imshow(img.squeeze().numpy(), cmap="gray")
#         ax.set_title(f"pred {r['pred']} ({r['conf']:.2f})\ntrue {r['true']}", fontsize=9, pad=2)
#         ax.axis("off")
#     for ax in axes[len(recs):]:
#         ax.axis("off")
#     plt.tight_layout(); plt.show()

@torch.inference_mode()
def collect_ensemble_misclassified(ensemble, loader, device="cpu"):
    """
    Return list of misclassified sample dicts:
    'img', 'true', 'pred', 'conf', 'probs', 'H_pred', 'H_exp', 'MI'
    """
    def entropy(p, eps=1e-9):
        p = p.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)

    for m in ensemble:
        m.to(device).eval()

    records = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        # Collect predictions from all ensemble members
        member_probs = []
        for model in ensemble:
            logits = model(xb)[0] if isinstance(model(xb), (tuple, list)) else model(xb)
            probs = torch.softmax(logits, dim=1)  # [B, C]
            member_probs.append(probs)
        probs_all = torch.stack(member_probs, dim=0)  # [M, B, C]

        p_hat = probs_all.mean(dim=0)     # [B, C]
        conf, pred = p_hat.max(dim=1)     # [B]
        H_pred = entropy(p_hat)           # [B]
        H_members = entropy(probs_all)    # [M, B]
        H_exp = H_members.mean(dim=0)     # [B]
        MI = H_pred - H_exp               # [B]

        mis_mask = pred.ne(yb)
        for i in mis_mask.nonzero(as_tuple=False).squeeze(1):
            records.append({
                "img": xb[i].detach().cpu(),
                "true": int(yb[i]),
                "pred": int(pred[i]),
                "conf": float(conf[i]),
                "probs": p_hat[i].cpu(),
                "H_pred": float(H_pred[i]),
                "H_exp": float(H_exp[i]),
                "MI": float(MI[i]),
            })

    return records





@torch.inference_mode()
def tsne_plot_test_embeddings(
    model,
    test_loader,
    title="t-SNE of test embeddings (○ correct, ■ mis)",
    perplexity=30,
    n_iter=1000,               # keep this for caller convenience
    random_state=0,
    learning_rate="auto",
):
    model.eval()

    feats_list, y_true_list, y_pred_list = [], [], []

    for xb, yb in test_loader:
        # Forward (logits only for inference)
        out = model(xb)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        preds = logits.argmax(dim=1)

        # Embeddings
        if not hasattr(model, "extract_embeddings"):
            raise AttributeError("Model must implement `extract_embeddings(x)` returning [B, D] features.")
        emb = model.extract_embeddings(xb)

        feats_list.append(emb.detach().cpu().numpy())
        y_true_list.append(yb.detach().cpu().numpy())
        y_pred_list.append(preds.detach().cpu().numpy())

    feats  = np.concatenate(feats_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # Build kwargs in a version-compatible way
    tsne_kwargs = dict(
        n_components=2,
        init="pca",
        perplexity=perplexity,
        random_state=random_state,
        learning_rate=learning_rate,
    )
    # Map n_iter -> max_iter if needed
    tsne_params = inspect.signature(TSNE.__init__).parameters
    if "n_iter" in tsne_params:
        tsne_kwargs["n_iter"] = n_iter
    else:
        tsne_kwargs["max_iter"] = n_iter

    # Fit t-SNE
    Z = TSNE(**tsne_kwargs).fit_transform(feats)

    # Plot
    correct = (y_true == y_pred)
    classes = np.unique(y_true)
    cmap = plt.cm.get_cmap("tab10", max(10, len(classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in classes:
        idx = (y_true == cls)
        idx_c = idx & correct
        idx_m = idx & ~correct
        color = [cmap(int(cls) % cmap.N)]
        if np.any(idx_c):
            ax.scatter(Z[idx_c, 0], Z[idx_c, 1], s=10, marker="o", alpha=0.85, linewidths=0, c=color)
        if np.any(idx_m):
            ax.scatter(Z[idx_m, 0], Z[idx_m, 1], s=16, marker="s", alpha=0.9, linewidths=0, c=color)

    color_handles = [
        mlines.Line2D([], [], marker='o', linestyle='None',
                      markerfacecolor=cmap(int(cls) % cmap.N), markeredgecolor='none', label=str(int(cls)))
        for cls in classes
    ]
    shape_handles = [
        mlines.Line2D([], [], marker='o', linestyle='None', color='black', label='Correct'),
        mlines.Line2D([], [], marker='s', linestyle='None', color='black', label='Misclassified'),
    ]
    leg1 = ax.legend(handles=color_handles, title="Class (color)",
                     bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, title="Prediction (marker)",
              bbox_to_anchor=(1.02, 0.45), loc='upper left', borderaxespad=0.)

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linewidth=0.3, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return Z, y_true, y_pred




@torch.inference_mode()
def tsne_plot_embeddings_with_ood(
    model,
    id_loader,
    ood_loaders,                    # dict name -> DataLoader, or list/tuple of loaders
    title="t-SNE: ID vs OOD (ID: color, OOD: black markers)",
    perplexity=30,
    n_iter=1000,
    random_state=0,
    learning_rate="auto",
    ood_markers=None,               # list of marker strings used for OOD groups
    ood_alpha=0.85,
    id_alpha=0.85,
    id_marker_correct="o",
    id_marker_mis="s",
    id_marker_size_correct=10,
    id_marker_size_mis=16,
    ood_marker_size=18,
):
    """
    Runs ONE t-SNE on (ID embeddings + OOD embeddings) and overlays them.
    - ID: colored by class; circle (correct) vs square (misclassified)
    - OOD: black markers (one shape per OOD group)

    Notes
    -----
    * t-SNE has no `.transform`, so we must fit on the concatenated features.
    * Model must implement `extract_embeddings(x)` -> [B, D] features.
    """
    model.eval()
    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else "cpu"

    # Normalize ood_loaders to an ordered list of (name, loader)
    if isinstance(ood_loaders, dict):
        ood_items = list(ood_loaders.items())
    else:
        ood_items = [(f"ood{i}", ld) for i, ld in enumerate(ood_loaders)]

    # ---------- 1) Collect ID features, labels, preds ----------
    id_feats, id_y_true, id_y_pred = [], [], []
    for xb, yb in id_loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        preds = logits.argmax(dim=1)
        if not hasattr(model, "extract_embeddings"):
            raise AttributeError("Model must implement `extract_embeddings(x)` returning [B, D] features.")
        emb = model.extract_embeddings(xb)

        id_feats.append(emb.detach().cpu().numpy())
        id_y_true.append(yb.detach().cpu().numpy())
        id_y_pred.append(preds.detach().cpu().numpy())

    id_feats  = np.concatenate(id_feats, axis=0) if id_feats else np.zeros((0, 1))
    id_y_true = np.concatenate(id_y_true, axis=0) if id_y_true else np.zeros((0,), dtype=int)
    id_y_pred = np.concatenate(id_y_pred, axis=0) if id_y_pred else np.zeros((0,), dtype=int)
    n_id = id_feats.shape[0]

    # ---------- 2) Collect OOD features (grouped) ----------
    ood_feats_groups = []
    ood_counts = []
    for name, loader in ood_items:
        feats = []
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            if not hasattr(model, "extract_embeddings"):
                raise AttributeError("Model must implement `extract_embeddings(x)` returning [B, D] features.")
            emb = model.extract_embeddings(xb)
            feats.append(emb.detach().cpu().numpy())
        feats = np.concatenate(feats, axis=0) if feats else np.zeros((0, id_feats.shape[1] if n_id else 1))
        ood_feats_groups.append(feats)
        ood_counts.append(feats.shape[0])

    n_ood_total = int(sum(ood_counts))
    feats_all = id_feats if n_id else np.zeros((0, ood_feats_groups[0].shape[1] if ood_counts else 1))
    for g in ood_feats_groups:
        feats_all = np.concatenate([feats_all, g], axis=0) if g.size else feats_all
    n_total = feats_all.shape[0]
    if n_total == 0:
        raise ValueError("No features found in ID or OOD loaders.")

    # ---------- 3) Build TSNE kwargs (version compatible) ----------
    tsne_kwargs = dict(
        n_components=2,
        init="pca",
        perplexity=float(perplexity),
        random_state=random_state,
        learning_rate=learning_rate,
    )
    tsne_params = inspect.signature(TSNE.__init__).parameters
    if "n_iter" in tsne_params:
        tsne_kwargs["n_iter"] = n_iter
    else:
        tsne_kwargs["max_iter"] = n_iter

    # Perplexity must be < (n_samples - 1)/3 per sklearn
    max_perp = max(5.0, (n_total - 1) / 3.0)
    if tsne_kwargs["perplexity"] >= max_perp:
        tsne_kwargs["perplexity"] = max(5.0, max_perp - 1e-3)

    # ---------- 4) Fit t-SNE on concatenated features ----------
    Z_all = TSNE(**tsne_kwargs).fit_transform(feats_all)

    # Split back into ID and OOD coordinate blocks
    Z_id = Z_all[:n_id] if n_id else np.zeros((0, 2))
    Z_ood_groups = []
    start = n_id
    for cnt in ood_counts:
        Z_ood_groups.append(Z_all[start:start+cnt] if cnt > 0 else np.zeros((0, 2)))
        start += cnt

    # ---------- 5) Plot ----------
    fig, ax = plt.subplots(figsize=(9, 7))

    # ID points: color by class, marker by correctness
    if n_id > 0:
        correct = (id_y_true == id_y_pred)
        classes = np.unique(id_y_true)
        cmap = plt.cm.get_cmap("tab10", max(10, len(classes)))

        for cls in classes:
            idx = (id_y_true == cls)
            idx_c = idx & correct
            idx_m = idx & ~correct
            color = [cmap(int(cls) % cmap.N)]
            if np.any(idx_c):
                ax.scatter(Z_id[idx_c, 0], Z_id[idx_c, 1],
                           s=id_marker_size_correct, marker=id_marker_correct,
                           alpha=id_alpha, linewidths=0, c=color, label=None)
            if np.any(idx_m):
                ax.scatter(Z_id[idx_m, 0], Z_id[idx_m, 1],
                           s=id_marker_size_mis, marker=id_marker_mis,
                           alpha=id_alpha, linewidths=0, c=color, label=None)

        # Legends for ID
        color_handles = [
            mlines.Line2D([], [], marker='o', linestyle='None',
                          markerfacecolor=cmap(int(cls) % cmap.N),
                          markeredgecolor='none', label=str(int(cls)))
            for cls in classes
        ]
        shape_handles = [
            mlines.Line2D([], [], marker=id_marker_correct, linestyle='None',
                          color='black', label='Correct'),
            mlines.Line2D([], [], marker=id_marker_mis, linestyle='None',
                          color='black', label='Misclassified'),
        ]
        leg1 = ax.legend(handles=color_handles, title="ID Class (color)",
                         bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        ax.add_artist(leg1)
        ax.legend(handles=shape_handles, title="ID Prediction (marker)",
                  bbox_to_anchor=(1.02, 0.62), loc='upper left', borderaxespad=0.)

    # OOD points: black markers with distinct shapes
    if ood_markers is None:
        ood_markers = ['o', 'D', '*', '^', 'X', 'P', 's', 'v', '<', '>']  # cycle if more groups
    ood_handles = []
    for (name, _), Zg, mk in zip(ood_items, Z_ood_groups, (ood_markers * ((len(ood_items)+len(ood_markers)-1)//len(ood_markers)))[:len(ood_items)]):
        if Zg.shape[0] == 0:
            continue
        ax.scatter(Zg[:, 0], Zg[:, 1],
                   s=ood_marker_size, marker=mk, c='black',
                   alpha=ood_alpha, linewidths=0, label=None)
        ood_handles.append(
            mlines.Line2D([], [], marker=mk, linestyle='None', color='black', label=name)
        )

    if ood_handles:
        ax.legend(handles=ood_handles, title="OOD (black markers)",
                  bbox_to_anchor=(1.02, 0.20), loc='upper left', borderaxespad=0.)

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linewidth=0.3, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Return everything useful for later analysis
    return {
        "Z_id": Z_id,                       # [N_id, 2]
        "id_y_true": id_y_true,             # [N_id]
        "id_y_pred": id_y_pred,             # [N_id]
        "Z_ood": {name: Z for (name, _), Z in zip(ood_items, Z_ood_groups)},  # dict name->[N,2]
        "counts": {"id": n_id, **{name: cnt for (name, _), cnt in zip(ood_items, ood_counts)}},
        "tsne_kwargs": tsne_kwargs,
    }



class OODDataset(Dataset):
    """
    Loads a saved OOD .pt file produced by Notebook A.
    File format: (data_uint8 [N,28,28], targets_long [N])
    Returns (PIL image, target) so torchvision transforms work.
    """
    def __init__(self, path, transform=None, target_transform=None):
        self.path = str(path)
        self.data, self.targets = torch.load(self.path, map_location="cpu")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        img_u8 = self.data[idx]                          # uint8 [28,28]
        target = int(self.targets[idx].item())           # -1
        img = Image.fromarray(img_u8.numpy(), mode="L")  # PIL grayscale
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class MLP(nn.Module):
    def __init__(self, n_hidden=128, n_labels=10, width=28, length=28, dropout=0.2):
        super().__init__()
        self.n_hidden = n_hidden
        self.width = width
        self.length = length
        self.fc1 = nn.Linear(width*length, n_hidden)
        self.relu = nn.ReLU()
        # wanna avoid exploding gradients and will use batch norm = but let's see if we need it
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_labels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, imgs, labels=None):
        x = imgs.view(-1, self.width * self.length)
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        logits = self.fc3(out) # batch x labels
        if labels is None:
            return logits, None 
        loss = F.cross_entropy(logits, labels)
        return logits, loss
    
    def extract_embeddings(self, imgs):
        x = imgs.view(-1, self.width * self.length)
        out = self.fc1(x)
        out = self.dropout(self.relu(out))
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        return out

def reliability_diagram(ax, logits, labels, n_bins=15, title=""):
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        acc = pred.eq(labels).float()

        edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_acc = torch.zeros(n_bins, device=logits.device)
        bin_conf = torch.zeros(n_bins, device=logits.device)
        bin_cnt = torch.zeros(n_bins, device=logits.device)

        for i in range(n_bins):
            m = (conf > edges[i]) & (conf <= edges[i+1])
            c = m.float().sum()
            if c > 0:
                bin_acc[i]  = acc[m].mean()
                bin_conf[i] = conf[m].mean()
                bin_cnt[i]  = c

        width = 1.0 / n_bins
        ax.bar(centers.cpu().numpy(), bin_acc.cpu().numpy(), width=width, align="center", edgecolor="black")
        # Gap shading between accuracy bar and mean confidence for each bin
        for i in range(n_bins):
            if bin_cnt[i] > 0:
                lower = min(bin_acc[i].item(), bin_conf[i].item())
                height = abs(bin_acc[i].item() - bin_conf[i].item())
                ax.add_patch(Rectangle((edges[i].item(), lower), width, height, alpha=0.3, hatch="//"))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("P[y]")
        ece = ((bin_cnt / bin_cnt.sum()) * torch.abs(bin_acc - bin_conf)).sum().item()
        ax.set_title(f"{title}\nECE = {ece:.3f}")





@torch.no_grad()
def evaluate_ensemble(
    split: str,
    models,                      # list[torch.nn.Module]
    train_loader,
    test_loader,
    num_classes: int = 10,
    scaler: torch.nn.Module | None = None,
    device: torch.device | None = None,
    weights: list[float] | None = None,   # optional weighted ensemble
    combine: str = "logits",              # "logits" (default) or "probs"
):
    """
    Evaluate an ensemble on 'train' or 'test'.
    Returns dict with: accuracy, f1_macro, confusion_matrix (torch.Tensor), loss.

    - Ensemble combination:
        * "logits": average member logits (default).
        * "probs" : average member softmax probabilities.
      After combining, an optional 'scaler' (e.g., temperature scaler) can be applied to logits.
      If combine=="probs", the scaler is ignored (calibration is defined on logits).

    - If 'weights' is provided, must have len == len(models).
    """
    assert split in {"train", "test"}
    loader = train_loader if split == "train" else test_loader

    # Choose device
    if device is None:
        device = next(models[0].parameters()).device

    # Put models/scaler/metrics on device, set to eval
    was_training = [m.training for m in models]
    for m in models:
        m.eval().to(device)
    if isinstance(scaler, torch.nn.Module):
        scaler = scaler.to(device).eval()

    acc = MulticlassAccuracy(num_classes=num_classes).to(device)
    f1  = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    cm  = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")
    lossi = []

    # Normalize weights if provided
    w = None
    if weights is not None:
        assert len(weights) == len(models), "weights must match number of models"
        w = torch.as_tensor(weights, dtype=torch.float32, device=device)
        w = w / w.sum()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        member_logits = []
        member_losses = []

        for m in models:
            out = m(imgs, labels)           # supports (logits, loss) or logits
            if isinstance(out, (tuple, list)):
                lgt, lss = out
                member_losses.append(lss.detach())
            else:
                lgt, lss = out, None
            member_logits.append(lgt)

        # Stack: [M, B, C]
        logits_stack = torch.stack(member_logits, dim=0)

        if combine == "probs":
            probs_stack = F.softmax(logits_stack, dim=-1)  # [M, B, C]
            if w is not None:
                probs_ens = (probs_stack * w[:, None, None]).sum(dim=0)  # [B, C]
            else:
                probs_ens = probs_stack.mean(dim=0)
            logits_ens_for_metrics = probs_ens.log()  # metrics accept logits/probs; log keeps argmax
            # Loss on probs requires logits; use log-probs via NLL if desired.
            # Simpler: convert back to logits via log; use NLL:
            loss_batch = F.nll_loss(logits_ens_for_metrics, labels, reduction="mean")
            logits_for_updates = logits_ens_for_metrics
        else:
            # combine == "logits" (recommended)
            if w is not None:
                logits_ens = (logits_stack * w[:, None, None]).sum(dim=0)
            else:
                logits_ens = logits_stack.mean(dim=0)

            # Optional post-hoc calibration on the ensemble logits
            if scaler is not None:
                logits_ens = scaler(logits_ens)

            loss_batch = ce_loss(logits_ens, labels)
            logits_for_updates = logits_ens

        # Metrics
        acc.update(logits_for_updates, labels)
        f1.update(logits_for_updates, labels)
        cm.update(logits_for_updates, labels)
        lossi.append(loss_batch.item())

    out = {
        "accuracy": float(acc.compute().item()),
        "f1_macro": float(f1.compute().item()),
        "confusion_matrix": cm.compute(),   # [C, C] on 'device'
        "loss": sum(lossi) / max(1, len(lossi)),
    }

    # Restore training flags if needed
    for m, was in zip(models, was_training):
        if was:
            m.train()

    return out

@torch.inference_mode()
def plot_ensemble_misclassified_with_uncertainty(
    ensemble, test_loader,
    n=20, cols=10, device="cpu",
    denorm=True, mean=0.1307, std=0.3081,
    sort_by="conf_desc"  # "conf_desc", "conf_asc", or "mi_desc"
):
    def entropy(p, eps=1e-9):
        p = p.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)

    # move models to device
    for m in ensemble:
        m.to(device).eval()

    records = []

    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)

        member_probs = []
        for model in ensemble:
            logits = model(xb)[0] if isinstance(model(xb), (tuple, list)) else model(xb)
            probs = torch.softmax(logits, dim=1)  # [B, C]
            member_probs.append(probs)
        probs_all = torch.stack(member_probs, dim=0)  # [M, B, C]
        p_hat = probs_all.mean(dim=0)                # [B, C]
        conf, pred = p_hat.max(dim=1)                # [B], [B]

        # uncertainties
        H_pred = entropy(p_hat)                      # [B]
        H_members = entropy(probs_all)               # [M, B]
        H_exp = H_members.mean(dim=0)                # [B]
        MI = H_pred - H_exp                          # [B]

        mis_mask = pred.ne(yb)
        if mis_mask.any():
            idxs = mis_mask.nonzero(as_tuple=False).squeeze(1).tolist()
            for i in idxs:
                records.append({
                    "img": xb[i].detach().cpu(),
                    "true": int(yb[i]),
                    "pred": int(pred[i]),
                    "conf": float(conf[i]),
                    "H_pred": float(H_pred[i]),
                    "H_exp": float(H_exp[i]),
                    "MI": float(MI[i]),
                })

    if not records:
        print("No misclassified samples found by the ensemble.")
        return

    # sorting
    reverse = sort_by.endswith("_desc")
    records = sorted(records, key=lambda r: r[sort_by.split("_")[0]], reverse=reverse)[:n]

    # plotting
    n = len(records)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.8*cols, 2.5*rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, records):
        img = r["img"]
        if denorm:
            img = img * std + mean
        ax.imshow(img.squeeze().numpy(), cmap="gray")
        ax.set_title(
            f"pred {r['pred']} ({r['conf']:.2f})\n"
            f"true {r['true']}\n"
            f"H={r['H_pred']:.2f} Ae={r['H_exp']:.2f} Ep={r['MI']:.2f}",
            fontsize=8.5
        )
        ax.axis("off")

    for ax in axes[len(records):]:
        ax.axis("off")

    title = f"Misclassified samples sorted by {sort_by.replace('_', ' ')}"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


@torch.inference_mode()
def plot_ood_total_entropy_MLP(
    model,
    ood_loaders,
    names,
    n: int = 20,
    cols: int = 10,
    device: str | torch.device = "cpu",
    denorm: bool = True,
    mean: float = 0.5,   # match your OOD transforms.Normalize((0.5,), (0.5,))
    std: float = 0.5,
    scaler: torch.nn.Module | None = None   # optional temperature/scaler on logits
):
    """
    Plot total predictive uncertainty H (entropy of the predictive distribution)
    for the first batch of each OOD loader, using a single model.

    Args:
        model: torch.nn.Module producing logits or (logits, loss)
        ood_loaders: list of DataLoaders (each may yield images, (images, labels), or dicts)
        names: list of names (same length as ood_loaders)
        n, cols: how many images to show and grid columns
        device: device to run on
        denorm, mean, std: denormalization for plotting
        scaler: optional calibration module applied to logits (e.g., TemperatureScaler)
    """

    def _entropy(p: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        p = p.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)  # [B]

    def _get_images_from_batch(batch):
        if isinstance(batch, (tuple, list)):
            return batch[0]
        if isinstance(batch, dict):
            for k in ("image", "img", "x", "input", "data"):
                if k in batch:
                    return batch[k]
            return next(iter(batch.values()))
        return batch  # tensor

    model = model.to(device).eval()
    if isinstance(scaler, torch.nn.Module):
        scaler = scaler.to(device).eval()

    for loader, name in zip(ood_loaders, names):
        batch = next(iter(loader))
        xb = _get_images_from_batch(batch).to(device)

        out = model(xb)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if scaler is not None:
            logits = scaler(logits)
        probs = torch.softmax(logits, dim=1)   # [B, C]
        H = _entropy(probs)                    # [B]

        # Slice to n
        n_show = min(n, xb.size(0))
        imgs = xb[:n_show].detach().cpu()
        Hs   = H[:n_show].detach().cpu()

        # Plot
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 2.7 * rows))
        axes = np.atleast_1d(axes).ravel()

        for i, ax in enumerate(axes):
            if i >= n_show:
                ax.axis("off")
                continue
            img = imgs[i]
            if denorm:
                img = img * std + mean
            ax.imshow(img.squeeze().numpy(), cmap="gray")
            ax.set_title(f"H={Hs[i]:.2f}", fontsize=9)
            ax.axis("off")

        fig.suptitle(f"{name} — Total Uncertainty (Predictive Entropy H)", fontsize=14)
        plt.tight_layout()
        plt.show()


@torch.inference_mode()
def plot_ood_uncertainties_ensemble(
    ensemble,
    ood_loaders,
    names,
    n=20,
    cols=10,
    device="cpu",
    denorm=True,
    mean=0.5,            # match your OOD transforms.Normalize((0.5,), (0.5,))
    std=0.5,
    scaler=None          # optional logits scaler (e.g., temperature scaling)
):
    """
    Plot predictive confidence and uncertainty decompositions (H, Ae, Ep) for the
    first batch of each OOD loader. Compatible with loaders that yield:
      - images
      - (images, labels)    # labels ignored
      - dicts containing an image tensor under common keys (image/img/x/input/data)
    Ensemble combines member *probabilities* (mean of softmax).
    If 'scaler' is provided, it's applied to each member's logits before softmax.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    def _entropy(p, eps=1e-9):
        p = p.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)

    def _get_images_from_batch(batch):
        # Accept Tensor, (Tensor, ...), or dict-like
        if isinstance(batch, (tuple, list)):
            xb = batch[0]
        elif isinstance(batch, dict):
            for k in ("image", "img", "x", "input", "data"):
                if k in batch:
                    xb = batch[k]
                    break
            else:
                # Fallback: first value
                xb = next(iter(batch.values()))
        else:
            xb = batch
        return xb

    # Move models (and scaler) to device and eval
    for m in ensemble:
        m.to(device).eval()
    if isinstance(scaler, torch.nn.Module):
        scaler = scaler.to(device).eval()

    for loader, name in zip(ood_loaders, names):
        # ---- fetch first batch (images only) ----
        batch = next(iter(loader))
        xb = _get_images_from_batch(batch).to(device)

        # ---- collect member probabilities ----
        member_probs = []
        for model in ensemble:
            out = model(xb)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if scaler is not None:
                logits = scaler(logits)
            probs = torch.softmax(logits, dim=1)
            member_probs.append(probs)

        probs_all = torch.stack(member_probs, dim=0)   # [M, B, C]

        # ---- uncertainty decomposition ----
        p_hat = probs_all.mean(dim=0)                  # [B, C]
        conf, pred = p_hat.max(dim=1)                  # [B], [B]
        H_pred = _entropy(p_hat)                       # predictive entropy, [B]
        H_members = _entropy(probs_all)                # per-member entropy, [M, B]
        H_exp = H_members.mean(dim=0)                  # expected entropy, [B]
        MI = H_pred - H_exp                            # mutual information (epistemic), [B]

        # ---- take first n ----
        n_show = min(n, xb.size(0))
        imgs = xb[:n_show].detach().cpu()
        preds = pred[:n_show].detach().cpu()
        confs = conf[:n_show].detach().cpu()
        H     = H_pred[:n_show].detach().cpu()
        Ae    = H_exp[:n_show].detach().cpu()
        Ep    = MI[:n_show].detach().cpu()

        # ---- plot grid ----
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 2.7 * rows))
        axes = np.atleast_1d(axes).ravel()

        for i, ax in enumerate(axes):
            if i >= n_show:
                ax.axis("off")
                continue
            img = imgs[i]
            if denorm:
                img = img * std + mean
            ax.imshow(img.squeeze().numpy(), cmap="gray")
            ax.set_title(
                f"pred {preds[i].item()} ({confs[i]:.2f})\n"
                f"H={H[i]:.2f} Ae={Ae[i]:.2f} Ep={Ep[i]:.2f}",
                fontsize=8.5
            )
            ax.axis("off")

        fig.suptitle(f"{name} — OOD Uncertainties (with confidence)", fontsize=14)
        plt.tight_layout()
        plt.show()
