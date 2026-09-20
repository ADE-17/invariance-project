"""Frozen-feature probes, independently selected without accessing test labels."""
from pathlib import Path
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
import joblib
from experiments.rq4_ptbxl.trial import balanced_bce


def bce(a, p):
    return float(np.mean([log_loss(a[a == g], p[a == g], labels=[0, 1]) for g in (0, 1)]))


def fit(ztrain, atrain, zval, aval, tests, folder, seed, device, smoke=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    assert len(np.unique(atrain)) == len(np.unique(aval)) == 2
    scaler = StandardScaler().fit(ztrain)
    x = scaler.transform(ztrain).astype('float32')
    v = scaler.transform(zval).astype('float32')
    testx = {k: scaler.transform(z).astype('float32') for k, z in tests.items()}
    joblib.dump(scaler, folder / 'scaler.joblib')
    result = {k: {} for k in tests}
    audit = {}
    candidates = []
    for c in [.1, 1., 10.]:
        model = LogisticRegression(C=c, max_iter=2000, class_weight='balanced', random_state=seed)
        model.fit(x, atrain)
        candidates.append((bce(aval, model.predict_proba(v)[:, 1]), model))
    loss, linear = min(candidates, key=lambda item: item[0])
    joblib.dump(linear, folder / 'linear.joblib')
    audit['linear'] = dict(validation_balanced_bce=loss, C=linear.C,
                           iterations=int(linear.n_iter_.max()), iteration_limit=linear.max_iter)
    # Permuted-label negative control; never participates in the invariance gate.
    perm = np.random.default_rng(seed + 900).permutation(atrain)
    negative = LogisticRegression(C=1., max_iter=2000, class_weight='balanced').fit(x, perm)
    joblib.dump(negative, folder / 'permuted_linear.joblib')
    tree = HistGradientBoostingClassifier(max_iter=200 if not smoke else 20, max_leaf_nodes=15,
                                          l2_regularization=1, early_stopping=False, random_state=seed,
                                          class_weight='balanced').fit(x, atrain)
    joblib.dump(tree, folder / 'tree.joblib')
    audit['tree'] = dict(validation_balanced_bce=bce(aval, tree.predict_proba(v)[:, 1]))
    for key in tests:
        result[key]['linear'] = linear.predict_proba(testx[key])[:, 1]
        result[key]['tree'] = tree.predict_proba(testx[key])[:, 1]
        result[key]['permuted_linear'] = negative.predict_proba(testx[key])[:, 1]
    for restart in range(3):
        torch.manual_seed(seed + 1000 + restart)
        model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.ReLU(), nn.Linear(128, 64),
                              nn.ReLU(), nn.Linear(64, 1), nn.Flatten(0)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
        dl = DataLoader(TensorDataset(torch.from_numpy(x), torch.tensor(atrain)),
                        batch_size=256, shuffle=True)
        vx, va = torch.from_numpy(v).to(device), torch.tensor(aval, device=device)
        best, state, best_epoch, stale = float('inf'), None, 0, 0
        epochs = 3 if smoke else 60
        for epoch in range(1, epochs+1):
            model.train()
            for bx, ba in dl:
                opt.zero_grad()
                loss = balanced_bce(model(bx.to(device)), ba.to(device))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(balanced_bce(model(vx), va))
            if vl < best - 1e-5:
                best, best_epoch, stale = vl, epoch, 0
                state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                stale += 1
            if stale >= 12:
                break
        model.load_state_dict(state)
        model.eval()
        name = f'mlp{restart}'
        torch.save(dict(state=state, input_dim=x.shape[1], seed=seed+1000+restart, epoch=best_epoch), folder / f'{name}.pt')
        audit[name] = dict(validation_balanced_bce=best, selected_epoch=best_epoch, epochs_run=epoch)
        with torch.no_grad():
            for key, xx in testx.items():
                result[key][name] = np.concatenate([torch.sigmoid(model(torch.from_numpy(xx[j:j+2048]).to(device))).cpu().numpy()
                                                   for j in range(0, len(xx), 2048)])
    (folder / 'audit.json').write_text(json.dumps(audit, indent=2))
    return result
