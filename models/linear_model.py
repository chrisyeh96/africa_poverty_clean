from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from typing import Any, Optional

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sklearn.linear_model


def train_ridge_logo(features: np.ndarray,
                     labels: np.ndarray,
                     group_labels: np.ndarray,
                     cv_groups: Sequence[int],
                     test_groups: Sequence[int],
                     weights: Optional[np.ndarray] = None,
                     plot: bool = True,
                     group_names: Optional[Sequence[str]] = None,
                     verbose: bool = False
                     ) -> tuple[np.ndarray, sklearn.linear_model.Ridge]:
    '''Leave-one-group-out cross-validated training of a linear model.

    Args
    - features: np.array, shape [N, D]
        each feature dim should be normalized to 0 mean, unit variance
    - labels: np.array, shape [N]
    - group_labels: np.array, shape [N], type np.int32
    - cv_groups: list of int, labels of groups to use for LOGO-CV
    - test_groups: list of int, labels of groups to test on
    - weights: np.array, shape [N], optional weights for each example to run
        weighted ridge regression
    - plot: bool, whether to plot MSE as a function of alpha
    - group_names: list of str, names of the groups, only used when plotting
    - verbose: bool

    Returns
    - test_preds: np.array, predictions on indices from test_groups
    - best_model: sklearn.linear_model.Ridge, fitted model with lowest MSE
    '''
    cv_indices = np.isin(group_labels, cv_groups).nonzero()[0]
    test_indices = np.isin(group_labels, test_groups).nonzero()[0]

    X = features[cv_indices]
    y = labels[cv_indices]
    groups = group_labels[cv_indices]
    w = None if weights is None else weights[cv_indices]

    alphas = 2**np.arange(-5, 35, 3.0)
    preds = np.zeros([len(alphas), len(cv_indices)], dtype=np.float64)
    group_mses = np.zeros([len(alphas), len(cv_groups)], dtype=np.float64)
    leftout_group_labels = np.zeros(len(cv_groups), dtype=np.int32)
    logo = sklearn.model_selection.LeaveOneGroupOut()

    for i, alpha in enumerate(alphas):
        if verbose:
            print(f'\rAlpha: {alpha} ({i+1}/{len(alphas)})', end='')

        # set random_state for deterministic data shuffling
        model = sklearn.linear_model.Ridge(alpha=alpha, random_state=123)

        for g, (train_indices, val_indices) in enumerate(logo.split(X, groups=groups)):
            train_X, val_X = X[train_indices], X[val_indices]
            train_y, val_y = y[train_indices], y[val_indices]
            train_w = None if w is None else w[train_indices]
            val_w = None if w is None else w[val_indices]
            model.fit(X=train_X, y=train_y, sample_weight=train_w)
            val_preds = model.predict(val_X)
            preds[i, val_indices] = val_preds
            group_mses[i, g] = np.average((val_preds - val_y) ** 2, weights=val_w)
            leftout_group_labels[g] = groups[val_indices[0]]

    if verbose:
        print()
    mses = np.average((preds - y) ** 2, axis=1, weights=w)  # shape [num_alphas]

    if plot:
        assert group_names is not None
        h = max(3, len(group_names) * 0.2)
        fig, ax = plt.subplots(1, 1, figsize=[h*2, h], constrained_layout=True)
        for g in range(len(cv_groups)):
            group_name = group_names[leftout_group_labels[g]]
            ax.scatter(x=alphas, y=group_mses[:, g], label=group_name,
                       c=[cm.tab20.colors[g % 20]])
        ax.plot(alphas, mses, 'g-', label='Overall val mse')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Left-out Group')
        ax.set(xlabel='alpha', ylabel='mse')
        ax.set_xscale('log')
        ax.grid(True)
        plt.show()

    best_alpha = alphas[np.argmin(mses)]
    best_model = sklearn.linear_model.Ridge(alpha=best_alpha)
    best_model.fit(X=X, y=y, sample_weight=w)
    test_X, test_y, = features[test_indices], labels[test_indices]
    test_preds = best_model.predict(test_X)

    best_val_mse = np.min(mses)
    test_w = None if weights is None else weights[test_indices]
    test_mse = np.average((test_preds - test_y) ** 2, weights=test_w)
    print(f'best val mse: {best_val_mse:.3f}, best alpha: {best_alpha}, test mse: {test_mse:.3f}')

    return test_preds, best_model


def ridge_cv(features: np.ndarray | Mapping[str, np.ndarray],
             labels: np.ndarray,
             group_labels: np.ndarray,
             group_names: Sequence[str],
             savedir: Optional[str] = None,
             weights: Optional[np.ndarray] = None,
             save_weights: bool = False,
             do_plot: bool = False,
             subset_indices: Optional[np.ndarray] = None,
             subset_name: Optional[str] = None,
             save_dict: Optional[Mapping[str, Any]] = None,
             verbose: bool = False
             ) -> np.ndarray:
    '''
    For every fold F (the test fold):
      1. uses leave-one-fold-out CV on all other folds
         to tune ridge model alpha parameter
      2. using best alpha, trains ridge model on all folds except F
      3. runs trained ridge model on F

    Saves predictions for each fold on test.
        savedir/test_preds_{subset_name}.npz if subset_name is given
        savedir/test_preds.npz otherwise
    Saves ridge regression weights to savedir/ridge_weights.npz
        if save_weight=True

    Args
    - features: either a dict or np.array
        - if dict: group_name => np.array, shape [N, D]
        - otherwise, just a single np.array, shape [N, D]
        - each feature dim should be normalized to 0 mean, unit variance
    - labels: np.array, shape [N]
    - group_labels: np.array, shape [N], type int
    - group_names: list of str, names corresponding to the group labels
    - savedir: str, path to directory to save predictions
    - weights: np.array, shape [N], optional weights for each example to run
        weighted ridge regression
    - save_weights: bool, whether to save the ridge regression weights
    - do_plot: bool, whether to plot alpha vs. mse curve for 1st fold
    - subset_indices: np.array, indices of examples to include for both
        training and testing
    - subset_name: str, name of the subset
    - save_dict: dict, str => np.array, data saved with test preds npz file
    - verbose: bool

    Returns
    - test_preds: np.array, shape [N]
    '''
    N = len(labels)
    if isinstance(features, np.ndarray):
        features = {f: features for f in group_names}
    for f in group_names:
        assert len(features[f]) == N

    if save_dict is None:
        save_dict = {}
    else:
        assert savedir is not None
        save_dict = dict(save_dict)  # make a copy

    if subset_indices is None:
        assert subset_name is None
        filename = 'test_preds.npz'
    else:
        assert subset_name is not None
        features = {f: feats[subset_indices] for f, feats in features.items()}
        labels = labels[subset_indices]
        group_labels = group_labels[subset_indices]

        filename = f'test_preds_{subset_name}.npz'
        for key in save_dict:
            save_dict[key] = save_dict[key][subset_indices]

    if savedir is None:
        assert not save_weights
    else:
        npz_path = os.path.join(savedir, filename)
        assert not os.path.exists(npz_path)
        if save_weights:
            weights_npz_path = os.path.join(savedir, 'ridge_weights.npz')
            assert not os.path.exists(weights_npz_path)

    test_preds = np.zeros_like(labels, dtype=np.float32)
    ridge_weights: dict[str, np.ndarray] = {}

    for i, f in enumerate(group_names):
        print('Group:', f)
        test_indices = np.where(group_labels == i)[0]
        preds, model = train_ridge_logo(
            features=features[f],
            labels=labels,
            group_labels=group_labels,
            cv_groups=[x for x in range(len(group_names)) if x != i],
            test_groups=[i],
            weights=weights,
            plot=do_plot,
            group_names=group_names,
            verbose=verbose)
        test_preds[test_indices] = preds
        ridge_weights[f + '_w'] = model.coef_
        ridge_weights[f + '_b'] = np.asarray([model.intercept_])

        # only plot the curve for the first group
        do_plot = False

    # save preds on the test set
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

        # build up save_dict
        if 'labels' in save_dict:
            assert np.array_equal(labels, save_dict['labels'])
        save_dict['labels'] = labels
        if weights is not None:
            save_dict['weights'] = weights
        save_dict['test_preds'] = test_preds

        print('saving test preds to:', npz_path)
        np.savez_compressed(npz_path, **save_dict)

        # save model weights
        if save_weights:
            print('saving ridge_weights to:', weights_npz_path)
            np.savez_compressed(weights_npz_path, **ridge_weights)

    return test_preds
