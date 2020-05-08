from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn.metrics


def calc_score(labels: np.ndarray, preds: np.ndarray, metric: str,
               weights: Optional[np.ndarray] = None) -> float:
    '''TODO

    See https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    for the weighted correlation coefficient formula.

    Args
    - labels: np.array, shape [N]
    - preds: np.array, shape [N]
    - metric: str, one of ['r2', 'R2', 'mse', 'rank']
        - 'r2': (weighted) squared Pearson correlation coefficient
        - 'R2': (weighted) coefficient of determination
        - 'mse': (weighted) mean squared-error
        - 'rank': (unweighted) Spearman rank correlation coefficient
    - weights: np.array, shape [N]

    Returns: float
    '''
    if metric == 'r2':
        if weights is None:
            return scipy.stats.pearsonr(labels, preds)[0] ** 2
        else:
            mx = np.average(preds, weights=weights)
            my = np.average(labels, weights=weights)
            cov_xy = np.average((preds - mx) * (labels - my), weights=weights)
            cov_xx = np.average((preds - mx) ** 2, weights=weights)
            cov_yy = np.average((labels - my) ** 2, weights=weights)
            return cov_xy ** 2 / (cov_xx * cov_yy)
    elif metric == 'R2':
        return sklearn.metrics.r2_score(y_true=labels, y_pred=preds,
                                        sample_weight=weights)
    elif metric == 'mse':
        return np.average((labels - preds) ** 2, weights=weights)
    elif metric == 'rank':
        return scipy.stats.spearmanr(labels, preds)[0]
    else:
        raise ValueError(f'Unknown metric: "{metric}"')


def calc_r2(x: np.ndarray, y: np.ndarray) -> float:
    return calc_score(labels=x, preds=y, metric='r2')


def evaluate(labels: np.ndarray,
             preds: np.ndarray,
             weights: Optional[np.ndarray] = None,
             do_print: bool = False,
             title: str = None
             ) -> Tuple[float, float, float, float]:
    '''TODO

    Args
    - labels: list of labels, length N
    - preds: list of preds, length N
    - weights: list of weights, length N
    - do_print: bool
    - title: str

    Returns: r^2, R^2, mse, rank_corr
    '''
    r2 = calc_score(labels=labels, preds=preds, metric='r2', weights=weights)
    R2 = calc_score(labels=labels, preds=preds, metric='R2', weights=weights)
    mse = calc_score(labels=labels, preds=preds, metric='mse', weights=weights)
    rank = calc_score(labels=labels, preds=preds, metric='rank', weights=weights)
    if do_print:
        if title is not None:
            print(f'{title}\t- ', end='')
        print(f'r^2: {r2:0.3f}, R^2: {R2:0.3f}, mse: {mse:0.3f}, rank: {rank:0.3f}')
    return r2, R2, mse, rank


def evaluate_df(df: pd.DataFrame, cols: List[str], labels_col: str = 'label',
                weights_col: Optional[str] = None,
                index_name: Optional[str] = None) -> pd.DataFrame:
    '''TODO

    Args
    - df: pd.DataFrame, columns include cols and labels_col
    - cols: list of str, names of cols in df to evaluate
    - labels_col: str, name of labels column
    - weights_col: str, name of weights column, optional
    - index_name: str, name of index for returned df

    Returns
    - results_df: pd.DataFrame, columns are ['r2', 'R2', 'mse', 'rank']
        row index are `cols`
    '''
    labels = df[labels_col]
    weights = None if weights_col is None else df[weights_col]
    records = []
    for col in cols:
        row = evaluate(labels=labels, preds=df[col], weights=weights)
        records.append(row)
    index = pd.Index(data=cols, name=index_name)
    results_df = pd.DataFrame.from_records(
        records, columns=['r2', 'R2', 'mse', 'rank'], index=index)
    return results_df


def plot_residuals(labels: np.ndarray, preds: np.ndarray,
                   title: Optional[str] = None,
                   ax: Optional[matplotlib.axes.Axes] = None) -> None:
    '''Plots residuals = preds - labels.

    Args
    - labels: np.array, shape [N]
    - preds: np.array, shape [N]
    - title: str, ie. 'train' or 'val'
    - ax: matplotlib.axes.Axes
    '''
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

    flat_line = np.array([-2, 3])
    ax.plot(flat_line, np.zeros_like(flat_line), color='black')

    residuals = preds - labels
    ax.scatter(x=labels, y=residuals, label='residuals')
    ax.legend()
    ax.grid()
    if title is not None:
        ax.set_title(title)
    ax.set(xlabel='label', ylabel='residual')


def sorted_scores(labels: np.ndarray, preds: np.ndarray, metric: str,
                  sort: str = 'increasing') -> Tuple[np.ndarray, np.ndarray]:
    '''Sorts (pred, label) datapoints by label using the given sorting
    direction, then calculates the chosen score over for the first k
    datapoints, for k = 1 to N.

    Args
    - labels: np.array, shape [N]
    - preds: np.array, shape [N]
    - metric: one of ['r2', 'R2', 'mse', 'rank']
    - sort: str, one of ['increasing', 'decreasing', 'random']

    Returns:
    - scores: np.array, shape [N]
    - labels_sorted: np.array, shape [N]
    '''
    if sort == 'increasing':
        sorted_indices = np.argsort(labels)
    elif sort == 'decreasing':
        sorted_indices = np.argsort(labels)[::-1]
    elif sort == 'random':
        sorted_indices = np.random.permutation(len(labels))
    else:
        raise ValueError(f'Unknown value for sort: {sort}')
    labels_sorted = labels[sorted_indices]
    preds_sorted = preds[sorted_indices]
    scores = np.zeros(len(labels))
    for i in range(1, len(labels) + 1):
        scores[i - 1] = calc_score(labels=labels_sorted[0:i], preds=preds_sorted[0:i], metric=metric)
    return scores, labels_sorted


def plot_label_vs_score(scores_list: List[np.ndarray],
                        labels_list: List[np.ndarray],
                        legends: List[str],
                        metric: str,
                        sort: str,
                        figsize: Tuple[float, float] = (5, 4)) -> None:
    '''TODO

    Args
    - scores_list: list of length num_models, each element is np.array of shape [num_examples]
    - labels_list: list of length num_models, each element is np.array of shape [num_examples]
    - legends: list of str, length num_models
    - metric: str, metric by which the scores were calculated
    - sort: str, one of ['increasing', 'decreasing', 'random']
    - figsize: tuple (width, height), in inches
    '''
    num_models = len(scores_list)
    assert len(labels_list) == num_models
    assert len(legends) == num_models

    f, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    for i in range(num_models):
        ax.scatter(x=labels_list[i], y=scores_list[i], s=2)

    ax.set(xlabel='wealthpooled', ylabel=metric)
    ax.set_title(f'Model Performance vs. cumulative {sort} label')
    ax.set_ylim(-0.05, 1.05)
    ax.grid()
    lgd = ax.legend(legends)
    for handle in lgd.legendHandles:
        handle.set_sizes([30.0])
    plt.show()


def plot_percdata_vs_score(scores_list: List[np.ndarray],
                           legends: List[str],
                           metric: str,
                           sort: str,
                           figsize: Tuple[float, float] = (5, 4)) -> None:
    '''TODO

    Args
    - scores_list: list of length num_models, each element is np.array of shape [num_examples]
    - legends: list of str, length num_models
    - metric: str, metric by which the scores were calculated
    - sort: str, one of ['increasing', 'decreasing', 'random']
    - figsize: tuple (width, height), in inches
    '''
    assert len(legends) == len(scores_list)

    f, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    for scores in scores_list:
        num_examples = len(scores)
        percdata = np.arange(1, num_examples + 1, dtype=np.float32) / num_examples
        ax.scatter(x=percdata, y=scores, s=1)
    ax.set(xlabel='% of data', ylabel='metric')
    ax.set_title(f'Model Performance vs. % of data by {sort} label')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid()
    lgd = ax.legend(legends)
    for handle in lgd.legendHandles:
        handle.set_sizes([30.0])
    plt.show()


def chunk_vs_score(labels: np.ndarray, preds: np.ndarray, nchunks: int,
                   metric: str, chunk_value: Optional[np.ndarray] = None
                   ) -> np.ndarray:
    '''TODO

    Args
    - labels: np.array, shape [N]
    - preds: np.array, shape [N]
    - nchunks: int
    - metric: str, one of ['r2', 'R2', 'mse', 'rank']
    - chunk_value: np.array, shape [N], TODO

    Returns
    - scores: np.array, shape [nchunks]
    '''
    if chunk_value is None:
        chunk_value = labels
    sorted_indices = np.argsort(chunk_value)
    chunk_indices = np.array_split(sorted_indices, nchunks)  # list of np.array
    scores = np.zeros(nchunks)
    for i in range(nchunks):
        chunk_labels = labels[chunk_indices[i]]
        chunk_preds = preds[chunk_indices[i]]
        scores[i] = calc_score(labels=chunk_labels, preds=chunk_preds, metric=metric)
    return scores


def plot_chunk_vs_score(scores: np.ndarray,
                        legends: List[str],
                        metric: str,
                        figsize: Tuple[float, float] = (5, 4),
                        cmap: Optional[str] = None,
                        sort: Optional[str] = None,
                        xlabel: Optional[str] = 'chunk of data') -> None:
    '''TODO

    Args
    - scores: np.array, shape [num_models, nchunks]
    - legends: list of str, length num_models
    - metric: str, metric by which the scores were calculated
    - figsize: tuple (width, height), in inches
    - cmap: str, name of matplotlib colormap
    - sort: str, one of ['increasing', 'decreasing', None], how to sort models by metric
    - xlabel: str
    '''
    assert len(scores) == len(legends)
    num_models, nchunks = scores.shape

    xticklabels = []
    start = 0
    for i in range(nchunks):
        end = int(np.ceil(100.0 / nchunks * (i + 1)))
        label = f'{start:d}-{end:d}%'
        xticklabels.append(label)
        start = end

    df = pd.DataFrame(data=scores.T, columns=legends, index=xticklabels)

    col_order = df.mean(axis=0).argsort().values
    if sort == 'increasing':
        df = df.iloc[:, col_order]
    elif sort == 'decreasing':
        df = df.iloc[:, col_order[::-1]]

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    df.plot(kind='bar', ax=ax, width=0.8, cmap=cmap)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', rotation_mode='anchor')
    ax.set(xlabel=xlabel, ylabel=metric, title='Model Performance vs. % chunk of data')
    ax.grid()
    plt.show()
