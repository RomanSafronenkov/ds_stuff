from torch import Tensor, sort
from math import log2


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    ys_true, ys_pred = ys_true.flatten(), ys_pred.flatten()  # flattering tensors just in case
    
    _, sorted_ys_true_idx = sort(ys_true, descending=True)

    sorted_preds_by_true = ys_pred[sort(ys_true, descending=True)[1]]
    count = 0

    for i in range(len(sorted_preds_by_true)):
        for j in range(i+1, len(sorted_preds_by_true)):
            if sorted_preds_by_true[i] < sorted_preds_by_true[j]:
                count += 1
    return count


def compute_gain(y_value: float, gain_scheme: str) -> float:
    assert gain_scheme in ['const', 'exp2']
    if gain_scheme == 'const':
        return y_value
    elif gain_scheme == 'exp2':
        return 2 ** y_value - 1


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str, k=None) -> float:
    ys_true, ys_pred = ys_true.flatten(), ys_pred.flatten()  # flattering tensors just in case

    if k is None:
        k = len(ys_true)
    
    dcg_value = 0
    _, sorted_ys_pred_idx = sort(ys_pred, descending=True)
    for i, rel in enumerate(ys_true[sorted_ys_pred_idx][:k]):
        dcg_value += compute_gain(rel, gain_scheme=gain_scheme) / log2(i+2)
    return dcg_value.item()
    

def compute_ideal_dcg(ys_true: Tensor, gain_scheme: str= 'const', k=None):
    return dcg(ys_true, ys_true, gain_scheme=gain_scheme, k=k)


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const', k=None) -> float:
    dcg_value = dcg(ys_true, ys_pred, gain_scheme=gain_scheme, k=k)
    perfect_dcg = compute_ideal_dcg(ys_true, gain_scheme=gain_scheme, k=k)
    
    return dcg_value / perfect_dcg
