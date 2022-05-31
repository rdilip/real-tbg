import numpy as np

def partition_indices_by_value(lst):
    """ Partitions a list into indices, where the value in each group of indices is the same.
    Args:
        lst (list): List of values.
    Returns:
        list: List of lists of indices.
    """
    ixs = np.argsort(lst)
    eq_indices = [[ixs[0]]]
    for i in range(1, len(lst)):
        prev_val = lst[eq_indices[-1][-1]]
        if np.isclose(prev_val, lst[ixs[i]]):
            eq_indices[-1].append(ixs[i])
        else:
            eq_indices.append([ixs[i]])
    return eq_indices
