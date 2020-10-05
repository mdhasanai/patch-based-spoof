import numpy as np

n = 1


def get_top_indices(accuracy, n):
    i = 1
    prev = 100
    top_n_indices = []

    while True:
        top_n_val = np.partition(accuracy, -2)[-i]
        if top_n_val == prev:
            conitnue

        lis = np.argwhere(np.logical_and(accuracy >= top_n_val, accuracy<prev))
#         print(top_n_val)
        top_n_indices.extend(lis.flatten())
        if len(top_n_indices) >= n:
            break
        prev = top_n_val
        i += 1
    return top_n_indices[:n]
if __name__=='__main__':
    msu = np.load('patch_dicts/combined.npy').flatten()
    oulu = np.load('patch_dicts/oulu.npy').flatten()
    top_msu = get_top_indices(msu, n)
    #top_oulu = get_top_indices(oulu, n)
    print("Top indices for msu: ", top_msu)
    #print("Top indices for oulu: ", top_oulu)

    print("Top accuracy for msu: ", msu[top_msu])
    #print("Top accuracy for oulu: ", oulu[top_oulu])