"""Assemble hi-D BCs from all generations"""
import numpy as np
import pandas as pd

def assemble(start_iter, end_iter, path, *, bc_dim, ds_ratio):
    """Assemble hi-D BCs from all generations"""
    print("Assembling {}-D BCs... with ds_ratio={}".format(bc_dim, ds_ratio))

    X, parent_options, child_options, labels = [], [], [], []
    for gen in range(start_iter, end_iter+1):
        print('processing iter {}...'.format(gen))
        parent_file = '{}/snapshots/snapshot_gen_{:04d}/snapshot_parent_{:04d}.dat'.format(path, gen, gen)
        pdata = np.loadtxt(parent_file)

        p_bc = pdata[:bc_dim]
        X.append(p_bc)
        parent_options.append(pdata[bc_dim:])
        labels.append(pdata[bc_dim:bc_dim+1])

        offspring_file = '{}/snapshots/snapshot_gen_{:04d}/snapshot_offspring_{:04d}.dat'.format(path, gen, gen)
        odata = pd.read_csv(offspring_file, sep=' ', header=None).values

        num_rows = odata.shape[0]
        selected = list(range(num_rows))
        if num_rows >= 10 and ds_ratio < 1.0:
            rndperm = np.random.permutation(num_rows)
            n_ds = max(10, int(num_rows*ds_ratio))
            selected = rndperm[:n_ds]

        o_bc = odata[selected, :bc_dim]
        num_os = o_bc.shape[0]
        X.append(o_bc)
        child_options.append(odata[selected, bc_dim:])
        labels.append(odata[selected, bc_dim:bc_dim+1])

    return np.vstack(X), parent_options, child_options, num_os, np.vstack(labels)
