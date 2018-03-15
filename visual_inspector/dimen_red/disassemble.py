"""disassemble into files by generation"""
import os
import os.path as osp
import fnmatch
from shutil import copyfile
import numpy as np


def disassemble(X, parent_options, child_options, method, *,
                start_iter, end_iter, path, chunk, copy_file_patterns):
    """Disassemble reduced BCs into each generation"""
    print("Disassembling and writing ...")

    assert len(parent_options) == len(child_options) == end_iter - start_iter + 1
    num_gens = len(parent_options)
    dir_name = "reduced_{}".format(method)

    for i in range(num_gens):
        gen = i + start_iter
        print('processing iter {}...'.format(gen))

        dir_name_gen = '{}/{}/snapshots/snapshot_gen_{:04d}'.format(path, dir_name, gen)
        if not osp.exists(dir_name_gen):
            os.makedirs(dir_name_gen)

        pfile_name = '{}/snapshot_parent_{:04d}.dat'.format(dir_name_gen, gen)
        X_pdata = np.hstack((X[i*chunk, :], parent_options[i]))
        len_pdata = len(X_pdata)
        np.savetxt(pfile_name, X_pdata.reshape(1, len_pdata))

        ofile_name = '{}/{}/snapshots/snapshot_gen_{:04d}/snapshot_offspring_{:04d}.dat'.format(path, dir_name, gen, gen)
        X_osdata = np.hstack((X[i*chunk+1:(i+1)*chunk, :], child_options[i]))
        np.savetxt(ofile_name, X_osdata)

        if copy_file_patterns is not None:
            src_dir = '{}/snapshots/snapshot_gen_{:04d}'.format(path, gen)
            for pattern in copy_file_patterns:
                for file in os.listdir(src_dir):
                    if fnmatch.fnmatch(file, pattern):
                        copyfile('{}/{}'.format(src_dir, file),
                                 '{}/{}'.format(dir_name_gen, file))
