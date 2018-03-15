"""interactive analysis"""
import click
from dimen_red.assemble import assemble
from dimen_red.reduce import reduce_dim
from dimen_red.disassemble import disassemble
import numpy as np

@click.command()
@click.argument('start_iter', nargs=1)
@click.argument('end_iter', nargs=1)
@click.argument('snapshots_path', nargs=1)
@click.argument('bc_dim', nargs=1)
@click.option('--method', default='pca',
              help='Methods of dimensionality reduction or downsampling.')
@click.option('--downsampling_ratio', default=1.0,
              help='Downsampling ratio (<1) when method=downsampling.')
@click.option('--copy_files',
              help='Files to copy over. Support Unix-style wildcards, separated in spaces')
def main(start_iter, end_iter, snapshots_path, bc_dim, method, downsampling_ratio, copy_files):
    """
    Apply dimensionality reduction or downsampling to hi-dimensional data.

    START_ITER: Process data that begins at this iteration (generation)\n
    END_ITER: Process data that ends at this iteration (generation)\n
    SNAPSHOTS_PATH: Path to hi-dimensional BC
    """
    start_iter, end_iter, bc_dim = int(start_iter), int(end_iter), int(bc_dim)

    if method != 'downsampling':
        downsampling_ratio = 1.0

    #step 1: Assemble hi-D BCs from all generations
    X, p_opt, ch_opt, num_os_per_gen, labels = assemble(start_iter, end_iter, snapshots_path,
                                                bc_dim=bc_dim, ds_ratio=downsampling_ratio)
    print('Assembling Completed! X.shape={} #OS_per_gen={}'.format(X.shape, num_os_per_gen))
    c_labels = np.round(labels/100)

    print(c_labels, c_labels.shape)
    X = X / 255.0
    #step 2: Hi-D BCs to 2-D BCs if method != downsampling
    X_r = reduce_dim(X, labels=np.ravel(c_labels), method=method)

    #step 3: Disassemble reduced BCs into each generation
    search_patterns = None
    if copy_files is not None:
        search_patterns = copy_files.split(' ')

    disassemble(X_r, p_opt, ch_opt, method,
                start_iter=start_iter, end_iter=end_iter, path=snapshots_path,
                chunk=1+num_os_per_gen, copy_file_patterns=search_patterns)

if __name__ == '__main__':
    main()
