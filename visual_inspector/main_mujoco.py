"""interactive analysis"""
import click
import matplotlib.pyplot as p
import figure_base.settings as gs
from figure_base.figure_control import FigureControl
from figure_base.fitness_figures import FitnessPlot
from figure_custom.cloud_figures_custom import CloudPlotHDBC, CloudPlotRollout


@click.command()
@click.argument('start_iter', nargs=1)
@click.argument('end_iter', nargs=1)
@click.argument('snapshots_path', nargs=-1)
@click.option('--visible_range', help='Up to how many generations visible on one plot.')
@click.option('--hi_dim_bc', type=(str, int), default=(None, None),
              help='Path to high-dimensional (> 2-D) BC and its dimension')
def main(start_iter, end_iter, snapshots_path, visible_range, hi_dim_bc):
    """
    START_ITER: Plot data that begins at this iteration (generation)\n
    END_ITER: Plot data that ends at this iteration (generation)\n
    SNAPSHOTS_PATH: Path(s) to One or multiple 2-D BCs
    """
    start_iter = int(start_iter)
    end_iter = int(end_iter)

    FigureControl.init(start_iter, end_iter, visible_range)

    for idx, path in enumerate(snapshots_path):
        print("Generating Cloud Plot {} from {}".format(idx, path))
        cplot = CloudPlotRollout("Cloud Plot {} ({})".format(idx, path),
                                  start_iter, end_iter, path, visible_range)
        gs.cloud_plots.add(cplot)
        gs.canvas2cloud_plot[cplot.fig.canvas] = cplot

    gs.fitness_plot = FitnessPlot("Fitness Plot", start_iter, end_iter, snapshots_path[0])


    hbc_path, hbc_dim = hi_dim_bc
    if hbc_path != None and hbc_dim != None:
        print("Generating Cloud Plot H-D from {}".format(hbc_path))
        hbcplot = CloudPlotHDBC("Cloud Plot {}-D BC ({})".format(hbc_dim, hbc_path),
                                 start_iter, end_iter, hbc_path, visible_range, hbc_dim)
        gs.cloud_plots.add(hbcplot)
        gs.canvas2cloud_plot[hbcplot.fig.canvas] = hbcplot

    p.show()

if __name__ == '__main__':
    main()
