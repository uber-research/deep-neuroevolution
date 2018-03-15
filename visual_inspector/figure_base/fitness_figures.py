"""generation# v.s. fitness plot"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from figure_base.figure_control import FigureControl
import figure_base.settings as gs
from figure_base.mouse_event import FitnessPlotClick, MouseMove
from figure_base.load_data import loadParentData

class FitnessPlot():
    """generation# v.s. fitness plot"""
    def __init__(self, title, start_iter, end_iter, snapshots_path):
        x, y = [], []
        for iteration in range(start_iter, end_iter+1):
            parent, _, _ = loadParentData(snapshots_path, iteration)
            x.append(iteration)
            y.append(parent[0].fitness)

        self.inc = 1
        self.fig = plt.figure(title)
        self.ax = self.fig.add_subplot(111)

        self.sliderax = self.fig.add_axes([0.125, 0.02, 0.775, 0.03],
                                          facecolor='yellow')

        self.slider = DiscreteSlider(self.sliderax, 'Gen', x[0], x[-1],
                                     increment=self.inc, valinit=-x[-1], valfmt='%0.0f')
        self.slider.on_changed(self.update)
        self.x = x
        self.y = y
        self.curve, = self.ax.plot(self.x, self.y, '--', picker=3)

        self.floating_annot = self.ax.annotate("", xy=(0, 0), xytext=(0, -40),
                                               textcoords="offset points",
                                               arrowprops=dict(arrowstyle="->"))
        self.floating_annot.set_visible(False)
        self.floating_annot.set_fontsize(18)
        self.floating_annot.set_color('b')

        self.ax.set_xlim(x[0], x[-1])
        maxy, miny = max(y), min(y)
        self.ax.set_ylim(miny - 0.05*abs(miny), maxy + 0.05 * abs(maxy))
        self.ax.set_ylabel("Fitness")
        self.ax.grid(True)
        self.dot, = self.ax.plot(-x[-1], -1, 'o', markersize=15, markerfacecolor="None",
                                 markeredgecolor='red', markeredgewidth=3)
        self.mapOfGenToArtist = {}
        self.fig.canvas.mpl_connect('pick_event', FitnessPlotClick.onpick)
        self.fig.canvas.mpl_connect("motion_notify_event", MouseMove.hover)
        self.fig.canvas.mpl_connect('close_event', FigureControl.handle_close)

    def update(self, value):
        """update the fitness plot"""
        if value < 0:
            self.dot.set_data([[value], [-1]])
            self.ax.set_title("")
        else:
            self.dot.set_data([[value], [self.y[value-self.x[0]]]])
            self.ax.set_title("Gen {}  Fitness {:.8f} ".format(value, self.y[value-self.x[0]]))

            vis_now = FigureControl.isVisible(value)
            if not vis_now:
                FigureControl.makeGenVisible(value, True, "dist",
                                             skip_fitness_plot=True)

        self.fig.canvas.draw()

    def setVal(self, val):
        self.slider.set_val(val)

    def reset(self):
        """reset the slider"""
        self.slider.reset()

    def markVisible(self, gen, visible):
        """mark a generation visible"""
        if not gen in self.mapOfGenToArtist:
            this_marker = gs.MARKERS[gen%gs.numMarkers]
            this_color = gs.COLOR_HEX_LISTS[gen%gs.numColors]
            pt, = self.ax.plot(gen, self.y[gen-self.x[0]],
                               this_marker,
                               color=this_color[-1],
                               markersize=10)

            numdigits = int(np.log10(gen)) + 1 if gen > 0 else 1
            annot = self.ax.annotate(gen, xy=(gen, self.y[gen-self.x[0]]),
                                     xytext=(-5.5*numdigits, 40), textcoords="offset points",
                                     arrowprops=dict(arrowstyle="->"), fontsize=18)

            self.mapOfGenToArtist[gen] = (pt, annot)

        self.mapOfGenToArtist[gen][0].set_visible(visible)
        self.mapOfGenToArtist[gen][1].set_visible(visible)

class DiscreteSlider(Slider):
    """This class is slightly adapted from the following Subscriber Content from the Stack Exchange Network
       https://stackoverflow.com/questions/13656387

       The question was asked by J Knight (https://stackoverflow.com/users/1547090/j-knight).
       The answer so used was answered by Joe Kington (https://stackoverflow.com/users/325565/joe-kington)
       and edited by Ian Campbell (https://stackoverflow.com/users/1008353/ian-campbell)

       Stack Exchange Network Terms of Service can be found at
       https://stackexchange.com/legal/terms-of-service
    """
    """A matplotlib slider widget with discrete steps."""
    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 0.5)
        Slider.__init__(self, *args, **kwargs)
        self.valtext.set_text('')

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        if discrete_val >= 0:
            self.valtext.set_text(self.valfmt % discrete_val)
        else:
            self.valtext.set_text('')
        if self.drawon:
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson:
            return
        for _, func in self.observers.items():
            func(discrete_val)
