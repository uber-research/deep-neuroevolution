"""cloud plots"""
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as p
import matplotlib.animation as animation
import figure_base.settings as gs
from figure_base.buttons import ButtonArea
from figure_base.mouse_event import PointClick, MouseMove
from figure_base.load_data import loadParentData, loadOffspringData, GenStat, color_index
from figure_base.load_data import DataPoint, generateMessage
from figure_base.figure_control import FigureControl


class CloudPlot():
    '''plot with pseudo-offspring cloud'''
    def __init__(self, title, start_iter, end_iter, snapshots_path, visible_range, bc_dim=2):
        self.path = snapshots_path
        self.bc_dim = bc_dim
        self.title = title
        self.fig = p.figure(title)
        self.main_ax = self.fig.add_subplot(111)
        self.main_ax.grid(True)

        self.main_curve = None
        # annot is for parent, controlled by MouseMove
        self.annot = self.main_ax.annotate("", xy=(0, 0), xytext=(0, -40),
                                           textcoords="offset points",
                                           arrowprops=dict(arrowstyle="->"))
        self.annot.set_fontsize(18)
        self.annot.set_color('b')
        self.annot.set_visible(False)
        # annot_offspring is for selected pseudo-offsprings
        self.annot_offspring = self.main_ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                                     textcoords="offset points",
                                                     arrowprops=dict(facecolor='blue',
                                                                     shrink=0.05, alpha=0.5))
        self.annot_offspring.set_visible(False)

        self.xlim, self.ylim = None, None

        self.parent2offsprings = {} # parent artist to offspring artist
        self.artist2data = {} # artist to data
        self.artist2gen = {}
        self.genStatMap = {} # "generate number" to "information (status)
                             # about the generation, i.e. GenStat"

        self.button_area = ButtonArea(self.fig, visible_range)
        self.text_area = TextArea(self.fig)
        self.colorbar = ColorBar(self.fig)

        self.load_data_and_plot(start_iter, end_iter, snapshots_path)

        self.fig.canvas.mpl_connect('pick_event', PointClick.onpick)
        self.fig.canvas.mpl_connect("motion_notify_event", MouseMove.hover)
        self.fig.canvas.mpl_connect('close_event', FigureControl.handle_close)

    def load_data_and_plot(self, start_iter, end_iter, snapshots_path):
        parent_xs, parent_ys = [], []
        scores = []

        for iteration in range(start_iter, end_iter+1):
            print('processing iter {}...'.format(iteration))

            this_marker = gs.MARKERS[iteration%gs.numMarkers]
            this_color = gs.COLOR_HEX_LISTS[iteration%gs.numColors]

            parent, op_data, filename = loadParentData(snapshots_path, iteration, self.bc_dim)
            parent_xs.append(parent[0].x[-1])
            parent_ys.append(parent[0].y[-1])
            pfit = parent[0].fitness
            scores.append(pfit)

            osDataTable, osIndBins, maxfit, minfit = loadOffspringData(snapshots_path, iteration,
                                                                       pfit, self.bc_dim)
            #prepareTrajectoryData(osDataTable, iteration, op_data[0], filename, parent[0])
            # plot the parents
            pt, = self.main_ax.plot(parent[0].x[-1], parent[0].y[-1],
                                    color=this_color[color_index(pfit, minfit, maxfit)],
                                    picker=3,
                                    marker=this_marker
                                   )
            self.colorbar.gen2max_minfit[iteration] = (maxfit, minfit)
            self.genStatMap[iteration] = GenStat(pt, osDataTable, filename, op_data)
            self.parent2offsprings[pt] = []
            self.artist2data[pt] = parent
            self.artist2gen[pt] = iteration

            # plot the offsprings
            if len(osIndBins) != gs.numBins:
                assert len(osIndBins) == gs.numBins + 1
                lastBinIdx = gs.numBins
            else:
                lastBinIdx = gs.numBins - 1

            cidx = 0
            for ind_bin in osIndBins:
                childx, childy = [], []
                for ind in ind_bin:
                    x = osDataTable[ind, self.bc_dim//2 - 1]
                    y = osDataTable[ind, self.bc_dim - 1]
                    childx.append(x)
                    childy.append(y)

                if len(childx) > 0:
                    msize = 10 if cidx == lastBinIdx else 6
                    if cidx >= gs.numBins:
                        assert cidx == gs.numBins
                        cidx = gs.numBins - 1

                    ospt, = self.main_ax.plot(childx, childy, this_marker,
                                              color=this_color[cidx], markersize=msize)
                    ospt.set_visible(False)

                    self.parent2offsprings[pt].append(ospt)
                    self.artist2data[ospt] = ind_bin
                    self.artist2gen[ospt] = iteration
                cidx = cidx + 1

        self.main_curve, = self.main_ax.plot(parent_xs, parent_ys, 'grey', linestyle='--')
        self.update_xy_lim()

    def update_xy_lim(self):
        self.xlim = self.main_ax.get_xlim()
        self.ylim = self.main_ax.get_ylim()

    def reset_xy_lim(self):
        left, right = self.xlim
        bottom, top = self.ylim
        self.main_ax.set_xlim(left, right)
        self.main_ax.set_ylim(bottom, top)

    def plotOffSprings(self, thisGenNumber):
        thisArtist = self.genStatMap[thisGenNumber].parentArtist
        thisArtist.set_markersize(15)

        annot = self.genStatMap[thisGenNumber].annotation
        if annot is None:
            parentDP = self.artist2data[thisArtist][0]
            numdigits = int(np.log10(thisGenNumber)) + 1 if thisGenNumber > 0 else 1
            annot = self.main_ax.annotate(thisGenNumber, xy=(parentDP.x[-1], parentDP.y[-1]),
                                          xytext=(-5.5*numdigits, 40), textcoords="offset points",
                                          arrowprops=dict(arrowstyle="->"), fontsize=18)
            self.genStatMap[thisGenNumber].annotation = annot
        annot.set_visible(True)

        if FigureControl.cloudMode == 'TopOnly':
            child = self.parent2offsprings[thisArtist][-1]
            child.set_visible(True)
            child.set_picker(5)
        elif FigureControl.cloudMode == 'AllCloud':
            for child in self.parent2offsprings[thisArtist]:
                child.set_visible(True)
                child.set_picker(2)

    def hideOffSprings(self, thisGenNumber):
        thisArtist = self.genStatMap[thisGenNumber].parentArtist
        thisArtist.set_markersize(6)

        annot = self.genStatMap[thisGenNumber].annotation
        assert annot != None
        annot.set_visible(False)
        for child in self.parent2offsprings[thisArtist]:
            child.set_visible(False)
            child.set_picker(None)

    def clear_labels(self):
        self.colorbar.hide()
        self.text_area.hide()
        self.annot_offspring.set_visible(False)

    def show_new_labels_gen(self, gen):
        parent_artist = self.genStatMap[gen].parentArtist
        parent_data, = self.artist2data[parent_artist]
        self.show_new_labels_dp(parent_data)

    def show_new_labels_dp(self, thisData):
        self.colorbar.show(thisData.gen)
        self.text_area.show(thisData.message)
        if not thisData.parentOrNot:
            self.annot_offspring.xy = (thisData.x[-1], thisData.y[-1])
            self.annot_offspring.set_visible(True)
        else:
            self.annot_offspring.set_visible(False)

    def fetch_child_data_point(self, genNum, rowIdx):
        newf = self.genStatMap[genNum].osDataTable[rowIdx]
        x = newf[0: self.bc_dim//2]
        y = newf[self.bc_dim//2 : self.bc_dim]
        fitness = newf[self.bc_dim]
        op_data = newf[self.bc_dim+1:]
        message = generateMessage(genNum, False, x[-1], y[-1], fitness)
        return DataPoint(x, y, fitness, genNum, False, message, op_data)

    def is_parent_artist(self, thisArtist, thisInd):
        thisData = self.artist2data[thisArtist][thisInd]
        return isinstance(thisData, DataPoint)

    def fetch_data_point(self, thisArtist, thisInd):
        thisData = self.artist2data[thisArtist][thisInd]

        if not isinstance(thisData, DataPoint):
            rowIdx = thisData
            genNum = self.artist2gen[thisArtist]
            return self.fetch_child_data_point(genNum, rowIdx)

        return thisData

    def update_annot(self, gen):
        text = "{}".format(gen)
        thisArtist = self.genStatMap[gen].parentArtist
        parentDP = self.artist2data[thisArtist][0]

        self.annot.xy = (parentDP.x[-1], parentDP.y[-1])
        self.annot.set_text(text)
        return parentDP.fitness

    def play_movie(self, start, stop):
        m_start = time.time()
        ims = []
        parent_xs, parent_ys = [], []
        fig = p.figure(self.title + " Movie")
        ax = fig.add_subplot(111)

        for genNumber in range(start, stop+1):
            print('processing gen {} for movie...'.format(genNumber))
            pa = self.genStatMap[genNumber].parentArtist
            parent_data = self.artist2data[pa][0]
            parent_xs.append(parent_data.x[-1])
            parent_ys.append(parent_data.y[-1])
            ax.plot(parent_data.x[-1], parent_data.y[-1],
                    color=pa.get_color(),
                    picker=None,
                    marker=pa.get_marker()
                   )

            if (genNumber == start or genNumber == stop or
                    (genNumber - start)%FigureControl.step == 0):
                t = ax.annotate("Gen {}".format(genNumber), (0, 0),
                                xycoords='axes points', fontsize=32, color=pa.get_color())
                im = [t]
                for child in self.parent2offsprings[pa]:
                    x, y = child.get_data()
                    pt, = ax.plot(x, y, child.get_marker(),
                                  color=child.get_color(),
                                  markersize=child.get_markersize()
                                 )
                    im.append(pt)
                ims.append(im)

        ax.plot(parent_xs, parent_ys, 'grey', linestyle='--')
        ax.grid(True)
        numFrames = len(ims)
        interval = min(1000, 30000/numFrames)
        ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=1000,
                                        repeat=True)
        if FigureControl.save_movie:
            print("saving movie ...")
            ani.save("test_{}.mp4".format(time.time()))
        print("movie processing time: ", time.time() - m_start)
        fig.show()

    def get_policy_file(self, gen):
        return self.genStatMap[gen].filename

    def get_parent_op_data(self, gen):
        return self.genStatMap[gen].parent_op_data

    def button_1(self, artist, ind):
        pass

    def button_3(self, artist, ind):
        pass

class ColorBar():
    '''color bar'''
    def __init__(self, fig):
        self.ax = fig.add_axes([0.91, 0.1, 0.03, 0.8])
        self.ax.set_visible(False)
        self.gen2max_minfit = {}

    def show(self, genNumber):
        self.ax.set_visible(True)
        this_color = gs.COLOR_HEX_LISTS[genNumber%gs.numColors]
        cmap = mpl.colors.ListedColormap(this_color)
        maxv, minv = self.gen2max_minfit[genNumber]
        bounds = np.around(np.linspace(minv, maxv, num=gs.numBins+1))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        mpl.colorbar.ColorbarBase(self.ax, cmap=cmap, norm=norm, orientation='vertical')

    def hide(self):
        self.ax.set_visible(False)

class TextArea():
    '''text area'''
    def __init__(self, fig):
        self.fig = fig

    def show(self, message):
        self.fig.suptitle(message)

    def hide(self):
        self.show("")
