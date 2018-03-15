"""Main figure components"""
import numpy as np
import figure_base.settings as gs
import matplotlib.pyplot as p


class FigureControl():
    """Central control for all figures"""
    @classmethod
    def init(cls, start_iter, end_iter, visible_range):
        cls.minPossibleGenNumber = start_iter
        cls.maxPossibleGenNumber = end_iter
        cls.setOfVisibleGenNumber = set()
        cls.cloudMode = 'AllCloud'
        cls.offspring_stochastic = False
        cls.save_movie = False

        cls.step = 1
        if cls.maxPossibleGenNumber - cls.minPossibleGenNumber >= 100:
            cls.step = int((cls.maxPossibleGenNumber - cls.minPossibleGenNumber)/10)

        cls.maxVisibleRangeSize = 1
        if visible_range:
            visible_range = int(visible_range)
            cls.maxVisibleRangeSize = max(1, visible_range)

    @classmethod
    def numVisibleGenNumber(cls):
        return len(cls.setOfVisibleGenNumber)

    @classmethod
    def minVisibleGenNumber(cls):
        return min(cls.setOfVisibleGenNumber)

    @classmethod
    def maxVisibleGenNumber(cls):
        return max(cls.setOfVisibleGenNumber)

    @classmethod
    def isVisible(cls, thisGenNumber):
        return thisGenNumber in cls.setOfVisibleGenNumber

    @classmethod
    def plotOffSprings(cls, thisGenNumber):
        cls.setOfVisibleGenNumber.add(thisGenNumber)
        for cplot in gs.cloud_plots:
            cplot.plotOffSprings(thisGenNumber)
        gs.fitness_plot.markVisible(thisGenNumber, True)

    @classmethod
    def hideOffSprings(cls, thisGenNumber):
        cls.setOfVisibleGenNumber.remove(thisGenNumber)
        for cplot in gs.cloud_plots:
            cplot.hideOffSprings(thisGenNumber)
        gs.fitness_plot.markVisible(thisGenNumber, False)

    @classmethod
    def applyVisibleRange(cls, mode, newGen):
        print("calling applyVisibleRange")
        while cls.numVisibleGenNumber() >= cls.maxVisibleRangeSize:
            minVG, maxVG = cls.minVisibleGenNumber(), cls.maxVisibleGenNumber()
            if mode == "next":
                drop_gen = minVG
            elif mode == "prev":
                drop_gen = maxVG
            elif mode == "dist":
                dist_minVG, dist_maxVG = np.abs(newGen - minVG), np.abs(newGen - maxVG)
                drop_gen = minVG if dist_minVG >= dist_maxVG else maxVG
            print("hiding Gen {}", drop_gen)
            cls.hideOffSprings(drop_gen)

    @classmethod
    def pickVR(cls, label):
        hzdict = {'1': 1, '2': 2, '3': 3}
        cls.maxVisibleRangeSize = hzdict[label]
        print("you select {}".format(cls.maxVisibleRangeSize))
        for cplot in gs.cloud_plots:
            cplot.button_area.radio.enforce(cls.maxVisibleRangeSize)

    @classmethod
    def pickCloud(cls, label):
        hzdict = {'All': 'AllCloud', 'Top': 'TopOnly', 'None': 'NoCloud'}
        selectedMode = hzdict[label]
        oldMode = cls.cloudMode
        cls.cloudMode = selectedMode
        print("you select {} vs old {}".format(selectedMode, oldMode))
        for cplot in gs.cloud_plots:
            cplot.button_area.radio_cloud.enforce(cls.cloudMode)
        if oldMode != cls.cloudMode and cls.numVisibleGenNumber() > 0:
            for gen in cls.setOfVisibleGenNumber:
                cls.hideOffSprings(gen)
                cls.plotOffSprings(gen)
            cls.draw_all_cloud_plots()

    @classmethod
    def stochastic(cls, label):
        oldstoc = cls.offspring_stochastic
        cls.offspring_stochastic = not oldstoc
        print("offspring_stochastic_seed: ", cls.offspring_stochastic)
        for cplot in gs.cloud_plots:
            cplot.button_area.check.enforce(cls.offspring_stochastic, 0)

    @classmethod
    def saveMovie(cls, label):
        oldstoc = cls.save_movie
        cls.save_movie = not oldstoc
        print("save movie: ", cls.save_movie)
        for cplot in gs.cloud_plots:
            cplot.button_area.check_savem.enforce(cls.save_movie, 0)

    @classmethod
    def fastMove(cls, label):
        if cls.step > 1:
            cls.step = 1
        else:
            cls.step = int((cls.maxPossibleGenNumber - cls.minPossibleGenNumber)/10)
            cls.step = max(cls.step, 1)
        print("current step size: ", cls.step)
        for cplot in gs.cloud_plots:
            cplot.button_area.check_pace.enforce(cls.step > 1, 0)

    @classmethod
    def draw_all_cloud_plots(cls):
        '''draw all cloud plots'''
        for cplot in gs.cloud_plots:
            cplot.fig.canvas.draw()

    @classmethod
    def makeGenVisible(cls, gen, visNow, mode, *, skip_fitness_plot=False):
        if visNow:
            for cplot in gs.cloud_plots:
                cplot.show_new_labels_gen(gen)
            if cls.numVisibleGenNumber() > 0:
                cls.applyVisibleRange(mode, gen)
            cls.plotOffSprings(gen)
            if not skip_fitness_plot:
                gs.fitness_plot.setVal(gen)
        else:
            cls.hideOffSprings(gen)
            if not skip_fitness_plot:
                gs.fitness_plot.fig.canvas.draw()

        cls.draw_all_cloud_plots()

    @classmethod
    def print_error(cls, err):
        for cplot in gs.cloud_plots:
            cplot.text_area.show(err)
        cls.draw_all_cloud_plots()

    @classmethod
    def clear_labels(cls):
        for cplot in gs.cloud_plots:
            cplot.clear_labels()
        gs.fitness_plot.reset()

    @classmethod
    def set_home(cls):
        for cplot in gs.cloud_plots:
            cplot.reset_xy_lim()
        cls.draw_all_cloud_plots()

    @classmethod
    def movie(cls, event):
        print("you clicked movie. will be showing movie in another figure")
        movie_start = cls.minPossibleGenNumber
        if cls.numVisibleGenNumber() > 0:
            movie_start = cls.minVisibleGenNumber()
        movie_end = cls.maxPossibleGenNumber
        print(movie_start, movie_end)
        cplot = gs.canvas2cloud_plot[event.canvas]
        cplot.play_movie(movie_start, movie_end)

    @classmethod
    def handle_close(cls, event):
        print("figure closed")
        if event.canvas == gs.fitness_plot.fig.canvas:
            print("close fitness plot")
            p.close('all')
        else:
            cplot = gs.canvas2cloud_plot[event.canvas]
            print(cplot.title)
            gs.cloud_plots.remove(cplot)
            gs.canvas2cloud_plot.pop(event.canvas)
            if len(gs.cloud_plots) == 0:
                p.close('all')
