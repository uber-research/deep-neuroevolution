"""mouse event"""
import time
from figure_base.figure_control import FigureControl
import figure_base.settings as gs


class FitnessPlotClick():
    """mouse pick event on fitness plot"""
    @classmethod
    def onpick(cls, event):
        """mouse pick event on fitness plot"""
        event_len = len(event.ind)
        if not event_len:
            return True
        value = event.ind[-1] + FigureControl.minPossibleGenNumber
        vis_now = FigureControl.isVisible(value)
        FigureControl.makeGenVisible(value, not vis_now, "dist")

class PointClick():
    """mouse pick event on cloud plot"""
    last_click_time = None

    @classmethod
    def rate_limiting(cls):
        """limit the rate of clicking"""
        this_click_time = time.time()
        time_to_last_click = None
        if cls.last_click_time:
            time_to_last_click = this_click_time - cls.last_click_time
        cls.last_click_time = this_click_time
        return time_to_last_click and time_to_last_click < 0.7

    @classmethod
    def button_1(cls, cloud_plot, artist, ind):
        """click with button 1, i.e., left button"""
        is_parent = cloud_plot.is_parent_artist(artist, ind)
        gen = cloud_plot.artist2gen[artist]
        if is_parent:
            vis_now = FigureControl.isVisible(gen)
            FigureControl.makeGenVisible(gen, not vis_now, "dist")
        else:
            row_idx = cloud_plot.artist2data[artist][ind]
            for cpl in gs.cloud_plots:
                this_data = cpl.fetch_child_data_point(gen, row_idx)
                cpl.show_new_labels_dp(this_data)
            FigureControl.draw_all_cloud_plots()
        cloud_plot.button_1(artist, ind)

    @classmethod
    def button_3(cls, cloud_plot, artist, ind):
        """click with button 3, i.e., right button"""
        is_parent = cloud_plot.is_parent_artist(artist, ind)
        gen = cloud_plot.artist2gen[artist]

        for cpl in gs.cloud_plots:
            if is_parent:
                cpl.show_new_labels_gen(gen)
            else:
                row_idx = cloud_plot.artist2data[artist][ind]
                this_data = cpl.fetch_child_data_point(gen, row_idx)
                cpl.show_new_labels_dp(this_data)
        FigureControl.draw_all_cloud_plots()
        cloud_plot.button_3(artist, ind)

    @classmethod
    def onpick(cls, event):
        """mouse pick event on cloud plot"""
        if cls.rate_limiting():
            return True

        if len(event.ind) != 1:
            print("Two or more points are too close! Please zoom in.")
            print("Showing the one with higher fitness score")

        cloud_plot = gs.canvas2cloud_plot[event.canvas]
        artist = event.artist
        ind = event.ind[-1]
        button = event.mouseevent.button

        if button == 1:
            cls.button_1(cloud_plot, artist, ind)
        elif button == 3:
            cls.button_3(cloud_plot, artist, ind)

class MouseMove():
    """mouse move event on plots"""
    @classmethod
    def update_annot(cls, ind):
        """update the parent floating annotations"""
        gen = ind + FigureControl.minPossibleGenNumber
        for cplot in gs.cloud_plots:
            fitness = cplot.update_annot(gen)

        text = "{}".format(gen)
        gs.fitness_plot.floating_annot.xy = (gen, fitness)
        gs.fitness_plot.floating_annot.set_text(text)

    @classmethod
    def update_plot(cls, vis):
        """update the plots"""
        for cplot in gs.cloud_plots:
            cplot.annot.set_visible(vis)
        gs.fitness_plot.floating_annot.set_visible(vis)
        FigureControl.draw_all_cloud_plots()
        gs.fitness_plot.fig.canvas.draw_idle()

    @classmethod
    def update(cls, event, curve, preferred_idx):
        """update the plots and/or annotations"""
        cont, ind = curve.contains(event)
        if cont:
            idx = ind['ind'][preferred_idx]
            cls.update_annot(idx)
            cls.update_plot(True)
        elif gs.fitness_plot.floating_annot.get_visible():
            cls.update_plot(False)

    @classmethod
    def hover(cls, event):
        """mouse move event on plots"""
        if event.canvas == gs.fitness_plot.fig.canvas:
            if event.inaxes == gs.fitness_plot.ax:
                cls.update(event, gs.fitness_plot.curve, -1)
        else:
            cplot = gs.canvas2cloud_plot[event.canvas]
            if event.inaxes == cplot.main_ax:
                cls.update(event, cplot.main_curve, 0)
