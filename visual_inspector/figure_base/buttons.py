"""buttons"""
from matplotlib.widgets import Button, CheckButtons, RadioButtons
from figure_base.figure_control import FigureControl


class _CheckButtons(CheckButtons):
    def enforce(self, bval, index):
        print(bval)
        if 0 > index >= len(self.labels):
            raise ValueError("Invalid CheckButton index: %d" % index)

        l1, l2 = self.lines[index]
        l1.set_visible(bval)
        l2.set_visible(bval)

        if self.drawon:
            self.ax.figure.canvas.draw()

class _RadioButtons(RadioButtons):
    def __init__(self, *args, **kwargs):
        self.val2index = kwargs.pop('val2index')
        RadioButtons.__init__(self, *args, **kwargs)

    def enforce(self, val):
        index = self.val2index[val]

        if 0 > index >= len(self.labels):
            raise ValueError("Invalid RadioButton index: %d" % index)

        self.value_selected = self.labels[index].get_text()

        for i, p in enumerate(self.circles):
            if i == index:
                color = self.activecolor
            else:
                color = self.ax.get_facecolor()
            p.set_facecolor(color)

        if self.drawon:
            self.ax.figure.canvas.draw()

class ButtonArea():
    def __init__(self, fig, visible_range):

        self.axhome = fig.add_axes([0.46, 0.01, 0.08, 0.05])
        self.axreset = fig.add_axes([0.55, 0.01, 0.08, 0.05])
        self.axmovie = fig.add_axes([0.64, 0.01, 0.08, 0.05])
        self.axprev = fig.add_axes([0.73, 0.01, 0.08, 0.05])
        self.axnext = fig.add_axes([0.82, 0.01, 0.08, 0.05])
        self.bhome = Button(self.axhome, 'Home')
        self.bhome.on_clicked(self.home)
        self.breset = Button(self.axreset, 'Reset')
        self.breset.on_clicked(self.reset)
        self.bmovie = Button(self.axmovie, 'Movie')
        self.bmovie.on_clicked(self.movie)
        self.bnext = Button(self.axnext, 'Next')
        self.bnext.on_clicked(self.next)
        self.bprev = Button(self.axprev, 'Prev')
        self.bprev.on_clicked(self.prev)


        self.checkb_ax = fig.add_axes([0., 0.0, 0.1, 0.09])
        self.checkb_ax.axis('off')

        self.checkb_ax_pace = fig.add_axes([0.1, 0.0, 0.1, 0.09])
        self.checkb_ax_pace.axis('off')

        self.checkb_ax_savem = fig.add_axes([0.2, 0.0, 0.1, 0.09])
        self.checkb_ax_savem.axis('off')

        curr_stoc = FigureControl.offspring_stochastic
        self.check = _CheckButtons(self.checkb_ax, ['Random\nSeed'], [curr_stoc])
        self.check.on_clicked(FigureControl.stochastic)

        self.check_pace = _CheckButtons(self.checkb_ax_pace, ['Fast\nPace'],
                                        [FigureControl.step > 1])
        self.check_pace.on_clicked(FigureControl.fastMove)

        self.check_savem = _CheckButtons(self.checkb_ax_savem,
                                         ['Save\nMovie'], [FigureControl.save_movie])
        self.check_savem.on_clicked(FigureControl.saveMovie)

        if not visible_range:
            self.rb_ax = fig.add_axes([0, 0.8, 0.15, 0.15])
            self.rb_ax.axis('off')
            self.radio = _RadioButtons(self.rb_ax, ('1', '2', '3'), val2index={1:0, 2:1, 3:2})
            self.radio.on_clicked(FigureControl.pickVR)

        self.rb_ax_cloud = fig.add_axes([0, 0.6, 0.15, 0.15])
        self.rb_ax_cloud.axis('off')
        self.radio_cloud = _RadioButtons(self.rb_ax_cloud, ('All', 'Top', 'None'),
                                         val2index={'AllCloud':0, 'TopOnly':1, 'NoCloud':2})
        self.radio_cloud.on_clicked(FigureControl.pickCloud)

    def eligibleClick(self, buttonClicked):
        if buttonClicked == "next":
            return (not FigureControl.isVisible(FigureControl.maxPossibleGenNumber),
                    "max gen already displayed")
        elif buttonClicked == "prev":
            return (not FigureControl.isVisible(FigureControl.minPossibleGenNumber),
                    "min gen already displayed")
        elif buttonClicked == "movie":
            return True, ""
        else:
            return False, "bad button"

    def next(self, event=None):
        ok, err = self.eligibleClick("next")
        if not ok:
            FigureControl.print_error(err)
        else:
            print("showing nextGen")
            nextGenNum = FigureControl.minPossibleGenNumber
            if FigureControl.numVisibleGenNumber() > 0:
                nextGenNum = min(FigureControl.maxVisibleGenNumber() + FigureControl.step,
                                 FigureControl.maxPossibleGenNumber)
            FigureControl.makeGenVisible(nextGenNum, True, "next")

    def prev(self, event=None):
        ok, err = self.eligibleClick("prev")
        if not ok:
            FigureControl.print_error(err)
        else:
            print("showing prevGen")
            nextGenNum = FigureControl.maxPossibleGenNumber
            if FigureControl.numVisibleGenNumber() > 0:
                nextGenNum = max(FigureControl.minVisibleGenNumber() - FigureControl.step,
                                 FigureControl.minPossibleGenNumber)
            FigureControl.makeGenVisible(nextGenNum, True, "prev")

    def movie(self, event):
        FigureControl.movie(event)

    def reset(self, event=None):
        #t1 = time.time()

        if FigureControl.numVisibleGenNumber() != 0:
            while FigureControl.numVisibleGenNumber() != 0:
                genNumber = FigureControl.maxVisibleGenNumber()
                print("cleaning ...", genNumber)
                FigureControl.hideOffSprings(genNumber)

        FigureControl.clear_labels()
        self.home()

    def home(self, event=None):
        FigureControl.set_home()
