"""rollout and obtain the trajectories"""
import time
import matplotlib.pyplot as p
from figure_base.figure_control import FigureControl
import figure_base.settings as gs
import numpy as np
from figure_custom.rollout_custom import RolloutMujoco

def extract_traj(traj):
    """extract the trajectory"""
    length = int(len(traj) / 2)
    tx, ty = traj[0:length], traj[length:]
    tx = np.insert(tx, 0, 0.)
    ty = np.insert(ty, 0, 0.)
    return tx, ty


class rolloutMaker():
    def __init__(self, gen, thisData, cloud_plot):
        self.fig = p.figure()
        self.thisData = thisData
        self.ax_list, self.artist_list = [], []
        self.artist_fixed = None
        self.selected_rollout = None
        self.policy_file = cloud_plot.get_policy_file(gen)
        self.noise_stdev, = cloud_plot.get_parent_op_data(gen)
        print(self.noise_stdev)

        self.fixed_seed = None
        if not thisData.parentOrNot and len(thisData.child_op_data) > 0:
            self.fixed_seed = thisData.child_op_data[0]

        self.fxs, self.fys, self.fts, self.fscores, self.fseeds = None, None, None, None, None
        if self.fixed_seed:
            self.fxs, self.fys, self.fts, self.fscores, self.fseeds = RolloutMujoco.setup_and_rollout_policy(self.policy_file, thisData, noise_stdev=self.noise_stdev,
                                            fixed_seed=int(self.fixed_seed), bc_choice="traj")

        self.xs, self.ys, self.ts, self.scores, self.seeds = None, None, None, None, None
        if self.fixed_seed is None or FigureControl.offspring_stochastic:
            self.xs, self.ys, self.ts, self.scores, self.seeds = RolloutMujoco.setup_and_rollout_policy(self.policy_file, thisData, noise_stdev=self.noise_stdev,
                                            num_rollouts=9, bc_choice="traj")

        self.ax1 = p.subplot2grid((3, 6), (0, 0), rowspan=3, colspan=3)
        self.ax1.plot(0, 0, 'ro', markersize=12, label="Origin")
        #self.ax1.plot(thisData.x[-1], thisData.y[-1], 'bo', markersize=12, label="Final (Fixed Seed)")
        self.ax1.grid(True)

        if self.fxs:
            self.artist_fixed, = self.ax1.plot(self.fxs, self.fys, 'bo', markersize=12, picker=5, label="Final (Fixed Seed)")
            traj = self.fts[0]
            tx, ty = extract_traj(traj)
            self.ax1.plot(tx, ty, 'b--')

        if self.xs:
            for idx, traj in enumerate(self.ts):
                tx, ty = extract_traj(traj)
                label_words = "Final (Random Seed)" if idx == 0 else None
                pt, = self.ax1.plot(self.xs[idx], self.ys[idx], 'C1X', markersize=12, picker=5, label=label_words)
                self.artist_list.append(pt)
                annot=self.ax1.annotate(idx+1, xy=(self.xs[idx], self.ys[idx]), xytext=(5,5),textcoords="offset points")
                annot.set_fontsize(16)
                annot.set_color('r')
                self.ax1.plot(tx, ty, 'C{}'.format(idx%10))

                ax2 = p.subplot2grid((3, 6), (int(idx/3), idx%3+3))
                ax2.plot(0, 0, 'ro', markersize=10)
                ax2.plot(self.fxs, self.fys, 'bo', markersize=10)
                ax2.plot(tx[-1], ty[-1], 'C1X', markersize=10)
                ax2.plot(tx, ty, 'C{}'.format(idx%10))

                left, right = ax2.get_xlim()
                bottom, top = ax2.get_ylim()
                ax2.text(0.5*(left+right), 0.5*(bottom+top), '{}'.format(idx+1),
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize=32, color='red', alpha=0.5)

                ax2.grid(True)
                self.ax_list.append(ax2)
        self.ax1.legend()
        #self.ax1.set_xlim(cloud_plot.xlim)
        #self.ax1.set_ylim(cloud_plot.ylim)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

        self.fig.show()

    def on_pick(self, event):
        thisevent = event.mouseevent
        thisArtist = event.artist
        self.reset()
        if self.artist_fixed and thisArtist == self.artist_fixed:
            print("you pick the fixed seed")
            self.fig.suptitle("x:{:.6f}  y:{:.6f}  fitness:{:.8f}".format(self.fxs[0],
                              self.fys[0], self.fscores[0]))
            self.artist_fixed.set_markersize(18)
        else:
            for i, art_sub in enumerate(self.artist_list):
                if thisArtist == art_sub:
                    self.select(i)
                    break
        self.fig.canvas.draw()
        if thisevent.button == 3:
            if self.selected_rollout != None:
                RolloutMujoco.setup_and_rollout_policy(self.policy_file, self.thisData,
                                                       noise_stdev=self.noise_stdev,
                                                       fixed_seed=self.seeds[self.selected_rollout], render=True)
            else:
                RolloutMujoco.setup_and_rollout_policy(self.policy_file, self.thisData,
                                                       noise_stdev=self.noise_stdev,
                                                       fixed_seed=int(self.fixed_seed), render=True)

    def reset(self):
        if self.artist_fixed:
            self.artist_fixed.set_markersize(12)
        self.fig.suptitle("")
        if self.selected_rollout != None:
            rIdx = self.selected_rollout
            self.artist_list[rIdx].set_markersize(12)
            self.ax_list[rIdx].set_facecolor('1')
            self.selected_rollout = None

    def select(self, rIdx):
        self.fig.suptitle("#{}  x:{:.6f}  y:{:.6f}  fitness:{:.8f}".format(rIdx+1, self.xs[rIdx],  self.ys[rIdx], self.scores[rIdx]))
        self.artist_list[rIdx].set_markersize(18)
        self.ax_list[rIdx].set_facecolor('0.9')
        self.selected_rollout = rIdx

    def on_press(self, event):
        print('you pressed', event.button, event.xdata, event.ydata)
        ax_on_press = event.inaxes
        if ax_on_press == self.ax1:
            return

        self.reset()

        if ax_on_press:
            for i, ax_sub in enumerate(self.ax_list):
                if ax_on_press == ax_sub:
                    self.select(i)
                    break
        self.fig.canvas.draw()

        #print(event.button, self.selected_rollout)
        if event.button == 3 and self.selected_rollout != None:
            RolloutMujoco.setup_and_rollout_policy(self.policy_file, self.thisData,
                                                   noise_stdev=self.noise_stdev,
                                                   fixed_seed=self.seeds[self.selected_rollout], render=True)
