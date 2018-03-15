"""Customerized Cloud Figures"""
from figure_base.cloud_figures import CloudPlot


class CloudPlotHDBC(CloudPlot):
    """Cloud plot to show trajectory as Hi Dim BCs"""
    def __init__(self, *args, **kwargs):
        CloudPlot.__init__(self, *args, **kwargs)
        self.hd_bc, = self.main_ax.plot([], [], color='k', linewidth=3)

    def show_new_labels_dp(self, thisData):
        CloudPlot.show_new_labels_dp(self, thisData)
        self.hd_bc.set_data(thisData.x, thisData.y)

    def clear_labels(self):
        CloudPlot.clear_labels(self)
        self.hd_bc.set_data([], [])

class CloudPlotRollout(CloudPlot):
    """Cloud plot with policy rollout"""
    def __init__(self, *args, **kwargs):
        CloudPlot.__init__(self, *args, **kwargs)
        self.traj_plots = []

    def button_3(self, artist, ind):
        from figure_custom.rollout_trajectory import rolloutMaker
        print("rolling out!!")
        gen = self.artist2gen[artist]
        this_data = self.fetch_data_point(artist, ind)
        if self.get_policy_file(gen) != None:
            self.traj_plots.append(rolloutMaker(gen, this_data, self))

class CloudPlotRolloutAtari(CloudPlot):
    """Cloud plot with policy rollout"""

    def button_3(self, artist, ind):
        from figure_custom.rollout_custom import RolloutAtari
        print("rolling out!!")
        gen = self.artist2gen[artist]
        print(gen)
        this_data = self.fetch_data_point(artist, ind)
        policy_file = self.get_policy_file(gen)
        if policy_file is None:
            return
        noise_stdev = self.get_parent_op_data(gen)[-1]

        if this_data.parentOrNot:
            seed = int(self.get_parent_op_data(gen)[-2])
            print(self.get_parent_op_data(gen))
        else:
            seed = int(this_data.child_op_data[-2])
            print(this_data.child_op_data)

        x, y, f = this_data.x[-1], this_data.y[-1], this_data.fitness
        record = "snapshots/snapshot_gen_{:04}/clips/x_{:.2f}_y_{:.2f}_f{:.2f}".format(
                this_data.gen, x, y, f)
        RolloutAtari.setup_and_rollout_policy(policy_file, this_data,
                                              noise_stdev=noise_stdev, fixed_seed=seed,
                                              render=True, path=self.path, record=record)


        import subprocess
        subprocess.call(["open {}/*.mp4".format(self.path+record)], shell=True)
