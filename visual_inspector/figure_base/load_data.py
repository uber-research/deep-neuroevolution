"""load data from file"""
import numpy as np
import figure_base.settings as gs
import pandas as pd
import os.path as osp

def color_index(fitness, minfit, maxfit):
    if maxfit == minfit:
        cind = 0.0
    else:    
        cind = (fitness - minfit)/(maxfit - minfit) * gs.numBins
    cind = int(cind)
    if cind >= gs.numBins:
        cind = gs.numBins-1
    elif cind < 0:
        cind = 0

    return cind


class GenStat:
    def __init__(self, artist, table, filename, op_data=None):
        self.parentArtist = artist
        self.osDataTable = table
        self.filename = filename
        self.parent_op_data = op_data
        self.annotation = None # annotation that indicates the selected generation

class DataPoint:
    def __init__(self, x, y, fitness, gen, parentOrNot, message, op_data=None):
        self.x = x
        self.y = y
        self.fitness = fitness
        self.gen = gen
        self.parentOrNot = parentOrNot
        self.message = message
        self.child_op_data = op_data

def generateMessage(thisGenNumber, parentOrNot, x, y, fitness):
    title_message = 'Gen {} '.format(thisGenNumber)

    if parentOrNot:
        title_message = title_message + 'Parent '
    else:
        title_message = title_message + 'Offspring '

    title_message = title_message + 'x = {:.6f}  y = {:.6f} fitness (on record) = {:.8f} '.format(
        x, y, fitness
        )

    return title_message

def loadParentData(path, gen, bc_dim=2):
    filename = '{}/snapshots/snapshot_gen_{:04d}/snapshot_parent_{:04d}.dat'.format(path, gen, gen)
    newf = np.loadtxt(filename)

    x_pt = newf[0: bc_dim//2]
    y_pt = newf[bc_dim//2 : bc_dim]
    area_pt = newf[bc_dim]
    op_data = newf[bc_dim+1:]
    f_pt = '{}/snapshots/snapshot_gen_{:04d}/snapshot_parent_{:04d}.h5'.format(path, gen, gen)
    if not osp.exists(f_pt):
        f_pt = None
    message = generateMessage(gen, True, x_pt[-1], y_pt[-1], area_pt)
    return [DataPoint(x_pt, y_pt, area_pt, gen, True, message)], op_data, f_pt

def loadOffspringData(path, gen, pfit, bc_dim=2):
    filename = '{}/snapshots/snapshot_gen_{:04d}/snapshot_offspring_{:04d}.dat'.format(path, gen, gen)
    newf = pd.read_csv(filename, sep=' ', header=None).values

    if gen not in gs.gen2sorted_indices:
        gs.gen2sorted_indices[gen] = newf[:, bc_dim].argsort()

    newf = newf[gs.gen2sorted_indices[gen]]
    area = newf[:, bc_dim]

    maxfit = max(pfit, area[-1])
    minfit = min(pfit, area[0])

    v = np.linspace(minfit, maxfit, num=gs.numBins+1)
    ind = (np.searchsorted(area, v[1:gs.numBins], side='right'))
    assert len(ind) == gs.numBins - 1

    ind_bins = []
    ind_bins.append(range(0, ind[0]))
    for i in range(0, len(ind)-1):
        ind_bins.append(range(ind[i], ind[i+1]))

    left, right = ind[-1], len(area)

    if right - left <= 10:
        ind_bins.append(range(left, right))
        assert len(ind_bins) == gs.numBins
    else:
        ind_bins.append(range(left, right-10))
        ind_bins.append(range(right-10, right))
        assert len(ind_bins) == gs.numBins+1

    return newf, ind_bins, maxfit, minfit
