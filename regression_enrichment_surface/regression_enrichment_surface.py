"""Main module."""
import matplotlib.pyplot as plt
import numpy as np

def erf(x_pred, x_true, r, y, indexs_pred, indexs_true):
    return len(
        set(indexs_pred[:int(r * indexs_pred.shape[0])]).intersection(set(indexs_true[:int(y * indexs_pred.shape[0])])))


def erfmax(x_pred, x_true, r, y, indexs_pred, indexs_true):
    return (int(min(r, y) * indexs_pred.shape[0]))


def nefr(*i):
    return erf(*i) / erfmax(*i)


def nefrcurve(points_, p, t, min_sample=-3, reverse_sort=False):
    xs = np.logspace(min_sample, 0, points_, base=10)
    ys = np.logspace(min_sample, 0, points_, base=10)

    indexs_pred = np.argsort(p)
    indexs_true = np.argsort(t)
    if reverse_sort:
        indexs_pred = indexs_pred[::-1]
        indexs_true = indexs_true[::-1]

    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros(xx.shape)
    for i in range(points_):
        for j in range(points_):
            zz[i, j] = nefr(p, t, xx[i, j], yy[i, j], indexs_pred, indexs_true)

    return xx, yy, zz


class RegressionEnrichmentSurface:
    """
    This is a conceptual class representation of a simple BLE device (GATT Server). It is essentially an extended combination of the :class:`bluepy.btle.Peripheral` and :class:`bluepy.btle.ScanEntry` classes
        :param percent_min: sets the axis bounds. Must be reasonable for your data size (i.e. cannot be data size 100 if you set -3)
        :type client: int
    """

    def __init__(self, percent_min=-3):
        self.min = percent_min
        self.nefr = None
        self.stratify = False

    def compute(self, trues, preds, stratify=None, samples=30):
        self.stratify = stratify is not None
        if not self.stratify:
            self.nefr = nefrcurve(samples, preds, trues, self.min)
        else:
            x, y, z = [], [], []
            u, indices = np.unique(stratify, return_inverse=True)
            for i in (range(u.shape[0])):
                locs = np.argwhere(indices == i).flatten()
                preds_strat = preds[locs]
                trues_strat = trues[locs]
                try:
                    x_2, y_2, z_2 = nefrcurve(30, preds_strat, trues_strat)
                except ZeroDivisionError:
                    continue
                x.append(x_2)
                y.append(y_2)
                z.append(z_2)
            self.nefr = (x, y, z)

        return self.nefr

    def plot(self, save_file=None, levels=10, title="RDS", cmap='Blues', figsize=(8, 5)):
        """Returns a list of :class:`bluepy.blte.Service` objects representing the services offered by the device. This will perform Bluetooth service discovery if this has not already been done; otherwise it will return a cached list of services immediately..

                :param save_file: if None uses plt show otherwise saves png to file
                :param levels: used for contour
                :param title: sets plot title
                :param cmap: uses matplotlib color maps
                :param figsize: sets figure size
                """
        plt.figure(figsize=figsize)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Screen top x%")
        plt.ylabel("True top x%")
        plt.contourf(np.stack(self.nefr[0]).mean(0) if self.stratify else self.nefr[0],
                     np.stack(self.nefr[1]).mean(0) if self.stratify else self.nefr[1],
                     np.stack(self.nefr[2]).mean(0) if self.stratify else self.nefr[2],
                     vmin=0,
                     vmax=1,
                     cmap=cmap,
                     levels=levels)

        plt.colorbar()
        plt.title(title)

        if save_file is None:
            plt.show()
        else:
            plt.savefig(save_file, bbox_inches='tight', dpi=300)
