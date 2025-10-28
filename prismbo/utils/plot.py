import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot2D(model, X, Y, range_x, c='blue', ls='', marker='o', fillstyle=None, label=None, ax=None, file=None, show=False, show_legend=False, bounds=None,title=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    sampled_X = np.linspace(range_x[0], range_x[1], 1000)[:, None]
    pred_mean, pred_var = model.predict(sampled_X)
    pred_mean = np.squeeze(pred_mean)
    pred_std = np.sqrt(np.squeeze(pred_var))
    ax.plot(sampled_X, pred_mean, c=c, ls=ls, label=label)
    ax.fill_between(
        sampled_X.flatten(), 
        pred_mean - pred_std, 
        pred_mean + pred_std, 
        color=c, alpha=0.2
    )
    
    ax.plot(X, Y, c=c, ls=ls, marker=marker, label=label,fillstyle=fillstyle)
    ax.set_xlabel('$f_1(\mathbf{x})$', fontsize=13)
    ax.set_ylabel('$f_2(\mathbf{x})$', fontsize=13)
    ax.tick_params(axis='both', labelsize=13)

    if show or file is not None:
        plt.grid()
        if show_legend:
            plt.legend()
        if bounds is not None:
            plt.xlim((bounds[0, 0], bounds[0, 1]))
            plt.ylim((bounds[1, 0], bounds[1, 1]))
    if title:
        plt.title(title)
    if file is not None:
        plt.savefig(file, format='pdf')
    if show:
        plt.show()
    if file is None and show is False:
        return ax
    return None


def plot3D(X, Y, Z, c='black', ls='', marker='o', fillstyle=None, label=None, ax=None, file=None, show=False,
           show_legend=False, bounds=None,title=None):
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot(X, Y, Z, c=c, ls=ls, marker=marker, label=label,fillstyle=fillstyle)
    ax.set_xlabel('$f_1(\mathbf{x})$', fontsize=13)
    ax.set_ylabel('$f_2(\mathbf{x})$', fontsize=13)
    ax.set_zlabel('$f_3(\mathbf{x})$', fontsize=13)
    ax.tick_params(axis='both', labelsize=13)

    if show or file is not None:
        plt.grid()
        if show_legend:
            plt.legend()
        if bounds is not None:
            ax.set_xlim((bounds[0, 0], bounds[0, 1]))
            ax.set_ylim((bounds[1, 0], bounds[1, 1]))
            ax.set_zlim((bounds[2, 0], bounds[2, 1]))
    if title:
        plt.title(title)
    if file is not None:
        plt.savefig(file, format='pdf')
    if show:
        plt.show()
    if file is None and show is False:
        return ax
    return None


def surface3D(X_grid, Y_grid, cmap=cm.Blues, ax=None, file=None, show=False,label=None):
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('$x_1$', fontsize=13)
    ax.set_ylabel('$x_2$', fontsize=13)
    ax.set_zlabel('f(\mathbf{x})', fontsize=13)
    ax.plot_surface(X_grid, X_grid.T, Y_grid, cmap=cmap,label=label)
    if file is not None:
        plt.grid()
        plt.savefig(file, format='pdf')
    if show:
        plt.grid()
        plt.show()
    if file is None and show is False:
        return ax
    return None