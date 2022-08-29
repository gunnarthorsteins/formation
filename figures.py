from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

# The following line is the only thing I've taken from the original code.
# It's actually the only thing I checked in the code.
# Everything is done fully from scratch.
# Need it for the figs to looks as much alike as possible
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 12})
plt.rcParams['legend.title_fontsize'] = 12


class Figure(ABC):

    @abstractmethod
    def scatterplot(self):
        pass

    @abstractmethod
    def lineplot(self):
        pass

    @abstractmethod
    def boxplot(self):
        pass

    @abstractmethod
    def properties(self):
        pass

    @abstractmethod
    def showme(self):
        pass


class Figure1(Figure):

    def __init__(self):
        self.fig, self.axs = plt.subplots(nrows=2,
                                          ncols=2,
                                          dpi=800,
                                          gridspec_kw={'width_ratios': [3, 1]},
                                          figsize=(9, 5.5))

        self.axs[1, 0].set_xlabel('Cycle Number')

    def scatterplot(self, row: int, x_pos: float, shifter: int,
                    cycles_to_70: int, c: str):
        # Scatterplot on right
        x = x_pos - 0.2 + shifter / 100  # TODO: Make less hacky

        self.axs[row, 1].scatter(x=x,
                                 y=cycles_to_70,
                                 facecolor=c,
                                 edgecolor='k',
                                 s=10,
                                 zorder=2)

    def lineplot(self, discharge_capacity: np.array, hppc_indices: list,
                 i: int, c: str, linestyle: str, row: int):
        x = np.linspace(0, len(discharge_capacity), len(discharge_capacity))
        self.axs[row, 0].plot(
            x[hppc_indices[i - 1]:hppc_indices[i]],
            discharge_capacity[hppc_indices[i - 1]:hppc_indices[i]],
            c=c,
            linestyle=linestyle,
            linewidth=1,
            zorder=2)

    def boxplot(self, categorized_cycles_to_70: list, row: int, x_pos: int):
        self.axs[row, 1].boxplot(categorized_cycles_to_70[row][x_pos],
                                 positions=[x_pos],
                                 showfliers=False,
                                 medianprops={
                                     'color': 'k',
                                     'zorder': 1
                                 },
                                 boxprops={'facecolor': '#E2E2E2'},
                                 patch_artist=True,
                                 widths=0.8,
                                 zorder=0)

        self.axs[row, 1].set_xticks([0, 1])
        self.axs[row, 1].set_xticklabels(['Fast', 'Baseline'])
        self.axs[row, 1].set_xlim([-1, 2])

    def legend(self, legend_title: str, row: int, c):
        custom_lines = [
            Line2D([0], [0], color='k', lw=1, linestyle='dotted'),
            Line2D([0], [0], color=c, lw=1, linestyle='solid')
        ]
        legend = self.axs[row, 0].legend(
            custom_lines,
            ['Baseline Formation', 'Fast Formation'],
            title=legend_title,
            loc='lower left',
            frameon=False,
            fontsize=10,
        )
        legend._legend_box.align = 'left'

    def properties(self, row):
        self.axs[row, 0].set_xlim(right=700)
        self.axs[row, 0].set_ylim(bottom=0)
        self.axs[row, 0].set_ylabel('Discharge\nCapacity (Ah)')

        self.axs[row, 1].yaxis.set_label_position('right')
        self.axs[row, 1].yaxis.tick_right()
        self.axs[row, 1].set_ylabel('Cycles\nto 70%')
        self.axs[row, 1].set_ylim([250, 650])

        col = row
        self.axs[0, col].get_xaxis().set_visible(False)

    def capacity_fade_line(self, row, nominal_capacity: float = 2.36):
        self.axs[row, 0].text(x=620, y=0.72 * nominal_capacity, s='70%')
        self.axs[row, 0].axhline(y=nominal_capacity * 0.7,
                                 c='gray',
                                 linewidth=0.75,
                                 linestyle='dashed',
                                 dashes=(5, 10),
                                 zorder=1)

    def showme(self):
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        self.fig.show()


class Figure2(Figure):

    def __init__(self):
        self.fig, self.axs = plt.subplots(nrows=2,
                                          ncols=3,
                                          dpi=800,
                                          figsize=(9, 5))

        self.axs[-1, -1].set_ylabel('$\mathrm{R_{10s, 5\% SoC}\,(m\Omega)}$')
        self.axs[-1, -1].set_xticks([0, 1])
        self.axs[-1, -1].set_xticklabels(['Fast', 'Baseline'])
        self.axs[-1, -1].get_xticklabels()[0].set_color("g")
        self.axs[-1, 1].vlines(5,
                               0,
                               43,
                               color='gray',
                               linestyle='dashed',
                               linewidth=1)

        # Setting axes lims to match original
        self.axs[1, 1].set_xlim([3, 10])
        self.axs[0, 0].set_ylim([2.34, 2.4])
        self.axs[0, 1].set_ylim([0.3, 0.46])
        self.axs[0, 2].set_ylim([0.84, 0.89])

    def scatterplot(self, metric, col, x_pos, c: str):
        shifter = np.linspace(-0.1, 0.1, len(metric))
        x = [x_pos] * len(metric) + shifter
        self.axs[0, col].scatter(x,
                                 metric,
                                 facecolor=c,
                                 edgecolor='k',
                                 s=20,
                                 zorder=2,
                                 linewidth=1)

    def boxplot(self, metric, x_pos: int, col, ylabel):
        self.axs[0, col].boxplot(metric,
                                 positions=[x_pos],
                                 showfliers=False,
                                 medianprops={
                                     'color': 'k',
                                     'zorder': 1
                                 },
                                 boxprops={'facecolor': '#E2E2E2'},
                                 patch_artist=True,
                                 widths=0.8,
                                 zorder=0)
        self.axs[0, col].set_xticks([0, 1])
        self.axs[0, col].set_xticklabels(['Fast', 'Baseline'])
        self.axs[0, col].set_xlim([-1, 2])
        self.axs[0, col].get_xticklabels()[0].set_color("g")
        self.axs[0, col].set_ylabel(ylabel)

    def lineplot(self, R, xlim: list, col: int, c: str):
        x = np.linspace(3.5, 103.5, len(R))
        self.axs[1, col].plot(x, R, c=c, linewidth=1)
        self.axs[1, col].scatter(x, R, c=c, s=10, linewidth=1)

        self.axs[1, col].set_xlim(xlim)
        self.axs[1, col].set_ylim([0, 65])
        self.axs[1, col].set_ylabel('$\mathrm{R_{10s}\,(m\Omega)}$')
        self.axs[1, col].set_xlabel('SoC (%)')

    def scatterplot_lower(self, R_5SoC, x_pos: int, c: str):
        shifter = np.linspace(-0.1, 0.1, len(R_5SoC))
        self.axs[-1, -1].scatter([x_pos] * len(R_5SoC) + shifter,
                                 R_5SoC,
                                 c=c,
                                 edgecolor='k')

    def boxplot_lower(self, R_5SoC, x_pos):
        self.axs[-1, -1].boxplot(R_5SoC,
                                 positions=[x_pos],
                                 showfliers=False,
                                 medianprops={
                                     'color': 'k',
                                     'zorder': 1
                                 },
                                 boxprops={'facecolor': '#E2E2E2'},
                                 patch_artist=True,
                                 widths=0.8,
                                 zorder=0)

        self.axs[-1, -1].set_xticks([0, 1])
        self.axs[-1, -1].set_xticklabels(['Fast', 'Baseline'])
        self.axs[-1, -1].set_xlim([-1, 2])
        self.axs[-1, -1].set_ylim([40, 52])

    def properties(self, col, ylabel):
        pass

    def showme(self):
        plt.tight_layout()
        self.fig.show()


class Figure3(Figure):

    def __init__(self):
        self.fig, self.axs = plt.subplots(nrows=2,
                                          ncols=4,
                                          dpi=800,
                                          figsize=(10, 5))
        for ax in self.axs.ravel():
            ax.set_ylim([100, 700])

    def scatterplot(self, variable, cycles_to_70, row: int, col: int, c: str):
        self.axs[row, col].scatter(variable,
                                   cycles_to_70,
                                   facecolor=c,
                                   edgecolor='k',
                                   s=20,
                                   zorder=2,
                                   linewidth=1)

    def boxplot(self):
        pass

    def lineplot(self):
        pass

    def properties(self, xlabel: str, xlim: list, col: int, ylabel: int):
        self.axs[1, col].set_xlabel(xlabel)
        self.axs[0, col].get_xaxis().set_visible(False)

        for row in range(2):
            self.axs[row, col].set_xlim(xlim)

            if col > 0:
                self.axs[row, col].get_yaxis().set_visible(False)
            else:
                self.axs[row, 0].set_ylabel(ylabel)

    def showme(self):
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        self.fig.show()


class FormationFigure:

    def __init__(self, no_cols: int, no_curves):
        self.fig, self.axs = plt.subplots(1, no_cols, figsize=(16, 4))
        self.colormap = plt.cm.coolwarm(np.linspace(0, 1, no_curves))

    def scatterplot(self):
        pass

    def boxplot(self):
        pass

    def lineplot(self, col: int, df: pd.DataFrame, i: int):
        self.axs[col].plot(df.index / 3600, df.h_potential, c=self.colormap[i])

    def properties(self, no_cols: int, titles: list[str]):
        for col in range(no_cols):
            self.axs[col].set_xlabel('t [h]')
            self.axs[col].set_title(titles[col])

        self.axs[1].get_yaxis().set_visible(False)
        plt.suptitle('Formation')

    def showme(self):
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        self.fig.show()


class AdolescenceFigure:
    def __init__(self, no_cols: int, no_curves: int):
        self.fig, self.axs = plt.subplots(1, no_cols, figsize=(16,4))
        self.colormap = plt.cm.coolwarm(np.linspace(0, 1, no_curves))

    def scatterplot(self):
        pass

    def boxplot(self):
        pass

    def lineplot(self, col: int, df: pd.DataFrame, i: int):
        self.axs[col].plot(df.index/3600, df.h_potential, c=self.colormap[i])

    def properties(self, no_cols: int, titles: list[str]):
        for col in range(no_cols):
            self.axs[col].set_xlabel('t [h]')
            self.axs[col].set_title(titles[col])

        self.axs[1].get_yaxis().set_visible(False)
        plt.suptitle('Cycling')

    def showme(self):
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        self.fig.show()


def _explanatory_fig(df: pd.DataFrame, indices: list):
    """Plots a single current pulse.
    
    Zooms in on the temporal axis for clarification.
    Helper function for get_resistance().
    
    Args:
        df (pd.DataFrame): HPPC data
        indices (list[list, list]): Indices of V_min and V_max.
    """

    fields = ['h_potential', 'h_current']
    c = ['b', 'r']
    ylims = [[3.3, 3.5], [-2.8, 0.4]]
    ylabels = ['V [V]', 'I [A]']
    text = ['V_1', 'V_0']

    df.index /= 3600  # s to h
    df.index -= df.index[0]

    fig, axs = plt.subplots(2, figsize=(4, 3), dpi=150)
    for row, field in enumerate(fields):
        axs[row].plot(df.index, df[field], 'k')
        axs[row].set_ylim(ylims[row])
        axs[row].set_ylabel(ylabels[row])
        axs[row].set_xlim([1.8, 2.0])

        for j, index in enumerate(indices):
            axs[row].scatter(df.index[index],
                             df[field].iloc[index],
                             s=20,
                             c=c[j])

    axs[0].set_title('Showing $\Delta \mathrm{V}$')
    axs[0].get_xaxis().set_visible(False)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.show()