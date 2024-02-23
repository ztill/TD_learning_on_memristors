"""
Created by Flavio Martinelli at 17:45 15/05/2020
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GridworldPlotter:
    def __init__(self, full_grid_size=None, custom_grid=None):

        self.mode = None

        # Checking input correctness and setting a dictionary (state2coo) that maps state number to coordinates in
        # the grid
        if full_grid_size is not None:
            self.mode = 'full'
            grid = np.arange(full_grid_size[0] * full_grid_size[1]).reshape(full_grid_size)

            self.state2coo = dict(zip(grid.flatten(),
                                      [np.array(np.where(grid == i)).squeeze() for i in grid.flatten()]))
            self.empty_coo = np.array([])
            self.size = (full_grid_size[1], full_grid_size[0])

        if custom_grid is not None:
            if self.mode is not None:
                warnings.warn("Cannot specify both full_grid_size and custom_grid, by default using custom_grid",
                              stacklevel=2)
            self.mode = 'custom'

            self.state2coo = dict(zip(custom_grid.flatten(),
                                      [np.array(np.where(custom_grid == i)).squeeze() if i != -1 else None for i in
                                       custom_grid.flatten()]))
            del self.state2coo[-1]

            self.empty_coo = np.array(np.where(custom_grid == -1)).T
            self.size = (custom_grid.shape[1], custom_grid.shape[0])

        if self.mode is None:
            raise ValueError('Need to specify either a full_grid_size or a custom_grid')

    def color_square(self, ax, coo, color, **kwargs):
        """ Colors a square of the maze (set alpha<1 to see better number of cell)
            coo: array of x,y coordinates
        """
        ax.add_patch(patches.Rectangle((coo[1], coo[0]),
                                       1,  # width
                                       1,  # height
                                       facecolor=color,
                                       **kwargs))

    def color_state(self, ax, state, color, **kwargs):
        """ Colors a state of the maze (set alpha<1 to see better number of cell)
            state: state number
        """
        try:
            coo = self.state2coo[state]
        except:
            raise ValueError(f'The state selected [{state}] is not in the environment')

        self.color_square(ax, coo, color, edgecolor='k', **kwargs)

    def base_fig(self, ax=None, background_col=(0.7, 0.7, 0.7)):
        """ Creates and returns fig, axis of the empty maze. Ticks are properly set up and each cell is size 1x1,
            the center of each cell is at coordinates x+0.5, y+0.5

            size: array of x,y size of the grid"""

        fig = None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.size)

        ax.grid(True, 'major')

        ax.set_ylim([0, self.size[1]])
        ax.set_yticks(np.arange(self.size[1]), minor=False)
        ax.set_yticks(np.arange(self.size[1]) + 0.5, minor=True)
        ax.set_yticklabels('', minor=False)
        ax.set_yticklabels([str(n) for n in np.arange(self.size[1])], minor=True)
        ax.invert_yaxis()
        ax.yaxis.tick_left()

        ax.set_xlim([0, self.size[0]])
        ax.set_xticks(np.arange(self.size[0]), minor=False)
        ax.set_xticks(np.arange(self.size[0]) + 0.5, minor=True)
        ax.set_xticklabels('', minor=False)
        ax.set_xticklabels([str(n) for n in np.arange(self.size[0])], minor=True)
        ax.xaxis.tick_top()

        if self.mode == 'custom':
            # grey out empty cells
            for coo in self.empty_coo:
                self.color_square(ax, coo, background_col)

            # border non empty cells
            for key, coo in self.state2coo.items():
                self.color_square(ax, coo, background_col, fill=False)

            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        return fig, ax

    def draw_cell_numbers(self, ax, state_val_dict, **kwargs):
        """ Print values defined by the states present in the dictionary """

        for state, val in state_val_dict.items():
            try:
                coo = self.state2coo[state]
            except:
                raise ValueError(f'The state selected [{state}] is not in the environment')

            ax.annotate(val,
                        xy=(coo[1] + 0.5, coo[0] + 0.5),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontweight='bold', **kwargs)

    def draw_arrow(self, ax, start_s, end_s, color='red', width=0.5, **kwargs):
        """ Draws arrow from start to end using states"""

        start_coo = self.state2coo[start_s]
        end_coo = self.state2coo[end_s]

        self.draw_arrow_coo(ax, start_coo, end_coo, color, width=width, **kwargs)

    def draw_arrow_coo(self, ax, start_coo, end_coo, color, **kwargs):
        """ Draws arrow from start to end with coordinates"""

        start_coo = np.array((start_coo[1], start_coo[0])) + 0.5
        end_coo = np.array((end_coo[1], end_coo[0])) + 0.5

        vec = end_coo - start_coo

        if vec[0] != 0 or vec[1] != 0:
            skip_vec = vec / (4 * np.linalg.norm(vec))
            start_arrow = start_coo + skip_vec
            dv_arrow = vec - 2 * skip_vec

            ax.add_patch(patches.Arrow(start_arrow[0], start_arrow[1],
                                       dv_arrow[0], dv_arrow[1],
                                       facecolor=color, **kwargs))

    @staticmethod
    def tmaze_grid(size=(5, 5)):
        """Returns a tmaze shaped custom grid"""

        if not size[1] % 2:
            raise ValueError('Cannot have even number of columns in t-maze')
        custom_grid = -np.ones(size)
        custom_grid[:, int(np.floor(size[1] / 2))] = np.arange(size[0], 0, -1) - 1
        left_leg = np.arange(int(np.floor(size[1] / 2)), -1, -1) + np.max(custom_grid)
        right_leg = np.arange(0, int(np.floor(size[1] / 2))) + np.max(left_leg) + 1
        custom_grid[0, :] = np.concatenate([left_leg, right_leg])

        return custom_grid.astype(int)

