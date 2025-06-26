from _context import *

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
from scipy.special import softmax
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker


class TMaze:
    """
    Class that handles the transitions and current state of an agent in the TMaze environment
    """
    def __init__(self, reward_s=6):
        size = (5, 5)
        self.n_states = size[0]+size[1]-1
        self.n_actions = 2
        self.reward_s = reward_s
        
        # sets up mapping between states, actions and next state (works with size=(5,5))
        self.t_map = np.zeros([self.n_states, self.n_actions], dtype=int)
        for s in range(self.n_states):
            self.t_map[s, 0] = np.max([0, s-1])
            self.t_map[s, 1] = s+1
        self.t_map[4, 0] = 7
        self.t_map[7, 0] = 4
        self.t_map[6, 1] = 6
        self.t_map[8, 1] = 8
        
        self.s = 0
        
    def step(self, action):
        """
        Performs a step in the environment, updates current agent state and returns it.
        """
        assert(action < self.n_actions)
        
        self.s = self.t_map[self.s, action]
        r = 1 if self.s == self.reward_s else 0
        
        return self.s, r
    
    def vstep(self, state, action):
        """
        Performs a virtual step in the environment from state, returns the new state without updating the current state.
        """
        assert(action < self.n_actions)
        assert(state < self.n_states)
        
        r = 1 if self.s == self.reward_s else 0

        return self.t_map[self.s, action], r
    
    def reset(self, s=0):
        """
        Reset to initial state s=0 or a defined state s
        """
        self.s = s
        return s


MOVES = [
    (0, 1),
    ((1/2)**0.5, (1/2)**0.5),
    (1, 0),
    ((1/2)**0.5, -(1/2)**0.5),
    (0, -1),
    (-(1/2)**0.5, -(1/2)**0.5),
    (-1, 0),
    (-(1/2)**0.5, (1/2)**0.5),
]


class WaterMaze:
    def __init__(self, n_states, st_dev=1.0, step_size=0.2, w_scale=1):
        self._step_size = step_size
        self._end = False
        self.reset()

        # variables for the rbf sampling
        self.st_dev = st_dev
        try:
            self.n_points = int(np.sqrt(n_states))
        except:
            print("Need to provide a number of states that has integer square root")

        positions = np.linspace(-6, 6, self.n_points)
        x_pos, y_pos = np.meshgrid(positions, positions)
        self.x_pos = x_pos.flatten()
        self.y_pos = y_pos.flatten()

        self.w_cache_pos = None
        self.w_cache = None
        self.w_scale = w_scale

    @property
    def end(self) -> bool:
        return self._end

    def step(self, action):
        if self._end:
            raise ValueError('The goal state has already been reached, please reset the maze!')

        move = MOVES[action]
        new_state = (self._state[0] + self._step_size*move[0], self._state[1] + self._step_size*move[1])
        self._history.append(self._state)

        # check if along the path the agent did not go out of the allowed area
        # get some points along the path
        x = np.linspace(self._state[0], new_state[0], 10)
        y = np.linspace(self._state[1], new_state[1], 10)
        for i in range(10):
            if WaterMaze.is_outside_of_area(x[i], y[i]):
                # reject moves that go out of the allowed area
                new_state = (x[i-1], y[i-1])
                reward = 0
                break
            if WaterMaze.is_in_objective(x[i], y[i]):
                reward = 1
                self._end = True
                break
            else:
                reward = 0

        self._state = new_state
        self._last_reward = reward

        return new_state, reward


    def get_rbf(self, position):
        """
        This returns an array of size N, weight for each i: exp(-||s-x_i||^2/(2\sigma^2))
        position: a list of two values: first for x pos; second for y pos
        """
        # We do a bit of caching to not compute this every time
        if self.w_cache_pos == position:
            return self.w_cache

        norm_gauss = 1/(self.st_dev*np.sqrt(2*np.pi))
        w = norm_gauss *\
            np.exp(- ((self.x_pos - position[0]) ** 2 + (self.y_pos - position[1]) ** 2) / (2*self.st_dev ** 2))

        w *= self.w_scale
        self.w_cache = w
        self.w_cache_pos = position
        return w


    @staticmethod
    def is_in_objective(x, y):
        """
        :returns: true if the position is in the target (circle of radius 1/2 around the origin)
        """
        return x*x + y*y <= 1/4

    @staticmethod
    def is_outside_of_area(x, y):
        """
        :returns: true if the position is outside the accessible area
        """
        # Check outer border
        if abs(x) > 6 or abs(y) > 6:
            return True
        # U sides
        if 1.5 <= abs(x) <= 2.5 and -2.5 < y < 2:
            return True
        # U bottom
        if abs(x) <= 1.5 and -2.5 <= y <= -1.5:
            return True
        return False

    def find_starting_pos(self):
        """
        This picks a random starting position inside the bounds of the environment.
        """
        pos_candidate = None
        while pos_candidate is None:
            pos_candidate = (np.random.uniform(-6, 6), np.random.uniform(-6, 6))
            # pos_candidate = (random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5))
            if self.is_outside_of_area(*pos_candidate) or self.is_in_objective(*pos_candidate):
                pos_candidate = None
        return pos_candidate

    def get_state(self):
        return self._state

    def reward(self):
        return self._last_reward

    def reset(self):
        self._end = False
        self._state = self.find_starting_pos()
        self._history = []

    def render(self, draw_rbf=False):
        
        plt.figure()
        ax = plt.gca()
        ax.axis('off')

        # Plot objective
        ax.add_patch(plt.Circle((0, 0), 0.5, color="red"))
        # Plot U shape
        ax.add_patch(plt.Rectangle((-2.5, -2.5), 1, 4.5, color="gray"))
        ax.add_patch(plt.Rectangle((1.5, -2.5), 1, 4.5, color="gray"))
        ax.add_patch(plt.Rectangle((-1.5, -2.5), 3, 1, color="gray"))

        norm_gauss = 1/(self.st_dev*np.sqrt(2*np.pi))
        halfway_x = self.st_dev*np.sqrt(2*np.log(2))

        rbf_1std = self.st_dev
        # Plot points
        for x in self.x_pos:
            for y in self.y_pos:
                plt.plot(x, y, 'b.', markersize=3)
                if draw_rbf:
                    # draw circle of radius r
                    ax.add_patch(plt.Circle((x, y), halfway_x,
                                            edgecolor=(0.85, 0.85, 1, 0.01), fill=False, linewidth=0.5))


        # Plot outside boundaries
        plt.plot([-6, -6, 6, 6, -6], [-6, 6, 6, -6, -6], 'gray', alpha=0.5)

        # Plot history

        #Add starting point
        if len(self._history)>0:
            pos0 = self._history[0]
            plt.plot(pos0[0],pos0[1], 'o', c='black', markersize=10,zorder=10)

        for i in range(len(self._history)):
            pos1 = self._history[i]
            pos2 = self._history[i+1] if i+1 < len(self._history) else self._state
            plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'b')
            if i>0:
                plt.plot(pos1[0],pos1[1],'o',c='black',markersize=2) #add point for every stop


        # Plot current position
        plt.plot(*self._state, 'kx', markersize=10)

        plt.show()

    def render_actions(self, net, ax, dense=False, double_arrows=0.0, magnitude=False,n_points=21,colormap=matplotlib.cm.get_cmap("jet")): #n_points was 21, 22 prb. best
        """
        This visualises the most probable action in each state using a colored arrow.
        Color of the arrow indicates probability.
        Set dense to True to sample more actions than rbf centers.
        """
        # colormap = matplotlib.cm.get_cmap("jet")
        ax.axis('off')

        if dense is False: # plot as background
            # Plot U shape
            ax.add_patch(plt.Rectangle((-2.5, -2.5), 1, 4.5, color="gray"))
            ax.add_patch(plt.Rectangle((1.5, -2.5), 1, 4.5, color="gray"))
            ax.add_patch(plt.Rectangle((-1.5, -2.5), 3, 1, color="gray"))
            # Plot outside boundaries
            # ax.plot([-6, -6, 6, 6, -6], [-6, 6, 6, -6, -6], 'black', alpha=1,linewidth=0.3)


        if dense is False:
            pos = (self.x_pos, self.y_pos)
        else:
            # pos = [np.linspace(-5, 5, n_points),
            #        np.linspace(-5, 5, n_points)]
            pos = [np.linspace(-6, 6, n_points),
                   np.linspace(-6, 6, n_points)]
            
        moves = np.array(MOVES)

        # def scaling_fun(x, p=3):
        def scaling_fun(x, p=1):
            q = 2**(p-1)
            r = 0.5 + q*(-0.5)**p * (-1)**p
            if x < 0.5:
                return q*x**p
            else:
                return -(-1)**p * q * (x-1)**p + r


        for x in pos[0]:
            for y in pos[1]:
                _, h_a = net.action(self.get_rbf((x, y)))
                action_probabilities = softmax(h_a/net.T)
                maxproba = np.max(action_probabilities)
                if magnitude is False and double_arrows != 0 and maxproba < double_arrows:
                    # if one action is not strong enough, plot the best two actions
                    sorted_actions = np.argsort(action_probabilities)
                    for i in range(1, 3):
                        action_x = moves[sorted_actions[-i], 0]
                        action_y = moves[sorted_actions[-i], 1]
                        action = np.array([action_x, action_y])
                        ax.arrow(x, y, 0.3 * action[0], 0.3 * action[1], head_width=0.03, head_length=0.03,
                                  color=colormap(action_probabilities[sorted_actions[-i]]), linewidth=1)
                elif magnitude is False:
                    action_x = np.mean(action_probabilities*moves[:, 0])
                    action_y = np.mean(action_probabilities*moves[:, 1])
                    mean_action = np.array([action_x, action_y])
                    normalized_action = mean_action/np.linalg.norm(mean_action)
                    ax.arrow(x, y, 0.3 * normalized_action[0], 0.3 * normalized_action[1], head_width=0.1, head_length=0.1,
                              color=colormap(maxproba), linewidth=1)
                elif magnitude is True:
                    scaled_action_probabilities = np.array(list(map(scaling_fun, action_probabilities)))
                    action_x = np.mean(scaled_action_probabilities*moves[:, 0])
                    action_y = np.mean(scaled_action_probabilities*moves[:, 1])
                    mean_action = np.array([action_x, action_y])
                    # mean_action = np.array(list(map(scaling_fun, mean_action)))
                    norm_action = np.linalg.norm(mean_action)
                    # print(norm_action)

                    head_scale=sum(abs(mean_action))
                    # ax.arrow(x, y, 2.5 * mean_action[0], 2.5 * mean_action[1], head_width=0.5*norm_action,
                    #          head_length=0.5*norm_action, color=colormap(maxproba), linewidth=0.75)
                    # ax.arrow(x, y, 1 *np.sign(mean_action[0])*np.sqrt(abs(mean_action[0])),1 *np.sign(mean_action[1])*np.sqrt(abs(mean_action[1])), head_width=0.4*norm_action,
                    #          head_length=0.4*norm_action, color=colormap(maxproba), linewidth=0.7)
                    # ax.arrow(x, y, 2.5 * mean_action[0], 2.5 * mean_action[1], head_width=0.5*norm_action,
                    #          head_length=0.5*norm_action, color=colormap(maxproba), linewidth=0.85)
                    ax.arrow(x, y, 2.5 * mean_action[0], 2.5 * mean_action[1], head_width=0.5*norm_action,
                             head_length=0.5*norm_action, color=colormap(maxproba), linewidth=1)

        if dense: # plot on top of arrows
            # Plot U shape
            ax.add_patch(plt.Rectangle((-2.5, -2.5), 1, 4.5, color="gray"))
            ax.add_patch(plt.Rectangle((1.5, -2.5), 1, 4.5, color="gray"))
            ax.add_patch(plt.Rectangle((-1.5, -2.5), 3, 1, color="gray"))
            # Plot outside boundaries
            # ax.plot([-6, -6, 6, 6, -6], [-6, 6, 6, -6, -6], 'black', alpha=1,linewidth=0.3)


        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        cbar = plt.colorbar(
            matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=colormap), ax=ax)
        # cbar = plt.colorbar(colormap, ax=ax)
        # cbar.set_label("Action max probability")
        cbar.set_label("Action probability")
        # ax.set_title("Policy map")

    def render_values(self, net, ax, show_rbf=False,lvls=10,ticks=None,colormap=matplotlib.cm.get_cmap("jet")):
        """
        Visualises the v(s) function for all possible states.
        """
        ax.axis('off')

        xs = np.linspace(-6, 6, 101)
        ys = np.linspace(-6, 6, 101)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        values = np.zeros_like(X)
        for i in range(101):
            for j in range(101):
                values[i, j] = net.value(self.get_rbf((xs[i], ys[j])))

        #Version not normalized
        cs = ax.contourf(X, Y, values, levels=lvls, cmap=colormap, zorder=-1)

        # Plot U shape
        ax.add_patch(plt.Rectangle((-2.5, -2.5), 1, 4.5, color="gray"))
        ax.add_patch(plt.Rectangle((1.5, -2.5), 1, 4.5, color="gray"))
        ax.add_patch(plt.Rectangle((-1.5, -2.5), 3, 1, color="gray"))
        # Plot outside boundaries
        ax.plot([-6, -6, 6, 6, -6], [-6, 6, 6, -6, -6], 'black', alpha=1,linewidth=0.3)


        #Colorbar
        cbar = plt.colorbar(cs, ax=ax)
        cbar.set_label("Value")
        if ticks is None:
            ticks= np.linspace(values.min(), values.max(), num=7)
        cbar.set_ticks(ticks)        
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
