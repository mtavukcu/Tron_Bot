#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
from collections import deque


# Throughout this file, ASP means adversarial search problem.


class StudentBot:
    """ Write your student bot here"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """

        # This is currently a random bot, will use a-b with cutoff in the future
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        """
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        pass

    """
    def dijkstra_search(self, state, start_cell):
        start_row, start_col = start_cell

        dists = np.zeros((len(state.board), len(state.board[0])))
        dists[:] = np.inf

        visited = {(start_row, start_col): 0.0}

        dists[start_row, start_col] = 0.0

        neighbors = self.neighbors(state, (start_row, start_col))
        for n in neighbors:
            row, col = n
            dists[row, col] = 1

        q = heapq.heapify(neighbors)

        while len(q) != 0:
            child_row, child_col = heapq.heappop(q)
            dist = dists[child_row, child_col] + 1
            for n in self.neighbors(state, (child_row, child_col)):
                neighbor_r, neighbor_c = n
                if state[neighbor_r, neighbor_c] != 0:
                    continue
                if dist < dists[neighbor_r, neighbor_c]:
                    dists[neighbor_r, neighbor_c] = dist
                if visited[(neighbor_r, neighbor_c)] == 0:
                    q.append(n)
                    visited[(neighbor_r, neighbor_c)] = 1
        return dists
    """

    def bi_directional_bfs(self, state, start_0, start_1):
        dist_0 = {start_0: 0.0}
        dist_1 = {start_1: 0.0}

        visited_0 = {start_0: False}
        visited_1 = {start_1: False}

        queue_0 = deque()
        queue_1 = deque()

        queue_0.append(start_0)
        queue_1.append(start_1)

        max_dist = math.inf
        goal_0 = None
        goal_1 = None

        while queue_0 and queue_1:
            curr_dist_0 = dist_0[start_0]
            curr_dist_1 = dist_1[start_1]

            if goal_0 and goal_1 and max_dist < curr_dist_0 + curr_dist_1:
                return dist_0, dist_1

            if curr_dist_0 < curr_dist_1:
                node = queue_0.popleft()
                visited_0[node] = True
                if max_dist > curr_dist_0 + curr_dist_1:
                    max_dist = curr_dist_0 + curr_dist_1
                    goal_0 = node
                for n in self.neighbors(state.board, node):
                    if n not in visited_0.keys():
                        dist_0[n] = dist_0[node] + 1
                        visited_0[n] = 1
                        queue_0.append(n)
            else:
                node = queue_1.popleft()
                visited_1[node] = True
                if max_dist > curr_dist_0 + curr_dist_1:
                    max_dist = curr_dist_0 + curr_dist_1
                    goal_1 = node
                for n in self.neighbors(state.board, node):
                    if n not in visited_1.keys():
                        dist_1[n] = dist_1[node] + 1
                        visited_1[n] = 1
                        queue_1.append(n)
            """
            TO-DO: Implement region comparison by comparing distance from start_0 to that from start_1
            """
        return dist_0, dist_1

    def get_index(self, row, col, length):
        return row * length + col

    def coords_from_index(self, index, length):
        col = index % length
        row = (index - col) // length
        return row, col

    def neighbors(self, gameboard, ind):
        board_x = len(gameboard[0])
        # board_y = len(gameboard)
        ind_y, ind_x = self.coords_from_index(ind, board_x)
        neighbors = []
        if gameboard[ind_y + 1][ind_x] is not CellType.WALL and gameboard[ind_y + 1][ind_x] is not CellType.BARRIER:
            neighbors += [self.get_index(ind_y + 1, ind_x, board_x)]
        if gameboard[ind_y - 1][ind_x] is not CellType.WALL and gameboard[ind_y + 1][ind_x] is not CellType.BARRIER:
            neighbors += [self.get_index(ind_y - 1, ind_x, board_x)]
        if gameboard[ind_y][ind_x + 1] is not CellType.WALL and gameboard[ind_y + 1][ind_x] is not CellType.BARRIER:
            neighbors += [self.get_index(ind_y, ind_x + 1, board_x)]
        if gameboard[ind_y][ind_x - 1] is not CellType.WALL and gameboard[ind_y + 1][ind_x] is not CellType.BARRIER:
            neighbors += [self.get_index(ind_y, ind_x - 1, board_x)]
        """
        if ind_y + 1 < board_y:
            neighbors += [(ind_y + 1, ind_x)]
        if ind_y > 0:
            neighbors += [(ind_y - 1, ind_x)]
        if ind_x + 1 < board_x:
            neighbors += [(ind_y, ind_x + 1)]
        if ind_x > 0:
            neighbors += [(ind_y, ind_x - 1)]
        """
        return neighbors

    # Our AB-cutoff function
    def alpha_beta_cutoff(self, asp, cutoff_ply, eval_func):
        """
        This function:
        - searches through the asp using alpha-beta pruning
        - cuts off the search after cutoff_ply moves have been made.

        Inputs:
                asp - an AdversarialSearchProblem
                cutoff_ply- an Integer that determines when to cutoff the search
                        and use eval_func.
                eval_func - a function that evaluates the state of the game and returns the how good it is for the
                player who uses ab-cutoff.

        Output: an action
        """

        moves = {}
        start = asp.get_start_state()
        actions = asp.get_available_actions(start)
        start_player = start.player_to_move()

        for a in actions:
            v = self.min_value_ab(asp, asp.transition(start, a), start_player, -math.inf, math.inf, cutoff_ply, 1, eval_func)
            moves[v] = a
        best = max(moves.keys())

        return moves[best]

    def eval_func(self, state, player_index):
        """
        TODO: Implement Voronoi regions as the first step of sophistication
        """
        p0 = state.player_locs[player_index]
        p1 = state.player_locs[(state.ptm + 1) % 2]

        start_p0 = self.get_index(p0[0], p0[1], len(state.board[0]))
        start_p1 = self.get_index(p1[0], p1[1], len(state.board[0]))

        Voronoi_p0 = 0 # voronoi region for first player
        Voronoi_p1 = 0 # voronoi region for second player

        dist_0, dist_1 = self.bi_directional_bfs(start_p0,start_p1)
        locations = dist_0.keys()
        for loc in locations:
            if dist_0[loc] > dist_1[loc]:
                Voronoi_p1 += 1
            if dist_0[loc] < dist_1[loc]:
                Voronoi_p0 += 1
            if dist_0[loc] == dist_1[loc]:
                continue

        if Voronoi_p0 > Voronoi_p1:
            if player_index == 0:
                return 1
            if player_index == 1:
                return -1

        if Voronoi_p0 < Voronoi_p1:
            if player_index == 1:
                return 1
            if player_index == 0:
                return -1

        if Voronoi_p0 == Voronoi_p1:
            return 0
        # 1. Calculate the regions based on state and player_index
        # 2. Make a comparison of these two regions
        # 3. Output a value for how good the state is for player with player_index


    # max_value helper for ab cutoff
    def max_value_ab(self, asp, state, player, alpha, beta, cutoff, depth, evaluation):
        if cutoff > 0 and cutoff == depth and evaluation is not None:
            if asp.is_terminal_state(state):
                return asp.evaluate_state(state)[player]
            else:
                return evaluation(state)

        if asp.is_terminal_state(state):
            return asp.evaluate_state(state)[player]
        v = -math.inf
        for a in asp.get_available_actions(state):
            v = max(v, self.min_value_ab(asp, asp.transition(state, a), player, alpha, beta, cutoff, depth + 1,
                                         evaluation))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    # min_value helper for ab cutoff
    def min_value_ab(self, asp, state, player, alpha, beta, cutoff, depth, evaluation):
        if cutoff > 0 and cutoff == depth and evaluation is not None:
            if asp.is_terminal_state(state):
                return asp.evaluate_state(state)[player]
            else:
                return evaluation(state)

        if asp.is_terminal_state(state):
            return asp.evaluate_state(state)[player]
        v = math.inf
        for a in asp.get_available_actions(state):
            v = min(v, self.max_value_ab(asp, asp.transition(state, a), player, alpha, beta, cutoff, depth + 1,
                                         evaluation))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v


class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision
