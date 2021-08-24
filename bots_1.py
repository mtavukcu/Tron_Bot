#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
from collections import deque
from queue import Queue


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
        """
        return self.alpha_beta_cutoff(asp, 6, self.eval_func)

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
    def bfs_helper(self, asp, state, player_start):
        dist_to_player = {player_start: 0.0}
        visited = {player_start: True}
        q = deque()
        q.append(player_start)

        while q:
            node = q.pop()
            # print("Node: ", node)
            # if node in visited.keys():
                # print("Node ", node, " already visited")
                # continue
            # print("Continue")
            # visited[node] = True

            for n in self.neighbors(asp, state.board, node):
                # print("Neighbor: ", n)
                if n not in visited.keys():
                    # print("Neighbor not visited")
                    dist_to_player[n] = dist_to_player[node] + 1
                    q.append(n)
                    visited[node] = True
            # print("Iteration complete")
        # print("Done")
        dist_to_player.pop(player_start)
        return dist_to_player

    def bi_directional_bfs(self, asp, state, start_p0, start_p1):
        visited_p0 = {start_p0: True}
        visited_p1 = {start_p1: True}

        queue_p0 = deque()
        queue_p1 = deque()

        queue_p0.append(start_p0)
        queue_p1.append(start_p1)

        while queue_p0 and queue_p1:
            if queue_p0:
                node0 = queue_p0.popleft()
            else:
                node0 = None
            if queue_p1:
                node1 = queue_p1.popleft()
            else:
                node1 = None

            """
            if node0 in visited_p0.keys() or node0 in visited_p1.keys():
                continue

            if node1 in visited_p0.keys() or node0 in visited_p1.keys():
                continue
            """

            neighbors_p0 = self.neighbors(asp, state.board, node0)
            neighbors_p1 = self.neighbors(asp, state.board, node1)

            if neighbors_p0:
                for n in neighbors_p0:
                    if n not in visited_p0.keys() and n not in visited_p1.keys() and n not in neighbors_p1:
                        visited_p0[n] = True
                        queue_p0.append(n)

            if neighbors_p1:
                for n in neighbors_p1:
                    if n not in visited_p1.keys() and n not in visited_p0.keys() and n not in neighbors_p0:
                        visited_p1[n] = True
                        queue_p1.append(n)

        if queue_p0 and not queue_p1:
            while queue_p0:
                node0 = queue_p0.popleft()
                neighbors_p0 = self.neighbors(asp, state.board, node0)
                for n in neighbors_p0:
                    if n not in visited_p0.keys() and n not in visited_p1.keys():
                        visited_p0[n] = True
                        queue_p0.append(n)
        if queue_p1 and not queue_p0:
            while queue_p1:
                node = queue_p1.popleft()
                neighbors_p1 = self.neighbors(asp, state.board, node)
                for n in neighbors_p1:
                    if n not in visited_p1.keys() and n not in visited_p0.keys():
                        visited_p1[n] = True
                        queue_p1.append(n)

        return visited_p0, visited_p1

        """
            curr_dist_0 = dist_0[queue_0[0]]
            curr_dist_1 = dist_1[queue_1[0]]

            
            if goal_0 and goal_1 and curr_dist_0 + curr_dist_1 > min_dist:
                return dist_0, dist_1
            

            if curr_dist_0 <= curr_dist_1:
                node = queue_0.popleft()
                
                if node in visited_0.keys() and curr_dist_0 + curr_dist_1 < min_dist:
                    min_dist = curr_dist_0 + curr_dist_1
                    goal_0 = node
                
                if node in visited_0.keys():
                    continue

                visited_0[node] = True

                for n in self.neighbors(asp, state.board, node):
                    if n not in visited_0.keys():
                        dist_0[n] = dist_0[node] + 1
                        queue_0.append(n)
            else:
                node = queue_1.popleft()
                
                if node in visited_1.keys() and curr_dist_0 + curr_dist_1 < min_dist:
                    min_dist = curr_dist_0 + curr_dist_1
                    goal_1 = node
                
                if node in visited_1.keys():
                    continue

                visited_1[node] = True

                for n in self.neighbors(asp, state.board, node):
                    if n not in visited_1.keys():
                        dist_1[n] = dist_1[node] + 1
                        queue_1.append(n)
        dist_0.pop(start_0)
        dist_1.pop(start_1)
        return dist_0, dist_1
        """

    def get_index(self, row, col, length):
        return row * length + col

    def coords_from_index(self, index, length):
        col = index % length
        row = (index - col) // length
        return row, col

    def neighbors(self, asp, gameboard, ind):
        if ind:
            board_x = len(gameboard[0])
            # board_y = len(gameboard)
            ind_y, ind_x = self.coords_from_index(ind, board_x)

            neighbors = []
            if not asp.is_cell_player(gameboard, (ind_y + 1, ind_x)) and gameboard[ind_y + 1][ind_x] is not CellType.WALL \
                    and gameboard[ind_y + 1][ind_x] is not CellType.BARRIER:
                # neighbors += [self.get_index(ind_y + 1, ind_x, board_x)]
                # print("Up: ", gameboard[ind_y + 1][ind_x])
                neighbors.append(self.get_index(ind_y + 1, ind_x, board_x))
            if not asp.is_cell_player(gameboard, (ind_y - 1, ind_x)) and gameboard[ind_y - 1][ind_x] is not CellType.WALL \
                    and gameboard[ind_y - 1][ind_x] is not CellType.BARRIER:
                # neighbors += [self.get_index(ind_y - 1, ind_x, board_x)]
                # print("Down: ", gameboard[ind_y - 1][ind_x])
                neighbors.append(self.get_index(ind_y - 1, ind_x, board_x))
            if not asp.is_cell_player(gameboard, (ind_y, ind_x + 1)) and gameboard[ind_y][ind_x + 1] is not CellType.WALL \
                    and gameboard[ind_y][ind_x + 1] is not CellType.BARRIER:
                # neighbors += [self.get_index(ind_y, ind_x + 1, board_x)]
                # print("Right: ", gameboard[ind_y][ind_x + 1])
                neighbors.append(self.get_index(ind_y, ind_x + 1, board_x))
            if not asp.is_cell_player(gameboard, (ind_y, ind_x - 1)) and gameboard[ind_y][ind_x - 1] is not CellType.WALL \
                    and gameboard[ind_y][ind_x - 1] is not CellType.BARRIER:
                # neighbors += [self.get_index(ind_y, ind_x - 1, board_x)]
                # print("Left: ", gameboard[ind_y][ind_x - 1])
                neighbors.append(self.get_index(ind_y, ind_x - 1, board_x))
            return neighbors
        else:
            return None

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
        start_player = start.ptm
        actions = list(TronProblem.get_safe_actions(start.board,start.player_locs[start_player],start.player_has_armor[start_player]))
        #actions = asp.get_safe_actions(start.board, start.player_locs[start_player],
        #                               start.player_has_armor(start_player))
        for a in actions:
            v = self.min_value_ab(asp, asp.transition(start, a), start_player, -math.inf, math.inf, cutoff_ply, 1,
                                  eval_func)

            print("Move value: ", v)
            moves[v] = a

        if moves.keys():
            best = max(moves.keys())
            print("Best move value: ", best)
            return moves[best]
        else:
            print("ono")
            return "U"

    def eval_func(self, asp, state):
        """
        TODO: Implement Voronoi regions as the first step of sophistication
        """

        p0 = state.player_locs[state.ptm]
        p1 = state.player_locs[(state.ptm + 1) % 2]

        board = state.board

        # print("Board: ", board)

        start_p0 = self.get_index(p0[0], p0[1], len(board[0]))
        start_p1 = self.get_index(p1[0], p1[1], len(board[0]))

        # voronoi_p0 = 0  # voronoi region for first player
        # voronoi_p1 = 0  # voronoi region for second player

        p0_area, p1_area = self.bi_directional_bfs(asp, state, start_p0, start_p1)

        # p0_dists = self.bfs_helper(asp, state, start_p0)
        # p1_dists = self.bfs_helper(asp, state, start_p1)

        # print("Done with bfs")

        # print("P0 distances: ", sorted(p0_dists.keys()))
        # print("P1 distances: ", sorted(p1_dists.keys()))

        """
        print("Player 0 dict: ", sorted(p0_dists.keys()))
        print("Player 1 dict: ", sorted(p1_dists.keys()))

        print("Player 0 dict: ",len(p0_dists.keys()))
        print("Player 1 dict: ", len(p1_dists.keys()))
        """

        max_x_bound = len(board[0]) - 2
        max_y_bound = len(board) - 2

        # dist_limit = max_x_bound + max_y_bound

        # print("Size of player 0's area: ", len(p0_area.keys()))
        # print("Size of player 1's area: ", len(p1_area.keys()))

        voronoi_p0 = len(p0_area.keys())
        voronoi_p1 = len(p1_area.keys())
        """
        for cell in p0_dists.keys():
            dist_from_p0 = p0_dists[cell]
            dist_from_p1 = p1_dists[cell]

            if dist_from_p0 < dist_from_p1:
                voronoi_p0 += 1
            elif dist_from_p1 < dist_from_p0:
                voronoi_p1 += 1
            else:
                continue
        """
        # print(max_x_bound)
        # print(max_y_bound)
        # print("Voronoi region for player 0: ", voronoi_p0)
        # print("Voronoi region for player 1: ", voronoi_p1)

        max_area = (max_x_bound * max_y_bound)
        # min_area = -max_area
        min_area = 0

        v_diff = voronoi_p0 - voronoi_p1
        scaled_v_diff = (v_diff - min_area) / (max_area - min_area)

        # v_ratio = voronoi_p0 / voronoi_p1
        # scaled_v_ratio = v_ratio / max_area

        print("Voronoi score for player" , scaled_v_diff)

        return scaled_v_diff

    # max_value helper for ab cutoff
    def max_value_ab(self, asp, state, player, alpha, beta, cutoff, depth, evaluation):
        if cutoff > 0 and cutoff == depth and evaluation is not None:
            if asp.is_terminal_state(state):
                return asp.evaluate_state(state)[player]
            else:
                return evaluation(asp, state)

        if asp.is_terminal_state(state):
            return asp.evaluate_state(state)[player]
        v = -math.inf
        for a in asp.get_safe_actions(state.board, state.player_locs[player], state.player_has_armor(player)):
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
                return evaluation(asp, state)

        if asp.is_terminal_state(state):
            return asp.evaluate_state(state)[player]
        v = math.inf
        for a in asp.get_safe_actions(state.board, state.player_locs[player], state.player_has_armor(player)):
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
