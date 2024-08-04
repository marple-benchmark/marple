import math
import time
import unittest

from abc import ABC, abstractmethod
from heapq import heappush, heappop
from typing import Iterable, Union, TypeVar, Generic

# infinity as a constant
Infinite = float("inf")

# introduce generic type
T = TypeVar("T")


class AStar(ABC, Generic[T]):

    __slots__ = ()

    class SearchNode:
        """Representation of a search node"""

        __slots__ = ("data", "gscore", "fscore",
                     "closed", "came_from", "out_openset")

        def __init__(
            self, data: T, gscore: float = Infinite, fscore: float = Infinite
        ) -> None:
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, b: "AStar.SearchNode") -> bool:
            return self.fscore < b.fscore

    class SearchNodeDict(dict):
        def __missing__(self, k):
            v = AStar.SearchNode(k)
            self.__setitem__(k, v)
            return v

    @abstractmethod
    def heuristic_cost_estimate(self, current: T, goal: T) -> float:
        """
        Computes the estimated (rough) distance between a node and the goal.
        The second parameter is always the goal.
        This method must be implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def distance_between(self, n1: T, n2: T) -> float:
        """
        Gives the real distance between two adjacent nodes n1 and n2 (i.e n2
        belongs to the list of n1's neighbors).
        n2 is guaranteed to belong to the list returned by the call to neighbors(n1).
        This method must be implemented in a subclass.
        """

    @abstractmethod
    def neighbors(self, node: T) -> Iterable[T]:
        """
        For a given node, returns (or yields) the list of its neighbors.
        This method must be implemented in a subclass.
        """
        raise NotImplementedError

    def is_goal_reached(self, current: T, goal: T) -> bool:
        """
        Returns true when we can consider that 'current' is the goal.
        The default implementation simply compares `current == goal`, but this
        method can be overwritten in a subclass to provide more refined checks.
        """
        return current == goal

    def reconstruct_path(self, last: SearchNode, reversePath=False) -> Iterable[T]:
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from

        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def astar(
        self, start: T, start_dir: T, goal: T, reversePath: bool = False
    ) -> Union[Iterable[T], None]:
        if self.is_goal_reached(start, goal):
            return [start]
        searchNodes = AStar.SearchNodeDict()

        # 0 right, 1 down, 2 left, 3 up
        if start_dir == 0:
            last_move = (1, 0)
        elif start_dir == 1:
            last_move = (0, 1)
        elif start_dir == 2:
            last_move = (-1, 0)
        elif start_dir == 3:
            last_move = (0, -1)
        else:
            last_move = None
        startNode = searchNodes[start] = AStar.SearchNode(
            start, gscore=0.0, fscore=self.heuristic_cost_estimate(start, goal, last_move)
        )
        openSet: list = []
        heappush(openSet, startNode)
        maximum_time = 120
        start_time = time.time()
        while openSet:
            current = heappop(openSet)
            # print("current", current.data, current.came_from.data if current.came_from else None,
            #       current.gscore, current.fscore, current.closed, current.out_openset)
            last_move = (current.data[0] - current.came_from.data[0], current.data[1] -
                         current.came_from.data[1]) if current.came_from else last_move
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath)
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + self.distance_between(
                    current.data, neighbor.data, last_move
                )
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                last_move = (neighbor.data[0] - current.data[0], neighbor.data[1] -
                             current.data[1]) if current else last_move
                neighbor.fscore = tentative_gscore + self.heuristic_cost_estimate(
                    neighbor.data, goal, last_move)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
            if time.time() - start_time > maximum_time:
                return None
        return None


class MazeSolver(AStar):

    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self, maze, step_size):
        # self.lines = maze.strip().split('\n')
        self.m = maze
        self.width = maze.grid.width
        self.height = maze.grid.height
        self.step_size = step_size
        self.last_move = None
        # self.width = len(maze[0])
        # self.height = len(maze)

    def normalize_vector(self, v):
        norm = math.hypot(v[0], v[1])
        if norm == 0:
            return [0, 0]  # Handle the zero vector case
        return [v[0] / norm, v[1] / norm]

    def heuristic_cost_estimate(self, n1, n2, last_move=None):
        """computes the 'direct' distance between two (x,y) tuples"""
        (x1, y1) = n1
        (x2, y2) = n2
        need_move = [None, None]
        if x2 - x1 > 0:
            need_move[0] = 1
        elif x2 - x1 < 0:
            need_move[0] = -1
        else:
            need_move[0] = 0
        if y2 - y1 > 0:
            need_move[1] = 1
        elif y2 - y1 < 0:
            need_move[1] = -1
        else:
            need_move[1] = 0

        need_move = tuple(self.normalize_vector(need_move))

        # must_cost = math.hypot(x2 - x1, y2 - y1)
        must_cost = abs(x2 - x1) + abs(y2 - y1)

        if last_move and last_move != need_move:
            extra_cost = 0
            # calculate the dot product of last_move and need_move
            dot_product = last_move[0] * \
                need_move[0] + last_move[1] * need_move[1]
            if dot_product > 0 and dot_product <= 1:
                extra_cost = 1
            elif dot_product == 1:
                extra_cost = 0
            elif dot_product < 0:
                extra_cost = 2

            extra_cost = extra_cost * 0

            # print("--------------calculate heuristic cost estimate",
            #       n1, n2, last_move, f"{must_cost} + {extra_cost} = {must_cost + extra_cost}")
            return must_cost + extra_cost
        # print("--------------calculate heuristic cost estimate",
        #       n1, n2, last_move, must_cost)

        return must_cost

    def distance_between(self, n1, n2, last_move=None):
        """this method returns Infinity if two 'neighbors' are blocked by objects other than door"""
        (x1, y1) = n1
        (x2, y2) = n2
        if x1 == x2:
            for y in range(min(y1, y2), max(y1, y2)+1):
                if 0 <= y < self.m.grid.height and not self.m.grid.is_empty(x1, y):
                    furniture, obj = self.m.grid.get(x1, y)
                    if not (furniture is not None and furniture.type == 'door' and obj is None):
                        return Infinite
        else:
            for x in range(min(x1, x2), max(x1, x2)+1):
                if 0 <= x < self.m.grid.width and not self.m.grid.is_empty(x, y1):
                    furniture, obj = self.m.grid.get(x, y1)
                    if not (furniture is not None and furniture.type == 'door' and obj is None):
                        return Infinite

        # Add an extra step if a turn is required
        if last_move and last_move != (x2 - x1, y2 - y1):
            # print("--------------calculate distance between",
            #       n1, n2, last_move, math.hypot(x2 - x1, y2 - y1) + max(abs(last_move[0] - (x2 - x1)),  abs(last_move[1] - (y2 - y1))))
            return math.hypot(x2 - x1, y2 - y1) + max(abs(last_move[0] - (x2 - x1)),  abs(last_move[1] - (y2 - y1)))

        # print("--------------calculate distance between",
        #       n1, n2, last_move, math.hypot(x2 - x1, y2 - y1))
        return math.hypot(x2 - x1, y2 - y1)

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        x, y = node
        neighbors = []
        base_action_space = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        action_space = base_action_space.copy()
        for step in range(2, self.step_size+1):
            action_space += [(x * step, y * step)
                             for x, y in base_action_space]
        candidate_neighbors = [(x + dx, y + dy) for dx, dy in action_space]
        for nx, ny in candidate_neighbors:
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if x == nx:
                    for dy in range(min(y, ny), max(y, ny) + 1):
                        if dy == y:
                            continue
                        if 0 <= dy < self.m.grid.height:
                            if not self.m.grid.is_empty(x, dy):
                                furniture, obj = self.m.grid.get(x, dy)
                                if not (furniture is not None and furniture.type == 'door' and obj is None):
                                    candidate_neighbors.remove((nx, ny))
                                    break
                        else:
                            break
                else:
                    for dx in range(min(x, nx), max(x, nx) + 1):
                        if dx == x:
                            continue
                        if 0 <= dx < self.m.grid.width:
                            if not self.m.grid.is_empty(dx, y):
                                furniture, obj = self.m.grid.get(dx, y)
                                if not (furniture is not None and furniture.type == 'door' and obj is None):
                                    candidate_neighbors.remove((nx, ny))
                                    break
                        else:
                            break
        return candidate_neighbors


# def make_maze(w=20, h=20):
#     """returns an ascii maze as a string"""
#     from random import shuffle, randrange
#     vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
#     ver = [["|  "] * w + ['|'] for _ in range(h)] + [[]]
#     hor = [["+--"] * w + ['+'] for _ in range(h + 1)]

#     def walk(x, y):
#         vis[y][x] = 1

#         d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
#         shuffle(d)
#         for (xx, yy) in d:
#             if vis[yy][xx]:
#                 continue
#             if xx == x:
#                 hor[max(y, yy)][x] = "+  "
#             if yy == y:
#                 ver[y][max(x, xx)] = "   "
#             walk(xx, yy)

#     walk(randrange(w), randrange(h))
#     result = ''
#     for (a, b) in zip(hor, ver):
#         result = result + (''.join(a + ['\n'] + b)) + '\n'

#     print(result)
#     return result.strip()

def make_maze(w=20, h=20):
    """returns an ascii maze as a string"""
    m = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         ]
    return m


def drawmaze(maze, set1=[], set2=[], c='#', c2='*'):
    """returns an ascii maze, drawing eventually one (or 2) sets of positions.
        useful to draw the solution found by the astar algorithm
    """
    set1 = list(set1)
    set2 = list(set2)
    # print("initial", maze)
    # lines = maze.strip().split('\n')
    width = len(maze[0])
    height = len(maze)
    result = ''
    for j in range(height):
        for i in range(width):
            if (i, j) in set1:
                maze[j][i] = c
            elif (i, j) in set2:
                maze[j][i] = c2
            else:
                pass
            result = result + str(maze[j][i])
        result = result + '\n'
    print(result)
    print(set1, set2)
    return result


def solve_maze():
    # generate an ascii maze
    size = 20
    m = make_maze(size, size)

    # what is the size of it?
    w = len(m[0])
    h = len(m)

    start = (10, 15)  # we choose to start at the upper left corner
    goal = (10, 1)  # we want to reach the lower right corner

    # let's solve it
    foundpath = MazeSolver(m).astar(start, goal)
    return drawmaze(m, list(foundpath))


class MazeTests(unittest.TestCase):
    def test_solve_maze(self):
        solve_maze()


if __name__ == '__main__':
    # print(solve_maze())
    solve_maze()
