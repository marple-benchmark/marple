import random


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end_pos, step_size):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    base_action_space = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # print("astart", start, end)
    # Env
    grid_width = maze.grid.width
    grid_height = maze.grid.height

    # grid_width = len(maze)
    # grid_height = len(maze[0])

    start = (start[0], start[1])
    dir = random.sample(base_action_space, 1)[0]
    end = (end_pos[0] + dir[0], end_pos[1] + dir[1])
    while end[0] < 0 or end[0] > grid_width \
            or end[1] < 0 or end[1] > grid_height or not maze.grid.is_empty(*end):
        # while maze[end[0]][end[1]]:
        dir = random.sample(base_action_space, 1)[0]
        end = (end_pos[0] + dir[0], end_pos[1] + dir[1])

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    max_iter = 1000
    iter = 0
    while len(open_list) > 0 or iter < max_iter:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        # Generate children
        children = []
        # Adjacent squares
        action_space = base_action_space.copy()
        step_size = 1
        for step in range(1, step_size+1):
            action_space += [(x * step, y * step)
                             for x, y in base_action_space]

        for new_position in action_space:
            # Get node position
            node_position = (
                current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (grid_width - 1) or node_position[0] < 0 or node_position[1] > (grid_height - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            walkable = True
            for i in range(min(current_node.position[0], node_position[0]), max(current_node.position[0], node_position[0]) + 1):
                for j in range(min(current_node.position[1], node_position[1]), max(current_node.position[1], node_position[1]) + 1):
                    # if (i, j) != start and not maze.grid.is_empty(i, j):
                    #     furniture, obj = maze.grid.get(i, j)
                    #     if not (furniture is not None and furniture.type == 'door' and obj is None):
                    #         print(
                    #             f"{i}, {j} not equal to start or end {start}, {end} but is not empty")
                    #         print(f"not walkable {i, j}")
                    #         # if maze[i][j]:
                    #         walkable = False
                    if not maze.grid.is_empty(i, j):
                        furniture, obj = maze.grid.get(i, j)
                        if furniture is not None and furniture.type != 'door' or obj is not None:
                            print(
                                f"{i}, {j} not equal to start or end {start}, {end} but is not empty, {furniture}, {obj}")
                            walkable = False

            if not walkable:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) **
                       2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)
        iter += 1


def main():

    maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    start = (4, 12)
    end = (5, 8)

    path = astar(maze, start, end, 3)
    print(path)


if __name__ == '__main__':
    main()
