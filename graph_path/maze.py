import numpy as np
import itertools as it

import pygame
from pygame.locals import *

class MazeNode:
    def __init__(self):
        self.directions = [0, 0, 0, 0]



cell_colors = (255, 255, 255), (0, 255, 0)
cell_margin = 2

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def get_tuple(self):
        return (self.x, self.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

class Node:

    dir_deltas = [Point(0, 1),
                  Point(1, 0),
                  Point(0, -1),
                  Point(-1, 0)]

    def __init__(self, node_coord):
        # Up, right, down, left
        self.directions = [0, 0, 0, 0]
        self.coord      = node_coord
        self.next_nodes = [None, None, None, None]

    def get_source_idx(self, src_coord):
        delta = src_coord - self.coord
        # print('Delta: {} - {}'.format(self.coord.get_tuple(), src_coord.get_tuple()))

        for idx, dlt in enumerate(Node.dir_deltas):
            if delta.x == dlt.x and delta.y == dlt.y:
                return idx

        # exit(1)
        return -1 #Node.dir_deltas.index(delta) 

    def edge_get_next_point(self, src_coord):
        src_dir_idx = self.get_source_idx(src_coord)

        next_dir_idxs = [i for i, x in enumerate(self.directions) if i != src_dir_idx and x == 1]
        if len(next_dir_idxs) != 1:
            print('Achtung!')
            exit(1)

        next_dir_idx = next_dir_idxs[0]

        return self.coord + Node.dir_deltas[next_dir_idx]

class Maze:

    START_NODE_ID   = 1
    END_NODE_ID     = 2

    def __init__(self, structure):
        self.height, self.width = structure.shape
        self.idx_set = set(it.product(range(self.width), range(self.height)))
        self.maze_array = structure


        # self.end_coordinate     = (-1, -1)

        self.nodes = {}
        self.edges = {}

        for x in range(self.height):
            for y in range(self.width):
                point = Point(x, y)
                elem = self.get_maze_element(point)
                if self.is_element_vacant(elem):
                    node = Node(point)

                    for i, delta in enumerate(Node.dir_deltas):
                        neighbour = self.get_maze_element(point + delta)
                        if neighbour is not None and self.is_element_vacant(neighbour):
                            node.directions[i] = 1

                    if elem == Maze.START_NODE_ID:
                        self.start_node = node
                        self.nodes[point.get_tuple()] = node
                        continue

                    if elem == Maze.END_NODE_ID:
                        self.end_node   = node
                        self.nodes[point.get_tuple()] = node
                        continue

                    if sum(node.directions) == 2:
                        self.edges[point.get_tuple()] = node
                    else:
                        self.nodes[point.get_tuple()] = node

        for key in self.nodes:
            curr_node  = self.nodes[key]

            print('Current node: {}'.format(key))
            
            for dir_idx, d in enumerate(curr_node.directions):
                curr_point = curr_node.coord
                if d == 1:
                    print('Direction: {}'.format(dir_idx))
                    next_point = curr_point + Node.dir_deltas[dir_idx]

                    while (1):
                        print('Next: {}'.format(next_point.get_tuple()))
                        if next_point.get_tuple() in self.nodes:
                            print('>>Node')
                            curr_node.next_nodes[dir_idx] = self.nodes[next_point.get_tuple()]
                            break;
                        elif next_point.get_tuple() in self.edges:
                            print('>>Edge')
                            edge = self.edges[next_point.get_tuple()]
                            prev_point = curr_point
                            curr_point = next_point
                            next_point = edge.edge_get_next_point(prev_point)
                        else:
                            print('>>Failed')
                            return


        for key in sorted(self.nodes, key=lambda key: key[0]):
            print(key, self.nodes[key], self.nodes[key].directions, self.nodes[key].next_nodes)

    def is_element_vacant(self, elem):
        if elem == 0 or elem == 1 or elem == 2:
            return True

        return False

    def get_maze_element(self, p):
        if p.x < 0 or p.x >= self.width:
            return None

        if p.y < 0 or p.y >= self.height:
            return None

        return self.maze_array[self.height-p.y-1][p.x];


    def render_get_cell_rect(self, coordinates, screen):
        x, y = coordinates
        y = self.height - 1 - y
        cell_width = screen.get_width() / self.width
        adjusted_width = cell_width - cell_margin
        return pygame.Rect(x * cell_width + cell_margin / 2,
                           y * cell_width + cell_margin / 2,
                           adjusted_width, adjusted_width)


    def render_maze(self):
        screen = pygame.display.set_mode((320, 320))
        screen.fill((0, 0, 0))

        for key in self.edges:
            screen.fill(cell_colors[0], self.render_get_cell_rect(key, screen))

        for key in self.nodes:
            screen.fill(cell_colors[1], self.render_get_cell_rect(key, screen))

        pygame.display.update()



if __name__ == '__main__':

    pygame.init()

    structure = [[0, 0, 0, 0, 8, 8, 0, 8, 8, 0, 8, 8],
                 [8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [8, 0, 8, 8, 8, 8, 8, 0, 8, 8, 0, 8],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
                 [8, 0, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8],
                 [8, 0, 8, 0, 8, 0, 8, 8, 8, 8, 8, 8],
                 [0, 0, 0, 0, 8, 0, 8, 8, 8, 8, 8, 8],
                 [0, 8, 8, 0, 8, 2, 8, 8, 8, 8, 8, 8],
                 [0, 8, 8, 0, 8, 0, 8, 8, 8, 8, 8, 8],
                 [0, 0, 0, 0, 8, 0, 8, 8, 8, 8, 8, 8],
                 [0, 8, 8, 0, 8, 0, 8, 8, 8, 8, 8, 8],
                 [0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8]]
    structure = np.array(structure, np.uint8)

    maze = Maze(structure)

    # print(maze.maze_array)
    maze.render_maze()

    while True:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                exit()


    pygame.quit()

