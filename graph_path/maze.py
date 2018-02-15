import numpy as np
import itertools as it

import pygame
from pygame.locals import *

class MazeNode:
    def __init__(self):
        self.directions = [0, 0, 0, 0]



cell_colors = (255, 255, 255), (0, 0, 0), (255, 0, 0)
cell_margin = 2


class Maze:

    def __init__(self, structure):
        self.height, self.width = structure.shape
        self.idx_set = set(it.product(range(self.width), range(self.height)))
        self.maze_array = structure

        self.end_coordinate     = (-1, -1)


    def check_neighbor(self, p):
        return p.x >= 0 and p.y >= 0 and p.x < self.width and p.y < self.height

    def get_coordinates(self, elem):
        return elem[1], elem[0]

    def render_get_cell_rect(self, coordinates, screen):
        x, y = coordinates
        cell_width = screen.get_width() / self.width
        adjusted_width = cell_width - cell_margin
        return pygame.Rect(x * cell_width + cell_margin / 2,
                           y * cell_width + cell_margin / 2,
                           adjusted_width, adjusted_width)

    def render_maze(self):
        screen = pygame.display.set_mode((320, 320))
        screen.fill(cell_colors[1])

        for x in range(self.height):
            for y in range(self.width):
                screen.fill(cell_colors[self.maze_array[y][x]], self.render_get_cell_rect((x, y), screen))

        pygame.display.update()



if __name__ == '__main__':

    pygame.init()
    screen = pygame.display.set_mode((160, 160))
    screen.fill(cell_colors[1])

    structure = [[0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                 [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                 [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    structure = np.array(structure, np.uint8)

    maze = Maze(structure)

    print(maze.maze_array)
    maze.render_maze()

    while True:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                exit()


    pygame.quit()

