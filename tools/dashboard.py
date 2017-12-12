# coding: utf-8

import time

import pygame


class Dashboard(object):
    def __init__(self):
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        pygame.init()

        self.screen = pygame.display.set_mode([800, 240])
        pygame.display.set_caption("Dashboard")

        self.screen.fill(self.WHITE)
        self.font = pygame.font.Font(None, 30)

        self.last_time = time.time()

    def update(self, image, info):
        y_position = 10
        self.screen.fill(self.WHITE)

        image_surface = pygame.surfarray.make_surface(image)
        self.screen.blit(image_surface, (0, 0))

        current_time = time.time()
        fps = 1. / (current_time - self.last_time)
        self.screen_print('FPS:', [330, y_position])
        self.screen_print(str(fps), [530, y_position])
        self.last_time = current_time

        y_position += 30
        for head in info.keys():
            self.screen_print(str(head), [330, y_position])
            y_position += 30

        y_position = 10
        y_position += 30
        for data in info.values():
            self.screen_print(str(data), [530, y_position])
            y_position += 30

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

    def screen_print(self, string, position):
        text_bit_map = self.font.render(str(string), True, self.BLACK)
        self.screen.blit(text_bit_map, position)


if __name__ == '__main__':
    import numpy as np

    dashboard = Dashboard()

    done = False
    while not done:
        img = np.random.randint(0, 255, (320, 240, 3), np.uint8)

        name = ('Motor signal:', 'Steering signal:', 'Car speed:')
        num = (1 / 3., .2 / 3, .3 / 4)
        inf = dict(zip(name, num))

        done = dashboard.update(img, inf)
