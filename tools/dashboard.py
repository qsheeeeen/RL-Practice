# coding: utf-8

import time

import pygame


class Dashboard(object):
    def __init__(self):
        self.__BLACK = (0, 0, 0)
        self.__WHITE = (255, 255, 255)

        pygame.init()

        self.__screen = pygame.display.set_mode([600, 240])
        pygame.display.set_caption("Dashboard")

        self.__screen.fill(self.__WHITE)
        self.__font = pygame.font.Font(None, 30)

        self.__last_time = time.time()

    def update(self, image: np.ndarray, info: dict):
        y_position = 10
        self.__screen.fill(self.__WHITE)
        finish = False

        image_surface = pygame.surfarray.make_surface(image)
        self.__screen.blit(image_surface, (0, 0))

        current_time = time.time()
        fps = 1. / (current_time - self.__last_time)
        self.__screen_print('FPS:', [330, y_position])
        self.__screen_print(str(fps), [530, y_position])
        self.__last_time = current_time
        y_position += 30

        for head in info.keys():
            self.__screen_print(str(head), [330, y_position])
            y_position += 30

        y_position = 10
        y_position += 30
        for data in info.values():
            self.__screen_print(str(data), [530, y_position])
            y_position += 30

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finish = True

        return finish

    def __screen_print(self, string, position):
        text_bit_map = self.__font.render(str(string), True, self.__BLACK)
        self.__screen.blit(text_bit_map, position)


if __name__ == '__main__':
    import numpy as np

    dashboard = Dashboard()

    done = False
    while not done:
        image = np.random.randint(0, 255, (320, 240, 3), np.uint8)

        name = ('Motor signal:', 'Steering signal:', 'Car speed:')
        num = (1., .2, .3)
        info = dict(zip(name, num))

        done = dashboard.update(image, info)
