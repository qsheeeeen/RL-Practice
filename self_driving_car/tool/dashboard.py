import time

import pygame


class Dashboard(object):
    def __init__(self, width=600, high=260):
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        self.INFO_KET_LEFT = 260
        self.INFO_VALUE_LEFT = 450

        self.FRONT_SIZE = 30

        pygame.init()

        self.screen = pygame.display.set_mode([width, high])
        pygame.display.set_caption("Dashboard")

        self.screen.fill(self.WHITE)
        self.font = pygame.font.Font(None, self.FRONT_SIZE)

        self.last_time = time.time()

    def update(self, image, info=None):
        y_position = 10
        self.screen.fill(self.WHITE)

        image_surface = pygame.surfarray.make_surface(image)
        image_surface = pygame.transform.rotate(image_surface, -90)
        image_surface = pygame.transform.flip(image_surface, True, False)
        image_surface = pygame.transform.scale(image_surface, (240, 240))
        self.screen.blit(image_surface, (10, 10))

        current_time = time.time()
        fps = 1. / (current_time - self.last_time)
        self.last_time = current_time
        self._screen_print('FPS:', [self.INFO_KET_LEFT, y_position])
        self._screen_print(fps, [self.INFO_VALUE_LEFT, y_position])

        y_position += self.FRONT_SIZE
        for head in info.keys():
            self._screen_print(head + ':', [self.INFO_KET_LEFT, y_position])
            y_position += self.FRONT_SIZE

        y_position = 10
        y_position += self.FRONT_SIZE
        for data in info.values():
            self._screen_print(data, [self.INFO_VALUE_LEFT, y_position])
            y_position += self.FRONT_SIZE

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

    def _screen_print(self, string, position):
        if isinstance(string, float):
            text_bit_map = self.font.render(str(string)[:6], True, self.BLACK)
        else:
            text_bit_map = self.font.render(str(string), True, self.BLACK)
        self.screen.blit(text_bit_map, position)
