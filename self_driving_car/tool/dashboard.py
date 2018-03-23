import time

import pygame


class Dashboard(object):
    def __init__(self, width=600, high=240):
        self._BLACK = (0, 0, 0)
        self._WHITE = (255, 255, 255)

        pygame.init()

        self._screen = pygame.display.set_mode([width, high])
        pygame.display.set_caption("Dashboard")

        self._screen.fill(self._WHITE)
        self._font = pygame.font.Font(None, 30)

        self.last_time = time.time()

    def update(self, image, info):
        if image.max() <= 1:
            image *= 256

        y_position = 10
        self._screen.fill(self._WHITE)

        image_surface = pygame.surfarray.make_surface(image)
        self._screen.blit(image_surface, (0, 0))

        current_time = time.time()
        fps = 1. / (current_time - self.last_time)
        self._screen_print('FPS:', [330, y_position])
        self._screen_print(fps, [530, y_position])
        self.last_time = current_time

        y_position += 30
        for head in info.keys():
            self._screen_print(head + ':', [330, y_position])
            y_position += 30

        y_position = 10
        y_position += 30
        for data in info.values():
            self._screen_print(data, [530, y_position])
            y_position += 30

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

    def _screen_print(self, string, position):
        if isinstance(string, float):
            text_bit_map = self._font.render(str(string)[:6], True, self._BLACK)
        else:
            text_bit_map = self._font.render(str(string), True, self._BLACK)
        self._screen.blit(text_bit_map, position)
