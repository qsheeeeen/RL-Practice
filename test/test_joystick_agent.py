# coding: utf-8

# Copied from PyGame documents.

import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class text_print:
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def print(self, screen, text_string):
        text_bit_map = self.font.render(text_string, True, BLACK)
        screen.blit(text_bit_map, [self.x, self.y])
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10


pygame.init()

# Set the width and height of the screen [width,height]
size = [500, 700]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("My Game")

# Loop until the user clicks the close button.
done = False

# Initialize the joysticks
pygame.joystick.init()

# Get ready to print
text_print = text_print()

# -------- Main Program Loop -----------
while not done:
    # EVENT PROCESSING STEP
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop

        # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
        if event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
        if event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")

    # DRAWING STEP
    # First, clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.
    screen.fill(WHITE)
    text_print.reset()

    # Get count of joysticks
    joystick_count = pygame.joystick.get_count()

    text_print.print(screen, "Number of joysticks: {}".format(joystick_count))
    text_print.indent()

    # For each joystick:
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()

        text_print.print(screen, "Joystick {}".format(i))
        text_print.indent()

        # Get the name from the OS for the controller/joystick
        name = joystick.get_name()
        text_print.print(screen, "Joystick name: {}".format(name))

        # Usually axis run in pairs, up/down for one, and left/right for
        # the other.
        axes = joystick.get_numaxes()
        text_print.print(screen, "Number of axes: {}".format(axes))
        text_print.indent()

        for i in range(axes):
            axis = joystick.get_axis(i)
            text_print.print(screen, "Axis {} value: {:>6.3f}".format(i, axis))
        text_print.unindent()

        buttons = joystick.get_numbuttons()
        text_print.print(screen, "Number of buttons: {}".format(buttons))
        text_print.indent()

        for i in range(buttons):
            button = joystick.get_button(i)
            text_print.print(screen, "Button {:>2} value: {}".format(i, button))
        text_print.unindent()

        # Hat switch. All or nothing for direction, not like joysticks.
        # Value comes back in an array.
        hats = joystick.get_numhats()
        text_print.print(screen, "Number of hats: {}".format(hats))
        text_print.indent()

        for i in range(hats):
            hat = joystick.get_hat(i)
            text_print.print(screen, "Hat {} value: {}".format(i, str(hat)))
        text_print.unindent()

        text_print.unindent()

    pygame.display.flip()

pygame.quit()
