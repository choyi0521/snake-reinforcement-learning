import pygame
import numpy as np
from blocks import SnakeBlock

pygame.init()
size = (400, 300)
screen = pygame.display.set_mode(size)
pygame.display.set_caption('Game Title')

done = False
clock = pygame.time.Clock()

while not done:
    clock.tick(10)

    # Main Event Loop
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop

    # All drawing code happens after the for loop and but
    # inside the main while done==False loop.

    # Clear the screen and set the screen background
    screen.fill((255, 255, 255))

    '''
    Your Work.....
    '''
    pygame.draw.polygon(screen, SnakeBlock(1, 0).color, 20*SnakeBlock(0, 1).points)
    pygame.draw.polygon(screen, SnakeBlock(1, 0).color, 20*SnakeBlock(0, 2).points + np.array([20, 0]))
    pygame.draw.polygon(screen, SnakeBlock(1, 0).color, 20*SnakeBlock(0, 3).points + np.array([40, 0]))
    pygame.draw.polygon(screen, SnakeBlock(1, 0).color, 20*SnakeBlock(1, 2).points + np.array([60, 0]))
    pygame.draw.polygon(screen, SnakeBlock(1, 0).color, 20*SnakeBlock(1, 3).points + np.array([80, 0]))
    pygame.draw.polygon(screen, SnakeBlock(1, 0).color, 20*SnakeBlock(2, 3).points + np.array([100, 0]))
    # Go ahead and update the screen with what we've drawn.
    # This MUST happen after all the other drawing commands.
    pygame.display.flip()

pygame.quit()