import pygame
import numpy as np

# 1. Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))

# 2. Main Loop
running = True
drone_x = 400
while running:
    # Handle Close Button
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Logic: Move a "Drone" (Red Square)
    drone_x += np.random.normal(0, 1) # Browninan motion

    # Render
    screen.fill((0, 0, 0)) # Black background
    pygame.draw.rect(screen, (255, 0, 0), (drone_x, 300, 20, 20)) # Red Drone
    pygame.display.flip()

pygame.quit()