import pygame 
from PIL import Image 

pygame.init()
pygame.display.set_caption("Crop window")

def setUp(path):
    px = pygame.image.load(path)
    screen = pygame.display.set_mode(px.get_rect()[2:])
    screen.blit(px, px.get_rect())
    pygame.display.flip()

    return screen, px 

def displayImage(screen, px, top_left, prior):
    #ensure that the react always has a positive width, height
    x, y = top_left
    width = pygame.mouse.get_pos()[0] - top_left[0]
    height = pygame.mouse.get_pos()[1] - top_left[1]

    if width < 0:
        x += width 
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)
    
    #eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current 
    if current == prior:
        return current 
    
    #draw transparent box and blit it onto canvas 
    screen.blit(px, px.get_rect())
    img = pygame.Surface((width, height))
    img.fill((128, 128, 128))
    pygame.draw.rect(img, (32, 32, 32), img.get_rect(), 1)
    img.set_alpha(128)
    screen.blit(img, (x, y))
    pygame.display.flip()

    #return current box extents
    return (x, y, width, height)


def mainLoop(screen, px):
    top_left = None 
    bottom_right = None 
    prior = None
    n = 0
    while n != 1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not top_left:
                    top_left = event.pos 
                else:
                    bottom_right = event.pos
                    n = 1
        if top_left:
            prior = displayImage(screen, px, top_left, prior)
    return (top_left + bottom_right)

def crop_image_shape(img_url):
    screen, px = setUp(img_url)
    left, upper, right, lower = mainLoop(screen , px)

    # reset the position 
    if right < left:
        left, right = right, left 
    if lower < upper:
        lower, upper = upper, lower
    img = Image.open(img_url)
    to_crop_width, to_crop_height = img.size
    pygame.display.quit() 

    return (left, upper, right, lower, to_crop_width, to_crop_height)