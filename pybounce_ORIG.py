import os
import math
import sys
import random
import time
import pygame
import pygame.gfxdraw
import pygame.locals as loc
from pygame.colordict import THECOLORS
import pymunk
import pymunk.pygame_util
import pymunk.util
from pymunk import Vec2d

# PLAYER COLOR
player_color = THECOLORS['red']

# Opens game in the center of the screen
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Get previous highscores
HS_DICT = {}
if not os.path.exists('hs.txt'):
    open('hs.txt', 'w').close()
else:
    with open('hs.txt', 'r') as fid:
        pass


# Refresh rate
clock = pygame.time.Clock()
FPS = 60

collision_types = {'player': 1,
                   'obstacle': 2,
                   'wall_bottom': 3,
                   'wall_right': 4,
                   'wall_top': 5,
                   'wall_left': 6
                   }

IPHONE_RAT = 1.78
width = 1000

# Size of pygame window
win_size = (width, int(width / IPHONE_RAT))

# Main screen
pygame.init()
screen = pygame.display.set_mode(win_size)
pygame.display.update()

# Background color
background = THECOLORS["white"]

# Caption & icon
pygame.display.set_caption("pybounce")
pygame.display.set_icon(pygame.image.load("C:/users/mcowa/pictures/icon.png"))

# Hide mouse
pygame.mouse.set_visible(False)

# Font
font = pygame.font.SysFont("Courier", 14)
boldfont = pygame.font.SysFont("Courier", 16, True)

# Physics
space = pymunk.Space()
space.gravity = 0, -1800.0
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Walls
static_body = space.static_body

# Add walls around pygame screen
pos = win_size[0]-1, win_size[1]
sec = [0, 0]
i = 0
wall = ['bottom', 'right', 'top', 'left']
for j in range(4):
    first = sec[:]
    sec[i] += pos[i]
    seg = pymunk.Segment(static_body, first, sec, 0)
    seg.elasticity = 1.0
    seg.friction = 0
    seg.color = THECOLORS["black"]
    seg.collision_type = collision_types['wall_{}'.format(wall[j])]
    space.add(seg)
    if i:
        pos = [k * -1 for k in pos]
        i = 0
    else:
        i = 1

# Add the ball
mass = 1
radius = 25
inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
body = pymunk.Body(mass, inertia)
body.position = win_size[0] / 2, win_size[1] - 50
shape = pymunk.Circle(body, radius, (0,0))
shape.elasticity = 1.0
shape.friction = 0
shape.color = THECOLORS['white']
shape.collision_type = collision_types['player']
space.add(body, shape)


def random_rect():
    x1 = win_size[0]
    x2 = x1 + random.randint(win_size[0] // 15, win_size[0] // 10)

    y2 = random.randint(win_size[1] // 5, win_size[1] // 2)

    # Bottom or top of screen
    if random.random() < 0.5:
        top = False
        y1 = 0
    else:
        top = True
        y1 = win_size[1]
        y2 = y1 - y2

    return ([x1, y1], [x1, y2], [x2, y2], [x2, y1]), top


obstacles = []
# Create obstacle
def add_obstacle():
    obst_body = pymunk.Body(body_type=1)
    obst_body.velocity = Vec2d(-175, 0) # -150 ideal
    coords, top = random_rect()
    #obst = pymunk.Segment(obst_body, coords[0], coords[1], 0)
    obst = pymunk.Poly(obst_body, coords, None, 1)
    obst.elasticity = 1.0
    obst.color = THECOLORS["blue"]
    obst.collision_type = collision_types['obstacle']
    obst.top = top
    obstacles.append((obst_body, obst))
    space.add(obst_body, obst)


def next_obstacle():
    return random.randint(60, 100)  #60, 100


def draw_textbox(text, border=True, padding=15):
    if not isinstance(text, list):
        text = [text]

    text_y = (win_size[1] / 2) - (padding * len(text))
    max_x = max([boldfont.size(t)[0] for t in text])
    rect_x = ((win_size[0] - max_x) / 2) - padding
    dist_x = max_x + (2 * padding)
    rect_y = text_y - padding
    dist_y = (padding * 2 * len(text)) + padding
    pygame.draw.rect(screen, THECOLORS["white"], (rect_x, rect_y, dist_x, dist_y))
    if border:
        pygame.draw.rect(screen, THECOLORS["black"], (rect_x, rect_y, dist_x, dist_y), 2)

    for t in text:
        text_render = boldfont.render(t, 1, THECOLORS["black"])
        screen.blit(text_render, ((win_size[0] - boldfont.size(t)[0]) / 2, text_y))
        text_y += padding * 2


# Game globals
first = True
startscreen = True
gameover = True
add_tail = False
tail = []
tail_length = 20
count = 0
score = 0
highscore = 0
addnow = 1


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[1])**2 + (p1[1] - p2[1])**2)


def restart_game():
    global first, startscreen, tail, count, score, highscore, obstacles, addnow
    startscreen = True
    tail = []
    if score > highscore:
        highscore = score
    count = score = 0
    addnow = 1
    for o in obstacles:
        space.remove(o)
    obstacles = []
    screen.fill(background)
    body.position = win_size[0] / 2, win_size[1] - 50
    body.velocity = (0, 0)
    draw_score()
    draw_highscore(highscore)
    draw_startinfo(first)
    space.remove(body, shape)
    space.add(body, shape)
    space.debug_draw(draw_options)
    draw_player()
    pygame.display.flip()


def draw_player():
    pos = [int(body.position[0]), int(win_size[1] - body.position[1])]
    # draw smooth black outline of ball
    pygame.gfxdraw.aacircle(screen, pos[0], pos[1], 26, THECOLORS['black'])
    pygame.gfxdraw.filled_circle(screen, pos[0], pos[1], 26, THECOLORS['black'])

    # draw smooth circle over pymunk shape
    pygame.gfxdraw.aacircle(screen, pos[0], pos[1], 25, player_color)
    pygame.gfxdraw.filled_circle(screen, pos[0], pos[1], 25, player_color)

def draw_score():
    global score
    screen.blit(font.render("Score: %d" %(score), 1, THECOLORS["black"]), (10, 10))

def draw_highscore(s):
    text = "High Score: %d" %(s)
    text_render = font.render(text, 1, THECOLORS["black"])
    screen.blit(text_render, (win_size[0] - font.size(text)[0] - 10, 10))

def draw_startinfo(first_play):
    space_txt = '(PRESS SPACE TO START)'
    text = []
    if first_play:
        text = ["Welcome to pybounce!",
                "",
                "Move with L & R arrow keys to avoid the blue obstacles",
                "  while continuously bouncing off the walls & floor.  ",
                "",
                "SCORING",
                "+1 : bounce on floor or right wall",
                "+2 : bounce over floor obstacle   ",
                "+3 : bounce under ceiling obstacle",
                "",
                "GOOD LUCK!",
                space_txt
                ]
        draw_textbox(text)
    else:
        text = ['pybounce',
                '',
                space_txt
                ]
        draw_textbox(text, padding=20)

def collision(arbitor, space, data):
    global first, gameover, score, highscore
    first = False
    text = ["GAME OVER"]
    text.append("SCORE: {}".format(str(score)))
    if score > highscore:
        text.append("NEW HIGHSCORE!")
    text.append("(SPACE TO RESTART)")
    draw_textbox(text)
    gameover = True
    pygame.display.flip()

def add_point(arbitor, space, data):
    global score, startscreen
    if not startscreen:
        score += 1

# Function to run when player collides with an obstacle
h0 = space.add_collision_handler(collision_types['player'], collision_types['obstacle'])
h0.separate = collision

# +1 for every bounce on floor
h1 = space.add_collision_handler(collision_types['player'], collision_types['wall_bottom'])
h1.separate = add_point

# +1 for every bounce on right wall
h2 = space.add_collision_handler(collision_types['player'], collision_types['wall_right'])
h2.separate = add_point

# Main game loop
restart_game()
while 1:
    # Used to handle game events (user inputs)
    for event in pygame.event.get():
        # Clicking 'X' will close window
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == loc.KEYDOWN:
            if event.key == loc.K_SPACE and startscreen:
                startscreen = False
                gameover = False

    # Esc key will exit game window
    pressed = pygame.key.get_pressed()
    if pressed[loc.K_ESCAPE]:
        pygame.quit()
        sys.exit()
    elif not gameover:
        if pressed[loc.K_RIGHT]:
            body.velocity += Vec2d(60, 0) #60
        elif pressed[loc.K_LEFT]:
            body.velocity += Vec2d(-60, 0)
    else:
        if pressed[loc.K_SPACE] and gameover:
            restart_game()
            clock.tick(FPS)
            continue
        if not startscreen:
            clock.tick(FPS)
            continue

    MAX_VELOCITY = 1800
    if not (-MAX_VELOCITY < body.velocity[0] < MAX_VELOCITY):
        body.velocity = Vec2d(MAX_VELOCITY * (body.velocity[0] / abs(body.velocity[0])), body.velocity[1])

    if count == addnow:
        add_obstacle()
        count = 0
        addnow = next_obstacle()

    remove_obst = [o for o in obstacles if o[0].position[0] < -win_size[0] - (win_size[0] / 10)]
    for r in remove_obst:
        obstacles.remove(r)
        space.remove(r)

    # Blanks screen
    screen.fill(background)

    # Draw tail
    # ball_pos = [int(body.position[0]), int(win_size[1] - body.position[1])]
    # [pygame.draw.circle(screen, THECOLORS['red'], p, 2) for p in tail if distance(p, ball_pos) - 10 > radius]

    space.debug_draw(draw_options)

    if startscreen:
        draw_startinfo(first)
    else:
        count += 1

    # get position of pymunk player
    ball_pos = [int(body.position[0]), int(win_size[1] - body.position[1])]

    draw_player()

    # Update physics
    space.step(1.0 / FPS)

    # Store tail points if add_tail option is True
    if add_tail:
        tail.append([int(body.position[0]), int(win_size[1] - body.position[1])])

    if len(tail) == tail_length:
        tail.pop(0)

    # Player earns points for every obstacle moved past
    for o in obstacles:
        if o[1].color != THECOLORS["skyblue"]:
            if body.position[0] - win_size[0] > o[0].position[0]:
                o[1].color = THECOLORS["skyblue"]
                score += 3 if o[1].top else 2

    # Writes score & highscore to screen
    draw_score()
    draw_highscore(highscore)

    # Updates screen and pauses for specified frame rate
    pygame.display.flip()
    clock.tick(FPS)

time.sleep(0.75)
pygame.quit()
sys.exit()