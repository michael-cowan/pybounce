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

# Opens game in the center of the self.screen
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Get previous self.highscores
# HS_DICT = {}
# if not os.path.exists('hs.txt'):
#     open('hs.txt', 'w').close()
# else:
#     with open('hs.txt', 'r') as fid:
#         pass


# Refresh rate
CLOCK = pygame.time.Clock()
FPS = 60

collision_types = {'player': 1,
                   'obstacle': 2,
                   'wall_bottom': 3,
                   'wall_right': 4,
                   'wall_top': 5,
                   'wall_left': 6
                   }

IPHONE_RATIO = 1.78
WIDTH = 1000

# Size of pygame window
WIN_SIZE = (WIDTH, int(WIDTH / IPHONE_RATIO))


class PyBounce(object):
    def __init__(self):
        # Game globals
        self.first_play = True
        self.startscreen = True
        self.gameover = True
        self.add_tail = False
        self.tail = []
        self.tail_length = 20
        self.count = 0
        self.score = 0
        self.highscore = 0
        self.addnow = 1

        # Main self.screen
        pygame.init()
        self.screen = pygame.display.set_mode(WIN_SIZE)
        pygame.display.update()

        # Background color
        self.background = THECOLORS["white"]

        # Caption & icon
        pygame.display.set_caption("pybounce")
        pygame.display.set_icon(
                        pygame.image.load("C:/users/mcowa/pictures/icon.png"))

        # Hide mouse
        pygame.mouse.set_visible(False)

        # self.font
        self.font = pygame.font.SysFont("Courier", 14)
        self.boldfont = pygame.font.SysFont("Courier", 16, True)

        # Physics
        self.space = pymunk.Space()
        self.space.gravity = 0, -1800.0
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        # Walls
        self.static_body = self.space.static_body

        self.obstacles = []

        self.build_walls()
        self.build_ball()
        self.define_collisions()

    def build_walls(self):
        # Add walls around pygame self.screen
        pos = WIN_SIZE[0]-1, WIN_SIZE[1]
        sec = [0, 0]
        i = 0
        wall = ['bottom', 'right', 'top', 'left']
        for j in range(4):
            first = sec[:]
            sec[i] += pos[i]
            seg = pymunk.Segment(self.static_body, first, sec, 0)
            seg.elasticity = 1.0
            seg.friction = 0
            seg.color = THECOLORS["black"]
            seg.collision_type = collision_types['wall_{}'.format(wall[j])]
            self.space.add(seg)
            if i:
                pos = [k * -1 for k in pos]
                i = 0
            else:
                i = 1

    def build_ball(self):
        # Add the ball
        mass = 1
        radius = 25
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        self.body = pymunk.Body(mass, inertia)
        self.body.position = WIN_SIZE[0] / 2, WIN_SIZE[1] - 50
        self.shape = pymunk.Circle(self.body, radius, (0, 0))
        self.shape.elasticity = 1.0
        self.shape.friction = 0
        self.shape.color = THECOLORS['white']
        self.shape.collision_type = collision_types['player']
        self.space.add(self.body, self.shape)

    def random_rect(self):
        x1 = WIN_SIZE[0]
        x2 = x1 + random.randint(WIN_SIZE[0] // 15, WIN_SIZE[0] // 10)

        y2 = random.randint(WIN_SIZE[1] // 5, WIN_SIZE[1] // 2)

        # Bottom or top of self.screen
        if random.random() < 0.5:
            top = False
            y1 = 0
        else:
            top = True
            y1 = WIN_SIZE[1]
            y2 = y1 - y2

        return ([x1, y1], [x1, y2], [x2, y2], [x2, y1]), top

    def add_obstacle(self):
        # Create obstacle
        obst_body = pymunk.Body(body_type=1)
        obst_body.velocity = Vec2d(-175, 0)  # -150 ideal
        coords, top = self.random_rect()
        # obst = pymunk.Segment(obst_body, coords[0], coords[1], 0)
        obst = pymunk.Poly(obst_body, coords, None, 1)
        obst.elasticity = 1.0
        obst.color = THECOLORS["blue"]
        obst.collision_type = collision_types['obstacle']
        obst.top = top
        self.obstacles.append((obst_body, obst))
        self.space.add(obst_body, obst)

    def next_obstacle(self):
        return random.randint(60, 100)

    def draw_textbox(self, text, border=True, padding=15):
        if not isinstance(text, list):
            text = [text]

        text_y = (WIN_SIZE[1] / 2) - (padding * len(text))
        max_x = max([self.boldfont.size(t)[0] for t in text])
        rect_x = ((WIN_SIZE[0] - max_x) / 2) - padding
        dist_x = max_x + (2 * padding)
        rect_y = text_y - padding
        dist_y = (padding * 2 * len(text)) + padding
        pygame.draw.rect(self.screen, THECOLORS["white"],
                         (rect_x, rect_y, dist_x, dist_y))
        if border:
            pygame.draw.rect(self.screen, THECOLORS["black"],
                             (rect_x, rect_y, dist_x, dist_y), 2)

        for t in text:
            text_render = self.boldfont.render(t, 1, THECOLORS["black"])
            self.screen.blit(text_render,
                             ((WIN_SIZE[0] - self.boldfont.size(t)[0]) / 2,
                              text_y))
            text_y += padding * 2

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[1])**2 + (p1[1] - p2[1])**2)

    def restart_game(self):
        self.startscreen = True
        self.tail = []
        if self.score > self.highscore:
            self.highscore = self.score
        self.count = self.score = 0
        self.addnow = 1

        # clear self.obstacles
        for o in self.obstacles:
            self.space.remove(o)
        self.obstacles = []

        # blank out self.screen
        self.screen.fill(self.background)

        # move ball to center and remove velocity
        self.body.position = WIN_SIZE[0] / 2, WIN_SIZE[1] - 50
        self.body.velocity = (0, 0)

        self.draw_score()
        self.draw_highscore(self.highscore)
        self.draw_startinfo()
        self.space.remove(self.body, self.shape)
        self.space.add(self.body, self.shape)
        self.space.debug_draw(self.draw_options)
        self.draw_player()
        pygame.display.flip()

    def draw_player(self):
        pos = [int(self.body.position[0]),
               int(WIN_SIZE[1] - self.body.position[1])]
        # draw smooth black outline of ball
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 26,
                                THECOLORS['black'])
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 26,
                                     THECOLORS['black'])

        # draw smooth circle over pymunk self.shape
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 25, player_color)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 25,
                                     player_color)

    def draw_score(self):
        self.screen.blit(self.font.render("Score: %d" % (self.score), 1,
                         THECOLORS["black"]), (10, 10))

    def draw_highscore(self, s):
        text = "High Score: %d" % (s)
        text_render = self.font.render(text, 1, THECOLORS["black"])
        self.screen.blit(text_render, (WIN_SIZE[0] -
                                       self.font.size(text)[0] - 10,
                                       10))

    def draw_startinfo(self):
        space_txt = '(PRESS SPACE TO START)'
        text = []
        if self.first_play:
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
            self.draw_textbox(text)
        else:
            text = ['pybounce',
                    '',
                    space_txt
                    ]
            self.draw_textbox(text, padding=20)

    def collision(self, arbitor, space, data):
        text = ["GAME OVER"]
        text.append("SCORE: {}".format(str(self.score)))
        if self.score > self.highscore:
            text.append("NEW HIGHSCORE!")
        text.append("(SPACE TO RESTART)")
        self.draw_textbox(text)
        self.gameover = True
        pygame.display.flip()

    def add_point(self, arbitor, space, data):
        if not self.startscreen:
            self.score += 1

    def define_collisions(self):
        # Function to run when player collides with an obstacle
        h0 = self.space.add_collision_handler(collision_types['player'],
                                              collision_types['obstacle'])
        h0.separate = self.collision

        # +1 for every bounce on floor
        h1 = self.space.add_collision_handler(collision_types['player'],
                                              collision_types['wall_bottom'])
        h1.separate = self.add_point

        # +1 for every bounce on right wall
        h2 = self.space.add_collision_handler(collision_types['player'],
                                              collision_types['wall_right'])
        h2.separate = self.add_point

    def run(self):
        # Main game loop
        self.restart_game()

        while 1:
            # Used to handle game events (user inputs)
            for event in pygame.event.get():
                # Clicking 'X' will close window
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == loc.KEYDOWN:
                    if event.key == loc.K_SPACE and self.startscreen:
                        self.first_play = False
                        self.startscreen = False
                        self.gameover = False

            # Esc key will exit game window
            pressed = pygame.key.get_pressed()
            if pressed[loc.K_ESCAPE]:
                pygame.quit()
                sys.exit()
            elif not self.gameover:
                if pressed[loc.K_RIGHT]:
                    self.body.velocity += Vec2d(60, 0)  # 60
                elif pressed[loc.K_LEFT]:
                    self.body.velocity += Vec2d(-60, 0)
            else:
                if pressed[loc.K_SPACE] and self.gameover:
                    self.restart_game()
                    CLOCK.tick(FPS)
                    continue
                if not self.startscreen:
                    CLOCK.tick(FPS)
                    continue

            MAX_VELOCITY = 1800
            if not (-MAX_VELOCITY < self.body.velocity[0] < MAX_VELOCITY):
                scalex = (MAX_VELOCITY *
                          (self.body.velocity[0] / abs(self.body.velocity[0])))
                self.body.velocity = Vec2d(scalex, self.body.velocity[1])

            if self.count == self.addnow:
                self.add_obstacle()
                self.count = 0
                self.addnow = self.next_obstacle()

            remove_obst = [o for o in self.obstacles
                           if o[0].position[0] < -
                           WIN_SIZE[0] - (WIN_SIZE[0] / 10)]
            for r in remove_obst:
                self.obstacles.remove(r)
                self.space.remove(r)

            # Blanks self.screen
            self.screen.fill(self.background)

            # Draw tail
            if self.add_tail:
                ball_pos = [int(self.body.position[0]),
                            int(WIN_SIZE[1] - self.body.position[1])]
                [pygame.draw.circle(self.screen, THECOLORS['red'], p, 2)
                 for p in tail if distance(p, ball_pos) - 10 > radius]

            self.space.debug_draw(self.draw_options)

            if self.startscreen:
                self.draw_startinfo()
            else:
                self.count += 1

            # get position of pymunk player
            ball_pos = [int(self.body.position[0]),
                        int(WIN_SIZE[1] - self.body.position[1])]

            self.draw_player()

            # Update physics
            self.space.step(1.0 / FPS)

            # Store tail points if add_tail option is True
            if self.add_tail:
                self.tail.append([int(self.body.position[0]),
                                  int(WIN_SIZE[1] - self.body.position[1])])

            if len(self.tail) == self.tail_length:
                self.tail.pop(0)

            # Player earns points for every obstacle moved past
            for o in self.obstacles:
                if o[1].color != THECOLORS["skyblue"]:
                    if self.body.position[0] - WIN_SIZE[0] > o[0].position[0]:
                        o[1].color = THECOLORS["skyblue"]
                        self.score += 3 if o[1].top else 2

            # Writes self.score & self.highscore to self.screen
            self.draw_score()
            self.draw_highscore(self.highscore)

            # Updates self.screen and pauses for specified frame rate
            pygame.display.flip()
            # CLOCK.tick()
            CLOCK.tick(FPS)

        # end the game
        time.sleep(0.75)
        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    bounce = PyBounce()
    bounce.run()
