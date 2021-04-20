import os
import sys
import random
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import pygame
import pygame.gfxdraw
import pygame.locals as loc
from pygame.colordict import THECOLORS

import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam



def fitness_test(score, fitness):
    fig, axes = plt.subplots(1, 2)
    a1, a2 = axes
    a1.scatter(score, fitness)
    a1.set_xlabel('Score')
    a1.set_ylabel('Fitiness')
    a2.scatter(score, fitness / fitness.sum())
    a2.set_xlabel('Score')
    a2.set_ylabel('$\\rm p_i$')
    plt.show()
    return fig, axes


# match pymunk coordinates to pygame (positive = up/right)
pymunk.pygame_util.positive_y_is_up = True

# pygame directory
DIRPATH = os.path.dirname(os.path.abspath(__file__))

# training directory
TRAINPATH = os.path.join(DIRPATH, 'training_results')

# path to tracking data
MJCDATA = os.path.join(DIRPATH, 'MJC-DATA.txt')

# path to save neural network weights
NNPATH = os.path.join(DIRPATH, 'nn-weights.hdf')

if not os.path.isfile(MJCDATA):
    open(MJCDATA, 'w').close()

# Opens game in the center of the self.screen
os.environ['SDL_VIDEO_CENTERED'] = '1'

# GLOBALS
GRAVITY = (0, -1800.0)
MAX_VELOCITY = 1800

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


class BouncePhysics(object):
    def __init__(self):
        self.score = 0
        self.highscore = 0
        self.gameover = False
        self.startscreen = False

        # build PyMunk physics space
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY
        self.static_body = self.space.static_body

        self.build_walls()
        self.build_ball()
        self.define_collisions()

        self.obstacles = []
        self._addnow = 1
        self._count = 0

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

    def add_point(self, arbitor, space, data):
        if not self.startscreen:
            self.score += 1

    def build_walls(self):
        # Add walls around pygame self.screen
        pos = [WIN_SIZE[0] - 1, WIN_SIZE[1]]
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
        self.ball = pymunk.Body(mass, inertia)
        self.ball.position = WIN_SIZE[0] / 2, WIN_SIZE[1] - 50
        self.shape = pymunk.Circle(self.ball, radius, (0, 0))
        self.shape.elasticity = 1.0
        self.shape.friction = 0
        self.shape.color = THECOLORS['white']
        self.shape.collision_type = collision_types['player']
        self.space.add(self.ball, self.shape)

    def collision(self, arbitor, space, data):
        self.gameover = True

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

    def get_features(self, n_obst=6):
        """
        Creates feature vector with all data normalized: [0, 1]
        features:
        posx, posy, velx, vely, (obstacle bounding box) * n_obst
        """
        pos = np.array(self.ball.position)
        pos[0] /= WIN_SIZE[0]
        pos[1] /= WIN_SIZE[1]

        # get normalized velocity
        vel = np.array(self.ball.velocity)
        vel = (vel + MAX_VELOCITY) / (2 * MAX_VELOCITY)

        # get data for <n_obst> obstacles
        opos = np.zeros((n_obst, 4))
        for i, obst in enumerate(self.obstacles):
            if i == n_obst:
                break
            bb = obst[1].bb
            left = max(bb.left / WIN_SIZE[0], 0)
            right = min(bb.right / WIN_SIZE[0], 1)
            top = min(bb.top / WIN_SIZE[1], 1)
            bot = max(bb.bottom / WIN_SIZE[1], 0)
            opos[i, :] = [left, right, top, bot]

        # combine data into feature vector
        features = np.zeros(4 + 4 * n_obst)
        features[:2] = pos
        features[2:4] = vel
        features[4:] = opos.flatten()
        features = features.reshape((1, len(features)))
        return features

    def get_nn_prediction(self, nn):
        """
        nn (keras.Model): neural network to make prediction
        """
        features = self.get_features()
        res = nn(features, training=False)[0]

        # max output in NN determines input for game
        inp = int(tf.math.argmax(res)) - 1

        return inp

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

    def restart_game(self):
        if self.score > self.highscore:
            self.highscore = self.score
        self.score = 0
        self._addnow = 1
        self._count = 0

        # clear self.obstacles
        for o in self.obstacles:
            self.space.remove(*o)
        self.obstacles = []

        # move ball to center and remove velocity
        self.ball.position = WIN_SIZE[0] / 2, WIN_SIZE[1] - 50
        self.ball.velocity = (0, 0)
        self.gameover = False

    def sim(self, nn, num_sims=1):
        scores = np.zeros(num_sims)
        for i in range(num_sims):
            self.restart_game()
            while 1:
                # GET INPUT FROM NEURAL NETWORK
                inp = self.get_nn_prediction(nn)

                # INPUT TO MOVE BALL
                self.update_ball(inp)
                self.update_obstacles()
                self.update_score()

                # Update physics
                self.space.step(1.0 / FPS)

                # end game if collision occurs or if sim has already been done
                if self.gameover:
                    break

            # track score
            scores[i] = self.score
        return scores

    def update_ball(self, move=0):
        """
        move (int): 0 = no input
                    1 = move right
                   -1 = move left
        """
        self.ball.velocity += Vec2d(60 * move, 0)
        # do not let velocity exceed MAX_VELOCITY
        if not (-MAX_VELOCITY < self.ball.velocity[0] < MAX_VELOCITY):
            scalex = MAX_VELOCITY * (self.ball.velocity[0] /
                                     abs(self.ball.velocity[0]))
            self.ball.velocity = Vec2d(scalex, self.ball.velocity[1])

    def update_obstacles(self):
        """
        Update obstacles in game
        - adds new obstacles
        - removes obstacles off screen
        """
        # HANDLE OBSTACLES
        if self._count == self._addnow:
            self.add_obstacle()
            self._count = 0
            self._addnow = random.randint(60, 100)

        # remove obstacles off screen
        remove_obst = [o for o in self.obstacles
                       if o[0].position[0] < -
                       WIN_SIZE[0] - (WIN_SIZE[0] / 10)]
        for r in remove_obst:
            self.obstacles.remove(r)
            self.space.remove(*r)
        self._count += 1

    def update_score(self):
        # Player earns points for every obstacle moved past
        for o in self.obstacles:
            if o[1].color != THECOLORS["skyblue"]:
                if self.ball.position[0] - WIN_SIZE[0] > o[0].position[0]:
                    o[1].color = THECOLORS["skyblue"]
                    self.score += 3 if o[1].top else 2


class PyBounce(BouncePhysics):
    def __init__(self):
        # initialize physics
        super().__init__()

        self.highscore = 0
        self.startscreen = True
        self.firstplay = True

        # neural network bot
        self.nn = None

        # data collection attributes
        self.data_collected = 0

        # Main self.screen
        pygame.init()
        self.screen = pygame.display.set_mode(WIN_SIZE)
        pygame.display.update()

        # Background color
        self.background = THECOLORS["white"]

        # Caption & icon
        pygame.display.set_caption("pybounce")
        # icon = pygame.image.load('PATH2ICON')
        # pygame.display.set_icon(iconpath)

        # Hide mouse
        pygame.mouse.set_visible(False)

        # self.font
        self.font = pygame.font.SysFont("Courier", 14)
        self.boldfont = pygame.font.SysFont("Courier", 16, True)

        # make pymunk draw to pygame screen
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def collision(self, arbitor, space, data):
        text = ["GAME OVER"]
        text.append("SCORE: {}".format(str(self.score)))
        if self.score > self.highscore:
            text.append("NEW HIGHSCORE!")
        text.append("(SPACE TO RESTART)")
        self.draw_textbox(text)

        # reset data collected and delete last <n> lines from tracked data
        if self.nn is None:
            print(self.data_collected)
            self.delete_data(min(self.data_collected, 50))
            self.data_collected = 0

        self.gameover = True
        pygame.display.flip()

    def draw_ball(self):
        # get position of ball from pymunk
        pos = [int(self.ball.position[0]),
               int(WIN_SIZE[1] - self.ball.position[1])]
        x, y = pos

        # draw smooth black outline of ball
        pygame.gfxdraw.aacircle(self.screen, x, y, 26, THECOLORS['black'])
        pygame.gfxdraw.filled_circle(self.screen, x, y, 26, THECOLORS['black'])

        # draw smooth red circle over pymunk self.shape
        pygame.gfxdraw.aacircle(self.screen, x, y, 25, THECOLORS['red'])
        pygame.gfxdraw.filled_circle(self.screen, x, y, 25, THECOLORS['red'])

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
        if self.firstplay:
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

    def draw_restart(self):
        """
        Updates UI when game is restarting
        """
        self.startscreen = True

        # blank out self.screen
        self.screen.fill(self.background)
        self.draw_score()
        self.draw_highscore(self.highscore)
        self.draw_startinfo()
        self.space.remove(self.ball, self.shape)
        self.space.add(self.ball, self.shape)
        self.space.debug_draw(self.draw_options)
        self.draw_ball()
        pygame.display.flip()

    def run(self, nn=None):
        self.nn = nn
        self.restart_game()
        delay = 0
        while 1:
            for event in pygame.event.get():
                # Clicking 'X' will close window
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == loc.KEYDOWN:
                    if event.key == loc.K_SPACE and self.startscreen:
                        self.firstplay = False
                        self.startscreen = False
                        self.gameover = False

            # handle user input
            pressed = pygame.key.get_pressed()
            # Esc key will exit game window
            if pressed[loc.K_ESCAPE]:
                pygame.quit()
                sys.exit()
            elif not (self.gameover or self.startscreen):
                if self.nn is not None:
                    inp = self.get_nn_prediction(nn)
                else:
                    # 0: do nothing
                    inp = 0
                    if pressed[loc.K_RIGHT]:
                        # 1: move right
                        inp = 1
                    elif pressed[loc.K_LEFT]:
                        # -1: move left
                        inp = -1

                    # don't save the first <n> steps
                    if delay == 30:
                        # only take 25% of "no input" moves for now to
                        # balance out dataset
                        if inp != 0 or random.random() < 0.25:
                            feature_ls = list(self.get_features()[0]) + [inp]
                            data = ', '.join(list(map(str, feature_ls)))
                            self.write_data(data)
                            self.data_collected += 1
                    else:
                        delay += 1

                # update ball based on input
                self.update_ball(inp)
            else:
                if pressed[loc.K_SPACE] and self.gameover:
                    self.restart_game()
                    self.draw_restart()
                    CLOCK.tick(FPS)
                    continue
                if not self.startscreen:
                    CLOCK.tick(FPS)
                    continue

            # blank screen
            self.screen.fill(self.background)

            # draw pymunk objects
            self.space.debug_draw(self.draw_options)

            # draw the ball
            self.draw_ball()

            if self.startscreen:
                self.draw_startinfo()
            else:
                self.update_obstacles()
                self.update_score()

            # update physics
            self.space.step(1.0 / FPS)

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

    def write_data(self, data):
        with open(MJCDATA, 'a') as fidw:
            fidw.write(data + '\n')

    def delete_data(self, num_lines):
        with open(MJCDATA, 'r') as fidr:
            lines = fidr.readlines()
        with open(MJCDATA, 'w') as fidw:
            fidw.writelines(lines[:-num_lines])


"""
Neural Network Bot Functions
"""


def read_in_data(shuffle=True):
    """
    Reads in data from me playing PyBounce

    KArgs:
    shuffle (bool): if True, randomize order of data
                    DEFAULT: True

    Returns:
    x: (n x 29) matrix
    y: (n x 3) one-hot-encoded outputs
    """
    data = np.loadtxt(MJCDATA, delimiter=',')
    if shuffle:
        np.random.shuffle(data)
    x = data[:, :-1]
    y = np.zeros((len(x), 3))
    for i, ans in enumerate(data[:, -1]):
        y[i, int(ans + 1)] = 1
    return x, y


def build_nn(node_arch=None, weights_path=None):
    """
    Builds a keras Sequential NN model for pybounce

    KArgs:
    node_arch (str): number of nodes in each hidden layer separated by "-"
                     DEFAULT: 64-64 (creates a 3-layer model of 64-64-3 nodes)

    weights_path (str): path to load in trained weights for the model
                        DEFAULT: None
    """
    print('Building NN...')
    nn = Sequential()

    # default architecture = 64-64-3 nodes
    if node_arch is None:
        node_arch = '64-64'

    # convert node architecture string to list of num_nodes
    nodes = list(map(int, node_arch.split('-')))

    # add first layer with specific input_shape
    nn.add(Dense(nodes.pop(0), input_shape=(28, ), activation='relu'))

    # add in remaining Dense layers
    for node in nodes:
        nn.add(Dense(node, activation='relu'))

    # add in output layer
    nn.add(Dense(3, activation='softmax'))

    # compile model
    nn.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # read in weights if a path is given and the file exists
    if weights_path is not None and os.path.isfile(weights_path):
        print('Reading in saved weights...')
        nn.load_weights(weights_path)

    return nn


def train_nn(weights_path=NNPATH):
    print('Reading in data...')
    x, y = read_in_data()

    # create the NN classifier
    classifier = KerasClassifier(build_fn=build_nn, epochs=14, batch_size=16)

    # run k-fold cross validation
    kfold = KFold(10, shuffle=True)
    res = cross_val_score(classifier, x, y, cv=kfold)
    print(res)
    classifier.model.save_weights(weights_path)

    # if train:
    #     print('Training NN...')
    #     nn.fit(x, y, epochs=200, batch_size=10)
    #     nn.save_weights(weights_path)

    return classifier


def plot_training(epochs=100, node_arch=None, play=False):
    """
    Builds model, trains with 80-20 train-test split, plots learning curve,
    saves model and plots, runs PyBounce with trained model bot (optional)

    KArgs:
    epochs (int): number of epochs to train the model
                  DEFAULT: 100

    node_arch (str): number of nodes in each hidden layer separated by "-"
                     DEFAULT: 64-64 (creates a 3-layer model of 64-64-3 nodes)

    play (bool): if True, PyBounce starts with newly trained model
                 DEFAULT: False
    """
    x, y = read_in_data()
    model = build_nn(node_arch=node_arch)
    history = model.fit(x, y, validation_split=0.2, epochs=epochs,
                        batch_size=20)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    res_path = os.path.join(DIRPATH, 'training_results', timestamp)
    if not os.path.isdir(res_path):
        os.mkdir(res_path)

    # get number of nodes in each layer
    units = '-'.join(map(lambda z: str(z.units), model.layers))

    # save trained model (includes architecture and weights)
    model.save(os.path.join(res_path, 'model_%s.h5' % units))

    # save weights as a file (NOTE: weights are saved in model file)
    weights_path = os.path.join(res_path, timestamp + '_' + units)
    model.save_weights(weights_path)

    # plot/save train and test accuracy
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train')
    ax.plot(history.history['val_accuracy'], label='Validation')
    ax.set_title('Model Accuracy (%s)' % units)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend()
    fig.savefig(os.path.join(res_path, 'accuracy.png'))

    # plot/save train and test loss
    fig2, ax2 = plt.subplots()
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss (%s)' % units)
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    fig2.savefig(os.path.join(res_path, 'loss.png'))

    # run game with model
    if play:
        pb = PyBounce()
        pb.run(nn=model)


def load_last_model():
    """
    Get last model trained and saved in training_results dir
    """
    path = os.path.join(TRAINPATH, sorted(os.listdir(TRAINPATH))[-1])
    modelpath = [f for f in os.listdir(path) if f.endswith('.h5')][0]
    nn = load_model(os.path.join(path, modelpath))
    return nn


if __name__ == "__main__":
    # CURRENT BEST ARCHITECTURE: 256-256-256-256-3
    # node_arch = '-'.join(['256'] * 4)
    # plot_training(epochs=200, node_arch=node_arch, play=False)

    nn = load_last_model()
    PyBounce().run(nn=nn)
