import sys
import random
import gym 
import numpy as np 
import pygame
from gym import spaces
from scipy.spatial import distance


class SnakeGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    difficulty = 25
    
    # Window size --> smaller window is easier to find food
    frame_size_x = 480
    frame_size_y = 360
    
    # Colors (R, G, B)
    black = pygame.Color(0, 0, 0)
    white = pygame.Color(255, 255, 255)
    red = pygame.Color(255, 0, 0)
    green = pygame.Color(0, 255, 0)
    blue = pygame.Color(0, 0, 255)
    
    # Action 
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    
    # set maximum steps    
    max_steps = 10000
    
    def __init__(self):
        super().__init__()
        pygame.init()
        
        # Initialise game window
        pygame.display.set_caption('Snake Eater')
        
        # FPS (frames per second) controller
        self.fps_controller = pygame.time.Clock()
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
        
        # Game variables
        self.snake_pos = [100, 50]
        self.prev_snake_pos = [100, 50]
        self.snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]
        self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10, 
                         random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawn = True
        self.score = 0
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.step_count = 0
        
        # action --> UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(4)
        
        # (x_snake, y_snake, x_food, y_food, x_snake_to_food, y_snake_to_food, snake_length)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
    
    def step(self, action):
        # 0 --> LEFT
        # 1 --> UP
        # 2 --> RIGHT
        # 3 --> DOWN
        
        reward = 0
        done = False
        info = {}
        
        if action == self.UP:
          self.change_to = 'UP'
        elif action == self.DOWN:
            self.change_to = 'DOWN'
        elif action == self.LEFT:
            self.change_to = 'LEFT'
        elif action == self.RIGHT:
            self.change_to = 'RIGHT'
            
        # Making sure the snake cannot move in the opposite direction instantaneously
        if self.change_to == 'UP':
            if self.direction != 'DOWN': self.direction = 'UP'
            else: self.direction = 'DOWN'
            
        elif self.change_to == 'DOWN':
            if self.direction != 'UP': self.direction = 'DOWN'
            else: self.direction = 'UP'
            
        elif self.change_to == 'LEFT':
            if self.direction != 'RIGHT': self.direction = 'LEFT'
            else: self.direction = 'RIGHT'
            
        elif self.change_to == 'RIGHT': 
            if self.direction != 'LEFT': self.direction = 'RIGHT'
            else: self.direction = 'LEFT'
            
        # Moving the snake
        if self.direction == 'UP':
            self.snake_pos[1] -= 10
        elif self.direction == 'DOWN':
            self.snake_pos[1] += 10
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        elif self.direction == 'RIGHT':
            self.snake_pos[0] += 10
        reward -= 1 # reduce reward in every move
        self.step_count += 1
        
        # get reward if move closer to food
        prev_distance = distance.euclidean(self.prev_snake_pos, self.food_pos)
        curr_distance = distance.euclidean(self.snake_pos, self.food_pos)
        if curr_distance < prev_distance:
            reward += 1
        else:
            reward -= 1
        self.prev_snake_pos = self.snake_pos.copy()
            
         # Snake body growing mechanism
        self.snake_body.insert(0, list(self.snake_pos))
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            reward += 100 # get huge reward if eats food
            self.score += 1 # score when render
            self.food_spawn = False
        else:
            self.snake_body.pop()
        
        # Spawning food on the screen
        if not self.food_spawn:
          self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10, 
                           random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawn = True
        
        # Game Over conditions
            # Getting out of bounds
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x - 10:
            info['GameOver'] = 'Out of bound X'
            done = True
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y - 10:
            info['GameOver'] = 'Out of bound Y'
            done = True
            
            # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                info['GameOver'] = 'Eat itself'
                done = True
                
            # reach maximum steps
        if self.step_count >= self.max_steps:
            info['GameOver'] = 'reach maximum steps'

        # new observation
        x_snake_to_food = abs(self.snake_pos[0] - self.food_pos[0])
        y_snake_to_food = abs(self.snake_pos[1] - self.food_pos[1])
        observation = np.array([self.snake_pos[0], self.snake_pos[1],
                                self.food_pos[0], self.food_pos[1],
                                x_snake_to_food, y_snake_to_food,
                                len(self.snake_body)])
        
        return observation, reward, done, info
    
    def reset(self):
        self.snake_pos = [100, 50]
        self.prev_snake_pos = [100, 50]
        self.snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]
        self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10, 
                         random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawn = True
        self.score = 0
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.step_count = 0
        
        x_snake_to_food = abs(self.snake_pos[0] - self.food_pos[0])
        y_snake_to_food = abs(self.snake_pos[1] - self.food_pos[1])
        observation = np.array([self.snake_pos[0], self.snake_pos[1],
                                self.food_pos[0], self.food_pos[1],
                                x_snake_to_food, y_snake_to_food,
                                len(self.snake_body)])
        
        return observation
    
    def render(self, mode='human'):
        # GFX
        self.game_window.fill(self.black)
        for pos in self.snake_body:
            # Snake body
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
            pygame.draw.rect(self.game_window, self.green, pygame.Rect(pos[0], pos[1], 10, 10))

        # Snake food
        pygame.draw.rect(self.game_window, self.white, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))
        self.show_score(1, self.white, 'consolas', 20)
        # Refresh game screen
        pygame.display.update()
        # Refresh rate
        self.fps_controller.tick(self.difficulty)
        
    def close (self):
      pygame.quit()
      sys.exit()
      
    def show_score(self, choice, color, font, size):
      score_font = pygame.font.SysFont(font, size)
      score_surface = score_font.render('Score : ' + str(self.score), True, color)
      score_rect = score_surface.get_rect()
      if choice == 1:
          score_rect.midtop = (self.frame_size_x / 10, 15)
      else:
          score_rect.midtop = (self.frame_size_x / 2, self.frame_size_y / 1.25)
      self.game_window.blit(score_surface, score_rect)
      # pygame.display.flip()