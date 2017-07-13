import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

# PyGame init
width = 256
height = 256
num_obstacles = 0
radio = 3
radio_obs = 20
_crash_reward = -0.001
_goal_reward = 1
_normal_reward = -.00001
_goal = Vec2d(0,0)
_old_position = Vec2d(0,0)
terminal = False
_initial_position_agent_x = width/2
_initial_position_agent_y = height/2 - 80
_initial_position_goal_x = width/2
_initial_position_goal_y = height/2 + 80


pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = True
draw_screen = True


class GameState:
    def __init__(self):

        # Global-ish.
        self.crashed = False
        self.goal_hit = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Create the car.
        #self.create_car(random.randint(5, (width - 5)), random.randint(5, (height-5)), radio)
        self.create_car(_initial_position_agent_x, _initial_position_agent_y, radio)
        _old_position[0] = self.car_body.position.x
        _old_position[1] = self.car_body.position.y
        print("old position")
        print(_old_position)


        # Record steps.
        self.num_steps = 0

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1),
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create agents-obstacles, semi-randomly.
        self.obstacles = []
        for i in range(num_obstacles):
            self.obstacles.append(self.create_obstacle(width / 2, height / 2, radio_obs))
            #self.obstacles.append(self.create_obstacle(random.randint(5, (width - 5)), random.randint(5, (height-5)), radio))

        self.create_goal()


    def create_goal(self):
        if _goal[0] != 0:
            self.space.remove(self.goal_mark)

        _goal[0] = _initial_position_goal_x    #(random.randint(60, (width - 60)) )
        _goal[1] = _initial_position_goal_y    #(random.randint(60, (height - 60)) )
        print("Goal coordinates:")
        print(_goal)
        self.goal_mark = pymunk.Segment(
            self.space.static_body,
            (_goal[0] - 10, _goal[1] - 10), (_goal[0] + 10, _goal[1] + 10), 10)
        self.goal_mark.group = 2
        self.goal_mark.color = THECOLORS['green']
        self.space.add(self.goal_mark)

    def background(self, image_file, location):
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        c_shape = pymunk.Circle(c_body, r)

        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_body.group = 2
        c_shape.group= 2
        c_shape.color = THECOLORS["orange"]
        self.space.add(c_body, c_shape)
        return c_body

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_body.group = 2
        self.car_shape = pymunk.Circle(self.car_body, r)
        self.car_shape.color = THECOLORS["red"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    def frame_step(self, input_actions):

        terminal = False
        #print("input_actions: ", input_actions)
        #print("input_actions[2]: ", input_actions[2])
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        #Actions: 0 do nothing, 1 turn left, 2 turn right
        if input_actions[1] == 1:
            self.car_body.angle -= .2
        elif input_actions[2] == 1:  # Turn right.
            self.car_body.angle += .2

        # Move obstacles.
        #if self.num_steps % 50 == 0:
            #self.move_obstacles()

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = 100 * driving_direction

        # Update the screen and stuff.
        screen.fill(THECOLORS["white"])
        #load the map
        #self.background('scrSht.png', [0,0])
        #screen.blit(self.image, self.rect)
        draw(screen, self.space)
        self.space.step(1./10)
        if draw_screen:
            pygame.display.flip()
        clock.tick()

        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        #readings.append(self.car_body.position.get_dist_sqrd(_goal)/10000)
        #state = np.array([readings])
        state = np.array([[self.car_body.position[0]/width, self.car_body.position[1]/height]])
        '''
        print("============= State: ===================")
        print(state)
        print("calculando distancia")
        print(self.car_body.position.get_dist_sqrd(_goal)/10000)
        '''

        # Set the reward.
        # Car crashed when any reading == 1
        if self.car_is_crashed(readings):
            self.crashed = True
            reward = _crash_reward
            #terminal = True
            self.recover_from_crash(driving_direction)
            #print("=================== Craaaaasssshhhhhhhh!!! ==================")
        elif  ( (_goal[0] - 10) <= self.car_body.position[0] <= (_goal[0] + 10) ) and (_goal[1] - 10) <= self.car_body.position[1] <= (_goal[1] + 10):
            self.goal_hit = True
            reward = _goal_reward
            #print("*********************************************** It got the Goal!!! ************************************************************")
            self.recover_from_goal(driving_direction)
            terminal = True
        #elif self.car_body.position.get_dist_sqrd(_goal) < _old_position.get_dist_sqrd(_goal):
            #reward = 100 / (self.car_body.position.get_dist_sqrd(_goal)/100000)
            #print("Reward comp 2:")
            #print( 100 / (self.car_body.position.get_dist_sqrd(_goal)/100000) )
        else:
            #reward = - (self.car_body.position.get_dist_sqrd(_goal)/50000)
            reward = _normal_reward #( (200 - int(self.sum_readings(readings)) ) /50) - 2

        #print("Reward Total")
        #print(reward)
        #print(state)

        _old_position[0] = self.car_body.position.x
        _old_position[1] = self.car_body.position.y
        #print(self.car_body.position)

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        self.num_steps += 1

        #x_t, r_0, terminal = game_state.frame_step(do_nothing)

        return image_data, reward, terminal

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(5, 30)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def car_is_crashed(self, readings):
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1 or readings[3] == 1 or readings[4] == 1:
            return True
        else:
            return False

    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
        """
        while self.crashed:
            # Go backwards.
            self.car_body.velocity = -100 * driving_direction
            self.crashed = False
            #Return to initial position
            self.car_body.position.x = _initial_position_agent_x
            self.car_body.position.y = _initial_position_agent_y
            draw(screen, self.space)
            self.space.step(1./10)
            if draw_screen:
                pygame.display.flip()
            clock.tick()

            '''for i in range(3):
                self.car_body.angle += .2  # Turn a little.
                #screen.fill(THECOLORS["red"])  # Red is scary!
                draw(screen, self.space)
                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()'''


    def recover_from_goal(self, driving_direction):
        """
        We get the goal something, so re-init.
        """
        while self.goal_hit:
            #self.create_goal()
            # Go backwards.
            self.car_body.velocity = -100 * driving_direction
            self.goal_hit = False
            #Return to initial position
            self.car_body.position.x = _initial_position_agent_x
            self.car_body.position.y = _initial_position_agent_y
            draw(screen, self.space)
            self.space.step(1./10)
            if draw_screen:
                pygame.display.flip()
            clock.tick()

            '''for i in range(3):
                self.car_body.angle += .2  # Turn a little.
                #screen.fill(THECOLORS["red"])  # Red is scary!
                draw(screen, self.space)
                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()'''

    def sum_readings(self, readings):
        """Sum the number of non-zero readings."""
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm_left_1 = self.make_sonar_arm(x, y)
        arm_left_2 = arm_left_1
        arm_middle = arm_left_1
        arm_right_1 = arm_left_1
        arm_right_2 = arm_left_1


        # Rotate them and get readings.
        readings.append(self.get_arm_distance(arm_left_1, x, y, angle, 0.35))
        readings.append(self.get_arm_distance(arm_left_2, x, y, angle, 0.75))
        readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_right_1, x, y, angle, -0.35))
        readings.append(self.get_arm_distance(arm_right_2, x, y, angle, -0.75))


        if show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 20 or rotated_p[1] <= 20 \
                    or rotated_p[0] >= (width - 20) or rotated_p[1] >= (height - 20):
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 0, 0), (rotated_p), 1, 1)

        # Return the distance for the arm.
        return i

    def make_sonar_arm(self, x, y):
        spread = 5 # Default spread.
        distance = 3  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == THECOLORS['white'] or reading == THECOLORS['green']:
            return 0
        else:
            return 1

if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 2)))
