import sys
from maze import Maze
import numpy as np
import random as random
from operator import add, sub, itemgetter
from pprint import pprint

class Robot(object):
    def __init__(self, dimensions):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''

        # Help variables for counting and identifying runs 
        self.count = 0
        self.path_count = -1
        self.return_to_start = False
        self.run1 = False

        # Maze-related variables
        self.maze_dim = dimensions
        self.start = [0, 0]
        self.goal = [self.maze_dim/2 - 1, self.maze_dim/2]

        # Robot position and heading
        self.robot_pos = {'location': [0, 0], 'heading': 'u'}

        # A star variables
        self.cost_so_far = {}
        self.cost_so_far[str(self.robot_pos['location'])] = 0
        self.path = []
        self.came_from = {}
        self.came_from[str(self.robot_pos['location'])] = None

        # Help dictionaries for guiding the robot
        self.dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
                            'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u']}

        self.dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0]}

        self.dir_reverse = {'u': 'd', 'r': 'l', 'd': 'u', 'l': 'r'}

        # This dictionary helps to guide the robot from the goal to the start
        # and back at the end of run 0 and in run 1, respectively
        self.dir_reconstruct = {'u': {'u': [0, 1],   'l': [-90, 1],
                                      'r': [90, 1],  'd': [0, -1]},
                                'l': {'u': [90, 1],  'l': [0, 1],
                                      'r': [0, -1],  'd': [-90, 1]},
                                'r': {'u': [-90, 1], 'l': [0, -1],
                                      'r': [0, 1],   'd': [90, 1]},
                                'd': {'u': [0, -1],  'l': [90, 1],
                                      'r': [-90, 1], 'd': [0, 1]}}

    def change_heading(self, rotation):

        # Update the heading variable to point out to the correct direction
        # INPUT
        # rotation: -90, 0, 90 degree rotation of the robot
        #
        # OUTPUT
        # heading: depending on the current heading of the robot
        # the rotation inputed makes it look at one of the following directions
        # 'u', 'd', 'l', 'r'

        heading = None
        if rotation == -90:
            heading = self.dir_sensors[self.robot_pos['heading']][0]
        elif rotation == 90:
            heading = self.dir_sensors[self.robot_pos['heading']][2]
        elif rotation == 0:
            heading = self.robot_pos['heading']
        else:
            print "Invalid rotation value, no rotation performed."
        return heading

    def change_location(self, movement):

        # Update the location variable
        # INPUT
        # movement: integer between -3 and 3. Signifies how many blocks the
        # robot moved from its current position
        #
        # OUTPUT
        # location: a list of the robot's new coordinates [x, y]

        location = self.robot_pos['location']
        if abs(movement) > 3:
            print "Movement limited to three squares in a turn."
        movement = max(min(int(movement), 3), -3) # fix to range [-3, 3]
        while movement:
            if movement > 0:
                # Update for forward movement. Already checked if allowed by
                # the readings of the sensors.
                location[0] += self.dir_move[self.robot_pos['heading']][0]
                location[1] += self.dir_move[self.robot_pos['heading']][1]
                movement -= 1
            else:
                # Update for backwards movement. No need to check if permisible
                # because in this implementation the robot will go backwards
                # only once and this is to return to an already visited block.
                rev_heading = self.dir_reverse[self.robot_pos['heading']]
                location[0] += self.dir_move[rev_heading][0]
                location[1] += self.dir_move[rev_heading][1]
                movement += 1
        return location

    def distance_to_goal(self, position):

        # Physical distance from a current block to the goal
        # INPUT
        # position: the position list passed as a string '[x, y]'
        #
        # OUTPUT
        # distance: distance from the goal (float)

        # Transform '[x, y]' (str) to [x, y] (list of integers)
        position = map(int,position\
                       .replace('[','')\
                       .replace(']','')\
                       .strip()\
                       .split(','))

        distance = abs(position[0] - self.maze_dim/2 -1) + \
                   abs(position[1] - self.maze_dim/2)
        return distance

    def reconstruct_path(self, came_from, start, goal):

        # Reconstructs the path to the goal from all the 'came_from' values.
        # Taken from http://www.redblobgames.com/pathfinding/a-star/implementation.html
        # INPUT
        # came_from: The came_from dictionary, showing the previous position to
        # any of the locations the bot has been on
        # start: The start block as string '[0, 0]'
        # goal: The goal block as string '[x, y]'
        #
        # OUTPUT
        # path: A list of strings of coordinates pointing from the goal to the
        # start ['[x1, y1]', ..., '[0, 0]']

        current = goal
        path = [current]
        while current != start:
            current = came_from[current]
            path.append(current)
        path.append(start) # optional
        path.reverse() # optional
        return path

    def move_and_rotate(self, current_pos, next_pos):

        # This method takes as inputs the current and a next position and
        # returns the rotation and movement that the robot should perform
        # to move to the next position.
        # INPUT
        # current_pos: The current position of the robot as a list of integers
        # next_pos: The position that the robot should move to as a string of a
        # list of integers
        #
        # OUTPUT
        # rotation: The rotation that the robot should perform to reach next_pos
        # movement: The movement that the robot should perform to reach next_pos

        # Transform '[x, y]' (str) to [x, y] (list of integers)
        next_pos = map(int,
                       next_pos\
                       .replace('[','')\
                       .replace(']','')\
                       .strip()\
                       .split(','))

        # Calculate the difference in blocks between current and next position
        # and then find from self.dir_move the direction that the robot should
        # move towards
        pos_difference = map(sub, next_pos, current_pos)

        for key, value in self.dir_move.iteritems():
            if value == pos_difference:
                temp_dir = key

        # Calculate rotation and movement values given the current heading and
        # the direction that the robot should move towards. This happens with
        # the help of the self.dir_reconstruct dictionary
        rotation = self.dir_reconstruct[self.robot_pos['heading']]\
                                       [temp_dir][0]
        movement = self.dir_reconstruct[self.robot_pos['heading']]\
                                       [temp_dir][1]

        return rotation, movement

    def next_move(self, sensors):
        '''
        Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''

        # Help variables
        rot = [-90, 0, 90]
        mov = [-1, 0, 1]
        ind = []

        # Find the directions the robot is allowed to move towards, if any
        for i in range(3):
            if sensors[i] > 0:
                ind.append(i)

        if not ind:
            # If the sensors see only walls then the robot turns to the right
            # by 90 degrees without moving forward
            rotation = 90
            movement = 0
        elif self.return_to_start == True and self.run1 == False:
            # If the robot has found its goal, then it returns to the start
            # following the path recunstructed from the reconstruct_path
            # method

            if self.robot_pos['location'] == self.start and\
               self.robot_pos['heading'] == 'u':
                # When the robot has returned to the start and its heading is
                # u', then run 1 starts
                rotation = 'Reset'
                movement = 'Reset'
                self.robot_pos['heading'] = 'u'
                self.robot_pos['location'] = [0, 0]
                self.path_count = 1
                self.run1 = True
            elif self.robot_pos['location'] == self.start and\
                 self.robot_pos['heading'] != 'u':
                 # When the robot reaches the start it rotates until its
                 # heading is 'u'
                 rotation = 90
                 movement = 0
            else:
                # This makes the robot follow the path from the goal to the
                # start
                current_position = self.robot_pos['location']
                self.path_count -= 1
                next_position = self.path[self.path_count]
                rotation, movement= self.move_and_rotate(current_position,
                                                          next_position)
        elif self.run1 == True:
            # This performs the final run to find the goal
            current_position = self.robot_pos['location']
            self.path_count += 1
            next_position = self.path[self.path_count]
            rotation, movement= self.move_and_rotate(current_position,
                                                      next_position)
        else:
            # One of the algorithms should be commented
            rotation = None
            cost_temp = []

            ###################
            ## A STAR ALGORITHM
            ###################
            for i in ind:
                # Calculation of the cost of a possible new position
                temp_heading = self.change_heading(rot[i])
                current_position = str(self.robot_pos['location'])
                next_position = str(map(add,self.robot_pos['location'],
                                            self.dir_move[temp_heading]))
                cost = self.cost_so_far[current_position] + \
                       self.distance_to_goal(next_position)

                if next_position not in self.cost_so_far or \
                   cost < self.cost_so_far[next_position]:
                    # If a position has not been explored or it has a lower
                    # cost, then it is prefered over others
                    self.cost_so_far[next_position] = cost
                    self.came_from[next_position] = current_position
                    cost_temp.append([cost, i])
                    rotation = 0
                else:
                    # The cost appended is a high value that will not be
                    # considered for the minumum
                    cost_temp.append([1000000, i])

            # Find the direction of the minimum cost and go towards there
            if rotation is None:
                # If all positions are already visited or they don't have
                # a lower cost than self.cost_so_far, then the robot
                # selects one in random
                rotation = rot[random.choice(ind)]
                movement = 1
            else:
                rotation = rot[min(cost_temp, key=itemgetter(0))[1]]
                movement = 1

            #######################
            ## BREADTH FIRST SEARCH
            #######################
            # for i in ind:
            #     # Calculation of location of a new block to visit
            #     # Among equally not visited positions, the robot will always
            #     # go towards the one on the right
            #     temp_heading = self.change_heading(rot[i])
            #     current_position = str(self.robot_pos['location'])
            #     next_position = str(map(add,self.robot_pos['location'],
            #                                 self.dir_move[temp_heading]))

            #     if next_position not in self.came_from:
            #         # If a position has not been explored then it is prefered
            #         # over others
            #         self.came_from[next_position] = current_position
            #         rotation = rot[i]
            #         movement = 1
            # if rotation is None:
            #     # If all positions are already visited, then the robot
            #     # selects one in random
            #     rotation = rot[random.choice(ind)]
            #     movement = 1

        # Update time-step counter
        self.count += 1

        # Update robot heading and location keys
        if not [rotation, movement] == ['Reset', 'Reset']:
            self.robot_pos['heading'] = self.change_heading(rotation)
            self.robot_pos['location'] = self.change_location(movement)

        # If robot reached the goal then reconstruct path to the start and make
        # the robot go towards it
        if self.robot_pos['location'][0] in self.goal and \
           self.robot_pos['location'][1] in self.goal:
            self.path = self.reconstruct_path(self.came_from,
                                              str(self.start),
                                              str(self.robot_pos['location']))
            self.return_to_start = True

        return rotation, movement
