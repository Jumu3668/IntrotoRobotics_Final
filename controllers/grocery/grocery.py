
"""Final Lab Controller"""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard, Display

import math
import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667 #number of rays
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240) #given to us how?


##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# display = Display("New_Display")
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

# range = robot.getDevice('range-finder')
# range.enable(timestep)
# camera = robot.getDevice('camera')
# camera.enable(timestep)
# camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 480x900 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

#set up keybaord call
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS) #position in radians of all lidar bins on robot
# print(lidar_offsets)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
# mode = 'manual' # Part 1.1: manual mode
mode = 'manual'
# mode = 'autonomous'





###################
#
# Planner
#
###################
if mode == 'planner':
    map = np.load('map.npy')
    pixels = 0
    #map = np.fliplr(map)
    map = np.rot90(map, 3)
    plt.imshow(map)
    print(map)
    plt.show()
    kernel_size = 12
    Kernel = np.ones((kernel_size, kernel_size))
    Convolved_map = convolve2d(map, Kernel, mode='same')
    Convolved_map = (Convolved_map > 0.3).astype(int)
    # Convolved_map = np.fliplr(Convolved_map)
    # Convolved_map = np.rot90(Convolved_map)
    # boxy_map = map.tolist()
    # for x in range(len(boxy_map)):
        # for y in range(len(boxy_map)):
            # if boxy_map[x][y] == 1:
                # pixels+= 1
                #12-16 pixels
                                                        
                # draw box around 
    print(pixels)
    np.set_printoptions(threshold=sys.maxsize)
    # print(Convolved_map.reshape(360,360).tolist())
    plt.imshow(Convolved_map)
    plt.show()
    # print("test0")
            
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    start_w = (4.46793, 8.05674) # (Pose_X, Pose_Z) in meters
    end_w = (10.342,1.34638) # (Pose_X, Pose_Z) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    ratio = 12/360
    start = (int(round(start_w[0] / ratio)), int(round(start_w[1] / ratio))) # (x, y) in 360x360 map
    end = (int(round(end_w[0] / ratio)), int(round(end_w[1] / ratio))) # (x, y) in 360x360 map
    # start = (167,185)
    print(str(start))
    print(str(end))
    # end = (293, 316)
    # print("test")
    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    
    def path_planner(map, start, end):
        #a* search from https://towardsdatascience.com/a-star-a-search-algorithm-eb495fb156bb
        print("called")
        '''
        :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
        :param start: A tuple of indices representing the start cell in the map
        :param end: A tuple of indices representing the end cell in the map
        :return: A list of tuples as a path from the given start to the given end in the given maze
        '''
        class Node():
            """A node class for A* Pathfinding"""

            def __init__(self, parent=None, position=None):
                self.parent = parent
                self.position = position

                self.distance_from_start = 0
                self.heuristic = 0
                self.total_cost = 0

            def __eq__(self, other):
                return self.position == other.position



        def shortest_path_grid(grid, start, goal):
            '''
            Function that returns the length of the shortest path in a grid
            that HAS obstacles represented by 1s. The length is simply the number
            of cells on the path including the 'start' and the 'goal'
        
            :param grid: list of lists (represents a square grid where 0 represents free space and 1s obstacles)
            :param start: tuple of start index
            :param goal: tuple of goal index
            :return: length of path
            '''
            start_node = Node(None, start)
        
            start_node.distance_from_start = 0
            start_node.heuristic = 0
            start_node.total_cost = 0
        
            goal_node = Node(None, goal)
        
            goal_node.distance_from_start = 0
            goal_node.heuristic = 0
            goal_node.total_cost = 0
        
            open_list = []
            closed_list = []
        
            # Add the start node
            open_list.append(start_node)
        
            # Loop until open list of explorable squares is 0
            while len(open_list) > 0:
        
                current_node = open_list[0]
                current_index = 0
                for index, item in enumerate(open_list):
                    if item.total_cost < current_node.total_cost:
                        current_node = item
                        current_index = index
        
                # Pop current off open list, add to closed list
                open_list.pop(current_index)
                closed_list.append(current_node)
        
                # If we find the goal node, return length of path we took
                if current_node == goal_node:
                    path = []
                    current = current_node
                    while current is not None:
                        path.append(current.position)
                        current = current.parent
                    # return len(path) # Return length of path
                    return path[::-1]
                # Generate children
                children = []
                for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares
        
                    # Get node position
                    node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
        
                    # Make sure node is within range of grid
                    if node_position[0] > (len(grid) - 1) or node_position[0] < 0 or node_position[1] > (len(grid[len(grid)-1]) -1) or node_position[1] < 0:
                        continue
        
                    # Make sure walkable terrain
                    if grid[node_position[0]][node_position[1]] != 0:
                        continue
        
                    # Create new node
                    new_node = Node(current_node, node_position)
        
                    # append
                    children.append(new_node)
        
                # Loop through children
                for child in children:
        
                    # Child is on the closed list
                    for closed_child in closed_list:
                        if child == closed_child:
                            continue
        
                    # Create heuristic values 
                    child.distance_from_start = current_node.distance_from_start + 1
                    child.heuristic = ((child.position[0] - goal_node.position[0]) ** 2) + ((child.position[1] - goal_node.position[1]) ** 2)
                    child.total_cost = child.distance_from_start + child.heuristic
        
                    # Child is already in the open list
                    for open_node in open_list:
                        if child == open_node and child.distance_from_start > open_node.distance_from_start:
                            continue
        
                    # Add the child to the open list
                    open_list.append(child)
        
        #if goal cannot be found
            return -1
        return shortest_path_grid(map, start, end)

    # Part 2.1: Load map (map.npy) from disk and visualize it


    # Part 2.2: Compute an approximation of the “configuration space”


    # Part 2.3 continuation: Call path_planner
    map = map.tolist()
    stuff = path_planner(map, start, end)
    print(stuff)
    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    waypoints = []

######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.zeros((480,900)) # Replace None by a numpy 2D floating point array
waypoints = []
if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    waypoints = [] # Replace with code to load your path

state = 0 # use this to iterate through your path




while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    ################ v [Begin] Do not modify v ##################
    # Ground truth pose
    pose_x = gps.getValues()[2]
    pose_y = gps.getValues()[0]

    n = compass.getValues()
    pose_theta = -((math.atan2(n[0], n[2]))-1.5708)

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    # print(str(lidar_sensor_readings))
    print(len(lidar_sensor_readings))
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i] #get current lidar bin position in radians

        # rho is single lidar sensor reading at position 'i'
        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue #if rho value too far... skip to next number
        
        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        # rx = math.cos(alpha)*rho
        # ry = -math.sin(alpha)*rho

        # Convert detection from robot coordinates into world coordinates
        wx = -math.cos(pose_theta - alpha) * rho + pose_x
        wy = -math.sin(pose_theta - alpha) * rho + pose_y
        # wx =  math.cos(pose_theta)*rx - math.sin(pose_theta)*ry + pose_x
        # wy =  -(math.sin(pose_theta)*rx + math.cos(pose_theta)*ry) + pose_y
        # print("wx: " + str(wx))
        # print("wy: " + str(wy))
        # print(rho)
        ################ ^ [End] Do not modify ^ ##################

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))

        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.

            # You will eventually REPLACE the following 2 lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            # display.setColor(0xFFFFFF)
            
            #convert world coordinates into display coordinates
            dy = 320-int(wy*30)
            dx = 100+int(wx*30)
                          
            # if dy > 900:
                # dy = 900
            # if dx > 480:
                # dx = 480
            # print("dy: " + str(dy))
            # print("dx: " + str(dx))
            # grayscale code    
            val = map[dx-1][dy-1]
            if val >= 1:
                val = 1
            else:
                val += 0.0045
                map[dx-1][dy-1] = val
            
            g = int(val* 255) # converting [0,1] to grayscale intensity [0,255]
            color = g*256**2+g*256+g
            display.setColor(color)
            display.drawPixel(dx,dy) #draws from the top left corner(0,900)
            
    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(int(100+pose_x*30),int(320-pose_y*30))
    

    ###################
    #
    # Controller
    #
    ###################
    if mode == 'manual':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            # Part 1.4: Filter map and save to filesystem
            savemap = (map > 0.3).astype(int)
            np.save('map', savemap)
            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            print("Map loaded")
        else: # slow down
            vL *= 0.75
            vR *= 0.75
    else: # not manual mode
        # Part 3.2: Feedback controller
        #STEP 1: Calculate the error
        rho = 0
        alpha = -(math.atan2(waypoint[state][1]-pose_y,waypoint[state][0]-pose_x) + pose_theta)


        #STEP 2: Controller
        dX = 0
        dTheta = 0

        #STEP 3: Compute wheelspeeds
        vL = 0
        vR = 0

        # Normalize wheelspeed
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)


    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)

