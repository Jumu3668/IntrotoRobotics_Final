"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Display
import math
import keyboard as kb
import sys
import numpy as np
from matplotlib import pyplot as plt
import time
import random
import matplotlib.patches as patches
# Uncomment if you want to use something else for finding the configuration space
from scipy.signal import convolve2d
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633  # [m/s]
AXLE_LENGTH = 0.4044  # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667  # number of rays
LIDAR_SENSOR_MAX_RANGE = 2.75  # Meters
LIDAR_ANGLE_RANGE = math.radians(240)  # given to us how?


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
target_pos = (0.0, 0.0, 0, 0.07, 1.02, -3.16,
              1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts = []

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

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
# camera.recognitionEnable(timestep)

# Odometry

localization_mode = 'gps'
# localization_mode = 'odometry'

pose_x = 0
pose_y = 0
pose_theta = 0

vL = 0
vR = 0


lidar_sensor_readings = []  # List to hold sensor readings
# position in radians of all lidar bins on robot
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE /
                            2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
# print(lidar_offsets)
# Only keep lidar readings not blocked by robot chassis
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83]

map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
# rrt testing

# mode = 'manual'  # Part 1.1: manual mode
# mode = 'planner'
mode = 'autonomous'


########################
#
# Computer Vision Helper Functions
#
#########################

light_green = [158, 206, 127]
dark_green = [87, 144, 36]

color_ranges = []
def add_color_range_to_detect(lower_bound, upper_bound):
    '''
    @param lower_bound: Tuple of BGR values
    @param upper_bound: Tuple of BGR values
    '''
    global color_ranges
    # Add color range to global list of color ranges to detect
    color_ranges.append([lower_bound, upper_bound])


def check_if_color_in_range(bgr_tuple):
    '''
    @param bgr_tuple: Tuple of BGR values
    @returns Boolean: True if bgr_tuple is in any of the color ranges specified in color_ranges
    '''
    global color_ranges
    for entry in color_ranges:
        lower, upper = entry[0], entry[1]
        in_range = True
        for i in range(len(bgr_tuple)):
            if bgr_tuple[i] < lower[i] or bgr_tuple[i] > upper[i]:
                in_range = False
                break
        if in_range:
            return True
    return False


add_color_range_to_detect(dark_green, light_green)

###################
#
# RRT
#
###################
#  green cube pickup world coordinates WHITE SHELVES
#   ordered in (wx, wy, wtt)
cube_waypoints = [(6.39358, -13.4795099, 4.71195),
                  (8.11796, -7.76351, 3.14174),
                  (8.129499, -1.490345, 3.141773),
                  (8.1800349, -3.49444, -0.001),
                  (12.12526, -2.4653599, 3.14034),
                  (4.1244388, -2.509675, 3.1414869),
                  (4.1303769, 0.41514208, 3.141649),
                  (0.901962, 3.33814055, 0.0004155),
                  (0.9239285, -2.739372, -0.00824976),
                  (0.134933779, 0.3488725640, 3.1284947),
                  (0.13583087, 2.142451, 3.1260417)]


display_waypoints = []
temp_list = []

# convert to display coordinates from world coordinates in waypoint list
for coord in cube_waypoints:
    counter = 0
    temp_list = []
    for i, element in enumerate(coord):
        # print(element)

        if localization_mode == 'odometry':
            if i == 0:  # x value
                temp_list.append(245+int(element*30))
            if i == 1:  # y value
                temp_list.append(700-int(element*30))

        if localization_mode == 'gps':
            if i == 0:  # x value
                temp_list.append(115+int(element*30))
            if i == 1:  # y value
                temp_list.append(290-int(element*30))

        if i == 2:
            temp_list.append(element)

    display_waypoints.append(tuple(temp_list))

# print(display_waypoints)


# rrt in pixel coordnates
# define above to convert pixel coordnates to global coordniates
# class Node():
    #list parents
    #list coordinates
    # previous nodes

class Node:
    """
    Node for RRT Algorithm. This is what you'll make your graph with!
    """
    def __init__(self, pt, parent=None):
        self.point = pt # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = [] # List of points along the way from the parent node (for edge's collision checking)

def state_is_valid(state):
        '''
        Function that takes an n-dimensional point and checks if it is within the bounds and not inside the obstacle
        :param state: n-Dimensional point
        :return: Boolean whose value depends on whether the state/point is valid or not
        '''
        for dim in range(state_bounds.shape[0]):
            if state[dim] < state_bounds[dim][0]: return False
            if state[dim] >= state_bounds[dim][1]: return False
        for obs in obstacles:
            if np.linalg.norm(state - obs[0]) <= obs[1]: return False
        return True
        
def get_random_valid_vertex(state_is_valid, bounds):
    '''
    Function that samples a random n-dimensional point which is valid (i.e. collision free and within the bounds)
    :param state_valid: The state validity function that returns a boolean
    :param bounds: The world bounds to sample points from
    :return: n-Dimensional point/state
    '''
    vertex = None
    while vertex is None: # Get starting vertex
        pt = np.random.rand(bounds.shape[0]) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
        if state_is_valid(pt):
            vertex = pt
    return vertex
def get_nearest_vertex(node_list, q_point):
    '''
    Function that finds a node in node_list with closest node.point to query q_point
    :param node_list: List of Node objects
    :param q_point: n-dimensional array representing a point
    :return Node in node_list with closest node.point to query q_point
    '''

    # TODO: Your Code Here
    # raise NotImplementedError
    ret_node = node_list[0]

    dist = np.linalg.norm(node_list[0].point - q_point)

    for i in node_list:
        temp = i.point
        if dist > np.linalg.norm(temp - q_point):
            dist = np.linalg.norm(temp - q_point)
            ret_node = i

    return ret_node    
def steer(from_point, to_point, delta_q):
    '''
    :param from_point: n-Dimensional array (point) where the path to "to_point" is originating from (e.g., [1.,2.])
    :param to_point: n-Dimensional array (point) indicating destination (e.g., [0., 0.])
    :param delta_q: Max path-length to cover, possibly resulting in changes to "to_point" (e.g., 0.2)
    :return path: Array of points leading from "from_point" to "to_point" (inclusive of endpoints)  (e.g., [ [1.,2.], [1., 1.], [0., 0.] ])
    '''
    # TODO: Figure out if you can use "to_point" as-is, or if you need to move it so that it's only delta_q distance away

    # TODO Use the np.linspace function to get 10 points along the path from "from_point" to "to_point"
    diff = to_point-from_point
    euc_dist = np.linalg.norm(diff)
    new_to = (diff/euc_dist)*delta_q + from_point

    if delta_q < euc_dist:
        path = np.linspace(from_point, new_to, num = 10)
    else:
        path = np.linspace(from_point, to_point, num = 10)

    return path

def check_path_valid(path, state_is_valid):

    for i in path:
        if state_is_valid(i) == False:
            return False
    return True    

def rrt(state_bounds, state_is_valid, starting_point, goal_point, k, delta_q):
    node_list = []
    node_list.append(Node(starting_point, parent=None)) # Add Node at starting point with no parent

    for nothing in range(k):
        q_rand = get_random_valid_vertex(state_is_valid, state_bounds)
        if goal_point is not None and random.random() < 0.04:
            q_rand = goal_point
        
        near_point = get_nearest_vertex(node_list, q_rand)
        path = steer(near_point.point, q_rand, delta_q)

        # add created node to node list and set node parent and path from parent
        if check_path_valid(path, state_is_valid):
            new_node = Node(path[-1], parent= near_point)
            new_node.path_from_parent = path
            node_list.append(new_node)

        # if we are basically on the point
        if goal_point is not None and np.linalg.norm(Node(path[-1], parent= near_point).point-goal_point) < 0.000001:
            return node_list
    
    return node_list

###################
#
# Planner
#
###################
if mode == 'planner':
    map = np.load('map.npy')
    pixels = 0
    # map = np.flipud(map)
    # map = np.rot90(map, 3)
    plt.imshow(map)
    # print(map)
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
    # 12-16 pixels

    # draw box around
    # print(pixels)
    np.set_printoptions(threshold=sys.maxsize)
    # print(Convolved_map.reshape(360,360).tolist())
    plt.imshow(Convolved_map)
    plt.show()
    # print("test0")

    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    start_w = (4.46793, 8.05674)  # (Pose_X, Pose_Z) in meters
    end_w = (10.342, 1.34638)  # (Pose_X, Pose_Z) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    ratiow = 16/480
    ratioh = 30/900
    # (x, y) in 480/900 map
    start = (int(round(start_w[0] / ratiow)), int(round(start_w[1] / ratioh)))
    # (x, y) in 480/900 map
    end = (int(round(end_w[0] / ratiow)), int(round(end_w[1] / ratioh)))
    # start = (167,185)
    # print(str(start))
    # print(str(end))
    start = (240,745)
    end = (410,54)
    # end = (293, 316)
    # print("test")
    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    # rrt testing
    # print(rrt(start, end, Convolved_map))

    # Part 2.1: Load map (map.npy) from disk and visualize it

    # Part 2.2: Compute an approximation of the “configuration space”

    # Part 2.3 continuation: Call path_planner
    # map = map.tolist()
    # stuff = path_planner(map, start, end)
    # print(stuff)
    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    waypoints = []

######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.zeros((480, 900))  # Replace None by a numpy 2D floating point array

state = 0  # use this to iterate through your path


print(localization_mode)
frame_marker = 0
item_detected = False
bounds = [[0,16],[0,30]]
bounds = np.array([[0,1],[0,1]])
K = 250
xy = rrt(bounds, state_is_valid=True, starting_point=cube_waypoints[0], goal_point=cube_waypoints[2], k=K , delta_q=np.linalg.norm(bounds/10.))
print("xy")
print(xy)
for i in range(len(xy)-1):

    display.setColor(0x00FF00)
    display.drawLine(xy[i][0], xy[i][1], xy[i+1][0], xy[i+1][1])

    
map = np.load('map.npy')
kernel_size = 12
Kernel = np.ones((kernel_size, kernel_size))
Convolved_map = convolve2d(map, Kernel, mode='same')
map = (Convolved_map > 0.3).astype(int)
for x in range(len(map)):
    for y in range(len(map)):
        if map[x][y] >= 0.2:
            display.drawPixel(x,y)


green_prev_frame = False
bearing = 0
Finished_turning = False
while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    # GPS BASED Ground truth pose
    wxx = 0
    wyy = 0
    wtt = 0
    if localization_mode == 'gps':
        pose_x = gps.getValues()[2]
        pose_y = gps.getValues()[0]
        display.setColor(int(0xFF0000))
        display.drawPixel(int(115+pose_x*30), int(290-pose_y*30))

    n = compass.getValues()
    pose_theta = -((math.atan2(n[0], n[2]))-1.5708)

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    # print(str(lidar_sensor_readings))
    # print(len(lidar_sensor_readings))
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]  # get current lidar bin position in radians

        # rho is single lidar sensor reading at position 'i'
        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue  # if rho value too far... skip to next number

        # Convert detection from robot coordinates into world coordinates

        wy = -math.sin(pose_theta - alpha) * rho + pose_y
        wx = -math.cos(pose_theta - alpha) * rho + pose_x
        wxx = pose_x
        wyy = pose_y
        wtt = pose_theta
        # print("wx: " + str(wx))
        # print("wy: " + str(wy))
        # print("wtt: " + str(wtt))
        # convert world coordinates into display coordinates
        if localization_mode == 'gps':
            dy = 290-int(wy*30)
            dx = 115+int(wx*30)

        if localization_mode == 'odometry':
            dy = 700-int(wy*30)
            dx = 245+int(wx*30)

        # print("wx: " + str(wx))
        # print("wy: " + str(wy))
        # print(rho)

        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.

            # You will eventually REPLACE the following 2 lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            # display.setColor(0xFFFFFF)

            # convert world coordinates into display coordinates

            if dy >= 900:
                dy = 900
            if dx >= 480:
                dx = 480
            # print("dy: " + str(dy))
            # print("dx: " + str(dx))

            # Lidar Filter
            val = map[dx-1][dy-1]
            if val >= 1:
                val = 1
            else:
                val += 0.0045
                map[dx-1][dy-1] = val

            # converting [0,1] to grayscale intensity [0,255]
            g = int(val * 255)
            color = g*256**2+g*256+g
            display.setColor(color)
            display.drawPixel(dx, dy)  # draws from the top left corner(0,900)
            # print(display_waypoints)

    ###################
    #
    # Controller
    #
    ###################
    if mode == 'manual':
        if kb.is_pressed("a"):
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif kb.is_pressed("d"):
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif kb.is_pressed("w"):
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif kb.is_pressed("s"):
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif kb.is_pressed(' '):
            vL = 0
            vR = 0

        elif kb.is_pressed('t'):
            item_detected = True
            print("item detected")
        elif kb.is_pressed('q'):
            # Part 1.4: Filter map and save to filesystem
            savemap = (map > 0.3).astype(int)
            np.save('map', savemap)
            print("Map file saved")
        elif kb.is_pressed('l'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            map = np.rot90(map, 3)
            map = np.fliplr(map)
            plt.imshow(map)
            plt.show()
            print("Map loaded")
        elif kb.is_pressed('p'):
            camera.saveImage("image.png", 100)
            print("image saved")
        else:  # slow down
            vL *= 0.75
            vR *= 0.75
           
                 
    elif mode == 'autonomous':  # roomba mode
        
        vL = MAX_SPEED /2
        vR = MAX_SPEED /2
        front_obstacle = False
        left_obstacle = False
        right_obstacle = False
        # print(str(len(lidar_sensor_readings)))
        Collision_Detected = False
        left = lidar_sensor_readings[0:len(lidar_sensor_readings)//3]
        # print(str(len(left)))
        middle = lidar_sensor_readings[len(lidar_sensor_readings)//3+1 : len(lidar_sensor_readings)]
        # right =
        if Finished_turning == True:
            bearing = random.uniform(0, math.pi)
        # print(str(pose_theta))
        for i, rho in enumerate(middle):
            if rho != float('inf') and rho < 1.5:
                # print(str(rho))
                print("collision detected")
                Collision_Detected = True
                vL = MAX_SPEED /4
                vR = MAX_SPEED /4
                #go somewhere else
        if Collision_Detected:
            # print("pose theta" + str(pose_theta))
            # print("bearing" + str(bearing))
            if bearing - 0.2 <= pose_theta <= bearing + 0.2:
                # we are on the correct new bearing
                print("we are on the bearing")
                Collision_Detected = False
                Finished_turning = True
                
            #spin in either direction until pose_theta is close to bearing       
            elif pose_theta < bearing:
                vL = -MAX_SPEED / 3
                vR = MAX_SPEED / 3   
                Finished_turning = False
            else:
                vL = MAX_SPEED / 3
                vR = -MAX_SPEED / 3   
                Finished_turning = False
    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    if localization_mode == 'odometry':

        pose_x -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS * \
            timestep/1000.0*math.cos(pose_theta)
        pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS * \
            timestep/1000.0*math.sin(pose_theta)
        pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED * \
            MAX_SPEED_MS*timestep/1000.0
        display.setColor(int(0xFFF000))
        display.drawPixel(int(240+pose_x*30), int(700-pose_y*30))


    ####################
    #
    # Manipulation
    #
    ####################
    tpos_pos_0 = (0.07, 0, -1.7, 0.0, -1.7, 1.39, 1.7)
    scoop_pos_0 = (1.3, -0.15, -1.7, 0.8, -1.7, 1.39, 1.7)
    scoop_pos_1 = (0.07, 0, -1.4, 2.29, -1.7, 1.39, 1.7)
    # if item_detected:
    # print(frame_marker)

    # Scoop animation stage 1
    if frame_marker >= 0 and frame_marker <= 45 and item_detected == True:
        robot_parts[3].setPosition(float(tpos_pos_0[0]))
        robot_parts[4].setPosition(float(tpos_pos_0[1]))
        robot_parts[5].setPosition(float(tpos_pos_0[2]))
        robot_parts[6].setPosition(float(tpos_pos_0[3]))
        robot_parts[7].setPosition(float(tpos_pos_0[4]))
        robot_parts[8].setPosition(float(tpos_pos_0[5]))
        robot_parts[9].setPosition(float(tpos_pos_0[6]))

    # Scoop animation stage 2
    if frame_marker > 45 and frame_marker <= 110 and item_detected == True:
        robot_parts[3].setPosition(float(scoop_pos_0[0]))
        robot_parts[4].setPosition(float(scoop_pos_0[1]))
        robot_parts[5].setPosition(float(scoop_pos_0[2]))
        robot_parts[6].setPosition(float(scoop_pos_0[3]))
        robot_parts[7].setPosition(float(scoop_pos_0[4]))
        robot_parts[8].setPosition(float(scoop_pos_0[5]))
        robot_parts[9].setPosition(float(scoop_pos_0[6]))

    # Scoop animation stage 3
    if frame_marker > 110 and frame_marker <= 170 and item_detected == True:
        robot_parts[3].setPosition(float(scoop_pos_1[0]))
        robot_parts[4].setPosition(float(scoop_pos_1[1]))
        robot_parts[5].setPosition(float(scoop_pos_1[2]))
        robot_parts[6].setPosition(float(scoop_pos_1[3]))
        robot_parts[7].setPosition(float(scoop_pos_1[4]))
        robot_parts[8].setPosition(float(scoop_pos_1[5]))
        robot_parts[9].setPosition(float(scoop_pos_1[6]))

    if frame_marker > 170 and frame_marker <= 240:
        # reverse
        vL = -MAX_SPEED/2
        vR = -MAX_SPEED/2
    # Reset robo arm

    if frame_marker > 240 and frame_marker <= 290:
        for i in range(N_PARTS):
            robot_parts[i].setPosition(float(target_pos[i]))

    if frame_marker > 290:  # animation complete. Reset Vars
        item_detected = False
        frame_marker = 0
    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)

    if item_detected == True:
        frame_marker += 1

    #########################
    #
    # Computer Vision
    #
    #########################

    camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    image = camera.getImage()
    # green_prev_frame = False
    green_pres_frame = False
    if mode != 'autonomous':
    
        # Scan center-bottom rows of pixels for green in 240x135 picture
        for x in range(90, 150):  # columns
            for y in range(132, 134):  # rows
                g = camera.imageGetGreen(image, width, x, y)
                r = camera.imageGetRed(image, width, x, y)
                b = camera.imageGetBlue(image, width, x, y)
                # print("hello")
                color = (r,g,b)
                if check_if_color_in_range(color) == True:
                    # print("i see green")
                    green_pres_frame = True
                    # green_prev_frame = green_pres_frame
                    
        
    
        # if true and !false
        # print(str(green_prev_frame) + " " + str(green_pres_frame))
        if green_prev_frame and not green_pres_frame:
            # print("moving my arm")
            item_detected = True
            
        green_prev_frame = green_pres_frame
        # pixels x-axis 100-150
