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

#We are using Lidar to scan the world for manual and atonomous mapping
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# The display is used to display the map. We are using 480x900 pixels to
display = robot.getDevice("display")

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)

#Choose the localization mode, uncomment the one you want to use.
localization_mode = 'gps'
# localization_mode = 'odometry'

print("Current Localization Mode: ", localization_mode)

#Choose the mapping mode, uncomment the one you want to use.
# mode = 'manual'
# mode = 'planner'
mode = 'autonomous'

####################
#
#GLOBAL VARIABLES
#
####################
pose_x = 0
pose_y = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = []  # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE / 2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)# position in radians of all lidar bins on robot
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

frame_marker = 0
item_detected = False

color_ranges = []

#empty arrays used to convert map coords into display coords
display_waypoints = []
temp_list = []

green_prev_frame = False
bearing = 0
Finished_turning = False

map = None

###################################
#
# Computer Vision Helper Functions
#
###################################
#function from homework 3, used to add color ranges to detect between
def add_color_range_to_detect(lower_bound, upper_bound): 
    '''
    @param lower_bound: Tuple of BGR values
    @param upper_bound: Tuple of BGR values
    '''
    global color_ranges
    # Add color range to global list of color ranges to detect
    color_ranges.append([lower_bound, upper_bound])

#function from homework 3, used to check if passed in color is in the range previously defined
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

light_green = [158, 206, 127] #color values to be passed into the color ranges
dark_green = [87, 144, 36]
add_color_range_to_detect(dark_green, light_green) #adding colors to color range 

#List of X, Y, Theta of the robot when in the position to grab all cubes
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

# convert to display coordinates from world coordinates in waypoint list
for coord in cube_waypoints:
    counter = 0
    temp_list = []
    for i, element in enumerate(coord):
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
  
def rrt(start_pt, end_pt, map): #rrt function, used to map path needed to be taken
    # check pixels, modify to go faster
    delta_q = 0.5
    # checks single coord is valid
    def valid(pt): 
        return not map[int(pt[0])][int(pt[1])]
        # every tuple has coord, index of parent
    explored = [(start_pt, None)]
    for n in range(10000):
        # random coord within map's rows and columns
        q_rand = (np.random.randint(len(map)), np.random.randint(len(map[0])))
        while not valid(q_rand):
            q_rand = (np.random.randint(len(map)), np.random.randint(len(map[0])))
            
        if np.random.rand() < 0.05:
            q_rand = end_pt

        closest_index = -1 # closest to q_rand
        closest_dist = float("inf")
        # finds the closest point
        for i in range(len(explored)):
            pt = explored[i][0]
            dist = ((q_rand[0] - pt[0]) ** 2 + (q_rand[1] - pt[1]) ** 2) ** 0.5 # eucidean distance
            if dist < closest_dist:
                closest_dist = dist
                closest_index = i
        isValid = True  #path is valid
        closest_pt = explored[closest_index][0]
        # checks every point along the line between closest pt and q_rand
        for i in np.arange(0, closest_dist, delta_q):
            p = i / closest_dist

            if not valid((closest_pt[0] + p * (q_rand[0] - closest_pt[0]), closest_pt[1] + p * (q_rand[1] - closest_pt[1]))):
                isValid = False
                break
        if isValid:
            # adds a tuple to the explored array
            explored.append((q_rand, closest_index))
            # checks within one pixel of the goal
            dist = ((q_rand[0] - end_pt[0]) ** 2 +
                    (q_rand[1] - end_pt[1]) ** 2) ** 0.5
            if dist < 1:
                path = [q_rand]
                c = explored[len(explored) - 1]
                # unrolls path using parent pointer
                while not c[1] is None:
                    c = explored[c[1]]
                    path.insert(0, c[0])
                return path
    return -1

##########
#
# Planner
#
##########
if mode == 'planner': #setting the mode to a planning state
    map = np.load('map.npy') #load the map
    pixels = 0  #variable pixels set to 0
    plt.imshow(map) #show the drawn map
    plt.show() #show the plot
    kernel_size = 12 #size for kerneling, change value to widen or shrink thickness
    Kernel = np.ones((kernel_size, kernel_size)) #kernel is equal to an array of ones in a square the size of the kernel
    Convolved_map = convolve2d(map, Kernel, mode='same') #the convolved map is equal to the map with the kerneling
    Convolved_map = (Convolved_map > 0.3).astype(int) #update convolved map

    np.set_printoptions(threshold=sys.maxsize) #setting the size that an np array can print to in the console (show entire np array)

    plt.imshow(Convolved_map) #show the convolved map on screen
    plt.show() #show the plot

    start_w = (4.46793, 8.05674)  # (Pose_X, Pose_Z) in meters
    end_w = (10.342, 1.34638)  # (Pose_X, Pose_Z) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    ratiow = 16/480 #ratio of pixels per meter for the width
    ratioh = 30/900 #ratio of pixels per meter for the heignt
    start = (int(round(start_w[0] / ratiow)), int(round(start_w[1] / ratioh))) # (x, y) in 480/900 map
    end = (int(round(end_w[0] / ratiow)), int(round(end_w[1] / ratioh))) # (x, y) in 480/900 map
    start = (240,745) #start position x, y
    end = (410,54) #end position x, y
    waypoints = [] #empty array to gold waypoints

map = np.zeros((480, 900))  # Replace None by a numpy 2D floating point array

xy = rrt(display_waypoints[0], display_waypoints[5], map)

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
        # convert world coordinates into display coordinates
        if localization_mode == 'gps':
            dy = 290-int(wy*30)
            dx = 115+int(wx*30)

        if localization_mode == 'odometry':
            dy = 700-int(wy*30)
            dx = 245+int(wx*30)

        if rho < LIDAR_SENSOR_MAX_RANGE:

            if dy >= 900:
                dy = 900
            if dx >= 480:
                dx = 480

            # Lidar Filter
            val = map[dx-1][dy-1]
            if val >= 1:
                val = 1
            else:
                val += 0.0045
                map[dx-1][dy-1] = val

            g = int(val * 255) # converting [0,1] to grayscale intensity [0,255]
            color = g*256**2+g*256+g
            display.setColor(color)
            display.drawPixel(dx, dy)  # draws from the top left corner(0,900)

    ###################
    #
    # Controller
    #
    ###################
    #using keyboard presses to control the robot
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
            savemap = (map > 0.3).astype(int)
            np.save('map', savemap)
            print("Map file saved")
        elif kb.is_pressed('l'):
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
        Collision_Detected = False
        left = lidar_sensor_readings[0:len(lidar_sensor_readings)//3]
        middle = lidar_sensor_readings[len(lidar_sensor_readings)//3+1 : len(lidar_sensor_readings)]
        if Finished_turning == True:
            bearing = random.uniform(0, math.pi)
        for i, rho in enumerate(middle):
            if rho != float('inf') and rho < 1.5:
                print("collision detected")
                Collision_Detected = True
                vL = MAX_SPEED /4
                vR = MAX_SPEED /4
        if Collision_Detected:
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
    green_pres_frame = False
    if mode != 'autonomous':
    
        # Scan center-bottom rows of pixels for green in 240x135 picture
        for x in range(90, 150):  # columns
            for y in range(132, 134):  # rows
                g = camera.imageGetGreen(image, width, x, y)
                r = camera.imageGetRed(image, width, x, y)
                b = camera.imageGetBlue(image, width, x, y)
                color = (r,g,b)
                if check_if_color_in_range(color) == True:
                    green_pres_frame = True
                    
        if green_prev_frame and not green_pres_frame:
            item_detected = True
        green_prev_frame = green_pres_frame

