"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Display
import math
import keyboard as kb
import sys
import numpy as np
from matplotlib import pyplot as plt
import time

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
        pass

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


# localization_mode = 'gps'
localization_mode = 'odometry'

print(localization_mode) 

while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    # GPS BASED Ground truth pose
    
    if localization_mode == 'gps':
        pose_x = gps.getValues()[2]
        pose_y = gps.getValues()[0] 
        display.setColor(int(0xFF0000))
        display.drawPixel(int(100+pose_x*30),int(320-pose_y*30))
        
    n = compass.getValues()
    pose_theta = -((math.atan2(n[0], n[2]))-1.5708)

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    # print(str(lidar_sensor_readings))
    # print(len(lidar_sensor_readings))
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i] #get current lidar bin position in radians

        # rho is single lidar sensor reading at position 'i'
        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue #if rho value too far... skip to next number

        # Convert detection from robot coordinates into world coordinates
        
        wy = -math.sin(pose_theta - alpha) * rho + pose_y
        wx = -math.cos(pose_theta - alpha) * rho + pose_x
        #convert world coordinates into display coordinates
        if localization_mode == 'gps': 
            dy = 320-int(wy*30)
            dx = 100+int(wx*30)
            
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
            
            #convert world coordinates into display coordinates
            
                          
            # if dy > 900:
                # dy = 900
            # if dx > 480:
                # dx = 480
            # print("dy: " + str(dy))
            # print("dx: " + str(dx))
            
            # Lidar Filter    
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
    
    

    ###################
    #
    # Controller
    #
    ###################
    if mode == 'manual':
        if kb.is_pressed("a") :
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
        elif kb.is_pressed('s'):
            # Part 1.4: Filter map and save to filesystem
            savemap = (map > 0.3).astype(int)
            np.save('map', savemap)
            print("Map file saved")
        elif kb.is_pressed('l'):
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
    if localization_mode == 'odometry':
        
        pose_x -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
        pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
        pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0
        display.setColor(int(0xFFF000))
        display.drawPixel(int(240+pose_x*30),int(700-pose_y*30))

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))
    
    item_detected = True
    get_obj_pos = (1.5, -0.25, -1.7, 1.7, 0.0, 0.0, 0.0)
    drag_obj_pos = (1.5, -0.5, -1.7, 1.7, 0.0, 0.0, 0.0)
    res_arm_pos = (0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41)
    #if item_detected:
    if kb.is_pressed("g"):
        #move arm into position
        robot_parts[3].setPosition(float(get_obj_pos[0]))
        robot_parts[4].setPosition(float(get_obj_pos[1]))
        robot_parts[5].setPosition(float(get_obj_pos[2]))
        robot_parts[6].setPosition(float(get_obj_pos[3]))
        robot_parts[7].setPosition(float(get_obj_pos[4]))
        robot_parts[8].setPosition(float(get_obj_pos[5]))
        robot_parts[9].setPosition(float(get_obj_pos[6]))
        
    if kb.is_pressed("p"): #drag object into basket
        robot_parts[3].setPosition(float(drag_obj_pos[0]))
        robot_parts[4].setPosition(float(drag_obj_pos[1]))
        robot_parts[5].setPosition(float(drag_obj_pos[2]))
        robot_parts[6].setPosition(float(drag_obj_pos[3]))
        robot_parts[7].setPosition(float(drag_obj_pos[4]))
        robot_parts[8].setPosition(float(drag_obj_pos[5]))
        robot_parts[9].setPosition(float(drag_obj_pos[6]))
        
    if kb.is_pressed("r"):
        robot_parts[3].setPosition(float(res_arm_pos[0]))
        robot_parts[4].setPosition(float(res_arm_pos[1]))
        robot_parts[5].setPosition(float(res_arm_pos[2]))
        robot_parts[6].setPosition(float(res_arm_pos[3]))
        robot_parts[7].setPosition(float(res_arm_pos[4]))
        robot_parts[8].setPosition(float(res_arm_pos[5]))
        robot_parts[9].setPosition(float(res_arm_pos[6]))
    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)