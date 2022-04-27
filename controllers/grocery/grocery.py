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
target_pos = (0.0, 0.0, 0, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
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

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Odometry

localization_mode = 'gps'
# localization_mode = 'odometry'

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
#rrt testing

mode = 'manual' # Part 1.1: manual mode
# mode = 'planner'
# mode = 'autonomous'

###################
#
# RRT
#
###################
#  green cube pickup world coordinates WHITE SHELVES  
#   ordered in (wx, wy, wtt)
cube_waypoints = [(6.39358,-13.4795099,4.71195),
                (8.11796,-7.76351,3.14174),
                (8.129499, -1.490345,3.141773),
                (8.1800349,-3.49444,-0.001),
                (12.12526,-2.4653599,3.14034),
                (4.1244388, -2.509675, 3.1414869),
                (4.1303769, 0.41514208, 3.141649),
                (0.901962, 3.33814055, 0.0004155),
                (0.9239285, -2.739372, -0.00824976),
                (0.134933779,0.3488725640,3.1284947),
                (0.13583087,2.142451,3.1260417)]
            

display_waypoints = []
temp_list = []

#convert to display coordinates from world coordinates in waypoint list
for coord in cube_waypoints:
    counter=0
    temp_list = []
    for i, element in enumerate(coord):
        # print(element)
        
        if localization_mode == 'odometry':
            if i == 0: #x value
                temp_list.append(245+int(element*30))
            if i == 1: #y value
                temp_list.append(700-int(element*30))
            
        if localization_mode == 'gps':
            if i == 0: #x value
                temp_list.append(115+int(element*30))
            if i == 1: #y value
                temp_list.append(290-int(element*30))

        if i == 2:
            temp_list.append(element)

    display_waypoints.append(tuple(temp_list)) 

# print(display_waypoints)


#rrt in pixel coordnates 
#define above to convert pixel coordnates to global coordniates
def setup_random_2d_world(map):
    '''
    Function that sets a 2D world with fixed bounds and # of obstacles
    :return: The bounds, the obstacles, the state_is_valid() function
    '''
    state_bounds = np.array([[0,480],[0,900]]) # matrix of min/max values for each dimension
    obstacles = [] # [pt, radius] circular obstacles
    rows, cols = map.shape
    for row in rows:
        for cols in cols:
            if map[row][col] == 1:
                obstacles.append([row,cols], 1)
    
    # for n in range(30):
    #     obstacles.append(get_nd_obstacle(state_bounds))

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

    return state_bounds, obstacles, state_is_valid

setup_random_2d_world(map)
def check_path_valid(path, state_is_valid):
    '''
    Function that checks if a path (or edge that is made up of waypoints) is collision free or not
    :param path: A 1D array containing a few (10 in our case) n-dimensional points along an edge
    :param state_is_valid: Function that takes an n-dimensional point and checks if it is valid
    :return: Boolean based on whether the path is collision free or not
    '''
    class vars:
        v_path = False #boolean to check the path validity
        length = len(path) #length value for the while loop    
        count = 0 #iterator value for the while loop
    while vars.count < vars.length: #while count is less than the length of path
        if not state_is_valid(path[vars.count]): #check if path passes the given state_is_valid function
            vars.v_path = True #we update path validity to be true
        vars.count = vars.count + 1 #increse step in while loop

    return vars.v_path #return if the path is valid or not
    
def rrt(start_pt, end_pt, map):
    #check pixels, modify to go faster
    delta_q = 0.5
    #checks single coord is valid
    
    #every tuple has coord, index of parent
    explored = [(start_pt, None)]
    #print(explored)
    for n in range(10000):
        #random coord within map's rows and columns
        q_rand = (np.random.randint(len(map)), np.random.randint(len(map[0])))
        # print(q_rand)
        #0.05 chance to test if at end point
        if np.random.rand() < 0.05:
            q_rand = end_pt
        #closest to q_rand
        closest_index = -1
        closest_dist = float("inf")
        #finds the closest point
        for i in range(len(explored)):
            pt = explored[i][0]
            #eucidean distance
            dist = ((q_rand[0] - pt[0]) ** 2+ (q_rand[1] - pt[1]) ** 2) ** 0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_index = i
        #path is valid
        isValid = True
        closest_pt = explored[closest_index][0]
        #checks every point along the line between closest pt and q_rand
        for i in np.arange(0, closest_dist, delta_q):
            p = i  / closest_dist
            # print(p)
            if not valid((closest_pt[0] + p * (q_rand[0] - closest_pt[0]), closest_pt[1] + p * (q_rand[1] - closest_pt[1]))):
                isValid = False
                break
        if isValid:
            #adds a tuple to the explored array
            explored.append((q_rand, closest_index))
            #checks within one pixel of the goal
            dist = ((q_rand[0] - end_pt[0]) ** 2+ (q_rand[1] - end_pt[1]) ** 2) ** 0.5
            if dist < 1:
                #print(explored)
                path = [q_rand]
                c = explored[len(explored) - 1]
                #unrolls path using parent pointer
                while not c[1] is None:
                    c = explored[c[1]]
                    path.insert(0, c[0])
                return path
    #print(explored)
    return -1 


###################
#
# Planner
#
###################
if mode == 'planner':
    map = np.load('map.npy')
    pixels = 0
    map = np.flipud(map)
    map = np.rot90(map, 3)
    plt.imshow(map)
    #print(map)
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
    #print(pixels)
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
    start = (200, 200)
    end = (300, 200)
    # end = (293, 316)
    # print("test")
    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    #rrt testing 
    print(rrt(start, end, Convolved_map))

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
map = np.zeros((480,900)) # Replace None by a numpy 2D floating point array
waypoints = []
if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    waypoints = [] # Replace with code to load your path
    
state = 0 # use this to iterate through your path



print(localization_mode) 
frame_marker = 0
item_detected = False

xy = rrt(display_waypoints[0], display_waypoints[5], map)
print("xy")
print(xy)
for i in range(len(xy)-1):
    
    display.setColor(0x00FF00)
    display.drawLine(xy[i][0], xy[i][1], xy[i+1][0], xy[i+1][1])
    

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
        display.drawPixel(int(115+pose_x*30),int(290-pose_y*30))
        
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
        wxx = pose_x
        wyy = pose_y
        wtt = pose_theta
        # print("wx: " + str(wx))
        # print("wy: " + str(wy))
        # print("wtt: " + str(wtt))
        #convert world coordinates into display coordinates
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
            
            #convert world coordinates into display coordinates
            
                          
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
            
            g = int(val* 255) # converting [0,1] to grayscale intensity [0,255]
            color = g*256**2+g*256+g
            display.setColor(color)
            display.drawPixel(dx,dy) #draws from the top left corner(0,900)
            # print(display_waypoints)
            
            
            
    # Draw the robot's current pose on the 360x360 display
    
    ##################
    #
    # RRT
    #
    ##################
    # print("RRT")
    # for i in range(1,11):
    
        # print(rrt(display_waypoints[0], display_waypoints[i], map))
    
    
    
    
    
    
    
    
    
    
    
    

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
    
    ####################
    #
    # Manipulation
    #
    ####################
    tpos_pos_0     = (0.07, 0, -1.7, 0.0, -1.7, 1.39, 1.7)
    scoop_pos_0    = (1.3, -0.15, -1.7, 0.8, -1.7, 1.39, 1.7)
    scoop_pos_1 =    (0.07, 0, -1.4, 2.29, -1.7, 1.39, 1.7)
    #if item_detected:
    # print(frame_marker)
    
    #Scoop animation stage 1
    if frame_marker >= 0 and frame_marker <= 45 and item_detected == True:
        #move arm into position
        robot_parts[3].setPosition(float(tpos_pos_0[0]))
        robot_parts[4].setPosition(float(tpos_pos_0[1]))
        robot_parts[5].setPosition(float(tpos_pos_0[2]))
        robot_parts[6].setPosition(float(tpos_pos_0[3]))
        robot_parts[7].setPosition(float(tpos_pos_0[4]))
        robot_parts[8].setPosition(float(tpos_pos_0[5]))
        robot_parts[9].setPosition(float(tpos_pos_0[6]))
        
    # Scoop animation stage 2   
    if frame_marker > 45 and frame_marker <= 110 and item_detected == True:
        # print("h")
        robot_parts[3].setPosition(float(scoop_pos_0[0]))
        robot_parts[4].setPosition(float(scoop_pos_0[1]))
        robot_parts[5].setPosition(float(scoop_pos_0[2]))
        robot_parts[6].setPosition(float(scoop_pos_0[3]))
        robot_parts[7].setPosition(float(scoop_pos_0[4]))
        robot_parts[8].setPosition(float(scoop_pos_0[5]))
        robot_parts[9].setPosition(float(scoop_pos_0[6]))
     
    # Scoop animation stage 3    
    if frame_marker > 110 and frame_marker <= 170 and item_detected == True:
        # print("k")
        robot_parts[3].setPosition(float(scoop_pos_1[0]))
        robot_parts[4].setPosition(float(scoop_pos_1[1]))
        robot_parts[5].setPosition(float(scoop_pos_1[2]))
        robot_parts[6].setPosition(float(scoop_pos_1[3]))
        robot_parts[7].setPosition(float(scoop_pos_1[4]))
        robot_parts[8].setPosition(float(scoop_pos_1[5]))
        robot_parts[9].setPosition(float(scoop_pos_1[6]))
    
    if frame_marker > 170 and frame_marker <= 240:
        #reverse
        vL = -MAX_SPEED/2
        vR = -MAX_SPEED/2
    # Reset robo arm
    
    if frame_marker > 240 and frame_marker <= 290:
        for i in range(N_PARTS):
            robot_parts[i].setPosition(float(target_pos[i]))
       
    if frame_marker > 290: #animation complete. Reset Vars
        item_detected = False
        frame_marker = 0        
    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
    
    if item_detected == True:
        frame_marker+=1

    #########################
    #
    # Computer Vision
    #
    #########################
    
    camera.saveImage("image.png", 50)
    camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    image = camera.getImage()
    for i in range(0,width):
        for j in range(0,height):
            g = camera.imageGetGreen(image, width, i, j)
            r = camera.imageGetRed(image, width, i, j)
            b = camera.imageGetBlue(image, width, i, j)
            # print("hello")
    
