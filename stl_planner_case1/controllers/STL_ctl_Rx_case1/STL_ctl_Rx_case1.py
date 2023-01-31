"""Rx CAV controller"""

# ****************** RECEIVER CAV *******************

from vehicle import Driver
from controller import GPS, Compass, Emitter, Receiver


import struct
import math 
import numpy as np
import copy

import csv
import time

from path_optimizer import PathOptimizer
from math import sin, cos, pi, sqrt

# import scipy.spatial

#from pySTL import STLFormula

from STL_lattice_planner import plan_paths, transform_paths, get_closest_index,\
     get_goal_index, collision_check, clearance_check, select_best_path_index, \
         get_goal_state_set

def main():
    # create the Vehicle instance
    driver = Driver()

    # get the time step of the current world.
    timestep = int(driver.getBasicTimeStep())

    # GPS initial configuration 
    gp = driver.getDevice("global_gps_AV4")
    GPS.enable(gp,timestep)

    # compass initial configuration
    cp = driver.getDevice("compass_AV4")
    Compass.enable(cp,timestep)

    # communication Link initial configuration
    COMMUNICATION_CHANNEL = 2
    receiver = driver.getDevice("receiver_AV4")
    Receiver.enable(receiver,timestep)
    
    # verify that the channel is set correctly
    channel = receiver.getChannel();
    if (channel != COMMUNICATION_CHANNEL):
        receiver.setChannel(COMMUNICATION_CHANNEL)

    # linear velocity
    vf = 10 
    # distance between front and rear axis in the car
    l = 2

    # global route specification
    #global_waypoints = np.array([ [60.0, 8.0], [45.0, 8.0], [30.0, 8.0], \
    #    [15.0, 8.0], [0.0, 8.0]])
    global_waypoints = np.array([[0.0, 3.0],[15.0, 3.0], [30.0, 3.0], \
        [45.0, 3.0], [60.0, 3.0]])

    # obstacle's position specification
    obstacles = np.array([ [[37,1.0]], [[37,3]], [[39,3]], [[41,3]] ])

    number_wps = len(global_waypoints)

    # waypoint resolution for tracking controller e.g. next-WP[current + delta]
    delta = 3

    # reference distance value to enable a new path generation cycle
    threshold = 2

    # reference value for planning horizon
    lookahead = 7.0

    # variable for collision checking 
    circle_offsets = [-1.0, 1.0, 3.0] 
    circle_radii = [1.5, 1.5, 1.5]  
    circle_radii_x2 = [3.0, 3.0, 3.0] 

    # reference value to penalize lattice options
    weight = 10

    # flag variables to active path generation cycles
    enable_path_generation = True
    key = True

    # variable to keep track the number of generated paths
    counter_paths = 0

    # flag to enable communicatio between vehicles
    comm_flag = False


    header = ['distance', 'time']
    # open the file in the write mode
    #f = open('distances_to_obs.csv', 'w', encoding='UTF8')

    f = open('distances_wo_stl.csv', 'w', encoding='UTF8')

    # create the csv writer
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

    #start_time = time.perf_counter() #1
    # MAIN SIMULATION LOOP 
    counter_step = 0
    
    while driver.step() != -1:

        # *********** DEFINE CURRENT STATE OF THE EGO-VEHICLE ***********
        # get waypoints to follow: WP[i] and WP[i+delta]

        # set longitudinal velocity
        driver.setCruisingSpeed(vf)
        
        # define the current postion and orientation of the ego-vehicle
        
        # get GPS values    
        x = gp.getValues()[0]
        y = gp.getValues()[1]
        z = gp.getValues()[2]  

        # get compass values
        comp_x = cp.getValues()[0]
        comp_y = cp.getValues()[1]
        comp_z = cp.getValues()[2]

        # location of ego vehicle
        coord_ego = np.array([z, x])

        # get the heading of the vehicle
        angle2North = math.atan2(comp_x, comp_z) # atan2(Vertical,Horizontal)

        # case I --- angle needs correction
        if angle2North >= -math.pi and angle2North < -math.pi/2: 
            ang_heading = -3*math.pi/2 - angle2North # HEADING angle
        else:  # cases II, III, IV
            ang_heading = (math.pi/2) - angle2North # HEADING angle    

        # ******************* LOG DATA TASKS *********************
        # calculate the distance robot to the first obstacle
        distance_robot_to_obs = np.linalg.norm(coord_ego - obstacles[0])
        #print("distance_robot_to_obs: ", distance_robot_to_obs)
        #with open('distances_to_obs.csv', 'w', encoding='UTF8') as f:
        #    writer = csv.writer(f)
        #    # write the header
        #    writer.writerow([aux_dist])


        #end_time = time.perf_counter()      # 2
        #run_time = end_time - start_time    # 3
        #writer.writerow([distance_robot_to_obs,run_time])
        #start_time = time.perf_counter() #1

        time_sec = (timestep/1000)*counter_step
        writer.writerow([distance_robot_to_obs,time_sec])
        # ******************* COMMUNICATION TASKS *********************
        # receive the message
        if receiver.getQueueLength() > 0: 
            # read current packet's data 
            msg = receiver.getData()
            buffer = struct.unpack('ff?',msg)
            neighboor_car_pos = np.array([ [[buffer[0],buffer[1]]] ])
            comm_flag = True
            receiver.nextPacket()
        else:
            pass   
        
        # ********************** GENERATE LATTICES **********************

        # get initial and final points for the lattice path
        ego_state = np.array([coord_ego[0], coord_ego[1], ang_heading])

        # get current and next global waypoint
        distances_to_global_wps = np.linalg.norm(global_waypoints - \
            coord_ego,axis=1)

        if key and counter_paths < number_wps-1:        
            min_len_to_global_wp, index_min_global_wp = \
                get_closest_index(distances_to_global_wps)
            initial_global_wp = global_waypoints[index_min_global_wp]
            index_next_global = get_goal_index(global_waypoints, lookahead, \
                min_len_to_global_wp, index_min_global_wp)
            next_global_wp = global_waypoints[index_next_global]
            enable_path_generation = True
            key = False
        else:
            pass       

        # path generation cycle    
        if enable_path_generation:

            vel_ref = 0 # reference value (velocity) --- *REPLACE WITH IDM

            # calculate the goal position (goal_x,goal_y)
            goal_state = np.array([next_global_wp[0],next_global_wp[1],vel_ref])

            # get the set of states available for lattice paths
            goal_state_set = get_goal_state_set(goal_state, ego_state, \
                initial_global_wp, next_global_wp)

            # generate lattice path options
            my_paths, my_path_validity = plan_paths(goal_state_set)

            # transform the paths to match current ego-vehicle configuration
            my_transformed_paths = transform_paths(my_paths, ego_state) 
    
            # obstacle detection process with STL
            collision_check_array_var, robustness_robt2obs_array = \
                collision_check(my_transformed_paths, obstacles, \
                    circle_offsets, circle_radii)

            # PRINTING VALUES FOR DEBUG ONLY
            #print("collision_check_array_var: ",collision_check_array_var," robustness_robt2obs_array: ",robustness_robt2obs_array) 
            

            # clearance verification process with STL
            if comm_flag:
                clearance_check_array_var, robusteness_array = \
                    clearance_check(my_transformed_paths, neighboor_car_pos, \
                        circle_offsets, circle_radii, circle_radii_x2)
                comm_flag = False
                # PRINTING VALUES FOR DEBUG ONLY
                #print("clearance_check_array_var: ",clearance_check_array_var, \
                #     " robusteness_array: ",robusteness_array) 
                #print("counter_paths: ", counter_paths)
                # get the index for the selected lattice path option
                control_var = select_best_path_index(my_transformed_paths, \
                    collision_check_array_var, goal_state,robusteness_array, \
                        robustness_robt2obs_array, weight) 
            else:
                robusteness_array_init = np.zeros(len(my_transformed_paths))
                # get the index for the selected lattice path option
                control_var = select_best_path_index(my_transformed_paths, \
                    collision_check_array_var, goal_state,\
                        robusteness_array_init, robustness_robt2obs_array, weight) 
                #print("counter_paths XYZ: ", counter_paths)
            
            #print("control_var: ",control_var)  # *******************************************************************************
            # create the final reference path
            reference_path = \
                np.transpose(my_transformed_paths[control_var][0:2][:])
            #print("**************    NEW PATH GENERATED   *****************")
            counter_paths = counter_paths + 1
            enable_path_generation = False

        else:
            pass

        # condition to activate a new path generation cycle
        dist_last_wp_on_path = \
            np.linalg.norm(reference_path[len(reference_path)-1] - coord_ego)
        if  dist_last_wp_on_path < threshold:
            key = True
        else:
            key = False   

        # ********************** TRACKING CONTROLLER **********************
        distances_to_path = np.linalg.norm(reference_path - coord_ego,axis=1)

        # find index of minimum value from 2D numpy array
        result_aux = np.where(distances_to_path == np.amin(distances_to_path))
        index_min = result_aux[0][0]
        initial_wp = reference_path[index_min]
    
        # verify if the current waypoint is the last of the reference path
        if  ( (index_min+delta) > (len(reference_path)-1) ) and \
            counter_paths == number_wps-1:

            next_wp = reference_path[len(reference_path)-1]
            if index_min == len(reference_path)-1:
                vf = 0
            else:
                pass
        else:   # get the next waypoint to track   
            next_wp = reference_path[index_min+delta]

        waypoints = np.array([initial_wp,next_wp])

        # get angle to last waypoint --- waypoint[coordinate_VorH][#wp]
        ang2lastWP = math.atan2(x-waypoints[0][1], z-waypoints[0][0])

        # get angle of the reference path
        angPath = math.atan2(waypoints[1][1]-waypoints[0][1], \
            waypoints[1][0]-waypoints[0][0])

        # get crosstrack error
        crosstrack_error = np.amin(np.linalg.norm(coord_ego - waypoints,axis=1))

        # get the crosstrack error sign
        ang_diff = ang2lastWP - angPath

        if ang_diff > np.pi:
            ang_diff -= 2*np.pi
        if ang_diff < -np.pi:
            ang_diff += 2*np.pi

        if ang_diff>0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = -abs(crosstrack_error)

        # get the error in the angle to consider in the control task
        ang_error = ang_heading - angPath

        if ang_error > np.pi:
            ang_error -= 2*np.pi
        if ang_error < -np.pi:
            ang_error += 2*np.pi    

        # parameter for longitudinal control
        k_e = 0.8

        # get final steering angle
        ang_crosstrack = math.atan2(k_e * crosstrack_error , vf)    
        ro =  ang_error + ang_crosstrack 
        steering_ang = copy.copy(ro)
        steering_ang = min(0.8, steering_ang)
        steering_ang = max(-0.8, steering_ang)

        # set final steering angle for the ego-vehicles
        driver.setSteeringAngle(steering_ang)   

        counter_step = counter_step + 1 

    f.close()

if __name__ == "__main__":
    main()



