# **************** METHODS FOR THE STL LATTICE PLANNER *****************

import numpy as np

from path_optimizer import PathOptimizer
from math import sin, cos, pi, sqrt, exp

import scipy.spatial

import copy

def stl_formula_min_dist(min_dist,signal_h):
    return np.amin(signal_h[:,0] - min_dist)

def plan_paths(goal_state_set):
    """Plans the path set using the polynomial spiral optimization.
    Plans the path set using polynomial spiral optimization to each of the
    goal states.
    args:
        goal_state_set: Set of goal states (offsetted laterally from one
            another) to be used by the local planner to plan multiple
            proposal paths. These goals are with respect to the vehicle
            frame.
            format: [[x0, y0, t0, v0],
                     [x1, y1, t1, v1],
                     ...
                     [xm, ym, tm, vm]]
            , where m is the total number of goal states
              [x, y, t] are the position and yaw values at each goal
              v is the goal speed at the goal point.
              all units are in m, m/s and radians
    returns:
        paths: A list of optimized spiral paths which satisfies the set of 
            goal states. A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    x_points: List of x values (m) along the spiral
                    y_points: List of y values (m) along the spiral
                    t_points: List of yaw values (rad) along the spiral
                Example of accessing the ith path, jth point's t value:
                    paths[i][2][j]
            Note that this path is in the vehicle frame, since the
            optimize_spiral function assumes this to be the case.
        path_validity: List of booleans classifying whether a path is valid
            (true) or not (false) for the local planner to traverse. Each ith
            path_validity corresponds to the ith path in the path list.
    """
    paths         = []
    path_validity = []
    for goal_state in goal_state_set:
        path_optimizer = PathOptimizer()
        path = path_optimizer.optimize_spiral(goal_state[0], 
                                                    goal_state[1], 
                                                    goal_state[2])
        if np.linalg.norm([path[0][-1] - goal_state[0], 
                           path[1][-1] - goal_state[1], 
                           path[2][-1] - goal_state[2]]) > 0.1:
            path_validity.append(False)
        else:
            paths.append(path)
            path_validity.append(True)
    return paths, path_validity

def transform_paths(paths, ego_state):
    """ Converts the to the global coordinate frame.
    Converts the paths from the local (vehicle) coordinate frame to the
    global coordinate frame.
    args:
        paths: A list of paths in the local (vehicle) frame.  
            A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    , x_points: List of x values (m)
                    , y_points: List of y values (m)
                    , t_points: List of yaw values (rad)
                Example of accessing the ith path, jth point's t value:
                    paths[i][2][j]
        ego_state: ego state vector for the vehicle, in the global frame.
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)
    returns:
        transformed_paths: A list of transformed paths in the global frame.  
            A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    , x_points: List of x values (m)
                    , y_points: List of y values (m)
                    , t_points: List of yaw values (rad)
                Example of accessing the ith transformed path, jth point's 
                y value:
                    paths[i][1][j]
    """
    transformed_paths = []
    for path in paths:
        x_transformed = []
        y_transformed = []
        t_transformed = []
        for i in range(len(path[0])):
            x_transformed.append(ego_state[0] + path[0][i]*cos(ego_state[2]) - \
                                                path[1][i]*sin(ego_state[2]))
            y_transformed.append(ego_state[1] + path[0][i]*sin(ego_state[2]) + \
                                                path[1][i]*cos(ego_state[2]))
            t_transformed.append(path[2][i] + ego_state[2])

        transformed_paths.append([x_transformed, y_transformed, t_transformed])

    return transformed_paths

def get_closest_index(distances_to_global_wps):
    """Gets closest index from a given list of distances.
    
    args:
        distances_to_global_wps: distances in m from waypoints to the 
            ego-vehicle position.
            format: [dist_0,
                     dist_1,
                     ...
                     dist_n]
    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle
                i.e. waypoints[closest_index] gives the waypoint closest to the 
                vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0     
    # Find index of minimum value from 2D numpy array
    closest_len = np.amin(distances_to_global_wps)
    result_aux = np.where(distances_to_global_wps == closest_len)
    closest_index = result_aux[0][0]             

    return closest_len, closest_index
    
def get_goal_index(gb_waypoints, lookahead, closest_len, closest_index):
    """Gets the goal index for the vehicle. 
    
    Set to be the earliest waypoint that has accumulated arc length
    accumulated arc length (including closest_len) that is greater than or
    equal to "lookahead".
    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]: 
                returns the 3rd waypoint's y position
                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)
        closest_len: length (m) to the closest waypoint from the vehicle.
        closest_index: index of the waypoint which is closest to the vehicle.
            i.e. waypoints[closest_index] gives the waypoint closest to the 
            vehicle.
    returns:
        wp_index: Goal index for the vehicle to reach
            i.e. waypoints[wp_index] gives the goal waypoint
    """
    # Find the farthest point along the path that is within the
    # lookahead distance of the ego vehicle.
    # Take the distance from the ego vehicle to the closest waypoint into
    # consideration.
    arc_length = closest_len
    wp_index = closest_index
    
    # In this case, reaching the closest waypoint is already far enough for
    # the planner.  No need to check additional waypoints.
    if arc_length > lookahead:
        return wp_index
    else:
        pass

    #############  REVIEW THIS CONDITION, MAYBE SHOULD NOT BE HERE
    # We are already at the end of the path.
    #if wp_index == len(waypoints) - 1:    
    #    return wp_index
    
    # Otherwise, find our next waypoint. 
    while wp_index < len(gb_waypoints) - 1:
        arc_length += np.linalg.norm(gb_waypoints[wp_index+1] - \
            gb_waypoints[wp_index])

        if arc_length >= lookahead:
            wp_index += 1
            break
        else:
            wp_index += 1
    
    return wp_index


# *********************  OBSTACLE DETECTION WTIH STL *********************
def collision_check(paths, obstacles, circle_offsets, circle_radii):
    """Returns a bool array on whether each path is collision free, and the 
    associated robustness measure.
    args:
        paths: A list of paths in the global frame.  
            A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    x_points: List of x values (m)
                    y_points: List of y values (m)
                    t_points: List of yaw values (rad)
                Example of accessing the ith path, jth point's t value:
                    paths[i][2][j]
        obstacles: A list of [x, y] points that represent points along the
            border of obstacles, in the global frame.
            Format: [[x0, y0],
                     [x1, y1],
                     ...,
                     [xn, yn]]
            , where n is the number of obstacle points and units are [m, m]
        circle_offsets: A list of circle offsets
            Format: [circ_1, circ_2, ... ,circ_n] 
        circle_radii: A list of the radious corresponding to each circle
            Format: [rad_1, rad_2, ... ,rad_n] 
    returns:
        collision_check_array: A list of boolean values which classifies
            whether the path is collision-free (true), or not (false). The
            ith index in the collision_check_array list corresponds to the
            ith path in the paths list.
        robustness_robt2obs: A list of the robustness measure for each path
    """
    rob2obs_min_dist = 2*circle_radii[0]#/2
    collision_check_array = np.zeros(len(paths), dtype=bool)
    robustness_robt2obs = np.zeros( len(paths) )
    sample_path = paths[0]
    
    clearance_dist2obs = np.zeros( len(obstacles) ) 
    clearance_dist2obs_array = np.zeros( len(sample_path[0]) ) 
    
    for i in range(len(paths)):
        collision_free = True
        path           = paths[i]
        # Iterate over the points in the path.
        for j in range(len(path[0])):
            circle_locations = np.zeros((len(circle_offsets), 2))
            circle_locations[:, 0] = \
                [i * int(np.cos(path[2][j])) for i in circle_offsets]+path[0][j]
            circle_locations[:, 1] = \
                [i * int(np.sin(path[2][j])) for i in circle_offsets]+path[1][j]
            for k in range(len(obstacles)):
                collision_dists = \
                    scipy.spatial.distance.cdist(obstacles[k], circle_locations)
                collision_dists = np.subtract(collision_dists, circle_radii)              
                clearance_dist2obs[k] = np.amin(collision_dists[0])
                collision_free =collision_free and not np.any(collision_dists<0)
                if not collision_free:
                    break
                else:
                    pass         
            clearance_dist2obs_array[j] = np.amin(clearance_dist2obs)
            
            if not collision_free:
                break
            else:
                pass

        # get the signal that encodes distances robot-to-obstacles        
        clearance_distances_to_obs = np.array([[clearance_dist2obs_array[n], n]\
             for n in range(len(path[0]))])  
        # apply the STL formula       
        robustness_val = \
            stl_formula_min_dist(rob2obs_min_dist, clearance_distances_to_obs)    
        # get the robustness value for the current path
        robustness_robt2obs[i] =  robustness_val 
        # get the array that encodes possible collision with obstacles                      
        collision_check_array[i] = collision_free

    return collision_check_array, robustness_robt2obs    
    
# ********************** INTER-VEHICLE CLEARANCE **********************
def clearance_check(paths_op, neighboor_car_state, circle_offsets, \
    circle_radii, circle_radii_x2):
    """Returns a bool array on whether each path is collision free with other
    vehicles, and the associated robustness measure.
    args:
        paths_op: A list of paths in the global frame.  
            A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    x_points: List of x values (m)
                    y_points: List of y values (m)
                    t_points: List of yaw values (rad)
                Example of accessing the ith path, jth point's t value:
                    paths[i][2][j]
        neighboor_car_state: ego state vector for neighboor vehicle, in the 
            global frame.
            format: [neighboor_x, neighboor_y]
                neighboor_x and neighboor_y  : position (m)
                neighboor_yaw                : top-down orientation [-pi to pi]
                neighboor_open_loop_speed    : open loop speed (m/s)
        circle_offsets: A list of circle offsets
            Format: [circ_1, circ_2, ... ,circ_n] 
        circle_radii: A list of the radious corresponding to each circle
            Format: [rad_1, rad_2, ... ,rad_n] 
        circle_radii_x2: A list of the diameter corresponding to each circle
            Format: [radx2_1, radx2_2, ... ,radx2_n]
    returns:
        clearance_check_array: A list of boolean values which classifies
            whether the path is collision-free (true), or not (false) with the
            neighboor vehicle. The ith index in the clearance_check_array list
            corresponds to the ith path in the paths list.
        robusteness_values: A list of the robustness measure for each path
    """
    rob2rob_min_dist = 2*circle_radii[0]
    clearance_check_array = np.zeros(len(paths_op), dtype=bool)   
    robusteness_values = np.zeros( len(paths_op) )
    clearance_dist_rob2rob = np.zeros( len(neighboor_car_state) )   
    aux_path = paths_op[0]
    clearance_distances_array = np.zeros( len(aux_path[0]) )    
    
    for i in range(len(paths_op)):
        clearance_free = True
        path_op = paths_op[i]
        # Iterate over the points in the path.
        for j in range(len(path_op[0])):
            circle_locations_av = np.zeros((len(circle_offsets), 2))           
            circle_locations_av[:, 0] = [i * int(np.cos(path_op[2][j])) \
                for i in circle_offsets] + path_op[0][j]
            circle_locations_av[:, 1] = [i * int(np.sin(path_op[2][j])) \
                for i in circle_offsets] + path_op[1][j]
            for k in range(len(neighboor_car_state)):
                clearance_dists = \
                    scipy.spatial.distance.cdist(neighboor_car_state[k], \
                        circle_locations_av)
                clearance_dists = np.subtract(clearance_dists, circle_radii_x2)
                clearance_dist_rob2rob[k] = np.amin(clearance_dists[0])
                clearance_free =clearance_free and not np.any(clearance_dists<0)
                
                if not clearance_free:
                    break
                else:
                    pass
            
            clearance_distances_array[j] = np.amin(clearance_dist_rob2rob)
                    
            if not clearance_free:
                break
            else:
                pass

        # get the signal that encodes distances robot-to-robot          
        clearance_distances = np.array( [ [clearance_distances_array[n], n] \
            for n in range(len(path_op[0])) ] )    
        # apply the STL formula
        robustness = stl_formula_min_dist(rob2rob_min_dist, clearance_distances)    
        # get the robustness value for the current path
        robusteness_values[i] =  robustness     
        # get the array that encodes possible collision between robots
        clearance_check_array[i] = clearance_free

    return clearance_check_array, robusteness_values    
    
# ************ BEST PATH INDEX SELECTION WITH STL ************
def select_best_path_index(paths, collision_check_array, goal_state, \
    robusteness_array_val, robustness_robt2obs_val, weight):
    """Returns the path index which is best suited for the vehicle to
    traverse.
    Selects a path index which is closest to the center line as well as far
    away from collision paths.
    args:
        paths: A list of paths in the global frame.  
            A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    x_points: List of x values (m)
                    y_points: List of y values (m)
                    t_points: List of yaw values (rad)
                Example of accessing the ith path, jth point's t value:
                    paths[i][2][j]
        collision_check_array: A list of boolean values which classifies
            whether the path is collision-free (true), or not (false). The
            ith index in the collision_check_array list corresponds to the
            ith path in the paths list.
        goal_state: Goal state for the vehicle to reach (centerline goal).
            format: [x_goal, y_goal, v_goal], unit: [m, m, m/s]
        robusteness_array_val: A list of the robustness measure for each path 
            related to the robot to robot minimum distance.
        robustness_robt2obs_val: A list of the robustness measure for each path
            related to the robot to obstacles minimum distance.
        weight: Weight that is multiplied to the best index score.
    returns:
        best_index: The path index which is best suited for the vehicle to
            navigate with.
    """
    beta = 1.1
    best_index = None
    best_score = float('Inf')
    for i in range(len(paths)):
        # Handle the case of collision-free paths.
        if collision_check_array[i]:
            # Compute the "distance from centerline" score.
            # The centerline goal is given by goal_state.
            # The exact choice of objective function is up to you.
            # A lower score implies a more suitable path.
            score = np.sqrt((goal_state[0]-paths[i][0][len(paths[i][0])-1])**2 \
                 + (goal_state[1] - paths[i][1][len(paths[i][0])-1])**2) 

            # ---------------------------------------------------------------     
            # Compute the "proximity to other colliding paths" score and
            # add it to the "distance from centerline" score.
            # The exact choice of objective function is up to you.
            #for j in range(len(paths)):
            #    if j == i:
            #        continue
            #    else:
            #        if not collision_check_array[j]:
            #            score += weight * paths[i][2][j]
            #            pass
            #        
            # --------------------------------------------------------------- 

            # MODIFY THIS ....
            score += max(exp(-beta*robusteness_array_val[i]),1)

            #score *= max(exp(-beta*robustness_robt2obs_val[i]),1)
            #score -= 0.8*robustness_robt2obs_val[i]#+0.1*robusteness_array_val[i]
        # Handle the case of colliding paths.
        else:
            score = float('Inf')
        # Set the best index to be the path index with the lowest score
        if score < best_score:
            best_score = score
            best_index = i

    return best_index

def get_goal_state_set(goal_state,ego_state,initial_global_wp,next_global_wp):
    """Gets the goal states given a goal position.
    
    Gets the goal states given a goal position. The states 
    args:
        goal_state: Goal state for the vehicle to reach (global frame)
            format: [x_goal, y_goal, v_goal], in units [m, m, m/s]
        ego_state: ego state vector for the vehicle, in the global frame.
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)
        initial_global_wp = Initial waypoint
            format: [wp_x, wp_y]
        next_global_wp = Next waypoint
            format: [next_wp_x, next_wp_y]
    returns:
        goal_state_set: Set of goal states (offsetted laterally from one
            another) to be used by the local planner to plan multiple
            proposal paths. This goal state set is in the vehicle frame.
            format: [[x0, y0, t0, v0],
                     [x1, y1, t1, v1],
                     ...
                     [xm, ym, tm, vm]]
            , where m is the total number of goal states
              [x, y, t] are the position and yaw values at each goal
              v is the goal speed at the goal point.
              all units are in m, m/s and radians
    """
    # Compute the final heading based on the next index.
    # If the goal index is the last in the set of waypoints, use
    # the previous index instead.

    goal_state_local = copy.copy(goal_state)
    goal_state_local[0] -= ego_state[0]
    goal_state_local[1] -= ego_state[1]
    goal_x = goal_state_local[0] * np.cos(ego_state[2]) + \
        goal_state_local[1] * np.sin(ego_state[2])
    goal_y = goal_state_local[0] * -np.sin(ego_state[2]) + \
        goal_state_local[1] * np.cos(ego_state[2])
    # calculate the goal heading value
    delta_x = next_global_wp[0] - initial_global_wp[0] 
    delta_y = next_global_wp[1] - initial_global_wp[1] 
    heading_global_path = np.arctan2(delta_y, delta_x)
    goal_t = heading_global_path - ego_state[2]
    # keep goal heading within [-pi,pi] so the optimizer behaves well
    if goal_t > pi:        
        goal_t -= 2*pi
    elif goal_t < -pi:
        goal_t += 2*pi
    else:
        pass
    # lattice paths parameters
    path_offset = 1  #6
    num_paths = 9 #3
    goal_state_set = []
    # velocity is preserved after the transformation
    goal_v = 0.5 # constant velocity
    # get the offset points for the lattice path options
    for i in range(num_paths):
        j = num_paths - (i+1)
        offset = (j - num_paths // 2) * path_offset
        x_offset = offset * np.cos(goal_t + pi/2)
        y_offset = offset * np.sin(goal_t + pi/2)
        goal_state_set.append([goal_x+x_offset, goal_y+y_offset,goal_t, goal_v])    

    return goal_state_set
