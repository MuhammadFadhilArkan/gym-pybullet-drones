import math
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
import os
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
from gym_pybullet_drones.utils.utils import nnlsRPM
#X merah, Y hijau, Z biru


class PayloadCoop(BaseMultiagentAviary):
    
    ################################################################################

    def __init__(self,
                 dest_point: np.ndarray = np.array([0, 4, 0.5]),
                 episode_len_sec: float=60,
                 max_distance_between_drone: float=1,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=100,
                 aggregate_phy_steps: int=10,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.PAYLOAD_Z_CONST,
                 act: ActionType=ActionType.XY_YAW,                 
                 ):
                 
        if(initial_xyzs == None):
            initial_xyzs = self._initPositionOnCircle(num_drones, r = max_distance_between_drone/4, z = dest_point[2])

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )
        self.Z_CONST = dest_point[2]
        self.MAX_DISTANCE_BETWEEN_DRONE = max_distance_between_drone
        self.MAX_XY = 30
        self.MAX_Z = 3
        self.DEST_POINT = dest_point
        self.OBSTACLE_IDS = []
        self.EPISODE_LEN_SEC = episode_len_sec
        self.K_MOVE = 0.1 # For ActionType.Joystick
        self.TARGET_HISTORY = np.zeros((self.NUM_DRONES, int(self.SIM_FREQ * self.EPISODE_LEN_SEC / self.AGGR_PHY_STEPS), 3))
        self.POSITION_HISTORY = np.zeros((self.NUM_DRONES, int(self.SIM_FREQ * self.EPISODE_LEN_SEC / self.AGGR_PHY_STEPS), 3))

        # assert self.NUM_DRONES == 2, "NUM_DRONES is not 2"
        assert self.DEST_POINT[0] < self.MAX_XY and self.DEST_POINT[1] < self.MAX_XY, "1.5 * dest_point exceeds MAX_XY"
    ################################################################################

    def _actionSpace(self):
        if(self.ACT_TYPE == ActionType.JOYSTICK):
            return spaces.Dict({i: spaces.Discrete(5) for i in range(self.NUM_DRONES)})

        if self.ACT_TYPE in [ActionType.RPM, ActionType.DYN, ActionType.VEL, ActionType.XYZ_YAW]:
            size = 4
        elif self.ACT_TYPE in [ActionType.PID, ActionType.XY_YAW]:
            size = 3
        else:
            print("[ERROR] in BaseMultiagentAviary._actionSpace()")
            exit()
        return spaces.Dict({i: spaces.Box(low=-1*np.ones(size),
                                          high=np.ones(size),
                                          dtype=np.float32
                                          ) for i in range(self.NUM_DRONES)})
    ################################################################################

    def _observationSpace(self):
        if self.OBS_TYPE in [ObservationType.KIN, ObservationType.PAYLOAD, ObservationType.PAYLOAD_Z_CONST]:
            if self.OBS_TYPE == ObservationType.KIN:
                #(x,y,z, r,p,y, x_dot, y_dot, z_dot, r_dot,p_dot,y_dot, ob_x+,ob_y+,ob_x-,ob_y-, x_dest-x,y_dest-y,z_dest-z,  x_dr2-x,y_dr2-y,z_dr2-z .....)
                #12 + 4 state obstacle, 3 state distance to dest_point, 3N state distance between drone,
                dist_drone = np.ones((3*(self.NUM_DRONES - 1)))
                low = np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, 0,0,0,0, -1,-1,-1])
                high = np.array([1,1,1,   1,1,1,    1,1,1,    1,1,1,   1,1,1,1,  1,1,1])

            elif self.OBS_TYPE == ObservationType.PAYLOAD_Z_CONST:
                #(ob_x+,ob_y+,ob_x-,ob_y-, x_dest-x,y_dest-y, x_dr2-x,y_dr2-y .....)
                #4 state obstacle, 2 state distance to dest_point, 2N state distance between drone,
                dist_drone = np.ones((2*(self.NUM_DRONES - 1)))
                low = np.array([0,0,0,0, -1,-1])
                high = np.array([1,1,1,1,  1,1])

            elif self.OBS_TYPE == ObservationType.PAYLOAD:
                #(ob_x+,ob_y+,ob_x-,ob_y-, x_dest-x,y_dest-y, x_dr2-x,y_dr2-y .....)
                #4 state obstacle, 3 state distance to dest_point, 3N state distance between drone,
                dist_drone = np.ones((3*(self.NUM_DRONES - 1)))
                low = np.array([0,0,0,0, -1,-1,-1])
                high = np.array([1,1,1,1,  1,1,1])

            low = np.hstack([low, -1 * dist_drone])
            high = np.hstack([high, dist_drone])
            return spaces.Dict({i: spaces.Box(low=low,
                                                high=high,
                                                dtype=np.float32
                                                ) for i in range(self.NUM_DRONES)})
        else:
            print("[ERROR] in PayloadCoop._observationSpace()")

    ################################################################################

    def _computeObs(self):
        if self.OBS_TYPE in [ObservationType.KIN, ObservationType.PAYLOAD, ObservationType.PAYLOAD_Z_CONST]:
            obs_all = np.zeros((self.NUM_DRONES, 19+3*(self.NUM_DRONES - 1)), dtype = np.float32)
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                obs_all[i, 0:3] = state[0:3] # pos
                obs_all[i, 3:6] = state[7:10] # rpy
                obs_all[i, 6:9] = state[10:13] # pos_dot
                obs_all[i, 9:12] = state[13:16] # rpy_dot
                obs_all[i, 12:16] = self._isObstacleNear(i) # obstacle sensor
                obs_all[i, 16:19] = self.DEST_POINT - state[0:3] # distance to DEST_POINT
                obs_all[i, 19:] = self._getDistBetweenAllDrone(i) # distance between drone
                obs_all[i, :] = self._clipAndNormalizeState(obs_all[i, :])             
        
            if self.OBS_TYPE == ObservationType.KIN:
                return {i: obs_all[i, :] for i in range(self.NUM_DRONES)}

            elif self.OBS_TYPE == ObservationType.PAYLOAD_Z_CONST:
                mask = np.arange(3*(self.NUM_DRONES - 1))
                mask = (mask+1) % 3 != 0 # remove dist_betw_drone z state index
                mask = np.hstack([[False]*12, [True]*6, False, mask]) # 12 drone state, 4 obst + 3 dist2dest                
                return {i: obs_all[i, mask] for i in range(self.NUM_DRONES)}

            elif self.OBS_TYPE == ObservationType.PAYLOAD:
                return {i: obs_all[i, 12:] for i in range(self.NUM_DRONES)}
        else:
            print("[ERROR] in PayloadCoop._computeObs()")
            
    def _computeReward(self):
        drone_ids = self.getDroneIds()
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        reward = {}
        rwd_hit = -1e4
        rwd_toofar_drone = -1e3
        rwd_arrive = 1e4
        rwd_dest = -0.1 / self.SIM_FREQ * self.AGGR_PHY_STEPS
        rwd_time = -0/ self.SIM_FREQ * self.AGGR_PHY_STEPS # got -1 every second
        rwd_rpm = -0 / self.SIM_FREQ * self.AGGR_PHY_STEPS
        rwd_dist_z = -1 / self.SIM_FREQ * self.AGGR_PHY_STEPS

        # Approaching dest
        reward[0] = -1 * np.linalg.norm(states[0, 0:3] - self.DEST_POINT)**2

        if(self._isHitEverything(drone_ids)):
            reward[0] += rwd_hit

        if(self._isDroneTooFar(drone_ids)):
            reward[0] += rwd_toofar_drone   

        if(self._isArrive(drone_ids)):
            reward[0] += rwd_arrive
        
        # Time reward
        reward[0] += rwd_time
        
        # Keep Z constant
        reward[0] += -1 * np.linalg.norm(states[0,2] - self.Z_CONST)

        # Energy usage
        RPM_eq = ((self.M * self.G) / (4 * self.KF))**0.5 
  
        # rewards[i] += rwd_rpm * np.linalg.norm(drone_states[16:20] - RPM_eq) / 4
        reward[0] += rwd_rpm * np.sum(states[0,16:20]**2) / 4

        for i in range(1, self.NUM_DRONES):
            reward[i] = -1 * np.linalg.norm(states[i-1, 0:3] - states[i, 0:3])**2

        return reward

    ################################################################################
    
    def _computeDone(self):
        drone_ids = self.getDroneIds()
        bool_val = self._isArrive(drone_ids) \
            or self._isDroneTooFar(drone_ids) \
            or self._isHitEverything(drone_ids)
        bool_val = bool_val or (self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC)
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        done["__all__"] = True if True in done.values() else False
        return done

    ################################################################################
    
    def _computeInfo(self):
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    def reset(self):
        temp = super().reset()
        self._resetDestPoint()
        
        pos = self._initPositionOnCircle(self.NUM_DRONES, self.MAX_DISTANCE_BETWEEN_DRONE/3)
        for i in range(self.NUM_DRONES) :
            p.resetBasePositionAndOrientation(self.DRONE_IDS[i],
                                            pos[i, :],
                                            p.getBasePositionAndOrientation(self.DRONE_IDS[i])[1],
                                            physicsClientId=self.CLIENT
                                            )
        self._addObstaclesAll()
        return temp

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        K_MOVE = self.K_MOVE  
        rpm = np.zeros((self.NUM_DRONES,4))
        rpm = np.zeros((self.NUM_DRONES,4))
        for k, v in action.items():
            if self.ACT_TYPE == ActionType.JOYSTICK:
                state = self._getDroneStateVector(int(k))
                if v == 0: # Arah X pos
                    target_pos = state[0:3] + K_MOVE * np.array([1, 0, 0])
                elif v == 1: # Arah Y pos
                    target_pos = state[0:3] + K_MOVE * np.array([0, 1, 0])
                elif v == 2: # Arah X neg
                    target_pos = state[0:3] + K_MOVE * np.array([-1, 0, 0])
                elif v == 3: # Arah Y neg
                    target_pos = state[0:3] + K_MOVE * np.array([0, -1, 0])
                elif v == 4: # Diam
                    target_pos = state[0:3]
                else:
                    target_pos = state[0:3]
                    print("Aksi tidak diketahui, drone ke-{} akan diam\n".format(k))
                rpm_k, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=target_pos
                                                        )
                rpm[int(k),:] = rpm_k
            elif self.ACT_TYPE == ActionType.XY_YAW:
                state = self._getDroneStateVector(int(k))
                target_pos = state[0:3] + K_MOVE * np.hstack([v[0:2],0])
                target_rpy = state[7:10] +  2*np.pi*  np.hstack([0,0,v[2]])
                rpm_k, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=target_pos,
                                                        target_rpy=target_rpy
                                                        )
                rpm[int(k),:] = rpm_k
            elif self.ACT_TYPE == ActionType.XYZ_YAW:
                state = self._getDroneStateVector(int(k))
                target_pos = state[0:3] + K_MOVE* v[0:3]
                target_rpy = state[7:10] + 2*np.pi* np.hstack([0,0,v[3]])
                rpm_k, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=target_pos,
                                                        target_rpy=target_rpy
                                                        )
                rpm[int(k),:] = rpm_k
            else:
                print("[ERROR] in BaseMultiagentAviary._preprocessAction()")
                exit()

            # self.TARGET_HISTORY[int(k), self.step_counter, :] = target_pos
            # self.POSITION_HISTORY[int(k), self.step_counter, :] = state[0:3] 
        return rpm

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        COEF_TOL = 1.3
        MAX_LIN_VEL_XY = self.MAX_SPEED_KMH /3.6 * COEF_TOL
        MAX_LIN_VEL_Z = self.MAX_SPEED_KMH /3.6 * COEF_TOL
        MAX_ANG_VEL = self.MAX_SPEED_KMH /3.6 * COEF_TOL * 0.9 # dikali jari jari
        MAX_XY = self.MAX_XY * COEF_TOL
        MAX_Z = self.MAX_Z * COEF_TOL
        MAX_DIST_GOAL = np.sqrt(2*MAX_XY**2) * COEF_TOL
        MAX_DISTANCE_BETWEEN_DRONE = self.MAX_DISTANCE_BETWEEN_DRONE * COEF_TOL

        # MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        # MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[3:5], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[6:8], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[8], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        clipped_ang_vel = np.clip(state[9:12], -MAX_ANG_VEL, MAX_ANG_VEL)
        clipped_dist_goal = np.clip(state[16:19], -MAX_DIST_GOAL, MAX_DIST_GOAL)
        clipped_dist_drone = np.clip(state[19:], -MAX_DISTANCE_BETWEEN_DRONE, MAX_DISTANCE_BETWEEN_DRONE)
        

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )
        state[0:2] = clipped_pos_xy / MAX_XY
        state[2] = clipped_pos_z / MAX_Z
        state[3:5] = clipped_rp / MAX_PITCH_ROLL
        state[5] = state[5] / np.pi # No reason to clip
        state[6:8] = clipped_vel_xy / MAX_LIN_VEL_XY
        state[8] = clipped_vel_z / MAX_LIN_VEL_XY
        state[9:12] = clipped_ang_vel / MAX_ANG_VEL
        state[16:19] = clipped_dist_goal / MAX_DIST_GOAL
        state[19:]= clipped_dist_drone / MAX_DISTANCE_BETWEEN_DRONE                                       
        return state
        
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        pass

    def _getDistBetweenAllDrone(self, drone_id):
        obst_dist = np.zeros((3*(self.NUM_DRONES - 1)), dtype = np.float32)
        state_drone_main = self._getDroneStateVector(drone_id)
        j = 0
        for i in range(self.NUM_DRONES):
            if(i != drone_id):
                state_drone_sec = self._getDroneStateVector(i)
                obst_dist[3*j: 3*j+3] = state_drone_sec[0:3] - state_drone_main[0:3]
                j += 1
        return obst_dist

    ################################################################################

    def _isHitEverything(self, drone_ids):
        for i in range(len(drone_ids)):
            a, b = p.getAABB(drone_ids[i], physicsClientId = self.CLIENT) # Melihat batas posisi collision drone ke i
            list_obj = p.getOverlappingObjects(a, b, physicsClientId = self.CLIENT) # Melihat objek2 yang ada di batas posisi collision
            if(list_obj != None and len(list_obj) > 6): # 1 Quadcopter memiliki 6 link/bagian
                # print("Drone {}: _isHitEverything".format(i))
                return True
        return False

    ################################################################################

    def _isDroneTooFar(self, drone_ids, max_dist = None):
        # Looping untuk setiap pair of drone
        if(max_dist == None):
            max_dist = self.MAX_DISTANCE_BETWEEN_DRONE
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):
                dr1_states = self._getDroneStateVector(i)
                dr2_states = self._getDroneStateVector(j)
                dist = np.linalg.norm(dr1_states[0:3] - dr2_states[0:3])
                if (dist > max_dist):
                    # print("Drone {}-{}: _isDroneTooFar".format(i, j))
                    return True
        return False

    def _getCentroid(self, drone_ids):
        centroid = np.array([0, 0, 0], dtype=np.float32)
        for i in range(len(drone_ids)):
            dr_states = self._getDroneStateVector(i)
            centroid += dr_states[0:3]
        centroid /= len(drone_ids)
        return centroid

    def _isArrive(self, drone_ids, tol = 0.01, dest = None):
        # Menghitung centroid of points dari kumpulan drone
        if(dest == None):
            dest = self.DEST_POINT
        centroid = self._getCentroid(drone_ids)
        if(np.linalg.norm(centroid - dest) < tol):
            # print("Drone ALL: _isArrive")
            return True
        else:
            return False

    ################################################################################

    def _isObstacleNear(self, drone_id, max_sensor_dist = 2, max_sensor_angle = 10): # drone_id (ordinal)
        if(self.ACT_TYPE in [ActionType.JOYSTICK]):
            obstacle_state = np.zeros(4)
            drone_state = self._getDroneStateVector(drone_id)
            for obst_id in list(self.DRONE_IDS[:drone_id]) + list(self.DRONE_IDS[drone_id+1:]) + self.OBSTACLE_IDS:  # sensor proximity read drone
                list_cp = p.getClosestPoints(self.DRONE_IDS[drone_id], obst_id, max_sensor_dist, physicsClientId = self.CLIENT)            
                if(len(list_cp) != 0): # there is obstacle near drone
                    for cp in list_cp:
                        x_dr, y_dr, z_dr = drone_state[0:3]
                        x_ob, y_ob, z_ob = cp[6]
                        theta = np.arctan2(y_ob - y_dr, x_ob - x_dr) * 180 / np.pi
                        eps = max_sensor_angle
                        if(-eps < theta < eps): #x+
                            obstacle_state[0] = 1
                        elif((90-eps) < theta < (90+eps)): #y+
                            obstacle_state[1] = 1
                        elif(-180 < theta <= (-180+eps) or (180-eps) < theta <= 180): #x-
                            obstacle_state[2] = 1
                        elif((-90-eps) < theta <(-90+eps)): #y-
                            obstacle_state[3] = 1

        elif(self.ACT_TYPE in [ActionType.XY_YAW, ActionType.XYZ_YAW]):
            obstacle_state = np.ones(4)
            drone_state = self._getDroneStateVector(drone_id)
            for obst_id in list(self.DRONE_IDS[:drone_id]) + list(self.DRONE_IDS[drone_id+1:]) + self.OBSTACLE_IDS:  # sensor proximity read drone
                list_cp = p.getClosestPoints(self.DRONE_IDS[drone_id], obst_id, max_sensor_dist, physicsClientId = self.CLIENT)
                if(len(list_cp) != 0): # there is obstacle near drone
                    for cp in list_cp:
                        x_dr, y_dr, z_dr = drone_state[0:3]
                        x_ob, y_ob, z_ob = cp[6]
                        dist = np.linalg.norm([x_ob - x_dr, y_ob - y_dr])
                        theta = np.arctan2(y_ob - y_dr, x_ob - x_dr) * 180 / np.pi
                        eps = max_sensor_angle
                        yaw = drone_state[9] * 180 / np.pi
                        if(-eps < theta - yaw < eps): 
                            obstacle_state[0] = np.clip(dist, 0, max_sensor_dist) / max_sensor_dist
                        elif((90-eps) < theta - yaw < (90+eps)): 
                            obstacle_state[1] = np.clip(dist, 0, max_sensor_dist) / max_sensor_dist
                        elif(-180 < theta - yaw <= (-180+eps) or (180-eps) < theta - yaw <= 180): 
                            obstacle_state[2] = np.clip(dist, 0, max_sensor_dist) / max_sensor_dist
                        elif((-90-eps) < theta - yaw <(-90+eps)):
                            obstacle_state[3] = np.clip(dist, 0, max_sensor_dist) / max_sensor_dist
        return obstacle_state


        ################################################################################

    def _addObstaclesAt(self, position, orientation = [0, 0, 0], name = "cube_no_rotation.urdf"):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        id_ = p.loadURDF(name,
                    position,
                    p.getQuaternionFromEuler(orientation),
                    physicsClientId=self.CLIENT
                    )
        self.OBSTACLE_IDS.append(id_) 

    def _addObstaclesAll(self):
        min_obst_dist_created = 1.5
        p_obst = np.array(self.DEST_POINT) / np.random.uniform(1.1, np.linalg.norm(self.DEST_POINT[0:2])/min_obst_dist_created)
        or_obst = [0, 0, np.random.uniform(0, 2*np.pi)]
        self._addObstaclesAt(p_obst, or_obst, "cube_no_rotation.urdf")
        # self._addObstaclesAt(p_obst, or_obst, "cube_custom.urdf")
        # self._addObstaclesAt(self.dest_point, name = 'duck_vhacd.urdf')

    def _resetDestPoint(self):
        # r = np.random.uniform(0.5, 1.5) * np.linalg.norm(self.DEST_POINT[0:2])
        r = np.linalg.norm(self.DEST_POINT[0:2])
        t = np.random.uniform(0, 2*np.pi)
        self.DEST_POINT = [r * np.cos(t), r * np.sin(t), self.Z_CONST]


    def _initPositionOnCircle(self, n_drone, r = None, z = None, random = True):
        
        if(r == None):
            r = self.MAX_DISTANCE_BETWEEN_DRONE / 4
        if(z == None):
            z = self.Z_CONST
        ps = np.zeros((n_drone, 3))
        t0 = np.random.uniform(0, 2*np.pi)
        # t0 = 0
        for i in range(n_drone):
            x = r * np.cos((i*2*np.pi+t0)/n_drone)
            y = r * np.sin((i*2*np.pi+t0)/n_drone)
            ps[i, :] = x, y, z
        return ps