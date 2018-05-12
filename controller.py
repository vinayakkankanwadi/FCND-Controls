"""
PID Controller

components:
    follow attitude commands
    gps commands and yaw
    waypoint following
"""
import numpy as np
from frame_utils import euler2RM
import math


DRONE_MASS_KG = 0.5
GRAVITY = -9.81
MOI = np.array([0.005, 0.005, 0.01])
MAX_THRUST = 10.0
MAX_TORQUE = 1.0

class NonlinearController(object):

    def __init__(self):
        """Initialize the controller object and control gains"""

        # used
        self.k_p_p = 27.0
        self.k_p_q = 27.0
        self.k_p_r = 10.0
        self.z_k_p = 40.0
        self.z_k_d = 15.0
        self.x_k_p = 0.965
        self.x_k_d = 4.5
        self.y_k_p = 0.965
        self.y_k_d = 4.5
        self.k_p_roll = 4.0
        self.k_p_pitch = 4.0
        self.k_p_yaw = 5.0

        return

    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory

        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds

        Returns: tuple (commanded position, commanded velocity, commanded yaw)

        """

        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]


        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]

            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]

        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]

                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]

        position_cmd = (position1 - position0) * \
                        (current_time - time0) / (time1 - time0) + position0
        velocity_cmd = (position1 - position0) / (time1 - time0)


        return (position_cmd, velocity_cmd, yaw_cmd)

    def lateral_position_control(self, local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                               acceleration_ff = np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command

        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """

        # PD controller

        # extract x-north axis variables from parameters
        x_north_cmd = local_position_cmd[0]
        x_north = local_position[0]
        x_dot_north_cmd = local_velocity_cmd[0]
        x_dot_north = local_velocity[0]
        x_dot_dot_north_ff = acceleration_ff[0]

        # extract y-east axis variables from parameters
        y_east_cmd = local_position_cmd[1]
        y_east = local_position[1]
        y_dot_east_cmd = local_velocity_cmd[1]
        y_dot_east = local_velocity[1]
        y_dot_dot_east_ff = acceleration_ff[1]

        # PD controller formula to calculate desirednorth acceleration in local frame
        acc_north_cmd = self.x_k_p * (x_north_cmd - x_north) + self.x_k_d * (x_dot_north_cmd - x_dot_north) + x_dot_dot_north_ff

        # PD controller formula to calculate desired east acceleration in local frame
        acc_east_cmd = self.y_k_p * (y_east_cmd - y_east) + self.y_k_d * (y_dot_east_cmd - y_dot_east) + y_dot_dot_east_ff

        return np.array([acc_north_cmd, acc_east_cmd])

    def altitude_control(self, altitude_cmd, vertical_velocity_cmd, altitude, vertical_velocity, attitude, acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: the vehicle's current attitude, 3 element numpy array (roll, pitch, yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)

        Returns: thrust command for the vehicle (+up)
        """

        # Extract roll, pitch, yaw variables from attitude parameter
        [roll, pitch, yaw] = attitude

        # calculate rotation matrix from roll pitch yaw
        rot_mat = euler2RM(roll, pitch, yaw)

        # desired vertical acceleration in inertial frame is given by the following PD controller formula
        u_1_bar = self.z_k_p * (altitude_cmd - altitude) + self.z_k_d * (vertical_velocity_cmd - vertical_velocity) + acceleration_ff

        # calculate vertical acceleration in body frame, rot_mat[2][2] is z scaling factor in inertial frame
        c = (u_1_bar - 9.81) / rot_mat[2][2]

        # since thrust is in Newton, from F=ma we need to multiply by drone's mass
        thrust = DRONE_MASS_KG * c

        # ensure thrust stays below the MAX_THRUST limit
        thrust_limited = np.clip(thrust, 0.0, MAX_THRUST)

        return thrust_limited  # thrust command in Newton

    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame

        Args:
            target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll, pitch, yaw) in radians
            thrust_cmd: vehicle thruts command in Newton

        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """

        # Extract variable values from parameters
        [north_acceleration_cmd, east_acceleration_cmd] = acceleration_cmd
        [roll, pitch, yaw] = attitude

        # Avoid division by zero
        if thrust_cmd != 0:

            # Based on equation x_dot_dot = (1/m) * R[1,3] * (-F),
            # given
            # x_dot_dot = north_acceleration_cmd
            # F = thrust_cmd
            # m = DRONE_MASS_KG
            # R[1,3] = b_x_c, x scaling factor in inertial frame of the commanded acceleration (DRONE_MASS_KG/thrust_cmd) in body frame
            b_x_c = north_acceleration_cmd / -thrust_cmd * DRONE_MASS_KG

            # Same for b_y_c
            b_y_c = east_acceleration_cmd / -thrust_cmd * DRONE_MASS_KG
        else:
            b_x_c = 0
            b_y_c = 0

        # Rotation matrix To transform between body-frame accelerations and world frame accelerations
        rot_mat = euler2RM(roll, pitch, yaw)

        # x scaling factor in inertial frame from actual attitude
        b_x_a = rot_mat[0][2]

        # y scaling factor in inertial frame from actual attitude
        b_y_a = rot_mat[1][2]

        # rate of change of x scaling factor in inertial frame
        b_x_c_dot = self.k_p_roll * (b_x_c - b_x_a)

        # rate of change of y scaling factor in inertial frame
        b_y_c_dot = self.k_p_pitch * (b_y_c - b_y_a)

        # creation of rate of change vector from its x and y components
        b_dot_vector = np.array([b_x_c_dot, b_y_c_dot]).T

        # sub rotation matrix containing the first 4 cells
        sub_rot_mat = np.zeros([2, 2])
        sub_rot_mat[0][0] = rot_mat[1][0]
        sub_rot_mat[1][0] = rot_mat[1][1]

        sub_rot_mat[0][1] = -rot_mat[0][0]
        sub_rot_mat[1][1] = -rot_mat[0][1]

        b_rot = (1 / rot_mat[2][2]) * sub_rot_mat

        # Calculation of p_c and  q_c
        [p_c, q_c] = b_rot.dot(b_dot_vector)

        # making sure we are not going over the roll limits
        if roll > (math.pi / 4) and p_c > 0:
            p_c = -math.pi/3

        elif roll < (-math.pi / 4) and p_c < 0:
            p_c = math.pi/3

        # making sure we are not going over the pitch limits
        if pitch > (math.pi / 4) and q_c > 0:
            q_c = -math.pi/3

        elif pitch < (-math.pi / 4) and q_c < 0:
            q_c = math.pi/3

        return np.array([p_c, q_c])


    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame

        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            attitude: 3-element numpy array (p,q,r) in radians/second^2

        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        # a proportional controller on body rates to commanded moments
        # kg * m^2 * rad / sec^2 = Newtons*meters

        # Only proportional gain used, no derivative term
        gains = np.array([self.k_p_p, self.k_p_q, self.k_p_r])

        # Calculation is based on Moment = (Moment of Inertia) x (Angular Acceleration)
        # Angular acceleration  equals (Gain) x (body rate desired - observed)
        [tau_x_c, tau_y_c, tau_z_c] = MOI.T * (gains.T * (body_rate_cmd.T - body_rate.T))

        # Maximum commanded moments are limited to MAX_TORQUE
        tau_x_c_c = np.clip(tau_x_c, -MAX_TORQUE, MAX_TORQUE)
        tau_y_c_c = np.clip(tau_y_c, -MAX_TORQUE, MAX_TORQUE)
        tau_z_c_c = np.clip(tau_z_c, -MAX_TORQUE, MAX_TORQUE)

        return np.array([tau_x_c_c, tau_y_c_c, tau_z_c_c])

    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate

        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians

        Returns: target yawrate in radians/sec
        """
        # proportional controller

        # calculate yaw delta between commanded and observed yaw values
        yaw_delta = yaw_cmd - yaw

        # create new yaw delta variable to store temp values
        yaw_delta_2 = yaw_delta

        # Ensure you always pick the shortest angle towards commanded yaw from observed yaw
        if abs(yaw_delta) > (math.pi):
            if yaw_delta < 0:
                yaw_delta_2 = (2 * math.pi) + yaw_delta
            else:
                yaw_delta_2 = yaw_delta - (2 * math.pi)

        # Calculate yaw rate with the proportional yaw constant
        yawrate = self.k_p_yaw * (yaw_delta_2)

        return yawrate
