import os
import gymnasium as gym
import numpy as np
import copy
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition

from functions_stableMP_fabrics.parametrized_planner_extended import ParameterizedFabricPlannerExtended
from tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework
from agent.utils.dynamical_system_operations import normalize_state
from agent.utils.normalizations import normalizaton_sim_NN
from functions_stableMP_fabrics.plotting_functions import plotting_functions
from functions_stableMP_fabrics.combining_actions import combine_fabrics_safeMP
from functions_stableMP_fabrics.environments import trial_environments

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.
# #
# # todo: tune behavior.

class example_point_robot_safeMP_fabrics():
    def __init__(self):
        dt = 0.01
    def set_planner(self, goal: GoalComposition, ONLY_GOAL=False, bool_speed_control=True, mode="acc", dt=0.01):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        Params
        ----------
        goal: StaticSubGoal
            The goal to the motion planning problem.
        """
        degrees_of_freedom = 2
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/examples/urdfs/point_robot.urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        forward_kinematics = GenericURDFFk(
            urdf,
            root_link="world",
            end_links=["base_link_y"],
        )
        collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
        collision_finsler = "2.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
        planner = ParameterizedFabricPlannerExtended(
            degrees_of_freedom,
            forward_kinematics,
            time_step=dt,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler
            )
        collision_links = ["base_link_y"]
        # The planner hides all the logic behind the function set_components.
        if ONLY_GOAL == 1:
            planner.set_components(
                goal=goal,
            )
        else:
            planner.set_components(
                collision_links=collision_links,
                goal=goal,
                number_obstacles=2,
            )
        # planner.concretize(extensive_concretize=True, bool_speed_control=bool_speed_control)
        planner.concretize_extensive(mode=mode, time_step=dt, extensive_concretize=True, bool_speed_control=bool_speed_control)
        return planner

    def simple_plot(self, list_fabrics_goal, list_fabrics_avoidance, list_safeMP, list_diff, dt=0.01, results_path=""):
        time_x = np.arange(0.0, len(list_fabrics_goal)*dt, dt)
        fig, ax = plt.subplots(1, 3)
        ax[0].plot(time_x, list_fabrics_goal)
        ax[0].plot(time_x, list_safeMP, '--')
        ax[1].plot(time_x, list_fabrics_avoidance, '-')
        # ax[1].plot(time_x, list_fabrics_goal, "--")
        ax[2].plot(time_x, list_diff, '-')
        ax[2].plot()
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[0].set(xlabel="time (s)", ylabel="x [m]", title="Actions fabrics, safeMP")
        ax[1].set(xlabel="time (s)", ylabel="x [m]", title="Action fabr avoidance")
        ax[2].set(xlabel="time (s)", ylabel="x [m]", title="Action difference")
        ax[0].legend(["x", "y", "$x_{IL}$", "$y_{IL}$"])
        ax[1].legend(["x", "y"])
        ax[2].legend(["x", "y"])
        plt.savefig(results_path+"images/difference_fabrics_safeMP.png")

    def run_point_robot_urdf(self, n_steps=2000, env=None, goal=None, init_pos=np.array([-5, 5]), goal_pos=[-2.4355761, -7.5252747], mode="acc", mode_NN = "2nd", dt=0.01):
        """
        Set the gym environment, the planner and run point robot example.
        The initial zero action step is needed to initialize the sensor in the
        urdf environment.

        Params
        ----------
        n_steps
            Total number of simulation steps.
        render
            Boolean toggle to set rendering on (True) or off (False).
        """
        # --- parameters --- #
        dof = 2
        # mode = "vel"
        # dt = 0.01
        # init_pos = np.array([-0.9, -0.1, 0.0])
        # goal_pos = [3.5, 0.5]
        # scaling_factor = 10
        #scaling_room = {"x": [-scaling_factor, scaling_factor], "y":[-scaling_factor, scaling_factor]}
        # init_pos = np.array([-0.5, 0.5])
        # goal_pos = [-0.24355761, -0.75252747]
        # scaling_factor = 1
        # scaling_room = {"x": [-1, 1], "y":[-1, 1]}
        # init_pos = np.array([-5, 5])
        # goal_pos = [-2.4355761, -7.5252747]
        scaling_factor = 10
        scaling_room = {"x": [-10, 10], "y":[-10, 10]}
        if mode == "vel":
            str_mode = "velocity"
        elif mode == "acc":
            str_mode = "acceleration"
        else:
            print("this control mode is not defined")

        # (env, goal) = initalize_environment(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
        planner = self.set_planner(goal, bool_speed_control=True, mode=mode, dt=dt)
        planner_goal = self.set_planner(goal=goal, ONLY_GOAL=True, bool_speed_control=True, mode=mode, dt=dt)
        planner_avoidance = self.set_planner(goal=None, bool_speed_control=True, mode=mode, dt=dt)

        # create class for combined functions on fabrics + safeMP combination
        v_min = -50*np.ones((dof,))
        v_max = 50*np.ones((dof,))
        acc_min = -50*np.ones((dof,))
        acc_max = 50*np.ones((dof,))
        combined_geometry = combine_fabrics_safeMP(v_min = v_min, v_max=v_max, acc_min=acc_min, acc_max=acc_max)

        action_safeMP = np.array([0.0, 0.0, 0.0])
        action_fabrics = np.array([0.0, 0.0, 0.0])
        ob, *_ = env.step(action_safeMP)
        q_list = np.zeros((2, n_steps))

        # Parameters
        params_name = '2nd_order_2D'
        x_t_init = np.array([np.append(ob['robot_0']["joint_state"]["position"][0:2], ob['robot_0']["joint_state"]["velocity"][0:2])]) # initial states
        # simulation_length = 2000
        results_base_directory = './'

        # Load parameters
        Params = getattr(importlib.import_module('params.' + params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        learner, _, data = initialize_framework(params, params_name, verbose=False)
        goal_NN = data['goals training'][0]

        # Translation of goal:
        normalizations = normalizaton_sim_NN(scaling_room=scaling_room)
        goal_normalized = np.array((goal._sub_goals[0]._config["desired_position"]))/scaling_factor
        translation = normalizations.get_translation(goal_pos=goal_normalized, goal_pos_NN=goal_NN)
        translation_gpu = torch.FloatTensor(translation).cuda()

        # Initialize dynamical system
        min_vel = learner.min_vel
        max_vel = learner.max_vel
        x_init_gpu, x_init_cpu = normalizations.transformation_to_NN(x_t=x_t_init, translation_gpu=translation_gpu,
                                                      dt=dt, min_vel=min_vel, max_vel=max_vel)
        dynamical_system = learner.init_dynamical_system(initial_states=x_init_gpu, delta_t=1)
        # dynamical_system.saturate

        # Initialize trajectory plotter
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        fig.show()
        trajectory_plotter = TrajectoryPlotter(fig, x0=x_init_cpu, pause_time=1e-5, goal=data['goals training'][0])
        # x_t_NN = torch.FloatTensor(x_t_init_scaled).cuda()

        # Initialize lists
        list_diff = []
        list_fabr_goal = []
        list_fabr_avoidance = []
        list_safeMP = []

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:2]
            qdot = ob_robot["joint_state"]["velocity"][0:2]
            q_list[:, w] = q
            x_t = np.array([np.append(q, qdot)])

            # --- translate to axis system of NN ---#
            x_t_gpu, _ = normalizations.transformation_to_NN(x_t=x_t, translation_gpu=translation_gpu,
                                           dt=dt, min_vel=dynamical_system.min_vel, max_vel=dynamical_system.max_vel)

            # --- get action by NN --- #
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
            x_t_NN = transition_info["desired state"]
            if mode == "acc":
                action_t_gpu = transition_info["desired acceleration"]
                xddot_t_NN  = transition_info["desired acceleration"]
            else:
                action_t_gpu = transition_info["desired "+str_mode]
                xddot_t_NN = transition_info["desired acceleration"]

            action_safeMP = normalizations.reverse_transformation(action_gpu=action_t_gpu,  dt=dt, mode_NN=mode_NN)
            xddot_safeMP = normalizations.reverse_transformation(action_gpu=xddot_t_NN, dt=dt, mode_NN="2nd")

            # --- get action by fabrics --- #
            arguments_dict = dict(
                q=ob_robot["joint_state"]["position"][0:dof],
                qdot=ob_robot["joint_state"]["velocity"][0:dof],
                x_goal_0=ob_robot['FullSensor']['goals'][2]['position'][0:dof],
                weight_goal_0=ob_robot['FullSensor']['goals'][2]['weight'],
                x_obst_0=ob_robot['FullSensor']['obstacles'][3]['position'],
                radius_obst_0=ob_robot['FullSensor']['obstacles'][3]['size'],
                x_obst_1=ob_robot['FullSensor']['obstacles'][4]['position'],
                radius_obst_1=ob_robot['FullSensor']['obstacles'][4]['size'],
                # x_obst_2=ob_robot['FullSensor']['obstacles'][5]['position'],
                # radius_obst_2=ob_robot['FullSensor']['obstacles'][5]['size'],
                radius_body_base_link_y=np.array([0.2])
            )
            action_fabrics[0:dof] = planner.compute_action(**arguments_dict)
            M, f, action_forced, xddot_speed = planner.compute_M_f_action_avoidance(**arguments_dict)
            M_avoidance, f_avoidance, action_avoidance, xddot_speed_avoidance = planner_avoidance.compute_M_f_action_avoidance(**arguments_dict)
            M_attractor, f_attractor, action_attractor, xddot_speed_attractor = planner_goal.compute_M_f_action_attractor(**arguments_dict)

            action_combined = combined_geometry.combine_action(M_avoidance, M_attractor, f_avoidance, f_attractor, xddot_speed, planner,
                                             qdot=ob_robot["joint_state"]["velocity"][0:dof])
            xddot_speed = np.zeros((dof,)) #todo: think about what to do with speed regulation term!!
            M_safeMP = np.identity(dof,)
            action_fabrics_safeMP = combined_geometry.combine_action(M_avoidance, M_safeMP, f_avoidance, -xddot_safeMP[0:dof], xddot_speed, planner,
                                                   qdot=ob_robot["joint_state"]["velocity"][0:dof])

            # --- update environment ---#
            ob, *_, = env.step(np.append(action_fabrics_safeMP, 0))

            # --- Update plot ---#
            trajectory_plotter.update(x_t_gpu.T.cpu().detach().numpy())

            # --- Plot actions ---#
            list_diff.append(action_safeMP[0:dof] - action_attractor)
            list_fabr_goal.append(action_attractor)
            list_fabr_avoidance.append(copy.deepcopy(action_avoidance))
            list_safeMP.append(copy.deepcopy(action_safeMP[0:dof]))
        plt.savefig(params.results_path+"images/point_robot_safeMP_fabrics")
        env.close()
        self.simple_plot(list_fabrics_goal=list_fabr_goal, list_fabrics_avoidance=list_fabr_avoidance,
                    list_safeMP=list_safeMP, list_diff = list_diff, dt=dt, results_path=params.results_path)
        make_plots = plotting_functions(results_path=params.results_path)
        make_plots.plotting_q_values(q_list, dt=dt, q_start=q_list[:, 0], q_goal=np.array(goal_pos))
        return q_list

if __name__ == "__main__":
    # --- Initial parameters --- #
    mode = "acc"
    mode_NN = "2nd"
    dt = 0.01
    init_pos = np.array([0.0, 0.0])
    goal_pos = [-2.4355761, -7.5252747]
    render = True

    # --- generate environment --- #
    envir_trial = trial_environments()
    (env, goal) = envir_trial.initalize_environment_pointmass(render, mode=mode, dt=dt, init_pos=init_pos,
                                                              goal_pos=goal_pos)

    # --- run example --- #
    example_class = example_point_robot_safeMP_fabrics()
    res = example_class.run_point_robot_urdf(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos,
                               dt=dt, mode=mode, mode_NN=mode_NN)