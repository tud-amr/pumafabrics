import os
import numpy as np
import pybullet
import warnings
from pumafabrics.tamed_puma.utils.filters import PDController
from pumafabrics.tamed_puma.utils.normalizations_2 import normalization_functions
from pumafabrics.tamed_puma.create_environment.environments import trial_environments
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from pumafabrics.tamed_puma.utils.analysis_utils import UtilsAnalysis
import importlib
from pumafabrics.puma_extension.initializer import initialize_framework
from pumafabrics.tamed_puma.tamedpuma.example_generic import ExampleGeneric
import copy
import yaml
from pumafabrics.tamed_puma.modulation_ik.Modulation_ik import IKGomp
import time
import random
"""
Example of KUKA iiwa 14 running ModulationIK as a controller with lots of obstacles.
"""
class example_kuka_modulation_IK_1000(ExampleGeneric):
    def __init__(self, file_name="kuka_ModulationIK_tomato"):
        super(ExampleGeneric, self).__init__()
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = float("nan")
        self.solver_times = []
        with open("../pumafabrics/tamed_puma/config/"+file_name+".yaml", "r") as setup_stream:
            self.params = yaml.safe_load(setup_stream)
        self.dof = self.params["dof"]
        self.robot_name = self.params["robot_name"]
        warnings.filterwarnings("ignore")

    def initialize_environment(self):
        envir_trial = trial_environments()
        self.params["nr_obst"]=2
        (self.env, self.goal) = envir_trial.initialize_environment_kuka(params=self.params)
        self.params["nr_obst"]=100

    def construct_fk(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/../pumafabrics/tamed_puma/config/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        self.forward_kinematics = GenericURDFFk(
            urdf,
            root_link=self.params["root_link"],
            end_links=self.params["end_links"],
        )

    def construct_example(self):
        self.initialize_environment()

        # Construct classes:
        results_base_directory = '../pumafabrics/puma_extension/'
        self.kuka_kinematics = KinematicsKuka(dt=self.params["dt"], end_link_name=self.params["end_links"][0])
        self.pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=self.params["dt"])
        self.construct_fk()
        self.utils_analysis = UtilsAnalysis(forward_kinematics=self.forward_kinematics,
                                            collision_links=self.params["collision_links"],
                                            collision_radii=self.params["collision_radii"],
                                            kinematics=self.kuka_kinematics)

        # Parameters
        if self.params["mode_NN"] == "1st":
            self.params_name = self.params["params_name_1st"]
        else:
            self.params_name = self.params["params_name_2nd"]

        # Load parameters
        Params = getattr(importlib.import_module('pumafabrics.puma_extension.params.' + self.params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        self.learner, _, data = initialize_framework(params, self.params_name, verbose=False)
        self.goal_NN = data['goals training'][0]

        # Initialize GOMP
        self.gomp_class  = IKGomp(q_home=self.params["init_pos"]) #q_home=q_init)
        self.gomp_class.construct_ik(nr_obst=self.params["nr_obst"])

        # Normalization class
        self.normalizations = normalization_functions(x_min=data["x min"], x_max=data["x max"], dof_task=self.params["dim_task"], dt=self.params["dt"], mode_NN=self.params["mode_NN"])


    def run_kuka_example(self):
        dof = self.params["dof"]
        dt = self.params["dt"]
        offset_orientation = np.array(self.params["orientation_goal"])

        action = np.zeros(dof)
        ob, *_ = self.env.step(action)

        q_init = ob['robot_0']["joint_state"]["position"][0:dof]
        if self.params["nr_obst"] > 0:
            self.obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
        else:
            self.obstacles = []

        # Translation of goal:
        goal_pos = self.params["goal_pos"]
        translation_gpu, translation_cpu = self.normalizations.translation_goal(state_goal = np.array(goal_pos), goal_NN=self.goal_NN)

        # initial state:
        x_t_init = self.gomp_class.get_initial_pose(q_init=q_init, offset_orientation=offset_orientation)
        x_init_gpu = self.normalizations.normalize_state_to_NN(x_t=[x_t_init], translation_cpu=translation_cpu, offset_orientation=offset_orientation)
        dynamical_system = self.learner.init_dynamical_system(initial_states=x_init_gpu[:, :3].clone(), delta_t=1)

        # Initialize lists
        xee_list = []
        qdot_diff_list = []
        quat_prev = copy.deepcopy(x_t_init[3:])

        for w in range(self.params["n_steps"]):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            if self.params["nr_obst"] > 0 and self.params["nr_obst"] < 20:
                positions_obstacles = [ob_robot["FullSensor"]["obstacles"][self.params["nr_obst"]]["position"], ob_robot["FullSensor"]["obstacles"][self.params["nr_obst"]+1]["position"]]
            else:
                positions_obstacles = [ob_robot["FullSensor"]["obstacles"][2]["position"]*(1+0.01*random.uniform(0, 1)) for _ in range(self.params["nr_obst"])]
                positions_obstacles[1] = ob_robot["FullSensor"]["obstacles"][3]["position"]
                positions_obstacles[0] = ob_robot["FullSensor"]["obstacles"][2]["position"]

            # recompute translation to goal pose:
            goal_pos = [goal_pos[i] + self.params["goal_vel"][i]*self.params["dt"] for i in range(len(goal_pos))]
            translation_gpu, translation_cpu = self.normalizations.translation_goal(state_goal=np.array(goal_pos), goal_NN=self.goal_NN)
            pybullet.addUserDebugPoints([goal_pos], [[1, 0, 0]], 5, 0.1)

            # --- end-effector states and normalized states --- #
            x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, quat_prev)
            x_t, xee_orientation = self.gomp_class.get_current_pose(q, quat_prev=quat_prev)

            quat_prev = copy.deepcopy(xee_orientation)
            vel_ee, Jac_current = self.kuka_kinematics.get_state_velocity(q=q, qdot=qdot)
            x_t_gpu = self.normalizations.normalize_state_to_NN(x_t=x_t, translation_cpu=translation_cpu, offset_orientation=offset_orientation)

            # --- action by NN --- #
            time0 = time.perf_counter()
            centers_obstacles = [[100, 100, 100] for _ in range(self.params["nr_obst"])]
            for i in range(self.params["nr_obst"]):
                centers_obstacles[i][0:3] = self.params["positions_obstacles"][0]
            obstacles_struct = {"centers": centers_obstacles,
                               "axes": [[0.3, 0.3, 0.3]], "safety_margins": [[1., 1., 1.]]}
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu[:, :3].clone(), obstacles=obstacles_struct)
            x_t_NN = transition_info["desired state"]
            if self.params["mode_NN"] == "2nd":
                print("not implemented!!")
            else:
                action_t_gpu = transition_info["desired velocity"]
                action_cpu = action_t_gpu.T.cpu().detach().numpy()
            x_t_action = self.normalizations.reverse_transformation_position(position_gpu=x_t_NN) #, offset_orientation=offset_orientation)
            pybullet.addUserDebugPoints(list(x_t_action), [[1, 0, 0]], 5, 0.1)
            action_safeMP = self.normalizations.reverse_transformation(action_gpu=action_t_gpu)
            q_d, solver_flag = self.gomp_class.call_ik(x_t_action[0][0:3], self.params["orientation_goal"],
                                                  positions_obsts=positions_obstacles,
                                                  q_init_guess=q,
                                                  q_home=q)
            # if solver_flag == False:
            #     q_d = q
            xee_IK, _ = self.gomp_class.get_current_pose(q=q_d, quat_prev=quat_prev)
            # print("solver_flag:", solver_flag)
            action = self.pdcontroller.control(desired_velocity=q_d, current_velocity=q)
            self.solver_times.append(time.perf_counter() - time0)
            action = np.clip(action, -1*np.array(self.params["vel_limits"]), np.array(self.params["vel_limits"]))

            self.check_goal_reached(x_ee=x_t[0][0:3], x_goal=goal_pos)
            ob, *_ = self.env.step(action)

            # result analysis:
            x_ee, _ = self.utils_analysis._request_ee_state(q, quat_prev)
            xee_list.append(x_ee[0])
            qdot_diff_list.append(1)
            self.IN_COLLISION = self.utils_analysis.check_distance_collision(q=q, obstacles=self.obstacles)
            self.GOAL_REACHED, error = self.utils_analysis.check_goal_reaching(q, quat_prev, x_goal=goal_pos)

            if self.GOAL_REACHED:
                self.time_to_goal = w*self.params["dt"]
                break

            if self.IN_COLLISION:
                self.time_to_goal = float("nan")
                break

        self.env.close()

        results = {
            "min_distance": self.utils_analysis.get_min_dist(),
            "collision": self.IN_COLLISION,
            "goal_reached": self.GOAL_REACHED,
            "time_to_goal": self.time_to_goal,
            "xee_list": xee_list,
            "qdot_diff_list": qdot_diff_list,
            "solver_times": np.array(self.solver_times)*1000,
            "solver_time": np.mean(self.solver_times),
            "solver_time_std": np.std(self.solver_times),
        }
        return results

def main(render=True, n_steps=None):
    q_init_list = [
        np.array((0.531, 1.36, 0.070, -1.065, 0.294, -1.2, -0.242)),
    ]
    positions_obstacles_list = [
        [[0.5, 0.15, 0.05], [0.5, 0.15, 0.2]],
    ]

    example_class = example_kuka_modulation_IK_1000()
    example_class.overwrite_defaults(params=example_class.params, init_pos=q_init_list[0], positions_obstacles=positions_obstacles_list[0],
                                     render=render, n_steps=n_steps)
    example_class.construct_example()
    res = example_class.run_kuka_example()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])
    return {}

if __name__ == "__main__":
    main()