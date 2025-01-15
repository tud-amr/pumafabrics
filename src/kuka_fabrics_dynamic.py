import os
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from pumafabrics.tamed_puma.tamedpuma.parametrized_planner_extended import ParameterizedFabricPlannerExtended
from pumafabrics.tamed_puma.create_environment.environments import trial_environments
from pumafabrics.tamed_puma.utils.analysis_utils import UtilsAnalysis
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka
import pickle
import yaml
import time

class example_kuka_fabrics():
    def __init__(self, nr_obst_dyn=1):
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = -1
        self.obstacles = []
        self.solver_times = []
        with open("../pumafabrics/tamed_puma/config/kuka_fabrics.yaml", "r") as setup_stream:
            self.params = yaml.safe_load(setup_stream)
        self.dof = self.params["dof"]
        self.robot_name = self.params["robot_name"]
        self.params["nr_obst_dyn"] = nr_obst_dyn

    def overwrite_defaults(self, render=None, init_pos=None, goal_pos=None, nr_obst=None, bool_energy_regulator=None, positions_obstacles=None, orientation_goal=None, params_name_1st=None):
        if render is not None:
            self.params["render"] = render
        if init_pos is not None:
            self.params["init_pos"] = init_pos
        if goal_pos is not None:
            self.params["goal_pos"] = goal_pos
        if orientation_goal is not None:
            self.params["orientation_goal"] = orientation_goal
        if nr_obst is not None:
            self.params["nr_obst"] = nr_obst
        if bool_energy_regulator is not None:
            self.params["bool_energy_regulator"] = bool_energy_regulator
        if positions_obstacles is not None:
            self.params["positions_obstacles"] = positions_obstacles
        if params_name_1st is not None:
            self.params["params_name_1st"] = params_name_1st

    def initialize_environment(self):
        envir_trial = trial_environments()
        (self.env, self.goal) = envir_trial.initialize_environment_kuka(params=self.params)

    def check_goal_reached(self, x_ee, x_goal):
        dist = np.linalg.norm(x_ee - x_goal)
        if dist<0.02:
            self.GOAL_REACHED = True
            return True
        else:
            return False

    def set_planner(self):
        """
        Initializes the fabric planner for the panda robot.
        """
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/examples/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        self.forward_kinematics = GenericURDFFk(
            urdf,
            root_link=self.params["root_link"],
            end_links=self.params["end_links"],
        )
        planner = ParameterizedFabricPlannerExtended(
            self.params["dof"],
            self.forward_kinematics,
            time_step=self.params["dt"],
        )
        planner.set_components(
            collision_links=self.params["collision_links"],
            goal=self.goal,
            number_obstacles=self.params["nr_obst"],
            number_dynamic_obstacles=self.params["nr_obst_dyn"],
            number_plane_constraints=0,
            limits=self.params["iiwa_limits"],
        )
        planner.concretize_extensive(mode=self.params["mode"], time_step=self.params["dt"], extensive_concretize=False, bool_speed_control=self.params["bool_speed_control"])
        return planner, self.forward_kinematics

    def compute_action_fabrics(self, q, ob_robot, obstacles: list, nr_obst=0):
        time0 = time.perf_counter()
        nr_obst_tot = self.params["nr_obst"]+self.params["nr_obst_dyn"]
        obstacles_static = obstacles[0:self.params["nr_obst"]]
        obstacles_dynamic = obstacles[self.params["nr_obst"]:]
        arguments_dict = dict(
            q=q,
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0 = ob_robot['FullSensor']['goals'][2+nr_obst_tot ]['position'],
            weight_goal_0 = ob_robot['FullSensor']['goals'][2+nr_obst_tot ]['weight'],
            x_goal_1 = ob_robot['FullSensor']['goals'][3+nr_obst_tot ]['position'],
            weight_goal_1 = ob_robot['FullSensor']['goals'][3+nr_obst_tot ]['weight'],
            x_goal_2=ob_robot['FullSensor']['goals'][4 + nr_obst_tot ]['position'],
            weight_goal_2=ob_robot['FullSensor']['goals'][4 + nr_obst_tot ]['weight'],
            x_obsts = [obstacles[i]["position"] for i in range(len(obstacles_static))],
            radius_obsts=[obstacles[i]["size"] for i in range(len(obstacles_static))],
            x_obsts_dynamic=[obstacles[i]["position"] for i in range(len(obstacles_dynamic))],
            xdot_obsts_dynamic=[np.zeros(3) for i in range(len(obstacles_dynamic))],
            xddot_obsts_dynamic=[np.zeros(3) for i in range(len(obstacles_dynamic))],
            radius_obsts_dynamic=[obstacles[i]["size"] for i in range(len(obstacles_dynamic))],
            radius_body_links=self.params["collision_radii"],
        )

        action = self.planner.compute_action(
            **arguments_dict)
        self.solver_times.append(time.perf_counter() - time0)
        return action, [], [], []

    def run_kuka_example(self):
        # --- parameters --- #
        offset_orientation = np.array(self.params["orientation_goal"])
        goal_pos = self.params["goal_pos"]

        self.initialize_environment()
        action = np.zeros(self.dof)
        ob, *_ = self.env.step(action)

        self.planner, fk = self.set_planner()
        utils_analysis = UtilsAnalysis(forward_kinematics=self.forward_kinematics, collision_links=self.params["collision_links"], collision_radii=self.params["collision_radii"])
        kuka_kinematics = KinematicsKuka()
        x_t_init = kuka_kinematics.get_initial_state_task(q_init=ob["robot_0"]["joint_state"]["position"][0:self.dof],
                                                          offset_orientation=offset_orientation,
                                                          params_name="1")
        quat_prev = x_t_init[3:7]
        xee_list = []

        for w in range(self.params["n_steps"]):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:self.dof]
            qdot = ob_robot["joint_state"]["velocity"][0:self.dof]

            if self.params["nr_obst"]>0 or self.params["nr_obst_dyn"]>0:
                self.obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())

            # ----- Fabrics action ----#
            action, _, _, _ = self.compute_action_fabrics(q=q, ob_robot=ob_robot, nr_obst=self.params["nr_obst"], obstacles=self.obstacles)

            ob, *_ = self.env.step(action)

            # result analysis:
            x_ee, _ = utils_analysis._request_ee_state(q, quat_prev)
            xee_list.append(x_ee[0])
            self.IN_COLLISION = utils_analysis.check_distance_collision(q=q, obstacles=self.obstacles)
            self.GOAL_REACHED, error = utils_analysis.check_goal_reaching(q, quat_prev, x_goal=goal_pos)

            if self.GOAL_REACHED:
                self.time_to_goal = w*self.params["dt"]
                break

            if self.IN_COLLISION:
                self.time_to_goal = float("nan")
                break

        self.env.close()

        results = {
            "min_distance": utils_analysis.get_min_dist(),
            "collision": self.IN_COLLISION,
            "goal_reached": self.GOAL_REACHED,
            "time_to_goal": self.time_to_goal,
            "xee_list": xee_list,
            "solver_time": np.mean(self.solver_times),
            "solver_time_std": np.std(self.solver_times),
        }
        with open("simulation_fabrics_kuka.pkl", 'wb') as f:
            pickle.dump(results, f)
        return results


if __name__ == "__main__":
    example_class = example_kuka_fabrics()
    res = example_class.run_kuka_example()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])

