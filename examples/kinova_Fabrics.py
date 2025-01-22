import numpy as np
from pumafabrics.tamed_puma.tamedpuma.fabrics_controller import FabricsController
from pumafabrics.tamed_puma.create_environment.environments import trial_environments
from pumafabrics.tamed_puma.utils.analysis_utils import UtilsAnalysis
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka
from pumafabrics.tamed_puma.tamedpuma.example_generic import ExampleGeneric
import yaml
from pumafabrics.tamed_puma.nullspace_control.nullspace_controller import CartesianImpedanceController
import pybullet

class example_kinova_fabrics(ExampleGeneric):
    def __init__(self, file_name="kinova_TamedPUMA_tomato"):
        super(ExampleGeneric, self).__init__()
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = float("nan")
        self.obstacles = []
        with open("../pumafabrics/tamed_puma/config/" + file_name + ".yaml", "r") as setup_stream:
             self.params = yaml.safe_load(setup_stream)
        self.dof = self.params["dof"]
        self.params["bool_extensive_concretize"] = False
        self.robot_name = self.params["robot_name"]

    def initialize_environment(self):
        envir_trial = trial_environments()
        (self.env, self.goal) = envir_trial.initialize_environment_kinova(params=self.params)

    def construct_example(self):
        # --- parameters --- #
        self.offset_orientation = np.array(self.params["orientation_goal"])

        self.initialize_environment()
        self.fabrics_controller = FabricsController(params=self.params)
        self.planner, fk = self.fabrics_controller.set_full_planner(goal=self.goal)
        self.kinova_kinematics = KinematicsKuka(end_link_name=self.params["end_links"][0], robot_name=self.robot_name)
        self.utils_analysis = UtilsAnalysis(forward_kinematics=fk,
                                            collision_links=self.params["collision_links"],
                                            collision_radii=self.params["collision_radii"],
                                            kinematics=self.kinova_kinematics)

    def run_kinova_example(self):
        # --- parameters --- #
        offset_orientation = np.array(self.params["orientation_goal"])
        goal_pos = self.params["goal_pos"]
        dof = self.params["dof"]
        action = np.zeros(dof+1)
        ob, *_ = self.env.step(action)

        # initial state:
        q_init = ob['robot_0']["joint_state"]["position"][0:dof]
        x_t_init = self.kinova_kinematics.get_initial_state_task(q_init=q_init, qdot_init=np.zeros((dof, 1)), offset_orientation=offset_orientation, mode_NN=self.params["mode_NN"])
        ob, *_ = self.env.step(np.zeros(self.dof))
        quat_prev = x_t_init[3:7]
        xee_list = []

        for w in range(self.params["n_steps"]):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:self.dof]
            qdot = ob_robot["joint_state"]["velocity"][0:self.dof]

            if self.params["nr_obst"]>0:
                self.obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
            else:
                self.obstacles = []

            # recompute goal position
            goal_pos = [goal_pos[i] + self.params["goal_vel"][i] * self.params["dt"] for i in
                                 range(len(goal_pos))]
            pybullet.addUserDebugPoints([goal_pos], [[1, 0, 0]], 5, 0.1)

            # ----- Fabrics action ----#
            action, _, _, _ = self.fabrics_controller.compute_action_full(q=q, ob_robot=ob_robot,
                                                                             nr_obst=self.params["nr_obst"],
                                                                             obstacles=self.obstacles,
                                                                             goal_pos=goal_pos)

            if self.params["mode_env"] == "vel" and self.params["mode"]=="acc":  # todo: fix nicely or mode == "acc"): #mode_NN=="2nd":
                action = self.integrate_to_vel(qdot=qdot, action_acc=action, dt=self.params["dt"])
                action = np.clip(action, -1 * np.array(self.params["vel_limits"]), np.array(self.params["vel_limits"]))
            else:
                action = action
            print("action[-1]", action[-1])
            ob, *_ = self.env.step(action)

            # result analysis:
            x_ee, _ = self.utils_analysis._request_ee_state(q, quat_prev)
            xee_list.append(x_ee[0])
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
            "solver_times": np.array(self.fabrics_controller.request_solver_times())*1000,
            "solver_time": np.mean(self.fabrics_controller.request_solver_times()),
            "solver_time_std": np.std(self.fabrics_controller.request_solver_times()),
        }
        return results

def main(render=True):
    example_class = example_kinova_fabrics()
    q_init_list = [
        # with goal changing:
        np.array((0.87, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
        np.array((0.531, 1.36, 0.070, -1.065, 0.294, -1.2, -0.242)),
        np.array((-0.702, 0.355, -0.016, -1.212, 0.012, -0.502, -0.010)),
        np.array((0.531, 1.16, 0.070, -1.665, 0.294, -1.2, -0.242)),
        np.array((0.07, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
        # others:
        np.array((0.531, 0.836, 0.070, -1.665, 0.294, -0.877, -0.242)),
        np.array((0.531, 1.36, 0.070, -1.065, 0.294, -1.2, -0.242)),
        np.array((-0.702, 0.355, -0.016, -1.212, 0.012, -0.502, -0.010)),
        np.array((0.531, 1.16, 0.070, -1.665, 0.294, -1.2, -0.242)),
        np.array((0.07, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
        np.array((0.531, 0.836, 0.070, -1.665, 0.294, -0.877, -0.242)),
        np.array((0.51, 0.67, -0.17, -1.73, 0.25, -0.86, -0.11)),
        np.array((0.91, 0.79, -0.22, -1.33, 1.20, -1.76, -1.06)),
        np.array((0.83, 0.53, -0.11, -0.95, 1.05, -1.24, -1.45)),
        np.array((0.87, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
    ]
    positions_obstacles_list = [
        # # with goal changing:
        [[0.5, 0., 0.55], [0.5, 0., 10.1]],
        [[0.55, 0.15, 0.05], [0.55, 0.15, 0.23]],
        [[0.5, -0.35, 0.5], [0.24, 0.45, 10.2]],
        [[0.45, 0.02, 0.28], [0.7, 0.02, 0.28]],
        [[0.5, -0.0, 0.5], [0.3, -0.1, 10.5]],
        # others:
        [[0.5, 0., 0.55], [0.5, 0., 10.1]],
        [[0.5, 0.15, 0.05], [0.5, 0.15, 0.2]],
        [[0.5, -0.35, 0.5], [0.24, 0.45, 10.2]],
        [[0.45, 0.02, 0.2], [0.6, 0.02, 0.2]],
        [[0.5, -0.0, 0.5], [0.3, -0.1, 10.5]],
        [[0.5, -0.05, 0.3], [0.5, 0.2, 10.25]],
        [[0.5, -0.0, 0.2], [0.5, 0.2, 10.4]],
        [[0.5, -0.0, 0.28], [0.5, 0.2, 10.4]],
        [[0.5, 0.25, 0.55], [0.5, 0.2, 10.4]],
        [[0.5, 0.1, 0.45], [0.5, 0.2, 10.4]],
    ]
    speed_obstacles_list = [
        # # with goal changing:
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        # others:
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
    ]
    goal_pos_list = [
        # # #changing goal pose:
        [0.58, -0.014, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.7, -0.214, 0.315],
        [0.7, -0.214, 0.115],
        # others:
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.25, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
    ]
    goal_vel_list = [
        [0., 0., 0.] for _ in range(len(q_init_list))
    ]
    goal_vel_list[0] = [0., 0., 0.]
    if len(q_init_list) > 1:
        goal_vel_list[1] = [-0.01, 0., 0.]
    if len(q_init_list) > 2:
        goal_vel_list[2] = [-0.01, 0., 0.0]
    id_nr = 1
    example_class.overwrite_defaults(params = example_class.params,
                                     init_pos=q_init_list[id_nr],
                                     positions_obstacles=positions_obstacles_list[id_nr],
                                     speed_obstacles=speed_obstacles_list[id_nr],
                                     goal_pos=goal_pos_list[id_nr],
                                     goal_vel=goal_vel_list[id_nr],
                                     render=render,
                                     )
    example_class.construct_example()
    res = example_class.run_kinova_example()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])
    return {}

if __name__ == "__main__":
    main()