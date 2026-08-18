from pumafabrics.puma_extension.tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from pumafabrics.puma_extension.initializer import initialize_framework
import numpy as np
import yaml
from pathlib import Path
from pumafabrics.puma_extension.plot_s2 import plot_on_sphere
import time
import os

_timestamp_format = "%Y%m%d-%H%M%S"

def load_init_pos_and_demonstrations(shape="JShape"):
    directory = "/home/saray/PycharmProjects/optiman/out/LieFlows/initial_points/"
    path = directory + "S2_Lieflows_initial_points_"+ shape +".txt"
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    xyz_random_init = np.array(data.get("xyz_random", []))
    xyz_demonstrations = np.array(data.get("xyz_demonstrations", []))
    return xyz_demonstrations, xyz_random_init

def to_yaml(data, path):
    """Write the bundle to a YAML file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)

def get_cur_timestamp_str():
    return time.strftime(_timestamp_format)

def single_query_action():
    return

def run_1_shape(selected_primitives_ids=0, shape="Angle", WITH_PLOTTING=False):
    # Parameters
    params_name = "1st_order_S2"
    delta_t = 1.0
    print("selected shape is: ", shape)
    training_time_per_shape = [180.31786506000003, 210.64493306500003, 197.67337383900008, 218.25947039099992, 203.6054036769999, 216.36646221699993, 202.54241337699978, 193.5598050499998, 190.61209000400004, 197.12235043600003, 205.6878476869997, 195.62876557400023, 200.9325070949999, 210.01140105800005, 201.37659896000014, 237.20393716299986, 202.11500268500004, 209.5562655520007, 209.84212922200004, 201.4310735070003, 200.3903714070002, 209.69455988500067, 196.9118279940003, 206.0184611550003, 201.21456212500016, 198.46874541000034]

    demonstrations, x_t_init = (
        load_init_pos_and_demonstrations(shape = shape)
    )

    simulation_length = 200
    results_base_directory = './'

    # Load parameters
    Params = getattr(importlib.import_module('params.' + params_name), 'Params')
    params = Params(results_base_directory)
    params.results_path += params.selected_primitives_ids + '/'
    params.load_model = True

    # Initialize framework
    learner, _, data = initialize_framework(params, params_name, verbose=False)

    # Initialize dynamical system

    dynamical_system = learner.init_dynamical_system(initial_states=torch.FloatTensor(x_t_init).cuda(), delta_t = delta_t)

    # Initialize trajectory plotter
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    trajectory_plotter = TrajectoryPlotter(fig, x0=x_t_init.T, pause_time=1e-5, goal=data['goals training'][0])

    # Simulate dynamical system and plot
    # test:
    transition_info = dynamical_system.transition(space='task') #, obstacles=obstacle_struct)
    x_t = transition_info['desired state']
    # torch array
    trjs_random = torch.zeros((simulation_length, x_t_init.shape[0], x_t_init.shape[1]), device=x_t.device)
    for i in range(simulation_length):
        # Do transition
        transition_info = dynamical_system.transition(space='task') #, obstacles=obstacle_struct)
        x_t = transition_info['desired state']
        trjs_random[i, :, :] = x_t
        # Update plot
        trajectory_plotter.update(x_t.T.cpu().detach().numpy())

    plt.savefig(params.results_path + "images/results_simulate_ds.png")
    if WITH_PLOTTING:
        plt.show()
        plot_on_sphere(demonstrations, trajectory=trjs_random.cpu().detach().numpy(), goal=data['goals training'][0], savepath=params.results_path + "images/",  title="with init randomized")

    ##################################################################################3
    # now for demonstrated trajectories:
    # Initialize dynamical system
    dynamical_system = learner.init_dynamical_system(initial_states=torch.FloatTensor(demonstrations[:, 0, :]).cuda(), delta_t = delta_t)

    # Initialize trajectory plotter
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    trajectory_plotter = TrajectoryPlotter(fig, x0=x_t_init.T, pause_time=1e-5, goal=data['goals training'][0])

    # Simulate dynamical system and plot
    # test:
    transition_info = dynamical_system.transition(space='task') #, obstacles=obstacle_struct)
    x_t = transition_info['desired state']
    # torch array
    trjs_demo = torch.zeros((simulation_length, x_t_init.shape[0], x_t_init.shape[1]), device=x_t.device)
    for i in range(simulation_length):
        # Do transition
        transition_info = dynamical_system.transition(space='task') #, obstacles=obstacle_struct)
        x_t = transition_info['desired state']
        x_t_np = x_t.cpu().detach().numpy()
        if np.isnan(x_t_np).any():
            print("x_t_np is NaN")
        trjs_demo[i, :, :] = x_t
        # Update plot
        trajectory_plotter.update(x_t.T.cpu().detach().numpy())

    if WITH_PLOTTING:
        plot_on_sphere(demonstrations, trajectory=trjs_demo.cpu().detach().numpy(), goal=data['goals training'][0], savepath=params.results_path + "images/", title="with init demos")


    ##############################################################################################################################
    # -- for computation times, do the query time and computation time per demonstration separately, for consistency ---#
    computation_time_trjs = np.zeros(len(demonstrations))
    for z in range(len(demonstrations)):
        time0 = time.perf_counter()
        demo_start = demonstrations[z, 0, :][None, :]
        dynamical_system = learner.init_dynamical_system(initial_states=torch.FloatTensor(demo_start).cuda(), delta_t=delta_t)

        for t in range(simulation_length):
            transition_info = dynamical_system.transition(space='task')  # , obstacles=obstacle_struct)
            x_t = transition_info['desired state']

        computation_time_trjs[z] = time.perf_counter() - time0

    computation_time_trjs_random = np.zeros(len(x_t_init))
    computation_time_queries_random = np.zeros(len(x_t_init))
    for z in range(len(demonstrations)):
        time0 = time.perf_counter()
        x_t_start = x_t_init[z][None, :]
        dynamical_system = learner.init_dynamical_system(initial_states=torch.FloatTensor(x_t_start).cuda(), delta_t=delta_t)

        for t in range(simulation_length):
            time1 = time.perf_counter()
            transition_info = dynamical_system.transition(space='task')  # , obstacles=obstacle_struct)
            x_t = transition_info['desired state']
            time3 = time.perf_counter()
        time4 = time.perf_counter()
        computation_time_queries_random[z] = time3 - time1
        computation_time_trjs_random[z] =  time4- time0


    ############################# SAVING DATA #######################################################3
    # save data:
    data_saving = {"x_mfd": trjs_demo.cpu().detach().numpy().transpose(1, 0, 2).tolist(),
                   "trjs_random": trjs_random.cpu().detach().numpy().transpose(1, 0, 2).tolist(),
                   "computation_time_trjs": computation_time_trjs.tolist(),
                   "computation_time_random": computation_time_trjs_random.tolist(),
                   "computation_time_queries_random": computation_time_queries_random.tolist(),
                   "training_time": training_time_per_shape[selected_primitives_ids]}

    timestamp = get_cur_timestamp_str()
    dirname = os.path.abspath(os.path.dirname(__file__)+'/../../')
    to_yaml(data_saving, os.path.join(dirname, f"../optiman/out/PUMA/{timestamp}_PUMA_{shape}_S2.yaml"))

if __name__ == "__main__":
    # selected_primitives_id = 0
    shapes = ["Angle",
              "BendedLine",
              "CShape",
              "DoubleBendedLine",
              "GShape",
              "JShape",
              "JShape_2",
              "Khamesh",
              "LShape",
              "Leaf_1",
              "Leaf_2",
              "Line",
              "NShape",
              "PShape",
              "RShape",
              "Saeghe",
              "Sharpc",
              "Sine",
              "Spoon",
              "Snake",
              "Sshape",
              "Trapezoid",
              "WShape",
              "Worm",
              "Zshape",
              "heee",
              ]
    for selected_primitives_id in range(len(shapes)):
        shape = shapes[selected_primitives_id]
        run_1_shape(selected_primitives_ids=selected_primitives_id, shape=shape, WITH_PLOTTING=False)
