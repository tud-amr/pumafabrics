from pumafabrics.puma_extension.datasets.dataset_keys import LASA, LASA_S2, LAIR, ABB_R3S3, hammer, kuka, dingo_kinova, lasa_s2_puma
from spatialmath import SO3, UnitQuaternion
import os
import pickle
import numpy as np
import scipy.io as sio
import json
import copy

def load_demonstrations(dataset_name, selected_primitives_ids, dim_manifold):
    """
    Loads demonstrations
    """
    # Get names of primitives in dataset
    dataset_primitives_names = get_dataset_primitives_names(dataset_name)

    # Get names of selected primitives for training
    primitives_names, _ = select_primitives(dataset_primitives_names, selected_primitives_ids)

    # Get number of selected primitives
    n_primitives = len(primitives_names)

    # Get loading path
    dataset_path = 'datasets/' + dataset_name + '/'

    # Get data loader
    data_loader = get_data_loader(dataset_name, dim_manifold)

    # Load
    demonstrations, demonstrations_primitive_id, delta_t_eval = data_loader(dataset_path, primitives_names)

    # Out dictionary
    loaded_info = {'demonstrations raw': demonstrations,
                   'demonstrations primitive id': demonstrations_primitive_id,
                   'n primitives': n_primitives,
                   'delta t eval': delta_t_eval}
    return loaded_info


def get_dataset_primitives_names(dataset_name):
    """
    Chooses primitives keys
    """
    if dataset_name == 'LASA':
        dataset_primitives_names = LASA
    elif dataset_name == 'LASA_S2':
        dataset_primitives_names = LASA_S2
    elif dataset_name == 'lasa_s2_puma':
        dataset_primitives_names = lasa_s2_puma
    elif dataset_name == 'LAIR':
        dataset_primitives_names = LAIR
    elif dataset_name == 'ABB_R3S3':
        dataset_primitives_names = ABB_R3S3
    elif dataset_name == 'hammer':
        dataset_primitives_names = hammer
    elif dataset_name == 'kuka':
        dataset_primitives_names = kuka
    elif dataset_name == 'dingo_kinova':
        dataset_primitives_names = dingo_kinova
    else:
        raise NameError('Dataset %s does not exist' % dataset_name)

    return dataset_primitives_names


def select_primitives(dataset, selected_primitives_ids):
    """
    Gets selected primitives
    """
    selected_primitives_names = []
    selected_primitives_save_name = ''
    selected_primitives_ids = list(map(int, selected_primitives_ids.split(',')))  # map from string to list
    for id in selected_primitives_ids:
        selected_primitives_names.append(dataset[id])
        selected_primitives_save_name += str(id) + '_'

    return selected_primitives_names, selected_primitives_save_name[:-1]


def get_data_loader(dataset_name, dim_manifold):
    """
    Chooses data loader depending on the data type
    """
    if dataset_name == 'LASA':
        data_loader = load_LASA
    elif dataset_name == 'LAIR' or dataset_name == 'optitrack' or dataset_name == 'interpolation':
        data_loader = load_numpy_file
    elif dataset_name == 'joint_space':
        data_loader = load_from_dict
    elif dataset_name == 'LASA_S2':
        data_loader = load_LASA_S2
    elif dataset_name == 'lasa_s2_puma':
        data_loader = load_LASA_S2
    elif dataset_name == 'ABB_R3S3':
        data_loader = load_R3S3
    elif dataset_name == 'hammer':
        data_loader = load_hammer
    elif dataset_name == 'kuka' and dim_manifold == 3:
        data_loader = load_R3
    elif dataset_name == 'kuka' and dim_manifold == 6:
        data_loader = load_R3S3
    elif dataset_name == 'dingo_kinova' and dim_manifold == 3:
        data_loader = load_R3
    elif dataset_name == 'dingo_kinova' and dim_manifold == 6:
        data_loader = load_R3S3
    else:
        raise NameError('Dataset %s does not exist' % dataset_name)

    return data_loader


def load_LASA(dataset_dir, demonstrations_names):
    """
    Load LASA matlab models
    """
    s_x, s_y, demos, primitive_id, dt = [], [], [], [], []
    dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/../" + copy.deepcopy(dataset_dir)
    for i in range(len(demonstrations_names)):
        mat_file = sio.loadmat(dataset_dir + demonstrations_names[i])
        data = mat_file['demos']

        for j in range(data.shape[1]):  # iterate through demonstrations
            s_x = data[0, j]['pos'][0, 0][0]
            s_y = data[0, j]['pos'][0, 0][1]
            s = [s_x, s_y]
            demos.append(s)
            dt.append(data[0, j]['dt'][0, 0][0, 0])
            primitive_id.append(i)

    return demos, primitive_id, dt


def load_numpy_file(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in numpy files
    """
    demos, primitive_id, dt = [], [], []
    dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/../" + copy.deepcopy(dataset_dir)
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        for demo_primitive in demos_primitive:
            data = np.load(dataset_dir + demonstrations_names[i] + '/' + demo_primitive)
            if data.shape[0] == 1:
                # if extra dimension in demo, remove
                data = data[0]
            demos.append(data.T)
            primitive_id.append(i)
            dt.append(np.ones(len(data))*0.03)  # dummy time

    return demos, primitive_id, dt


def load_from_dict(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in dictionaries
    """
    demos, primitive_id, dt = [], [], []
    dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/../" + copy.deepcopy(dataset_dir)

    # Iterate in each primitive (multi model learning)
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        # Iterate over each demo in primitive
        for demo_primitive in demos_primitive:
            filename = dataset_dir + demonstrations_names[i] + '/' + demo_primitive
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            demos.append(data['q'].T)
            dt.append(data['delta_t'])
            primitive_id.append(i)

    return demos, primitive_id, dt


def load_R3S3(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in dictionaries for ABB
    """
    demos, primitive_id, dt = [], [], []
    dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/../" + copy.deepcopy(dataset_dir)

    # Iterate in each primitive (multi model learning)
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        # Iterate over each demo in primitive
        for demo_primitive in demos_primitive:
            filename = dataset_dir + demonstrations_names[i] + '/' + demo_primitive
            with open(filename, 'rb') as file:
                data = pickle.load(file)

            prev_quat = None
            quats = []
            for numpy_rot_mat in data['x_rot']:
                # Get quatenion array from data
                quat = UnitQuaternion(SO3(numpy_rot_mat)).A

                # Check if quaternion flip, and flip if necessary
                if prev_quat is None:
                    prev_quat = quat

                dist_quats = np.linalg.norm(quat - prev_quat)

                if dist_quats > 0.5:
                    quat *= -1

                # Append quaternion to list
                quats.append(quat)
                prev_quat = quat

            # Set identity quat as goal
            last_quat = UnitQuaternion(quats[-1])
            quats = [(UnitQuaternion(quats[i]) / last_quat).A for i in range(len(quats))]

            # Set zero as goal
            positions = np.array(data['x_pos']) - np.array(data['x_pos'])[-1]

            # Create demo out of positions and quats
            demo = np.concatenate([positions, quats], axis=1).T

            # Append demo to demo list
            demos.append(demo)
            dt.append(data['delta_t'])
            primitive_id.append(i)

    return demos, primitive_id, dt

def load_R3(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in 3D
    """
    demos, primitive_id, dt = [], [], []
    dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/../" + copy.deepcopy(dataset_dir)

    # Iterate in each primitive (multi model learning)
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        # Iterate over each demo in primitive
        for demo_primitive in demos_primitive:
            filename = dataset_dir + demonstrations_names[i] + '/' + demo_primitive
            with open(filename, 'rb') as file:
                data = pickle.load(file)

            # Set zero as goal
            positions = np.array(data['x_pos']) - np.array(data['x_pos'])[-1]

            # Append demo to demo list
            demos.append(positions.T)
            dt.append(data['delta_t'])
            primitive_id.append(i)

    return demos, primitive_id, dt


def load_LASA_S2(dataset_path, primitives_names):
    """
    Load LASA S2 models
    """
    demos, primitive_id = [], []
    for i in range(len(primitives_names)):
        path = dataset_path + primitives_names[i] + '.txt'
        # Read the data from the file
        with open(path) as f:
            data = f.read()

        # Reconstruct the data as a dictionary
        data = json.loads(data)

        # Iterate through demonstrations
        for j in range(len(data['xyz'])):
            s = np.array(data['xyz'][j]).T
            demos.append(s)
            primitive_id.append(i)

    dt = 1
    return demos, primitive_id, dt


def load_hammer(dataset_path, primitives_names):
    """
    Loads demos for poultry use case
    """
    demos, primitive_id, dt = [], [], []
    for i in range(len(primitives_names)):
        demos_primitive = os.listdir(dataset_path + primitives_names[i])
        # Iterate over each demo in primitive
        for demo_primitive in demos_primitive:
            path = dataset_path + primitives_names[i] + '/' + demo_primitive

            # Read the data from the file
            data = np.genfromtxt(path, delimiter=',')[1:, 4:].T

            # Check if quaternion flip, and flip if necessary
            prev_quat = None
            for j in range(data.shape[1]):
                quat = data[3:, j]
                if prev_quat is None:
                    prev_quat = quat

                dist_quats = np.linalg.norm(quat - prev_quat)

                if dist_quats > 1.0:
                    quat *= -1

                data[3:, j] = quat
                prev_quat = quat

            # Check if quaternion trajectory starts from predefined initial hemisphere
            if data[4, -1] < 0:  # if last element of hemisphere angle negative
                data[3:, :] *= -1  # flip trajectory such that it starts from predefined initial hemisphere

            demos.append(data)
            primitive_id.append(i)
            delta_t = data[0] * 0 + 0.097  # directly obtained from data
            dt.append(delta_t)
    return demos, primitive_id, dt