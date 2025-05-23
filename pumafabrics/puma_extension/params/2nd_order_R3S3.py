from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    """ General parameters """
    dataset_name: str = 'kuka'  # selects dataset, options: LASA, LAIR, optitrack, interpolation, joint_space, ABB_R3S3
    results_path: str = 'results/2nd_order_R3S3/'
    multi_motion: bool = False  # true when learning multiple motions together
    selected_primitives_ids: str = '6'  # id number from dataset_keys.py, e.g., '2' or '4,0,6'
    manifold_dimensions: int = 6  # dimensionality of the data manifold
    saturate_out_of_boundaries_transitions: bool = True  # True to enforce positively invariant set
    dynamical_system_order: int = 2  # options: 1, 2
    space: str = 'euclidean_sphere'  # data manifold shape

    """ Neural Network """
    latent_space_dim: int = 600  # dimensionality latent space
    neurons_hidden_layers: int = 600  # number of neurons per layer
    batch_size: int = 300  # sampling batch size
    learning_rate: float = 0.0001  # AdamW learning rate
    weight_decay: float = 0.0000  # AdamW weight decay

    """ Contrastive Imitation """
    triplet_type: str = 'spherical squared'  # distance metric used in triplet loss
    imitation_loss_weight: float = 1.0  # imitation loss weight
    stabilization_loss_weight: float = 1.0  # stability loss weight
    boundary_loss_weight: float = 10.0   # boundary loss weight
    imitation_window_size: int = 13  # imitation window size
    stabilization_window_size: int = 2  # stability window size
    triplet_margin: float = 1e-6  # 1.25e-4 for triplet  # triplet loss margin
    interpolation_sigma: float = 0.8  # percentage of points sampled in demonstrations space when multi-model learning

    """ Training """
    train: bool = True  # true when training
    load_model: bool = False  # true to load previously trained model
    max_iterations: int = 40000  # maximum number of training iterations

    """ Preprocessing """
    spline_sample_type: str = 'from data'  # resample from spline type, options: from data, evenly spaced
    workspace_boundaries_type: str = 'custom'  # options: from data, custom
    workspace_boundaries: np.ndarray = np.array([[-1.0, 1.0],
                                                 [-1.0, 1.0],
                                                 [-1.0, 1.0],
                                                 [-1.0, 1.0],
                                                 [-1.0, 1.0],
                                                 [-1.0, 1.0],
                                                 [-1.0, 1.0]])  # list to provide boundaries when custom boundaries
    trajectories_resample_length: int = 2000  # amount of points resampled from splines
    state_increment: float = 0.5  # when workspace_boundaries_type = from data, percentage to increment state-space size

    """ Evaluation """
    save_evaluation: bool = True  # true to save evaluation results
    evaluation_interval: int = 1000  # interval between training iterations to evaluate model
    quanti_eval: bool = True  # quantitative evaluation
    quali_eval: bool = True  # qualitative evaluation
    diffeo_quanti_eval: bool = False  # quantitative evaluation of diffeomorphism mismatch
    diffeo_quali_eval: bool = False  # qualitative evaluation of diffeomorphism mismatch
    ignore_n_spurious: bool = False  # when selecting best model, true to ignore amount of spurious attractors
    fixed_point_iteration_thr = 0.2  # distance threshold to consider that a point did not reach the goal
    density: int = 2  # density^workspace_dimension = amount of points sampled from state space for evaluation
    simulated_trajectory_length: int = 700  # integration length for evaluation
    evaluation_samples_length: int = 100  # integration steps skipped in quantitative evaluation for faster evaluation
    show_plotly: bool = False  # show evaluation during training

    """ Hyperparameter Optimization """
    gamma_objective = 3.5  # weight for hyperparameter evaluation
    optuna_n_trials = 1000  # maximum number of optuna trials

    """ Dataset training """
    length_dataset = 30  # number of primitives in dataset

    def __init__(self, results_base_directory):
        self.results_path = results_base_directory + self.results_path
