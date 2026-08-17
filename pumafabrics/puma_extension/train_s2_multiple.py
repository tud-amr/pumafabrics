import importlib
from simple_parsing import ArgumentParser
from pumafabrics.puma_extension.initializer import initialize_framework
from torch.utils.tensorboard import SummaryWriter
import time

# Get arguments
parser = ArgumentParser()
parser.add_argument('--params', type=str, default='1st_order_S2', help='')
parser.add_argument('--results-base-directory', type=str, default='./', help='')
args = parser.parse_args()


def train_1_shape(shape_id):
    time_0 = time.perf_counter()
    # Import parameters
    Params = getattr(importlib.import_module('params.' + args.params), 'Params')
    params = Params(args.results_base_directory)
    params.selected_primitives_ids = shape_id
    params.results_path += params.selected_primitives_ids + '/'

    # Initialize training objects
    learner, evaluator, data = initialize_framework(params, args.params)

    # Start tensorboard writer
    log_name = args.params + '_' + params.selected_primitives_ids
    writer = SummaryWriter(log_dir='results/tensorboard_runs/' + log_name)

    # Train
    for iteration in range(params.max_iterations + 1):
        # Evaluate model
        if iteration % params.evaluation_interval == 0:
            metrics_acc, metrics_stab = evaluator.run(iteration=iteration)

            if params.save_evaluation:
                evaluator.save_progress(params.results_path, iteration, learner.model, writer)

            print('Metrics sum:', metrics_acc['metrics sum'], '; Number of unsuccessful trajectories:', metrics_stab['n spurious'])

        # Training step
        loss, loss_list, losses_names = learner.train_step()

        # Print progress
        if iteration % 10 == 0:
            print(iteration, 'Total cost:', loss.item())

        # Log losses in tensorboard
        for j in range(len(losses_names)):
            writer.add_scalar('losses/' + losses_names[j], loss_list[j], iteration)
    time_2 = time.perf_counter() - time_0
    return time_2

if __name__ == '__main__':
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
    training_times = []
    for shape_id in range(len(shapes)):
        training_time = train_1_shape(str(shape_id))
        training_times.append(training_time)
    print("training times per shape: ", training_times)
