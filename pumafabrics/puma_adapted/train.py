import importlib
import time

from simple_parsing import ArgumentParser
from pumafabrics.puma_adapted.initializer import initialize_framework
from torch.utils.tensorboard import SummaryWriter

# Get arguments
parser = ArgumentParser()
parser.add_argument('--params', type=str, default='2nd_order_R3S3_kinova', help='')
parser.add_argument('--results-base-directory', type=str, default='./', help='')
args = parser.parse_args()

# Import parameters
Params = getattr(importlib.import_module('params.' + args.params), 'Params')
params = Params(args.results_base_directory)
params.results_path += params.selected_primitives_ids + '/'

# Initialize training objects
learner, evaluator, data = initialize_framework(params, args.params)

# Start tensorboard writer
log_name = args.params + '_' + params.selected_primitives_ids
writer = SummaryWriter(log_dir='results/tensorboard_runs/' + log_name)

# Train
time1 = time.perf_counter()
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
time2 = time.perf_counter()
print("timer:", time2 - time1)

