import os
import pickle
import torch
from src.dataset import Dataset
from argparse import ArgumentParser, RawTextHelpFormatter
from utils.common import set_seed
import random

########################################################################################################################

parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--split_folder', type=str, required=True, help='Path of the split folder'
)
parser.add_argument(
    '--samples_folder', type=str, required=True, help='Path of the samples folder'
)
parser.add_argument(
    '--max_sample_size', type=int, required=False, default=1000, help='Maximum sample size'
)
parser.add_argument(
    '--seed', type=int, required=False, default=19, help='Seed value'
)
args = parser.parse_args()

########################################################################################################################

split_folder = args.split_folder
samples_folder = args.samples_folder
log_file_path = os.path.join(samples_folder, "log.txt")
max_sample_size = args.max_sample_size
seed = args.seed

########################################################################################################################

# Set the seed value
set_seed(seed=seed)

########################################################################################################################

# If the samples folder does not exist, create it
if not os.path.exists(samples_folder):
    os.makedirs(samples_folder)

# Read the train dataset
dataset = Dataset()
dataset.read_edge_list(split_folder + "/residual.edges")
dataset.print_info()
init_time, last_time = dataset.get_init_time(), dataset.get_last_time()
nodes_num = dataset.get_nodes_num()
is_directed = dataset.is_directed()
is_signed = dataset.is_signed()
del dataset

MIN_INTERVAL_SIZE = (1e-2 * (last_time - init_time))  # 1e-2 # 8e-2 # 9e-2

log_file = open(log_file_path, "w")
log_file.write(f"+ Seed: {seed}\n")
log_file.write(f"+ Max sample size: {max_sample_size}\n")
log_file.write(f"+ Residual init time: {init_time}\n")
log_file.write(f"+ Residual last time: {last_time}\n")
log_file.write(f"+ Residual nodes num: {nodes_num}\n")
log_file.write(f"+ Residual is directed: {is_directed}\n")
log_file.write(f"+ Residual is signed: {is_signed}\n")
log_file.write(f"+ Min interval size: {MIN_INTERVAL_SIZE}\n")
########################################################################################################################


def durations2samples(pos_durations, zero_durations, max_sample_size):

    # Define the sample size
    sample_size = min(
        len(pos_durations[0]) + len(pos_durations[1]), len(zero_durations[0]) + len(zero_durations[1]), max_sample_size
    )

    # Shuffle the durations and select some samples
    zero_simple_durations = random.sample(zero_durations[0], len(zero_durations[0]))
    zero_hard_durations = random.sample(zero_durations[1], len(zero_durations[1]))
    zero_selected_durations = zero_hard_durations[:min(sample_size//2, len(zero_hard_durations))]
    zero_selected_durations += random.sample(
        zero_hard_durations[min(sample_size//2, len(zero_hard_durations)):] + zero_simple_durations,
        sample_size - len(zero_selected_durations)
    )
    # Construct zero pair and interval samples
    zero_pairs, zero_intervals = [], []
    for duration in zero_selected_durations:
        zero_pairs.append((duration[0], duration[1]))
        # Sample a random point in the duration
        rp = random.uniform(duration[2].item(), duration[3].item())
        zero_intervals.append(
            (max(duration[2].item(), rp - MIN_INTERVAL_SIZE / 2), min(rp + MIN_INTERVAL_SIZE / 2, duration[3].item()))
        )
        # zero_intervals.append((duration[2].item(), duration[3].item()))

    # Shuffle the durations and select some samples
    pos_simple_durations = random.sample(pos_durations[0], len(pos_durations[0]))
    pos_hard_durations = random.sample(pos_durations[1], len(pos_durations[1]))
    pos_selected_durations = pos_hard_durations[:min(sample_size // 2, len(pos_hard_durations))]
    pos_selected_durations += random.sample(
        pos_hard_durations[min(sample_size // 2, len(pos_hard_durations)):] + pos_simple_durations,
        sample_size - len(pos_selected_durations)
    )
    # Construct positive pair and interval samples
    pos_pairs, pos_intervals = [], []
    for duration in pos_selected_durations:
        pos_pairs.append((duration[0], duration[1]))
        # Sample a random point in the duration
        rp = random.uniform(duration[2].item(), duration[3].item())
        pos_intervals.append(
            ( max(duration[2].item(), rp - MIN_INTERVAL_SIZE / 2), min(rp + MIN_INTERVAL_SIZE / 2, duration[3].item()) )
        )
        # pos_intervals.append((duration[2].item(), duration[3].item()))

    # Construct the samples dictionary
    samples = {
        'zero': {'pairs': zero_pairs, 'intervals': zero_intervals, 'labels': [0]*len(zero_pairs)},
        'pos': {'pairs': pos_pairs, 'intervals': pos_intervals, 'labels': [1]*len(pos_pairs)}
    }

    return samples


def get_durations(data_dict, directed, nodes_num, init_time, last_time):
    """
    Get the durations from the data dictionary
    :param data_dict: the data dictionary
    :param directed: whether the graph is directed
    :param nodes_num: number of nodes
    :param init_time: initial time
    :param last_time: last time
    """

    pos_durations, neg_durations, zero_durations = [], [], []
    for i in range(nodes_num):

        for j in range(0 if directed else i+1, nodes_num):

            if i != j:

                # If the pair (i, j) contains any event
                if i in data_dict and j in data_dict[i]:

                    # Sort the time-state list by time
                    time_state_list = sorted(data_dict[i][j], key=lambda x: x[0])

                    for idx, (time, state) in enumerate(time_state_list):

                        # For the first event
                        if idx == 0:

                            # If the time is greater than the initial time, than add the duration [init_time, time]
                            if time > init_time:

                                # Assumption: the first event time must indicate a positive or negative link
                                duration = (i, j, init_time, time, 0)

                            # If the first time equals the initial time, then continue
                            else:
                                continue

                        # If it is not the first event
                        else:

                            # For duration [time_state_list[idx-1][0], time_state_list[idx][0]]
                            # The state of the duration is the state of the previous event
                            duration = (i, j, time_state_list[idx-1][0], time, time_state_list[idx-1][1].item())

                        # Add the durations to the corresponding list
                        if duration[3] - duration[2] >= MIN_INTERVAL_SIZE:
                            if duration[4] == 1:
                                pos_durations.append(duration)
                            elif duration[4] == -1:
                                neg_durations.append(duration)
                            else:
                                zero_durations.append(duration)

                    # If the last event time is smaller than the last time of the network
                    # Add the duration [time_state_list[-1][0], last_time]
                    if time_state_list[-1][0] != last_time:

                        duration = (i, j, time_state_list[-1][0], last_time, time_state_list[-1][1].item())

                        if duration[3] - duration[2] >= MIN_INTERVAL_SIZE:
                            if duration[4] == 1:
                                pos_durations.append(duration)
                            elif duration[4] == -1:
                                neg_durations.append(duration)
                            else:
                                zero_durations.append(duration)

                # If the pair (i, j) does not contain any event
                else:

                    duration = (i, j, init_time, last_time, 0)
                    if duration[3] - duration[2] >= MIN_INTERVAL_SIZE:
                        zero_durations.append(duration)

    return pos_durations, zero_durations, neg_durations


def get_simple_hard_durations(durations, init_t, last_t, set_type="", res_data_dict=None):
    '''
    Get the simple and hard durations

    init_t and last_t indicate the initial and last time of the residual or prediction network depending on the task
    '''

    simple_durations, hard_durations = [], []

    # For the future prediction task,
    if set_type == "future_prediction":

        for duration in durations:
            u, v, _, _, state = duration
            if u not in res_data_dict or v not in res_data_dict[u]:
                if state == 0:
                    simple_durations.append(duration)
                else:
                    hard_durations.append(duration)
            else:

                # If the residual network contains an event whose time differs from the initial and last time
                # of the residual network, then the duration is hard
                is_hard = False
                for edges in res_data_dict[u][v]:
                    if edges[0] != init_time or edges[1] != last_time:
                        is_hard = True
                        break
                if is_hard:
                    hard_durations.append(duration)
                else:
                    simple_durations.append(duration)

    # For the other tasks such as residual network reconstruction, completion and validation
    else:

        for duration in durations:
            if duration[2] == init_t and duration[3] == last_t:
                simple_durations.append(duration)
            else:
                hard_durations.append(duration)

    return simple_durations, hard_durations

########################################################################################################################

# Construct the durations for the reconstruction set
# Read the residual dataset
residual_dataset = Dataset()
residual_dataset.read_edge_list(os.path.join(split_folder, "./residual.edges"))
residual_data_dict = residual_dataset.get_data_dict(weights=True)
res_pos_dur, res_zero_dur, res_neg_dur = get_durations(
    residual_data_dict, is_directed, nodes_num, init_time=init_time, last_time=last_time
)
res_pos_dur = get_simple_hard_durations(res_pos_dur, init_time, last_time)
res_zero_dur = get_simple_hard_durations(res_zero_dur, init_time, last_time)
# Convert the durations to samples
res_samples = durations2samples(res_pos_dur, res_zero_dur, max_sample_size)
log_file.write(f"+ The statistics of the reconstruction set:\n")
log_file.write(f"\t- Zero durations: {len(res_zero_dur[0]) + len(res_zero_dur[1])} "
                f"| Pos durations: {len(res_pos_dur[0]) + len(res_pos_dur[1])}\n")
log_file.write(f"\t- Zero samples: {len(res_samples['zero']['labels'])} "
               f"| Pos samples: {len(res_samples['pos']['labels'])}\n")

# Construct the samples for the validation set
# Read the validation dataset
valid_dataset = Dataset()
valid_dataset.read_edge_list(os.path.join(split_folder, "./validation.edges"))
valid_data_dict = valid_dataset.get_data_dict(weights=True)
# Construct the durations for the validation network
valid_pos_dur, valid_zero_dur, valid_neg_dur = get_durations(
    valid_data_dict, is_directed, nodes_num, init_time=init_time, last_time=last_time
)
valid_pos_dur = get_simple_hard_durations(valid_pos_dur, init_time, last_time)
valid_zero_dur = get_simple_hard_durations(valid_zero_dur, init_time, last_time)
# Convert the durations to samples
valid_samples = durations2samples(valid_pos_dur, valid_zero_dur, max_sample_size)
log_file.write(f"+ The statistics of the validation set:")
log_file.write(f"\t- Zero durations: {len(valid_zero_dur[0]) + len(valid_zero_dur[1])} "
                f"| Pos durations: {len(valid_pos_dur[0])+len(valid_pos_dur[1])}\n")
log_file.write(f"\t- Zero samples: {len(valid_samples['zero']['labels'])} "
                f"| Pos samples: {len(valid_samples['pos']['labels'])}\n")

# Construct the samples for the completion set
# Read the completion dataset
comp_dataset = Dataset()
comp_dataset.read_edge_list(os.path.join(split_folder, "./completion.edges"))
comp_data_dict = comp_dataset.get_data_dict(weights=True)
# Construct the durations for the completion network
comp_pos_dur, comp_zero_dur, comp_neg_dur = get_durations(
    comp_data_dict, is_directed, nodes_num, init_time=init_time, last_time=last_time
)
comp_pos_dur = get_simple_hard_durations(comp_pos_dur, init_time, last_time)
comp_zero_dur = get_simple_hard_durations(comp_zero_dur, init_time, last_time)
# Convert the durations to samples
comp_samples = durations2samples(comp_pos_dur, comp_zero_dur, max_sample_size)
log_file.write(f"+ The statistics of the completion set:\n")
log_file.write(f"\t- Zero durations: {len(comp_zero_dur[0]) + len(comp_zero_dur[1])} "
               f"| Pos durations: {len(comp_pos_dur[0]) + len(comp_pos_dur[1])}\n")
log_file.write(f"\t- Zero samples: {len(comp_samples['zero']['labels'])} "
               f"| Pos samples: {len(comp_samples['pos']['labels'])}\n")

# Construct the samples for the prediction set
# Read the residual dataset
pred_dataset = Dataset()
pred_dataset.read_edge_list(os.path.join(split_folder, "./prediction.edges"))
pred_data_dict = pred_dataset.get_data_dict(weights=True)
# Construct the durations for the prediction network
pred_pos_dur, pred_zero_dur, pred_neg_dur = get_durations(
    pred_data_dict, is_directed, nodes_num, init_time=last_time, last_time=pred_dataset.get_last_time()
)
pred_pos_dur = get_simple_hard_durations(pred_pos_dur, None, None, "future_prediction", residual_data_dict)
pred_zero_dur = get_simple_hard_durations(pred_zero_dur, None, None, "future_prediction", residual_data_dict)
# Convert the durations to samples
pred_samples = durations2samples(pred_pos_dur, pred_zero_dur, max_sample_size)
# Add the last states for the prediction samples
pred_samples['zero']['last_states'] = [
    max(residual_data_dict.get(pair[0], dict()).get(pair[1], [(None, 0)]), key=lambda item: item[0])[1]
    for pair in pred_samples['zero']['pairs']
]
pred_samples['pos']['last_states'] = [
    max(residual_data_dict.get(pair[0], dict()).get(pair[1], [(None, 0)]), key=lambda item: item[0])[1]
    for pair in pred_samples['pos']['pairs']
]
log_file.write(f"+ The statistics of the future prediction set:\n")
log_file.write(f"\t- Zero durations: {len(pred_zero_dur[0])+len(pred_zero_dur[1])} "
               f"| Pos durations: {len(pred_pos_dur[0]) + len(pred_pos_dur[1])}\n")
log_file.write(f"\t- Zero samples: {len(pred_samples['zero']['labels'])} "
               f"| Pos samples: {len(pred_samples['pos']['labels'])}\n")

########################################################################################################################

# Save the reconstruction samples
with open(os.path.join(samples_folder, "reconstruction.samples"), 'wb') as f:
    pickle.dump(res_samples, f)
# Save the validation samples
with open(os.path.join(samples_folder, "validation.samples"), 'wb') as f:
    pickle.dump(valid_samples, f)
# Save the completion samples
with open(os.path.join(samples_folder, "completion.samples"), 'wb') as f:
    pickle.dump(comp_samples, f)
# Save the prediction samples
with open(os.path.join(samples_folder, "prediction.samples"), 'wb') as f:
    pickle.dump(pred_samples, f)

########################################################################################################################