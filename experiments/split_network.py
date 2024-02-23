import os
import sys
import networkx as nx
from argparse import ArgumentParser, RawTextHelpFormatter
import torch
import utils
from src.dataset import Dataset
from utils import set_seed, flatIdx2matIdx, matIdx2flatIdx

########################################################################################################################

parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--edges', type=str, required=True, help='Path of the edge list file'
)
parser.add_argument(
    '--output_folder', type=str, required=True, help='Path of the output dataset folder'
)
parser.add_argument(
    '--pr', type=float, required=False, default=0.1, help='Prediction ratio'
)
parser.add_argument(
    '--mr', type=float, required=False, default=0.1, help='Validation ratio'
)
parser.add_argument(
    '--cr', type=float, required=False, default=0.1, help='Completion ratio'
)
parser.add_argument(
    '--verbose', type=bool, required=False, default=True, help='Verbose'
)
parser.add_argument(
    '--seed', type=int, required=False, default=19, help='Seed value'
)
args = parser.parse_args()

########################################################################################################################

# Set some parameters
edges_file = args.edges
output_folder = args.output_folder
prediction_ratio = args.pr
validation_ratio = args.mr
completion_ratio = args.cr
verbose = args.verbose
seed = args.seed

########################################################################################################################

# Set the seed value
set_seed(seed=seed)

# Create the target folder
os.makedirs(output_folder)

log_file_path = os.path.join(output_folder, "log.txt")
# orig_stdout = sys.stdout
# f = open(log_file_path, 'w')
# sys.stdout = f

# Load the dataset
dataset = Dataset()
dataset.read_edge_list(edges_file)
nodes_num = dataset.get_nodes_num()
# Get the minimum and maximum time values
min_time, max_time = dataset.get_init_time(), dataset.get_last_time()

data_dict = dataset.get_data_dict(weights=True)
directed = dataset.is_directed()
signed = dataset.is_signed()
# edges, times, states = dataset.get_edges(), dataset.get_times(), dataset.get_states()

########################################################################################################################

if verbose:
    print("- The network is being divided into training and prediction sets for the future!")

# Firstly, the network will be split into two part
split_ratio = 1.0 - prediction_ratio

# Get the split time
split_time = int(min_time + split_ratio * (max_time - min_time))
first_half_pairs, first_half_times, first_half_states = [], [], []
second_half_pairs, second_half_times, second_half_states = [], [], []
for i in data_dict.keys():
    for j in data_dict[i].keys():

        first_half_pairs.append((i, j))
        first_half_times.append([])
        first_half_states.append([])

        second_half_pairs.append((i, j))
        second_half_times.append([])
        second_half_states.append([])

        for time, state in data_dict[i][j]:

            # Split the pair events into two parts
            if time <= split_time:
                first_half_times[-1].append(time)
                first_half_states[-1].append(state)
            else:
                second_half_times[-1].append(time)
                second_half_states[-1].append(state)

        # Order the event times and states with respect to the time in the first and second half
        if len(first_half_times[-1]) != 0:
            first_half_times[-1], first_half_states[-1] = zip(*sorted(zip(first_half_times[-1], first_half_states[-1])))
        if len(second_half_times[-1]) != 0:
            second_half_times[-1], second_half_states[-1] = zip(*sorted(zip(second_half_times[-1], second_half_states[-1])))

        # If the first half pair does not have any event but the second half contains any event,
        # then add a zero event to the first half
        if len(first_half_times[-1]) == 0 and len(second_half_times[-1]) != 0:
            first_half_times[-1].append(min_time)
            first_half_states[-1].append(0)

        # If the first half pair has any event, then get the last event time and state
        if len(first_half_times[-1]) != 0: #and len(second_half_times[-1]) == 0:
            # first_half_last_time = first_half_times[-1][-1]
            first_half_last_state = first_half_states[-1][-1]
            # raise ValueError(type(second_half_states[-1]), type(second_half_times[-1]))
            second_half_times[-1] = [split_time, ] + list(second_half_times[-1])
            second_half_states[-1] = [first_half_last_state, ] + list(second_half_states[-1])

# Construct a static graph from the non-zero state links in the training set
first_half_graph = nx.DiGraph() if directed else nx.Graph()
first_half_graph.add_nodes_from(range(nodes_num))
first_half_graph.add_edges_from(
    [pair for pair, pair_states in zip(first_half_pairs, first_half_states) if 1 in pair_states or -1 in pair_states]
)

if verbose:
    print(f"\t+ The static version of the first half graph has {first_half_graph.number_of_nodes()} nodes.")
    print(f"\t+ The static version of the first half has {first_half_graph.number_of_edges()} pairs.")
    print(f"\t+ The first half has {sum(map(len, first_half_times))} events.")
    print(f"\t+ The second half has {len(torch.unique(torch.asarray(second_half_pairs)))} unique nodes.")
    print(f"\t+ The second half has {len(second_half_pairs)} pairs.")
    print(f"\t+ The second half has {sum(map(len, first_half_times))} events.")

# If there are any nodes, which do not have any events during the first half of the timeline,
# the graph must be relabeled and these nodes must be removed from the testing set as well.
newlabel = None
if len(list(nx.isolates(first_half_graph))) != 0:

    isolated_nodes = list(nx.isolates(first_half_graph))
    if verbose:
        print(f"\t\t+ The first half graph has {len(isolated_nodes)} isolated nodes.")

    # Remove the isolated nodes from the first half pairs/events/states lists
    idx, count = 0, 0
    while idx < len(first_half_pairs):
        i, j = first_half_pairs[idx]
        if i in isolated_nodes or j in isolated_nodes:
            first_half_pairs.pop(idx)
            first_half_times.pop(idx)
            first_half_states.pop(idx)
            count += 1
        else:
            idx += 1

    # Remove the isolated nodes from the networkx graph
    first_half_graph.remove_nodes_from(isolated_nodes)

    # Remove the isolated nodes from the second half pairs/events/states lists
    idx, count = 0, 0
    while idx < len(second_half_pairs):
        i, j = second_half_pairs[idx]
        if i in isolated_nodes or j in isolated_nodes:
            second_half_pairs.pop(idx)
            second_half_times.pop(idx)
            second_half_states.pop(idx)
            count += 1
        else:
            idx += 1

    if verbose:
        print(f"\t\t+ {count} pairs have been removed from the first half set.")
        print(f"\t\t+ The prediction set has currently {len(torch.unique(torch.asarray(second_half_pairs)))} nodes.")
        print(f"\t\t+ The prediction set has currently {len(second_half_pairs)} pairs having at least one events.")

    # Set the number of nodes
    nodes_num = first_half_graph.number_of_nodes()

    if verbose:
        print(f"\t+ Nodes are being relabeled.")

    # Relabel nodes in the first half set
    newlabel = {node: idx for idx, node in enumerate(first_half_graph.nodes())}
    for idx, pair in enumerate(first_half_pairs):
        first_half_pairs[idx] = (newlabel[pair[0]], newlabel[pair[1]])

    # Relabel nodes in the second half set
    for idx, pair in enumerate(second_half_pairs):
        second_half_pairs[idx] = (newlabel[pair[0]], newlabel[pair[1]])

    # Relabel nodes in the networkx object
    first_half_graph = nx.relabel_nodes(G=first_half_graph, mapping=newlabel)

    if verbose:
        print(f"\t\t+ Completed.")

# Construct the first half dataset
first_half_dataset = Dataset(
    nodes_num=nodes_num, directed=directed, signed=signed,
    edges=torch.repeat_interleave(
        torch.as_tensor(first_half_pairs, dtype=torch.long).T,
        repeats=torch.as_tensor(list(map(len, first_half_times)), dtype=torch.long), dim=1
    ),
    edge_times=torch.as_tensor([t for pair_times in first_half_times for t in pair_times], dtype=torch.long),
    edge_weights=torch.as_tensor([s for pair_states in first_half_states for s in pair_states]).to(torch.long),
)
# Update the min and max times
min_time, max_time = int(first_half_dataset.get_init_time()), int(first_half_dataset.get_last_time())
########################################################################################################################

if verbose:
    print("- Sampling processes for the validation and completion pairs have just started!")

# Get the first half graph data dict
first_half_data_dict = first_half_dataset.get_data_dict(weights=True)

# Sample the validation and completion pairs
all_possible_pair_num = nodes_num * (nodes_num - 1)
if not directed:
    all_possible_pair_num = all_possible_pair_num // 2

validation_size = int(all_possible_pair_num * validation_ratio)
completion_size = int(all_possible_pair_num * completion_ratio)
total_sample_size = validation_size + completion_size


# Construct pair indices
all_pos_pairs = torch.as_tensor(list(utils.pair_iter(n=nodes_num, is_directed=directed)), dtype=torch.long).T
perm = torch.randperm(all_possible_pair_num)
all_pos_pairs = all_pos_pairs[:, perm]

# Sample node pairs such that each node in the residual has at least one event
sampled_pairs = []
for k, pair in enumerate(all_pos_pairs.T):
    i, j = pair[0].item(), pair[1].item()

    if first_half_graph.has_edge(i, j):
        first_half_graph.remove_edge(i, j)

        pair_events = [t for t, _ in first_half_data_dict[i][j]]
        # Check if the residual graph has any isolated nodes (we don't check if the residual graph is connected or not)
        if min_time == min(pair_events) and max_time == max(pair_events):
            first_half_graph.add_edge(i, j)
        elif len(list(nx.isolates(first_half_graph))) != 0:
            first_half_graph.add_edge(i, j)
        else:
            sampled_pairs.append((i, j))
    else:
        sampled_pairs.append((i, j))

    if len(sampled_pairs) == total_sample_size:
        break

assert len(sampled_pairs) == total_sample_size, "Enough number of sample pairs couldn't be found!"

# Set the completion and validation pairs
validation_pairs, completion_pairs = [], []
if validation_size:
    validation_pairs = sampled_pairs[:validation_size]
if completion_size:
    completion_pairs = sampled_pairs[validation_size:]

# Set the completion and validation events
validation_events = [
    [e for e, _ in first_half_data_dict[pair[0]][pair[1]]]
    if pair[0] in first_half_data_dict and pair[1] in first_half_data_dict[pair[0]]
    else [min_time]
    for pair in validation_pairs
]
validation_states = [
    [s for _, s in first_half_data_dict[pair[0]][pair[1]]]
    if pair[0] in first_half_data_dict and pair[1] in first_half_data_dict[pair[0]]
    else [0]
    for pair in validation_pairs
]
completion_events = [
    [e for e, _ in first_half_data_dict[pair[0]][pair[1]]]
    if pair[0] in first_half_data_dict and pair[1] in first_half_data_dict[pair[0]]
    else [min_time]
    for pair in completion_pairs
]
completion_states = [
    [s for _, s in first_half_data_dict[pair[0]][pair[1]]]
    if pair[0] in first_half_data_dict and pair[1] in first_half_data_dict[pair[0]]
    else [0] for pair in completion_pairs
]

# Construct the residual pairs and events
# Since we always checked in the previous process, every node has at least one event
residual_pairs = first_half_pairs.copy()
residual_times = first_half_times.copy()
residual_states = first_half_states.copy()

# Remove the validation pairs from the residual pairs
if validation_size:
    validation_pair_indices = [
        matIdx2flatIdx(
            i=torch.as_tensor([pair[0]], dtype=torch.long), j=torch.as_tensor([pair[1]], dtype=torch.long),
            n=nodes_num, is_directed=directed
        ).item()
        for pair in validation_pairs
    ]

    idx = 0
    while idx < len(residual_pairs):
        pair = residual_pairs[idx]
        pair_idx = matIdx2flatIdx(
            i=torch.as_tensor([pair[0]], dtype=torch.long), j=torch.as_tensor([pair[1]], dtype=torch.long),
            n=nodes_num, is_directed=directed
        ).item()
        if pair_idx in validation_pair_indices:
            residual_pairs.pop(idx)
            residual_times.pop(idx)
            residual_states.pop(idx)
        else:
            idx += 1

# Remove the completion pairs from the residual pairs
if completion_size:
    completion_pair_indices = [
        matIdx2flatIdx(
            i=torch.as_tensor([pair[0]], dtype=torch.long), j=torch.as_tensor([pair[1]], dtype=torch.long),
            n=nodes_num, is_directed=directed
        ).item()
        for pair in completion_pairs
    ]

    idx = 0
    while idx < len(residual_pairs):
        pair = residual_pairs[idx]
        pair_idx = matIdx2flatIdx(
            i=torch.as_tensor([pair[0]], dtype=torch.long), j=torch.as_tensor([pair[1]], dtype=torch.long),
            n=nodes_num, is_directed=directed
        ).item()
        if pair_idx in completion_pair_indices:
            residual_pairs.pop(idx)
            residual_times.pop(idx)
            residual_states.pop(idx)
        else:
            idx += 1

# Get the minimum and maximum of the residual times
residual_times_min = min([min(times) for times in residual_times])
residual_times_max = max([max(times) for times in residual_times])

# The minimum time of validation or completion sets should not be smaller than the minimum time of the residual set
if validation_size:
    for pair, pair_times, pair_states in zip(validation_pairs, validation_events, validation_states):

        min_idx = 0
        for time in pair_times:
            if time < residual_times_min:
                min_idx += 1

        max_idx = len(pair_times)
        for time in reversed(pair_times):
            if time > residual_times_max:
                max_idx -= 1




if verbose:
    print(f"\t+ Validation set has {validation_size} pairs.")
    print(f"\t\t+ {sum(map(len, validation_events))} validation pairs contain at least one event. ")

    print(f"\t+ Completion set has {completion_size} pairs.")
    print(f"\t\t+ {sum(map(len, completion_events))} validation pairs have at least one event. ")

    print(f"\t+ Residual network has {len(residual_pairs)} event pairs.")

########################################################################################################################

if verbose:
    print("- The files are being written...")

# Save the training pair and events
first_half_path = os.path.join(output_folder, "first_half.edges")
triplets = [
    (pair[0], pair[1], t, s)
    for pair, pair_times, pair_states in zip(first_half_pairs, first_half_times, first_half_states)
    for t, s in zip(pair_times, pair_states)
]
with open(first_half_path, 'w') as f:
    for element in sorted(triplets, key=lambda x: (x[2], x[0], x[1], x[3])):
        f.write(f"{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}\n")

# Save the residual pair and events
residual_path = os.path.join(output_folder, "residual.edges")
triplets = [
    (pair[0], pair[1], t, s)
    for pair, pair_times, pair_states in zip(residual_pairs, residual_times, residual_states)
    for t, s in zip(pair_times, pair_states)
]
with open(residual_path, 'w') as f:
    for element in sorted(triplets, key=lambda x: (x[2], x[0], x[1], x[3])):
        f.write(f"{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}\n")

# Save the completion pair and events
completion_path = os.path.join(output_folder, "completion.edges")
triplets = [
    (pair[0], pair[1], t, s)
    for pair, pair_times, pair_states in zip(completion_pairs, completion_events, completion_states)
    for t, s in zip(pair_times, pair_states)
]
with open(completion_path, 'w') as f:
    for element in sorted(triplets, key=lambda x: (x[2], x[0], x[1], x[3])):
        f.write(f"{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}\n")

# Save the validation pair and events
validation_path = os.path.join(output_folder, "validation.edges")
triplets = [
    (pair[0], pair[1], t, s)
    for pair, pair_times, pair_states in zip(validation_pairs, validation_events, validation_states)
    for t, s in zip(pair_times, pair_states)
]
with open(validation_path, 'w') as f:
    for element in sorted(triplets, key=lambda x: (x[2], x[0], x[1], x[3])):
        f.write(f"{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}\n")

# Save the prediction pairs
prediction_path = os.path.join(output_folder, "prediction.edges")
triplets = [
    (pair[0], pair[1], t, s)
    for pair, pair_times, pair_states in zip(second_half_pairs, second_half_times, second_half_states)
    for t, s in zip(pair_times, pair_states)
]
with open(prediction_path, 'w') as f:
    for element in sorted(triplets, key=lambda x: (x[2], x[0], x[1], x[3])):
        f.write(f"{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}\n")

if verbose:
    print(f"\t+ Completed.")

########################################################################################################################

# sys.stdout = orig_stdout
# f.close()