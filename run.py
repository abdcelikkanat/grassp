import torch
from src.dataset import Dataset
from src.learning import LearningModel
from argparse import ArgumentParser, RawTextHelpFormatter


def parse_arguments():
    """
    Parse the command line arguments
    """
    parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--edges', type=str, required=True, help='Path of the edge list file'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path of the model'
    )
    parser.add_argument(
        '--log_path', type=str, required=False, default=None, help='Path of the log file'
    )
    parser.add_argument(
        '--bins_num', type=int, default=100, required=False, help='Number of bins'
    )
    parser.add_argument(
        '--dim', type=int, default=2, required=False, help='Dimension size'
    )
    parser.add_argument(
        '--prior_lambda', type=float, default=1e10, required=False, help='Scaling coefficient of the covariance'
    )
    parser.add_argument(
        '--epoch_num', type=int, default=300, required=False, help='Number of epochs'
    )
    parser.add_argument(
        '--spe', type=int, default=10, required=False, help='Number of steps per epoch'
    )
    parser.add_argument(
        '--batch_size', type=int, default=100, required=False, help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=0.1, required=False, help='Learning rate'
    )
    parser.add_argument(
        '--bipartite', action='store_true', help='Bipartite mode (default)'
    )
    parser.add_argument('--no-bipartite', dest='bipartite', action='store_false', help='Non-bipartite mode')
    parser.set_defaults(bipartite=False)
    parser.add_argument(
        '--device', type=str, default="cuda", required=False, help='Device'
    )
    parser.add_argument(
        '--seed', type=int, default=19, required=False, help='Seed value to control the randomization'
    )
    parser.add_argument('--verbose', action='store_true', help='Verbose mode (default)')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='Non-verbose mode')
    parser.set_defaults(verbose=True)

    return parser.parse_args()


def process(parser):

    # Load the dataset
    dataset = Dataset(bipartite=parser.bipartite)
    dataset.read_edge_list(parser.edges)

    # Get the number of nodes
    nodes_num = dataset.get_nodes_num()
    # Check if the network is directed
    directed = dataset.is_directed()
    # Check if the network is signed
    signed = dataset.is_signed()

    # Print the information of the dataset
    if parser.verbose:
        dataset.print_info()

    # Define the learning model
    lm = LearningModel(
        nodes_num=nodes_num, directed=directed, signed=signed, bins_num=parser.bins_num, dim=parser.dim,
        prior_lambda=parser.prior_lambda, device=parser.device, verbose=parser.verbose, seed=parser.seed,
    )

    # Learn the hyper-parameters
    lm.learn(
        dataset=dataset, log_file_path=parser.log_path,
        lr=parser.lr, batch_size=parser.batch_size, epoch_num=parser.epoch_num, steps_per_epoch=parser.spe,
    )
    # Save the model
    lm.save(path=parser.model_path)


if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = parse_arguments()
    process(parser)

