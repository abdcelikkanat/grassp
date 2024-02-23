import torch
from argparse import ArgumentParser, RawTextHelpFormatter
from src.learning import LearningModel
from src.animation import Animation
from src.dataset import Dataset


def parse_arguments():
    parser = ArgumentParser(description="Examples: \n",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--edges', type=str, required=True, help='Path of the edge list file'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path of the model'
    )
    parser.add_argument(
        '--anim_path', type=str, required=True, default="", help='Animation path'
    )
    parser.add_argument('--title', action='store_true', help='Enable the title of the animation')
    parser.add_argument('--no-title', dest='title', action='store_false', help='Remove the title of the animation.')
    parser.set_defaults(title=True)
    parser.add_argument(
        '--font_size', type=int, required=False, default=16, help='Font size of the title'
    )
    parser.add_argument(
        '--fig_size', type=list, nargs='+', required=False, default=[12, 10], help='Figure size'
    )
    parser.add_argument(
        '--fps', type=int, required=False, default=12, help='Frame per second for the animation'
    )
    parser.add_argument('--axis', action='store_true', help='Keep the axis of the animation')
    parser.add_argument('--no-axis', dest='axis', action='store_false', help='Remove the axis of the animation.')
    parser.set_defaults(axis=False)
    parser.add_argument(
        '--padding', type=int, required=False, default=0, help='Padding for the animation'
    )
    parser.add_argument(
        '--edge_alpha', type=float, required=False, default=0.35, help='Alpha value for the edges'
    )
    parser.add_argument(
        '--edge_width', type=float, required=False, default=1., help='Width value for the edges'
    )
    parser.add_argument(
        '--edge_color', type=str, required=False, default='k', help='Color for the edges'
    )
    parser.add_argument(
        '--node_sizes', type=int, required=False, default=100, help='Size value for the nodes'
    )
    parser.add_argument(
        '--node_color', type=str, required=False, default='b', help='Color for the nodes'
    )
    parser.add_argument(
        '--color_palette', type=str, required=False, default='rocket_r', help='Color palette'
    )
    parser.add_argument(
        '--frames_num', type=int, required=False, default=100, help='Number of frames'
    )
    parser.add_argument(
        '--format', type=str, required=False, choices=["mp4", "gif"], default="mp4", help='Animation file format'
    )
    parser.add_argument('--verbose', action='store_true', help='Verbose mode (default)')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='Non-verbose mode')
    parser.set_defaults(verbose=True)

    return parser.parse_args()


def process(parser):

    # Read the arguments
    edges_path = parser.edges
    model_path = parser.model_path
    anim_path = parser.anim_path

    # Load the dataset
    dataset = Dataset()
    dataset.read_edge_list(edges_path)
    nodes_num = dataset.get_nodes_num()

    # Load the model
    kwargs, lm_state = torch.load(model_path, map_location=torch.device("cpu"))
    lm = LearningModel(**kwargs, device=torch.device("cpu"))
    lm.load_state_dict(lm_state)
    # Update the model parameters
    kwargs['device'] = torch.device("cpu")
    kwargs['verbose'] = parser.verbose

    # Get the dimension size and directed flag
    dim = kwargs['dim']
    directed = kwargs['directed']
    signed = kwargs['signed']

    # Get the initial and last time
    init_time = dataset.get_init_time()
    last_time = dataset.get_last_time()

    # Define the frame times and nodes
    frame_times = torch.linspace(init_time, last_time, steps=parser.frames_num)
    nodes = torch.arange(nodes_num).unsqueeze(1).expand(nodes_num, parser.frames_num)
    time_list = frame_times.unsqueeze(0).expand(nodes_num, parser.frames_num)

    rt_s = lm.get_rt_s(
        time_list=(time_list.flatten() - init_time ) / float(last_time - init_time), nodes=nodes.flatten()
    ).reshape(nodes_num, parser.frames_num, dim).transpose(0, 1)
    if directed:
        rt_r = lm.get_rt_r(
            time_list=(time_list.flatten() - init_time ) / float(last_time - init_time), nodes=nodes.flatten()
        ).reshape(nodes_num, parser.frames_num, dim).transpose(0, 1)
    else:
        rt_r = None

    anim = Animation(
        rt_s=rt_s, rt_r=rt_r, frame_times=frame_times.detach().numpy(), data_dict=dataset.get_data_dict(weights=True),
        title=parser.title, font_size=parser.font_size, fig_size=parser.fig_size, fps=parser.fps,
        axis=parser.axis, padding=parser.padding,
        edge_alpha=parser.edge_alpha, edge_width=parser.edge_width, edge_color=parser.edge_color,
        node_sizes=parser.node_sizes, node_colors=parser.node_color, color_palette=parser.color_palette
    )
    anim.save(anim_path, format=parser.format)


if __name__ == "__main__":
    args = parse_arguments()
    process(args)