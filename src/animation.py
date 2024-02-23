import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


class Animation:

    def __init__(self, rt_s: torch.Tensor, rt_r: torch.Tensor, frame_times: torch.Tensor, data_dict: dict = None,
                 title: bool = True, font_size: int = 16, fig_size: tuple = (12, 10), fps: int = 12,
                 axis: bool = True, padding: int = 0, edge_alpha: float = 0.35, edge_width=1, edge_color='k',
                 node_sizes=100, node_colors='b', color_palette: str = "rocket_r",
                 ):

        # Data properties
        self._nodes_num = rt_s.shape[1]
        self._frames_num = rt_s.shape[0]

        # Set the latent representations and frame times
        self._rt_s = rt_s
        self._rt_r = rt_r
        self._frame_times = frame_times

        # Other optional parameters
        self._data_dict = data_dict

        # Visual properties
        sns.set_theme(style="ticks")
        self._palette = sns.color_palette(color_palette,  n_colors=self._nodes_num)
        self._title = title
        self._font_size = font_size
        self._fig_size = fig_size
        self._fps = fps
        self._axis = axis
        self._padding = padding
        self._edge_alpha = edge_alpha
        self._edge_width = edge_width
        self._edge_color = edge_color
        self._node_sizes = node_sizes if type(node_sizes) is list else [node_sizes] * self._nodes_num
        self._node_colors = node_colors if type(node_colors) is list else self._palette.as_hex()

    def _render(self, fig, repeat=False):
        global sc, sc_r, ax

        def __set_canvas():
            """
            Set the canvas of the animation
            """

            # Find the minimum and maximum values of the data
            xy_min = self._rt_s.min(dim=0, keepdim=False)[0].min(dim=0, keepdim=False)[0]
            xy_max = self._rt_s.max(dim=0, keepdim=False)[0].max(dim=0, keepdim=False)[0]
            if self._rt_r is not None:
                xy_min_r = self._rt_r.min(dim=0, keepdims=False)[0].min(dim=0, keepdims=False)[0]
                xy_max_r = self._rt_r.max(dim=0, keepdims=False)[0].max(dim=0, keepdims=False)[0]

                xy_min = torch.min(xy_min, xy_min_r)
                xy_max = torch.max(xy_max, xy_max_r)

            # Set the canvas sizes and limits with the additional padding value
            ax.set_xlim([xy_min[0] - self._padding, xy_max[0] + self._padding])
            ax.set_ylim([xy_min[1] - self._padding, xy_max[1] + self._padding])

        def __init_func():
            global sc, sc_r, ax

            # Set the figure
            sc = ax.scatter(
                [0]*self._nodes_num, [0]*self._nodes_num,
                s=self._node_sizes, c=self._node_colors,
                linewidths=self._edge_width, edgecolors=self._edge_color
            )
            sc_r = None
            if self._rt_r is not None:
                sc_r = ax.scatter(
                    [0]*self._nodes_num, [0]*self._nodes_num,
                    s=self._node_sizes, c=self._node_colors,
                    linewidths=self._edge_width, edgecolors=self._edge_color,
                    marker='>'
                )

            # Set the canvas
            __set_canvas()

        def __func(frame_idx):
            global sc, sc_r, ax

            # Clear the previous edges
            for line in list(ax.lines):
                ax.lines.remove(line)

            # Get the current frame time
            current_frame_time = self._frame_times[frame_idx]

            # The title will be the current frame time
            if self._title:
                ax.set_title("Time (t={:0.2f})".format(current_frame_time, fontsize=self._font_size))

            # Plot the nodes
            sc.set_offsets(self._rt_s[frame_idx, :, :])
            if self._rt_r is not None:
                sc_r.set_offsets(self._rt_r[frame_idx, :, :])

            # Plot the edges if the dataset is given
            if self._data_dict is not None and self._edge_width > 0 and self._edge_alpha > 0:

                for i in self._data_dict.keys():
                    for j in self._data_dict[i].keys():

                        # Get the state of the largest event time which is smaller than the current frame time
                        index = torch.bucketize(
                            input=torch.as_tensor([current_frame_time]),
                            boundaries=torch.as_tensor([t for t, _ in self._data_dict[i][j]]), right=False
                        )
                        # If the frame time is smaller than the smallest event time, then the state is 0 by assumption
                        if index == 0:
                            ij_frame_state = 0
                            ij_diff = current_frame_time
                        else:
                            ij_frame_state = self._data_dict[i][j][index-1][1]
                            ij_diff = current_frame_time - self._data_dict[i][j][index-1][0]

                        if ij_frame_state > 0:
                            ax.plot(
                                [self._rt_s[frame_idx, i, 0], self._rt_s[frame_idx, j, 0]],
                                [self._rt_s[frame_idx, i, 1], self._rt_s[frame_idx, j, 1]],
                                color=self._edge_color,
                                alpha=self._edge_alpha,
                            )
                        else:
                            ax.plot(
                                [self._rt_s[frame_idx, i, 0], self._rt_s[frame_idx, j, 0]],
                                [self._rt_s[frame_idx, i, 1], self._rt_s[frame_idx, j, 1]],
                                color=self._edge_color,
                                alpha=0,
                            )

        anim = animation.FuncAnimation(
            fig=fig, init_func=__init_func, func=__func, frames=self._frames_num, interval=200, repeat=repeat
        )
        
        return anim

    def save(self, filepath, format="mp4"):
        global sc, sc_r, ax

        # Set the figure
        fig, ax = plt.subplots(figsize=self._fig_size, frameon=True)
        # Remove the axis
        if not self._axis:
            ax.set_axis_off()
            fig.subplots_adjust(left=0, bottom=0, right=1, top=0.95)

        # Runt the animation
        self._anim = self._render(fig)

        if format == "mp4":
            writer = animation.FFMpegWriter(fps=self._fps)
        elif format == "gif":
            writer = animation.PillowWriter(fps=self._fps)
        else:
            raise ValueError("Invalid format!")

        self._anim.save(filepath, writer)