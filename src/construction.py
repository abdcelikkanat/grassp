import torch
import utils
from src.ssp import SequentialSurviveProcess
from src.base import BaseModel
from src.dataset import Dataset
from src.animation import Animation


class ConstructionModel(torch.nn.Module):

    def __init__(self, cluster_sizes: list, bins_num: int, dim: int, directed: bool, prior_lambda: float,
                 prior_sigma_s: float, prior_sigma_r: float, beta_s: list, beta_r: list,
                 prior_B_x0_s: float, prior_B_x0_r: float, prior_B_ls_s: float, prior_B_ls_r: float,
                 max_time: int = 1000, min_time_diff: int = 10,
                 device: torch.device = "cpu", verbose: bool = False, seed: int = 0):

        super(ConstructionModel, self).__init__()

        # Set the model parameters
        self.__nodes_num = sum(cluster_sizes)
        self.__cluster_sizes = cluster_sizes
        self.__k = len(cluster_sizes)
        self.__dim = dim
        self.__bins_num = bins_num
        self.__directed = directed
        self.__signed = False

        # Sample the bias tensors
        self.__beta_s = torch.as_tensor(beta_s, dtype=torch.float, device=device)
        self.__beta_r = None

        # Set the prior hyper-parameters
        self.__prior_lambda = prior_lambda
        self.__prior_sigma_s = torch.as_tensor(prior_sigma_s, dtype=torch.float, device=device)
        self.__prior_sigma_r = None
        self.__prior_B_x0_s = torch.as_tensor([prior_B_x0_s], dtype=torch.float, device=device)
        self.__prior_B_x0_r = None
        self.__prior_B_ls_s = torch.as_tensor(prior_B_ls_s, dtype=torch.float, device=device)
        self.__prior_B_ls_r = None
        if self.__directed:
            self.__beta_r = torch.as_tensor(beta_r, dtype=torch.float, device=device)
            self.__prior_sigma_r = torch.as_tensor(prior_sigma_r, dtype=torch.float, device=device)
            self.__prior_B_ls_r = torch.as_tensor(prior_B_ls_r, dtype=torch.float, device=device)
            self.__prior_B_x0_r = torch.as_tensor([prior_B_x0_r], dtype=torch.float, device=device)

        # Construct the Q matrix for the node matrix C, (nodes)
        self.__prior_C_Q_s = -utils.INF * torch.ones(size=(self.__nodes_num, self.__k), dtype=torch.float)
        for k in range(self.__k):
            self.__prior_C_Q_s[sum(self.__cluster_sizes[:k]):sum(self.__cluster_sizes[:k + 1]), k] = 0
        self.__prior_C_Q_r = self.__prior_C_Q_s if self.__directed else None

        # Set max time and min time difference
        self.__max_time = max_time
        self.__min_time_diff = min_time_diff

        # Set the device, verbose and seed
        self.__device = device
        self.__verbose = verbose
        self.__seed = seed

        # Set the seed
        utils.set_seed(self.__seed)

        # Sample a graph
        self.__bm, self.__dataset = self.sample_graph()

    def sample_graph(self):
        """
        Sample a graph
        """

        # Sample the initial position and velocity tensors
        x0_s, v_s, x0_r, v_r = self.sample_x0_and_v()

        # Define the model
        bm = BaseModel(
            x0_s=x0_s, v_s=v_s, beta_s=self.__beta_s,
            directed=self.__directed, prior_lambda=self.__prior_lambda,
            x0_r=x0_r, v_r=v_r, beta_r=self.__beta_r,
            device=self.__device, verbose=self.__verbose, seed=self.__seed
        )

        # Sample the events and states for each pair
        edges, edge_times, edge_states = self.sample_events(bm=bm)

        # edges = torch.as_tensor([[0, 0], [1, 1.]])
        # edge_times = torch.as_tensor([0, 1000])
        # edge_states = torch.as_tensor([0, 1])

        # Map the event times sampled from [0,1] to [0, T]
        edge_times = (self.__max_time * torch.as_tensor(edge_times, device=self.__device)).to(torch.long).tolist()

        # Remove the links/events/states if they are too close
        data_dict = dict()
        for idx, (edge, time, state) in enumerate(zip(edges, edge_times, edge_states)):

            u, v = edge[0], edge[1]
            if self.__directed is False:
                u, v = v, u

            if u in data_dict:
                if v in data_dict[u]:
                    data_dict[u][v].append((idx, time, state))
                    # Check the previous state is different
                    assert data_dict[u][v][-1][2] != data_dict[u][v][-2][2], "The consecutive states are the same!"
                else:
                    data_dict[u][v] = [(idx, time, state)]
            else:
                data_dict[u] = {v: [(idx, time, state)]}

        remove_indices = []
        for u in data_dict.keys():
            for v in data_dict[u].keys():
                data_dict[u][v] = sorted(data_dict[u][v], key=lambda item: item[1])
                for k in range(1, len(data_dict[u][v])):
                    if data_dict[u][v][k][1] - data_dict[u][v][k-1][1] < self.__min_time_diff:
                        remove_indices.append(data_dict[u][v][k][0])
                        remove_indices.append(data_dict[u][v][k-1][0])

        for index in sorted(list(set(remove_indices)), reverse=True):
            del edges[index]
            del edge_times[index]
            del edge_states[index]

        # Construct the dataset
        dataset = Dataset(
            nodes_num=self.__nodes_num,
            edges=torch.as_tensor(edges, dtype=torch.long, device=self.__device).T,
            edge_times=torch.as_tensor(edge_times, device=self.__device).to(torch.long),
            edge_weights=torch.as_tensor(edge_states, device=self.__device, dtype=torch.long),
            directed=self.__directed, signed=self.__signed
        )

        return bm, dataset

    def sample_x0_and_v(self):
        """
        Sample the initial position and velocity of the nodes
        For directed graphs, B matrix (bins) are only different
        """

        # Define the bin centers
        bin_centers = torch.arange(0.5/self.__bins_num, 1.0, 1.0/self.__bins_num, device=self.__device).unsqueeze(0)

        # Define the final dimension size
        final_dim = (self.__bins_num+1) * self.__nodes_num * self.__dim

        # Construct the factor of B matrix (bins)
        B_factor_s = torch.linalg.cholesky(
            utils.EPS*torch.eye(self.__bins_num+1, device=self.__device) + torch.block_diag(
                self.__prior_B_x0_s**2,
                torch.exp(-((bin_centers - bin_centers.T)**2 / (2.0*(self.__prior_B_ls_s**2))))
            )
        )
        # Construct the factor of C matrix (nodes)
        C_factor_s = torch.softmax(self.__prior_C_Q_s, dim=1)
        # Get the factor of D matrix, (dimension)
        D_factor_s = torch.eye(self.__dim, device=self.__device)

        # Sample from the low rank multivariate normal distribution
        # covariance_matrix = cov_factor @ cov_factor.T + cov_diag
        cov_factor_s = self.__prior_lambda * torch.kron(B_factor_s.contiguous(), torch.kron(C_factor_s, D_factor_s))
        cov_diag_s = (self.__prior_lambda**2) * (self.__prior_sigma_s**2) * \
                     torch.ones(size=(final_dim, ), dtype=torch.float, device=self.__device)

        # Sample from the low rank multivariate normal distribution
        sample_s = torch.distributions.LowRankMultivariateNormal(
            loc=torch.zeros(size=(final_dim,)),
            cov_factor=cov_factor_s,
            cov_diag=cov_diag_s
        ).sample().reshape(shape=(self.__bins_num+1, self.__nodes_num, self.__dim))

        # Split the tensor into x0 and v
        x0_s, v_s = torch.split(sample_s, [1, self.__bins_num])
        x0_s = x0_s.squeeze(0)

        if self.__directed:

            # Construct the factor of B matrix (bins)
            B_factor_r = torch.linalg.cholesky(
                utils.EPS*torch.eye(self.__bins_num+1, device=self.__device) + torch.block_diag(
                    torch.sigmoid(self.__prior_B_x0_logit_c_r)**2,
                    torch.exp(-((bin_centers - bin_centers.T)**2 / (2.0*(self.__prior_B_ls_r**2))))
                )
            )
            # Construct the factor of C matrix (nodes)
            C_factor_r = torch.softmax(self.__prior_C_Q_r, dim=1)
            # Get the factor of D matrix, (dimension)
            D_factor_r = torch.eye(self.__dim, device=self.__device)

            # Sample from the low rank multivariate normal distribution
            # covariance_matrix = cov_factor @ cov_factor.T + cov_diag
            cov_factor_r = self.__prior_lambda * torch.kron(B_factor_r.contiguous(), torch.kron(C_factor_r, D_factor_r))
            cov_diag_r = (self.__prior_lambda**2) * (self.__prior_sigma_r**2) * \
                         torch.ones(size=(final_dim, ), dtype=torch.float, device=self.__device)

            # Sample from the low rank multivariate normal distribution
            sample_r = torch.distributions.LowRankMultivariateNormal(
                loc=torch.zeros(size=(final_dim,)),
                cov_factor=cov_factor_r,
                cov_diag=cov_diag_r
            ).sample().reshape(shape=(self.__bins_num+1, self.__nodes_num, self.__dim))

            # Split the tensor into x0 and v
            x0_r, v_r = torch.split(sample_r, [1, self.__bins_num])
            x0_r = x0_r.squeeze(0)

        else:
            x0_r, v_r = None, None

        return utils.standardize(x0_s), utils.standardize(v_s), utils.standardize(x0_r), utils.standardize(v_r)

    def sample_events(self, bm: BaseModel) -> tuple[list, list, list]:
        """
        Sample the events from the Hawkes process
        :param bm: BaseModel object
        :return: list of edges, event times and states
        """

        # Get the positions at the beginning of each time bin for every node
        rt_s = bm.get_rt_s(
            time_list=bm.get_bin_bounds()[:-1].repeat(self.__nodes_num),
            nodes=torch.repeat_interleave(torch.arange(self.__nodes_num), repeats=self.__bins_num)
        ).reshape((self.__nodes_num, self.__bins_num,  self.__dim)).transpose(0, 1)
        if self.__directed:
            rt_r = bm.get_rt_r(
                time_list=bm.get_bin_bounds()[:-1].repeat(self.__nodes_num),
                nodes=torch.repeat_interleave(torch.arange(self.__nodes_num), repeats=self.__bins_num)
            ).reshape((self.__nodes_num, self.__bins_num,  self.__dim)).transpose(0, 1)
        else:
            rt_r = rt_s

        edges, edge_times, edge_states = [], [], []
        for i, j in utils.pair_iter(self.__nodes_num, self.__directed):
            # print(i,j)

            # Define the intensity function for each node pair (i,j)
            intensity_func = lambda t, state: bm.get_intensity_at(
                time_list=torch.as_tensor([t], dtype=torch.float, device=self.__device),
                edges=torch.as_tensor([[i], [j]], dtype=torch.long, device=self.__device),
                edge_states=torch.as_tensor([state], dtype=torch.long, device=self.__device)
            ).item()

            # Get the flat index of the pair
            flat_idx = utils.matIdx2flatIdx(
                torch.as_tensor([i], dtype=torch.long), torch.as_tensor([j], dtype=torch.long),
                self.__nodes_num, is_directed=self.__directed
            )

            # Get the critical points
            v_s = bm.get_v_s()
            v_r = bm.get_v_r() if self.__directed else v_s
            critical_points = self.__get_critical_points(
                i=i, j=j, bin_bounds=bm.get_bin_bounds(), rt_s=rt_s, rt_r=rt_r, v_s=v_s, v_r=v_r
            )
            # Simulate the Survive or Die Process
            nhpp_ij = SequentialSurviveProcess(
                lambda_func=intensity_func, initial_state=0, critical_points=critical_points,
                seed=self.__seed + flat_idx
            )
            ij_edge_times, ij_edge_states = nhpp_ij.simulate()

            # Add edges, edge times and edge states to the list
            edges.extend([[i, j]]*len(ij_edge_times))
            edge_times.extend(ij_edge_times)
            edge_states.extend(ij_edge_states)

        return edges, edge_times, edge_states

    def __get_critical_points(self, i: int, j: int, bin_bounds: torch.Tensor,
                              rt_s: torch.Tensor, rt_r: torch.Tensor, v_s: torch.Tensor, v_r: torch.Tensor) -> list:

        # Add the initial time point
        critical_points = []

        for idx in range(self.__bins_num):

            interval_init_time = bin_bounds[idx]
            interval_last_time = bin_bounds[idx+1]

            # Add the initial time point of the interval
            critical_points.append(interval_init_time)

            # Get the differences
            delta_idx_x = rt_s[idx, i, :] - rt_r[idx, j, :]
            delta_idx_v = v_s[idx, i, :] - v_r[idx, j, :]

            # For the model containing only position and velocity
            # Find the point in which the derivative equal to 0
            t = - torch.dot(delta_idx_x, delta_idx_v) / (torch.dot(delta_idx_v, delta_idx_v) + utils.EPS) + interval_init_time

            if interval_init_time < t < interval_last_time:
                critical_points.append(t)

        # Add the last time point
        critical_points.append(bin_bounds[-1])

        return critical_points

    def get_model(self):

        return self.__bm

    def save_model(self, file_path: str):
        """
        Save the model
        :param file_path: the path to the file
        """

        kwargs = {
            'x0_s': self.__bm.get_x0_s(),
            'v_s': self.__bm.get_v_s(),
            'beta_s': self.__bm.get_beta_s(),
            'directed': self.__bm.is_directed(),
            'signed': self.__bm.is_signed(),
            'prior_lambda': self.__bm.get_prior_lambda(),
            'prior_b_sigma_s': self.__bm.get_prior_b_sigma_s(),
            'prior_b_sigma_r': self.__bm.get_prior_b_sigma_r(),
            'x0_r': self.__bm.get_x0_r(),
            'v_r': self.__bm.get_v_r(),
            'beta_r': self.__bm.get_beta_r(),
            'prior_c_sigma_s': self.__bm.get_prior_c_sigma_s(),
            'prior_c_sigma_r': self.__bm.get_prior_c_sigma_r(),
            'device': self.__bm.get_device(),
            'verbose': self.__bm.get_verbose(),
            'seed': self.__bm.get_seed()
        }
        torch.save(kwargs, file_path)

    def write_edges(self, file_path: str):
        """
        Write the edges
        :param file_path: the path to the file
        """

        self.__dataset.write_edges(file_path, weights=True)

    def write_info(self, file_path: str):
        """
        Write the info
        :param file_path: the path to the file
        """

        with open(file_path, 'w') as f:
            f.write(f"Nodes num: {self.__nodes_num}\n")
            f.write(f"Clusters num: {self.__k}\n")
            f.write(f"Dimension: {self.__dim}\n")
            f.write(f"Bins num: {self.__bins_num}\n")
            f.write(f"Directed: {self.__directed}\n")
            f.write(f"Signed: {self.__signed}\n")
            f.write(f"Prior lambda: {self.__prior_lambda}\n")
            f.write(f"Prior sigma_s: {self.__prior_sigma_s}\n")
            f.write(f"Prior sigma_r: {self.__prior_sigma_r}\n")
            f.write(f"Prior B_x0_s: {self.__prior_B_x0_s}\n")
            f.write(f"Prior B_x0_r: {self.__prior_B_x0_r}\n")
            f.write(f"Prior B_ls_s: {self.__prior_B_ls_s}\n")
            f.write(f"Prior B_ls_r: {self.__prior_B_ls_r}\n")
            f.write(f"Prior C Q_s:\n{self.__prior_C_Q_s}\n")
            f.write(f"Prior C Q_r:\n{self.__prior_C_Q_r}\n")
            f.write(f"Beta_s:\n{self.__beta_s}\n")
            f.write(f"Beta_r:\n{self.__beta_r}\n")
            f.write(f"Device: {self.__device}\n")
            f.write(f"Verbose: {self.__verbose}\n")
            f.write(f"Seed: {self.__seed}\n")

    def save_animation(self, file_path, frames_num = 100):
        """
        Save the animation
        :param file_path: the path to the file
        """

        # Get the initial and last time
        init_time = self.__dataset.get_init_time()
        last_time = self.__dataset.get_last_time()
        nodes_num = self.__dataset.get_nodes_num()

        # Define the frame times and nodes
        frame_times = torch.linspace(init_time, last_time, steps=frames_num)
        nodes = torch.arange(nodes_num).unsqueeze(1).expand(nodes_num, frames_num)
        time_list = frame_times.unsqueeze(0).expand(nodes_num, frames_num)

        rt_s = self.__bm.get_rt_s(
            time_list=(time_list.flatten() - init_time) / float(last_time - init_time), nodes=nodes.flatten()
        ).reshape(nodes_num, frames_num,  self.__dim).transpose(0, 1)
        rt_r = self.__bm.get_rt_r(
            time_list=(time_list.flatten() - init_time) / float(last_time - init_time), nodes=nodes.flatten()
        ).reshape(nodes_num, frames_num, self.__dim).transpose(0, 1) if self.__directed else None

        anim = Animation(
            rt_s=rt_s, rt_r=rt_r, frame_times=frame_times, axis=False,
            data_dict=self.__dataset.get_data_dict(weights=True),
        )
        anim.save(file_path, format="mp4")
