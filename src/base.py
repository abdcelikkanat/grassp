import utils
import torch


class BaseModel(torch.nn.Module):
    '''
    Description
    '''
    def __init__(self, x0_s: torch.Tensor, v_s: torch.Tensor, beta_s: torch.Tensor,
                 directed: bool = False, signed: bool = False,
                 prior_lambda: float = 1.0, prior_b_sigma_s: torch.Tensor = None, prior_b_sigma_r: torch.Tensor = None,
                 x0_r: torch.Tensor = None, v_r: torch.Tensor = None, beta_r: torch.Tensor = None,
                 prior_c_sigma_s: torch.Tensor = None, prior_c_sigma_r: torch.Tensor = None,
                 device: torch.device = "cpu", verbose: bool = False, seed: int = 19):

        super(BaseModel, self).__init__()

        # Define the constants
        self.__init_time = 0.
        self.__last_time = 1.

        # Set the model parameters
        self.__x0_s = x0_s
        self.__x0_r = x0_r
        self.__v_s = v_s
        self.__v_r = v_r
        self.__beta_s = beta_s
        self.__beta_r = beta_r

        # Set the number of bins, directed, and signed
        self.__bins_num = self.__v_s.shape[0]
        self.__directed = directed
        self.__signed = signed
        # Set the number of nodes, dimension size and bin width
        self.__nodes_num = self.__x0_s.shape[0]
        self.__dim = self.__x0_s.shape[1]

        # Set the model hyperparameters
        self.__prior_lambda = prior_lambda
        self.__prior_b_sigma_s = prior_b_sigma_s
        self.__prior_b_sigma_r = prior_b_sigma_r
        self.__prior_c_sigma_s = prior_c_sigma_s
        self.__prior_c_sigma_r = prior_c_sigma_r

        self.__device = device
        self.__verbose = verbose
        self.__seed = seed
        
        # Set the seed value for reproducibility
        utils.set_seed(self.__seed)

        if self.__verbose:
            print(f"+ Model parameters")
            print(f"\t- The number of nodes: {self.get_nodes_num()}")
            print(f"\t- The dimension size: {self.get_dim()}")
            print(f"\t- The number of bins: {self.get_bins_num()}")
            print(f"\t- The directed graph: {self.is_directed()}")
            print(f"\t- The signed graph: {self.is_signed()}")
            print(f"\t- The device: {self.get_device()}")
            print(f"\t- The prior lambda: {self.get_prior_lambda()}")
            print(f"\t- Seed: {self.__seed}")

    def is_directed(self):
        """
        Returns if the graph is directed or not
        """

        return self.__directed

    def is_signed(self):
        """
        Returns if the graph is signed or not
        """

        return self.__signed

    def get_x0_s(self, standardize=True):
        """
        Returns the initial positions of the sender nodes
        :param standardize: If True, the initial positions are standardized
        """

        if standardize:
            return utils.standardize(self.__x0_s)
        else:
            return self.__x0_s

    def get_x0_r(self, standardize=True):
        """
        Returns the initial positions of the receiver nodes
        :param standardize: If True, the initial positions are standardized
        """

        if standardize:
            return utils.standardize(self.__x0_r)
        else:
            return self.__x0_r

    def get_v_s(self, standardize=True):
        """
        Returns the velocity of the source nodes
        :param standardize: If True, the velocity nodes are standardized
        """

        if standardize:
            return utils.standardize(self.__v_s)
        else:
            return self.__v_s

    def get_v_r(self, standardize=True):
        """
        Returns the velocity of the receiver nodes
        :param standardize: If True, the velocity nodes are standardized
        """

        if standardize:
            return utils.standardize(self.__v_r)
        else:
            return self.__v_r

    def get_beta_s(self):
        """
        Returns the beta parameter of the source nodes
        """

        return self.__beta_s

    def get_beta_r(self):
        """
        Returns the beta parameter of the receiver nodes
        """

        return self.__beta_r

    def get_dim(self):
        """
        Returns the dimension size
        """

        return self.__dim

    def get_nodes_num(self):
        """
        Returns the number of nodes
        """

        return self.__nodes_num

    def get_bins_num(self):
        """
        Returns the number of bins
        """

        return self.__bins_num

    def get_bin_width(self):
        """
        Returns the bin width
        """

        return (self.__last_time - self.__init_time) / float(self.__bins_num)

    def get_seed(self):
        """
        Returns the seed value
        """

        return self.__seed

    def get_verbose(self):
        """
        Returns the verbose parameter
        """

        return self.__verbose

    def get_device(self):
        """
        Returns the device where the model is running
        """

        return self.__device

    def get_prior_lambda(self):
        """
        Returns the scaling factor of covariance of the prior distribution
        """
        return self.__prior_lambda

    def get_prior_b_sigma_s(self):
        """
        Returns the diagonal elements of B matrix for the source nodes of the covariance of the prior distribution
        """
        return self.__prior_b_sigma_s

    def get_prior_b_sigma_r(self):
        """
        Returns the diagonal elements of B matrix for the receiver nodes of the covariance of the prior distribution
        """
        return self.__prior_b_sigma_r

    def get_prior_c_sigma_s(self):
        """
        Returns the diagonal elements of C matrix for the source nodes of the covariance of the prior distribution
        """
        return self.__prior_c_sigma_s

    def get_prior_c_sigma_r(self):
        """
        Returns the diagonal elements of C matrix for the receiver nodes of the covariance of the prior distribution
        """
        return self.__prior_c_sigma_r

    def get_bin_bounds(self):
        """
        Computes the bin bounds of the model
        :return: a vector of shape B+1
        """
        bounds = torch.linspace(
            self.__init_time, self.__last_time, self.get_bins_num()+1, device=self.get_device(), dtype=torch.float
        )

        return bounds

    def get_bin_index(self, time_list: torch.Tensor):
        """
        Computes the bin indices of given times

        :param time_list: a vector of shape L
        :return: an index and residual vectors of of shapes L
        """

        # Compute the bin indices of the given time points
        bin_indices = utils.div(time_list, self.get_bin_width()).type(torch.long)
        # If there is a time equal to the last time, set its bin index to the last bin
        bin_indices[bin_indices == self.get_bins_num()] = self.get_bins_num() - 1

        return bin_indices

    def get_residual(self, time_list: torch.Tensor):
        """
        Computes the residuals of given times

        :param time_list: a vector of shape L
        :return: an index and residual vectors of of shapes L
        """

        # Compute the residual times
        residual_time = utils.remainder(time_list, self.get_bin_width())
        # If there is a time equal to the last time, set its residual time to bin width
        residual_time[time_list == self.__last_time] = self.get_bin_width()

        return residual_time

    def get_rt_s(self, time_list: torch.Tensor, nodes: torch.Tensor, standardize: bool = True) -> torch.Tensor:
        """
        Computes the locations at given times based on the initial position and velocity tensors.
        
        :param time_list: a vector of shape L
        :param nodes: a vector of shape L
        :return: A matrix of shape L X D
        """

        # Compute the bin indices and residul times of the given time points
        bin_indices = self.get_bin_index(time_list=time_list)
        residual_time = self.get_residual(time_list=time_list)

        # Get the initial position and velocity vectors
        x0 = self.get_x0_s(standardize=standardize)
        v = self.get_v_s(standardize=standardize)

        # Compute the displacement until the initial time of the corresponding bins
        cum_displacement = torch.cumsum(torch.cat((x0.unsqueeze(0),self.get_bin_width() * v)), dim=0)
        rt = cum_displacement[bin_indices, nodes, :]
        # Finally, add the displacement on the bin that time points lay on
        rt = rt + residual_time.view(-1, 1)*self.get_v_s(standardize=standardize)[bin_indices, nodes, :]
        return rt

    def get_rt_r(self, time_list: torch.Tensor, nodes: torch.Tensor, standardize: bool = True) -> torch.Tensor:
        """
        Computes the locations at given times based on the initial position and velocity tensors.
        
        :param time_list: a vector of shape L
        :param nodes: a vector of shape L
        :return: A matrix of shape L X D
        """

        # Compute the bin indices and residul times of the given time points
        bin_indices = self.get_bin_index(time_list=time_list)
        residual_time = self.get_residual(time_list=time_list)

        # Get the initial position and velocity vectors
        x0 = self.get_x0_r(standardize=standardize)
        v = self.get_v_r(standardize=standardize)

        # Compute the displacement until the initial time of the corresponding bins
        cum_displacement = torch.cumsum(torch.cat((x0.unsqueeze(0),self.get_bin_width() * v)), dim=0)
        rt = cum_displacement[bin_indices, nodes, :]
        # Finally, add the displacement on the bin that time points lay on
        rt = rt + residual_time.view(-1, 1)*self.get_v_r(standardize=standardize)[bin_indices, nodes, :]
        return rt

    def get_beta_ij(self, pair_states: torch.Tensor) -> torch.Tensor:
        """
        Computes the sum of the beta elements for given pair and states

        :param time_list: a vector of shape L
        :param pairs: a vector of shape 2 x L
        :param states: a vector of shape L
        :return: A vector of shape L
        """

        beta_s = self.get_beta_s()
        if self.is_directed():
            beta_r = self.get_beta_r()
            return beta_s[pair_states] + beta_r[pair_states]
        else:
            return beta_s[pair_states]

    def get_delta_v(self, bin_indices: torch.Tensor, pairs: torch.Tensor, standardize: bool = True) -> torch.Tensor:
        """
        Computes the velocity diffrences for given bin indices and pairs.
        
        :param bin_indices: a vector of shape L
        :param pairs: a vector of shape 2 x L
        :param standardize: a boolean parameter to set the standardization of velocity vectors
        :return: A matrix of shape L X D
        """

        v_s = self.get_v_s(standardize=standardize)
        v_r = self.get_v_r(standardize=standardize) if self.is_directed() else v_s

        return v_s[bin_indices, pairs[0], :] - v_r[bin_indices, pairs[1], :]

    def get_delta_rt(self, time_list: torch.Tensor, pairs: torch.Tensor, standardize: bool = True) -> torch.Tensor:
        """
        Computes the locations at given times based on the initial position and velocity tensors.
        
        :param time_list: a vector of shape L
        :param pairs: a vector of shape 2 x L
        :param standardize: a boolean parameter to set the standardization of the initial position and velocity vectors
        :return: A matrix of shape L X D
        """
        # Compute the bin indices and residul times of the given time points
        bin_indices = self.get_bin_index(time_list=time_list)
        residual_time = self.get_residual(time_list=time_list)

        # Get the initial position and velocity vectors
        x0_s = self.get_x0_s(standardize=standardize)
        v_s = self.get_v_s(standardize=standardize)

        if self.__directed:
            x0_r, v_r = self.get_x0_r(standardize=standardize), self.get_v_r(standardize=standardize)
        else:
            x0_r, v_r = x0_s, v_s

        # Compute the displacements
        cum_displacement_s = torch.cumsum(torch.cat((x0_s.unsqueeze(0), self.get_bin_width() * v_s)),  dim=0)
        cum_displacement_r = torch.cumsum(torch.cat((x0_r.unsqueeze(0), self.get_bin_width() * v_r)),  dim=0)

        # Select the bin indices and nodes
        delta_rt = cum_displacement_s[bin_indices, pairs[0], :] - cum_displacement_r[bin_indices, pairs[1], :]

        # Finally, add the displacement on the bin that time points lay on
        delta_rt += residual_time.view(-1, 1)*(v_s[bin_indices, pairs[0], :]-v_r[bin_indices, pairs[1], :])

        return delta_rt

    def get_log_intensity_at(self, time_list: torch.Tensor, edges: torch.Tensor, edge_states: torch.Tensor) -> torch.Tensor:
        """
        Computes the log of the intenstiy function for given times and pairs

        :param time_list: a vector of shape L
        :param pairs: a vector of shape 2 x L
        :param states: a vector of shape L
        :return: A vector of shape L
        """

        if self.is_signed():

            beta_ij_plus = self.get_beta_ij(
                pair_states=torch.ones(len(edge_states), dtype=torch.long, device=self.get_device())
            )
            beta_ij_neg = self.get_beta_ij(
                pair_states=torch.zeros(len(edge_states), dtype=torch.long, device=self.get_device())
            )

            beta_ij_pm = (1 - (edge_states.absolute() - edge_states) / 2) * beta_ij_plus + \
                         (1 - (edge_states.absolute() + edge_states) / 2) * beta_ij_neg

            intensity = beta_ij_pm + edge_states * torch.norm(
                self.get_delta_rt(time_list=time_list, pairs=edges), p=2, dim=1, keepdim=False
            )**2

        else:

            beta_ij = self.get_beta_ij(pair_states=edge_states)

            intensity = beta_ij + (2.*edge_states-1.) * torch.norm(
                self.get_delta_rt(time_list=time_list, pairs=edges), p=2, dim=1, keepdim=False
            ) ** 2

        return intensity

    def get_intensity_at(self, time_list: torch.Tensor, edges: torch.Tensor, edge_states: torch.Tensor) -> torch.Tensor:
        """
        Computes the intenstiy function for given times and pairs

        :param time_list: a vector of shape L
        :param pairs: a vector of shape 2 x L
        :return: A vector of shape L
        """
        return torch.exp(self.get_log_intensity_at(time_list=time_list, edges=edges, edge_states=edge_states))

    def get_intensity_integral(self, time_list: torch.Tensor, pairs: torch.Tensor, delta_t: torch.Tensor,
                               states: torch.Tensor, standardize: bool = True) -> torch.Tensor:
        """
        Computes the negative log-likelihood function of the model

        :param time_list: a vector of shape L
        :param pairs: a matrix of shape 2 x L
        :param delta_t: a vector of shape L
        :param states: a vector of shape L
        :return:
        """

        _UPPER_BOUND = 8

        # Compute the bin indices and residul times of the given time points
        bin_indices = self.get_bin_index(time_list=time_list)

        # Get the position and velocity differences
        delta_r = self.get_delta_rt(time_list=time_list, pairs=pairs, standardize=standardize)
        delta_v = self.get_delta_v(bin_indices=bin_indices, pairs=pairs, standardize=standardize)

        # Compute the beta sums
        beta_ij_plus = self.get_beta_ij(
            pair_states=torch.ones(len(states), dtype=torch.long, device=self.get_device())
        )
        beta_ij_neg = self.get_beta_ij(
            pair_states=torch.zeros(len(states), dtype=torch.long, device=self.get_device())
        )

        norm_delta_r = torch.norm(delta_r, p=2, dim=1, keepdim=False)
        norm_delta_v = torch.norm(delta_v, p=2, dim=1, keepdim=False) + utils.EPS
        inv_norm_delta_v = 1.0 / norm_delta_v
        delta_r_v = (delta_r * delta_v).sum(dim=1, keepdim=False)
        r = delta_r_v * inv_norm_delta_v

        term0 = 0.5 * torch.sqrt(torch.as_tensor(utils.PI, device=self.__device)) * inv_norm_delta_v

        # Because of numerical issues, we need to clamp the ratio vector, r, which is upper bounded by norm_delta_r
        # But it is not enough to bound only norm_delta_r since we don't update the values of delta_r.
        # Therefore, we also need to bound r.
        norm_delta_r = torch.clamp(norm_delta_r, max=_UPPER_BOUND)
        r = torch.clamp(r, min=-_UPPER_BOUND, max=_UPPER_BOUND)

        term1_plus = torch.exp(beta_ij_plus - (r ** 2 - norm_delta_r ** 2))
        term1_neg = torch.exp(beta_ij_neg + (r ** 2 - norm_delta_r ** 2))

        term2_u_plus = utils.erfi_approx(delta_t * norm_delta_v + r)
        term2_l_plus = utils.erfi_approx(r)

        term2_u_neg = torch.erf(delta_t * norm_delta_v + r)
        term2_l_neg = torch.erf(r)

        if self.is_signed():
            output = term0 * (
                    (1 - (states.absolute() - states) / 2) * term1_plus * (term2_u_plus - term2_l_plus) +
                    (1 - (states.absolute() + states) / 2) * term1_neg * (term2_u_neg - term2_l_neg)
            )
        else:
            output = term0 * (
                    states * term1_plus * (term2_u_plus - term2_l_plus) +
                    (1 - states) * term1_neg * (term2_u_neg - term2_l_neg)
            )

        return output

    def get_intensity_integral_for(self, time_list: torch.Tensor, pairs: torch.Tensor, delta_t: torch.Tensor,
                                   states: torch.Tensor, standardize: bool = True) -> torch.Tensor:
        """
        Computes the negative log-likelihood function of the model

        :param time_list: a vector of shape L
        :param pairs: a matrix of shape 2 x L
        :param delta_t: a vector of shape L
        :param states: a vector of shape L
        :return:
        """

        _UPPER_BOUND = 8

        # Before starting, we need to extend the time_list, pairs, and states for the model bin boundaries
        # Find the initial and the last bin indices of the integral interval
        init_bin_idx = utils.div(time_list, self.get_bin_width())
        last_bin_idx = utils.div(time_list + delta_t, self.get_bin_width())
        # Find the number of required bins for each event time
        repeat_counts = (last_bin_idx - init_bin_idx + 1).to(torch.long)

        # FInd out the start and last index of each event time for the extended 1D time_list vector
        repeats_cum_sum = torch.concat((
            torch.zeros(1, dtype=torch.long, device=self.get_device()), torch.cumsum(repeat_counts, dim=0)
        ))
        start_marker, last_marker = repeats_cum_sum[:-1], repeats_cum_sum[1:] - 1

        # Construct a vector for the regions of the corresponding event times of the form [0 1 2 0 1 0 1 2 3 0 1 ...]
        temp = torch.ones(sum(repeat_counts), device=self.get_device(), dtype=torch.long)
        temp[start_marker] = 0  # It has now the form of [0 1 1 0 1 0 1 1 1 0 1 ...]
        temp = temp.cumsum(0)  # It has now the form of [0 1 2 2 3 3 4 5 6 6 7 ...]
        temp = temp - torch.repeat_interleave(temp[start_marker], repeats=repeat_counts)

        # Construct the extended time list vector
        extended_time_list = torch.repeat_interleave(self.get_bin_width() * init_bin_idx, repeats=repeat_counts, dim=0)
        extended_time_list = extended_time_list + temp * self.get_bin_width()
        extended_time_list[start_marker] = time_list

        # Construct the extended delta_t vector
        extended_delta_t = torch.concat((
            extended_time_list[1:]-extended_time_list[:-1], torch.zeros(1, device=self.get_device(), dtype=torch.float)
        ))
        extended_delta_t[last_marker] = time_list + delta_t - extended_time_list[last_marker]

        extended_pairs = torch.repeat_interleave(pairs, repeats=repeat_counts, dim=1)
        extended_states = torch.repeat_interleave(states, repeats=repeat_counts, dim=0)

        # Compute the bin indices and residul times of the given time points
        bin_indices = self.get_bin_index(time_list=extended_time_list)

        # Get the position and velocity differences
        delta_r = self.get_delta_rt(time_list=extended_time_list, pairs=extended_pairs, standardize=standardize)
        delta_v = self.get_delta_v(bin_indices=bin_indices, pairs=extended_pairs, standardize=standardize)

        # Compute the beta sums
        beta_ij_plus = self.get_beta_ij(
            pair_states=torch.ones(len(extended_states), dtype=torch.long, device=self.get_device())
        )
        beta_ij_neg = self.get_beta_ij(
            pair_states=torch.zeros(len(extended_states), dtype=torch.long, device=self.get_device())
        )

        norm_delta_r = torch.norm(delta_r, p=2, dim=1, keepdim=False)
        norm_delta_v = torch.norm(delta_v, p=2, dim=1, keepdim=False) + utils.EPS

        inv_norm_delta_v = 1.0 / norm_delta_v
        delta_r_v = (delta_r * delta_v).sum(dim=1, keepdim=False)
        r = delta_r_v * inv_norm_delta_v

        # Because of numerical issues, we need to clamp the ratio vector, r, which is upper bounded by norm_delta_r
        # But it is not enough to bound only norm_delta_r since we don't update the values of delta_r.
        # Therefore, we also need to bound r.
        norm_delta_r = torch.clamp(norm_delta_r, max=_UPPER_BOUND)
        r = torch.clamp(r, min=-_UPPER_BOUND, max=_UPPER_BOUND)

        term0 = 0.5 * torch.sqrt(torch.as_tensor(utils.PI, device=self.__device)) * inv_norm_delta_v
        # term1 = torch.exp(beta_ij - (2*states-1)*(r**2 - norm_delta_r**2) )
        term1_plus = torch.exp(beta_ij_plus - (r ** 2 - norm_delta_r ** 2))
        term1_neg = torch.exp(beta_ij_neg + (r ** 2 - norm_delta_r ** 2))

        term2_u_plus = utils.erfi_approx(extended_delta_t * norm_delta_v + r)
        term2_l_plus = utils.erfi_approx(r)

        term2_u_neg = torch.erf(extended_delta_t * norm_delta_v + r)
        term2_l_neg = torch.erf(r)

        if self.is_signed():
            output = term0 * (
                    (1 - (extended_states.absolute() - extended_states) / 2) * term1_plus * (term2_u_plus - term2_l_plus) +
                    (1 - (extended_states.absolute() + extended_states) / 2) * term1_neg * (term2_u_neg - term2_l_neg)
            )
        else:
            output = term0 * (
                    extended_states * term1_plus * (term2_u_plus - term2_l_plus) +
                    (1 - extended_states) * term1_neg * (term2_u_neg - term2_l_neg)
            )

        # Sum the entries from start_marker to last_marker
        output = output.cumsum(0)
        output = output[last_marker]
        output[1:] = output[1:] - output[:-1]

        return output

    def get_nll(self, pairs: torch.Tensor, times: torch.FloatTensor, states: torch.LongTensor,
                event_states: torch.LongTensor, is_edge: torch.BoolTensor, delta_t: torch.FloatTensor) -> torch.Tensor:
        """
        Computes the negative log-likelihood function of the model
        :param pairs: a matrix of shape 2 x L
        :param times: a vector of shape L
        :param states: a vector of shape L
        :param is_edge: a vector of shape L
        :param delta_t: a vector of shape L
        :return:
        """

        non_integral_term = self.get_log_intensity_at(
            time_list=times[is_edge], edges=pairs[:, is_edge], edge_states=event_states[is_edge].to(torch.long)
        ).sum()

        # Then compute the integral term
        integral_term = self.get_intensity_integral(
            time_list=times[delta_t > 0], pairs=pairs[:, delta_t > 0],
            delta_t=delta_t[delta_t > 0], states=states[delta_t > 0]
        ).sum()

        return -(non_integral_term - integral_term)

    def get_neg_log_prior(self, nodes: torch.Tensor) -> torch.float:
        """
        Computes the negative log-prior of the model
        :param nodes: a vector of shape batch_size
        :return: negative log-prior
        """

        # Define the final dimension size
        final_dim = self.get_nodes_num() * (self.get_bins_num()+1) * self.get_dim()

        # Covariance scaling coefficient
        lambda_sq = self.get_prior_lambda()**2

        # Normalize and vectorize the initial position and velocity vectors
        v_s = torch.index_select(self.get_v_s(),  dim=1, index=nodes).flatten()
        if self.is_directed():
            v_r = torch.index_select(self.get_v_r(),  dim=1, index=nodes).flatten()

        # Compute the negative log-likelihood
        d_s = lambda_sq * torch.kron(
            torch.softmax(self.get_prior_b_sigma_s(), dim=0),
            torch.kron(
                torch.index_select(torch.softmax(self.get_prior_c_sigma_s(), dim=0), dim=0, index=nodes),
                torch.ones(self.get_dim(), device=self.get_device(), dtype=torch.float)
            )
        )
        log_prior_s = -0.5*(final_dim*utils.LOG2PI+torch.log(d_s).sum()+(v_s**2 @ (1. / d_s)))
        if self.is_directed():
            d_r = lambda_sq * torch.kron(
                torch.softmax(self.get_prior_b_sigma_r(), dim=0),
                torch.kron(
                    torch.index_select(torch.softmax(self.get_prior_c_sigma_r(), dim=0), dim=0, index=nodes),
                    torch.ones(self.get_dim(), device=self.get_device(), dtype=torch.float)
                )
            )
            log_prior_r = -0.5*(final_dim*utils.LOG2PI+torch.log(d_r).sum()+(v_r**2 @ (1. / d_r)))

        neg_log_prior = log_prior_s
        if self.is_directed():
            neg_log_prior += log_prior_r

        return -neg_log_prior.squeeze()
