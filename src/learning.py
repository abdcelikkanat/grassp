import sys
import math
import torch
import time
import utils
from src.base import BaseModel
from src.sampler import BatchSampler


class LearningModel(BaseModel, torch.nn.Module):

    def __init__(self, nodes_num: int, directed: bool, signed: bool,
                 bins_num: int, dim: int, prior_lambda: float = 1e5,
                 device: torch.device = 'cpu', verbose: bool = False, seed: int = 19):

        utils.set_seed(seed)

        super(LearningModel, self).__init__(
            x0_s=torch.nn.Parameter(
                2. * torch.rand(size=(nodes_num, dim), device=device) - 1., requires_grad=False
            ),
            x0_r=torch.nn.Parameter(
                2. * torch.rand(size=(nodes_num, dim), device=device) - 1., requires_grad=False
            ) if directed else None,
            v_s=torch.nn.Parameter(
                torch.randn(size=(bins_num, nodes_num, dim), device=device), requires_grad=False
            ),
            v_r=torch.nn.Parameter(
                torch.randn(size=(bins_num, nodes_num, dim), device=device), requires_grad=False
            ) if directed else None,
            beta_s=torch.nn.Parameter(
                torch.randn(size=(2, ), device=device), requires_grad=False
            ),
            beta_r=torch.nn.Parameter(
                torch.randn(size=(2, ), device=device), requires_grad=False
            ) if directed else None,
            prior_lambda=torch.as_tensor(
                prior_lambda, dtype=torch.float, device=device
            ),
            prior_b_sigma_s=torch.nn.Parameter(
                torch.ones(size=(bins_num, ), device=device), requires_grad=False
            ),
            prior_b_sigma_r=torch.nn.Parameter(
                torch.ones(size=(bins_num, ), device=device), requires_grad=False
            ) if directed else None,
            prior_c_sigma_s=torch.nn.Parameter(
                torch.ones(size=(nodes_num, ), device=device), requires_grad=False
            ),
            prior_c_sigma_r=torch.nn.Parameter(
                torch.ones(size=(nodes_num,), device=device), requires_grad=False
            ) if directed else None,
            directed=directed,
            signed=signed,
            device=device,
            verbose=verbose,
            seed=seed
        )

        self.__bipartite = False  # Bipartite flag

        # Learning parameters
        self.__lp = "sequential"  # Learning procedure
        self.__optimizer = torch.optim.Adam  # Optimizer

        # - The following parameters are set in the learn() method
        self.__min_time = None
        self.__max_time = None
        self.__lr = None  # Learning rate
        self.__batch_size = None   # Batch size
        self.__epoch_num = None  # Number of epochs
        self.__steps_per_epoch = None  # Number of steps per epoch

    def __set_gradients(self, beta_grad=None, x0_grad=None, v_grad=None, prior_grad=None):
        """
        Set the gradient status of the model parameters
        :param beta_grad: The gradient for the bias terms
        :param x0_grad: The gradient for the initial positions
        :param v_grad: The gradient for the velocities
        """

        # Set the gradient of the bias terms
        if beta_grad is not None:
            self.get_beta_s().requires_grad = beta_grad
            if self.is_directed():
                self.get_beta_r().requires_grad = beta_grad

        # Set the gradient of the initial positions
        if x0_grad is not None:
            self.get_x0_s(standardize=False).requires_grad = x0_grad
            if self.is_directed():
                self.get_x0_r(standardize=False).requires_grad = x0_grad

        # Set the gradient of the velocities
        if v_grad is not None:
            self.get_v_s(standardize=False).requires_grad = v_grad
            if self.is_directed():
                self.get_v_r(standardize=False).requires_grad = v_grad
                
        # Set the gradient of the all prior parameters
        if prior_grad is not None:
            for name, param in self.named_parameters():
                if '_prior' in name:
                    param.requires_grad = prior_grad

    def learn(self, dataset, lr: float = 0.1, batch_size: int = 0, epoch_num: int = 100, steps_per_epoch=1,
              log_file_path=None):
        """
        Learn the model parameters
        :param dataset: The dataset
        :param lr: The learning rate
        :param batch_size: The batch size
        :param epoch_num: The number of epochs
        :param steps_per_epoch: The number of steps per epoch
        :param masked_data: The masked data
        :param log_file_path: The log file path

        It is worth noting that the model ignores the direction of the links
        if the bipartite flag is set to True in the dataset object.

        """

        # Set if the model is bipartite
        self.__bipartite = True if dataset.is_bipartite() else False

        # Set the learning parameters
        self.__lr = lr
        self.__batch_size = self.get_nodes_num() if batch_size == 0 else batch_size
        self.__epoch_num = epoch_num
        self.__steps_per_epoch = steps_per_epoch

        # Scale the edge times to [0, 1]
        edge_times = dataset.get_times()
        self.__min_time = edge_times.min()
        self.__max_time = edge_times.max()
        edge_times = (edge_times - self.__min_time) / (self.__max_time - self.__min_time)

        # Define the batch sampler
        bs = BatchSampler(
            edges=dataset.get_edges().to(self.get_device()), edge_times=edge_times.to(self.get_device()),
            edge_states=dataset.get_weights().to(self.get_device()),
            bin_bounds=self.get_bin_bounds(), nodes_num=self.get_nodes_num(), batch_size=self.__batch_size, 
            directed=self.is_directed(), bipartite=dataset.is_bipartite(), device=self.get_device(), seed=self.get_seed()
        )

        # Check the learning procedure
        if self.__lp == "sequential":

            # Run sequential learning
            loss, nll = self.__sequential_learning(bs=bs)

        else:

            raise Exception(f"Unknown learning procedure: {self.__lp}")

        # Save the loss if the loss file path was given
        if log_file_path is not None:
            
            with open(log_file_path, 'w') as f:
                for batch_losses, nll_losses in zip(loss, nll):
                    f.write(f"Loss: {' '.join('{:.3f}'.format(loss) for loss in batch_losses)}\n")
                    f.write(f"Nll: {' '.join('{:.3f}'.format(loss) for loss in nll_losses)}\n")

        return loss, nll

    def __sequential_learning(self, bs: BatchSampler):

        if self.get_verbose():
            print("+ Training started (Procedure: Sequential Learning).")

        # Define the optimizers and parameter group names
        self.group_optimizers = []
        self.param_groups = [["v"], ["x0", "v"], ["x0", "v", "beta", "prior"]]
        self.group_epoch_weights = [1.0, 1.0, 1.0]

        # For each group of parameters, add a new optimizer
        for current_group in self.param_groups:
            
            # Set the gradients to True
            self.__set_gradients(**{f"{param_name}_grad": True for param_name in current_group})
        
            # Add a new optimizer
            self.group_optimizers.append(self.__optimizer(self.parameters(), lr=self.__lr))
            
            # Set the gradients to False
            self.__set_gradients(**{f"{param_name}_grad": False for param_name in current_group})

        # Determine the number of epochs for each parameter group
        group_epoch_counts = (self.__epoch_num * torch.cumsum(
            torch.as_tensor([0.] + self.group_epoch_weights, device=self.get_device(), dtype=torch.float), dim=0
        ) / sum(self.group_epoch_weights)).type(torch.int)
        group_epoch_counts = group_epoch_counts[1:] - group_epoch_counts[:-1]

        # Run the epochs
        loss, nll = [], []
        epoch_num = 0
        for current_epoch_count, optimizer, current_group in zip(group_epoch_counts, self.group_optimizers, self.param_groups):

            # Set the gradients to True
            self.__set_gradients(**{f"{param_name}_grad": True for param_name in current_group})

            # Run the epochs
            for _ in range(current_epoch_count):
                # Run one epoch
                epoch_loss, epoch_nll = self.__train_one_epoch(bs=bs, epoch_num=epoch_num, optimizer=optimizer)
                loss.append(epoch_loss)
                nll.append(epoch_nll)

                # Increase the epoch number by one
                epoch_num += 1

            # Set the gradients to False
            self.__set_gradients(**{f"{param_name}_grad": False for param_name in current_group})

        return loss, nll

    def __train_one_epoch(self, bs: BatchSampler, epoch_num: int, optimizer: torch.optim.Optimizer):

        init_time = time.time()

        total_batch_loss = 0.
        epoch_loss, epoch_nll = [], []
        for batch_num in range(self.__steps_per_epoch):

            batch_nll, batch_nlp = self.__train_one_batch(bs=bs)
            batch_loss = batch_nll + batch_nlp

            epoch_loss.append(batch_loss)
            epoch_nll.append(batch_nll)

            total_batch_loss += batch_loss

        # Set the gradients to 0
        optimizer.zero_grad()

        # Backward pass
        total_batch_loss.backward()

        # Perform a step
        optimizer.step()

        # Get the average epoch loss
        avg_loss = total_batch_loss / float(self.__steps_per_epoch)

        if torch.isnan(avg_loss):
            print(f"\t- Please restart the training with smaller epoch number or learning rate!")
            sys.exit(1)

        if not math.isfinite(avg_loss):
            print(f"\t- Epoch loss is {avg_loss}, stopping training")
            sys.exit(1)

        if self.get_verbose() and (epoch_num % 10 == 0 or epoch_num == self.__epoch_num - 1):
            time_diff = time.time() - init_time
            print("\t- Epoch = {} | Avg. Loss/train: {} | Elapsed time: {:.2f}".format(epoch_num, avg_loss, time_diff))

        return epoch_loss, epoch_nll

    def __train_one_batch(self, bs: BatchSampler) -> tuple[torch.Tensor, torch.Tensor]:

        self.train()

        # Sample a batch
        batch_nodes, expanded_pairs, expanded_times, expanded_states, event_states, is_edge, delta_t = bs.sample()

        # Finally, compute the negative log-likelihood and the negative log-prior for the batch
        batch_nll, batch_nlp = self.forward(
            nodes=batch_nodes, pairs=expanded_pairs, times=expanded_times, states=expanded_states,
            event_states=event_states, is_edge=is_edge, delta_t=delta_t
        )

        # Divide the batch loss by the number of all possible pairs
        average_batch_nll = batch_nll / float(self.__batch_size * (self.__batch_size - 1))
        average_batch_nlp = batch_nlp / float(self.__batch_size * (self.__batch_size - 1))
        if not self.is_directed() and not self.__bipartite:
            average_batch_nll /= 2.
            average_batch_nlp /= 2.

        return average_batch_nll, average_batch_nlp

    def forward(self, nodes: torch.Tensor, pairs: torch.LongTensor, times: torch.FloatTensor, states: torch.LongTensor,
                event_states: torch.LongTensor, is_edge: torch.BoolTensor, delta_t: torch.FloatTensor):

        # Get the negative log-likelihood
        nll = self.get_nll(
            pairs=pairs, times=times, states=states, event_states=event_states, is_edge=is_edge, delta_t=delta_t
        )

        # Get the negative log-prior and the R-factor inverse
        nlp = self.get_neg_log_prior(nodes=nodes)

        return nll, nlp

    def save(self, path):

        if self.get_verbose():
            print(f"+ Model file is saving.")
            print(f"\t- Target path: {path}")

        kwargs = {
            'nodes_num': self.get_nodes_num(),
            'directed': self.is_directed(),
            'signed': self.is_signed(),
            'bins_num': self.get_bins_num(),
            'dim': self.get_dim(),
            'prior_lambda': self.get_prior_lambda(),
            'verbose': self.get_verbose(), 
            'seed': self.get_seed()
        }
        
        torch.save([kwargs, self.state_dict()], path)

        if self.get_verbose():
            print(f"\t- Completed.")

