import torch
import utils


class SequentialSurviveProcess:

    def __init__(self, lambda_func, initial_state: int, critical_points: list, seed: int = 19):

        # It is supposed that the first and the last value of the 'critical_points'
        # correspond to initial and last time of the timeline
        self.__lambda_func = lambda_func
        self.__initial_state = initial_state
        self.__critical_points = critical_points

        self.__init_time = self.__critical_points[0]
        self.__num_of_bins = len(self.__critical_points)

        self.__seed = seed

        # Find the max lambda values for each interval
        # For indexing, add first elements which will not be used
        self.__lambda_max = ([0], [0])
        for idx in range(1, self.__num_of_bins):
            self.__lambda_max[0].append(
                max(
                    self.__lambda_func(t=self.__critical_points[idx-1], state=0),
                    self.__lambda_func(t=self.__critical_points[idx], state=0)
                )
            )
            self.__lambda_max[1].append(
                max(
                    self.__lambda_func(t=self.__critical_points[idx-1], state=1),
                    self.__lambda_func(t=self.__critical_points[idx], state=1)
                )
            )
        self.__lambda_max = torch.as_tensor(self.__lambda_max, dtype=torch.float)

        # Set seed
        utils.set_seed(seed=seed)

    def simulate(self) -> tuple[list, list]:

        t, J, S, states = self.__init_time, 1, [], [self.__initial_state]
        # Step 2
        U = torch.rand(1)  # Random number
        # Random variable from exponential dist.
        X = (-1.0 / (self.__lambda_max[states[-1], J] + utils.EPS)) * torch.log(U)

        while True:
            # Step 3
            if t + X < self.__critical_points[J]:
                # Step 4
                t = t + X
                # Step 5
                U = torch.rand(1)
                # Step 6
                if U <= self.__lambda_func(t, state=states[-1])/self.__lambda_max[states[-1], J]:
                    # Don't need I for index, because we append t to S
                    # if len(S) and abs(S[-1] - t.item()) < utils.EPS:
                    #     S.pop()
                    #     states.pop()
                    # else:
                    S.append(t.item())
                    states.append(1 - states[-1])
                # Step 7 -> Do step 2 then loop starts again at step 3
                U = torch.rand(1)  # Random number
                # Random variable from exponential dist.
                X = (-1./(self.__lambda_max[states[-1], J]+utils.EPS)) * torch.log(U)
                # If X is too small, then we can discard the current event and sample again
                if X < utils.EPS and len(S):
                    S.pop()
                    states.pop()
                    X = (-1. / (self.__lambda_max[states[-1], J] + utils.EPS)) * torch.log(U)

            else:
                # Step 8
                if J == self.__num_of_bins - 1:  # k +1 because of zero-indexing
                    break
                # Step 9
                X = (X-self.__critical_points[J]+t)*self.__lambda_max[states[-1], J]/self.__lambda_max[states[-1], J+1]
                t = self.__critical_points[J]
                J += 1

        return S, states[1:]