import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from src.dataset import Dataset
from src.learning import LearningModel

# Define the folder
BASE_FOLDER = "/gras2p/"
# Define the sample file
SAMPLE_FILE = "mr=0.1_cr=0.1_pr=0.1"

def get_scores(sample_folder: str, dataset: str, set_type:str, score_type: str, lambda_values: list,
               bins: int, dim: int, epoch_num: int, spe: int, batch_size: int, lr: float,
               init_time: int, last_time: int, seeds: list):

    # Load the samples
    with open(os.path.join(sample_folder, dataset, f"{set_type}.samples"), "rb") as f:
        samples = pickle.load(f)
    # zero_samples, pos_samples, neg_samples = samples["zero"], samples["pos"], samples["neg"]
    zero_samples, pos_samples = samples["zero"], samples["pos"]

    # Construct samples
    sample_pairs = torch.as_tensor(zero_samples['pairs'] + pos_samples['pairs'], dtype=torch.long).T
    sample_labels = zero_samples['labels'] + pos_samples['labels']
    sample_intervals = torch.as_tensor(zero_samples['intervals'] + pos_samples['intervals'], dtype=torch.float).T
    time_list = (sample_intervals[0] - init_time) / (last_time - init_time)
    delta_t = (sample_intervals[1] - sample_intervals[0]) / (last_time - init_time)
    # print( sample_intervals[1] - sample_intervals[0] )

    roc_auc_scores, pr_auc_scores = [[] for _ in range(len(seeds))], [[] for _ in range(len(seeds))]
    for seed_idx, seed in enumerate(seeds):
        for lambda_idx, lambda_value in enumerate(lambda_values):

            modelname = f"residual_{dataset}_B={bins}_lambda={lambda_value}_dim={dim}_epoch={epoch_num}"
            modelname += f"_spe={spe}_bs={batch_size}_lr={lr}_seed={seed}"
            model_path = os.path.join(BASE_FOLDER, "experiments", f"models_{SAMPLE_FILE}", modelname + ".model")

            # Load the model
            kwargs, lm_state = torch.load(model_path, map_location=torch.device('cpu'))
            kwargs['device'] = 'cpu'
            kwargs['verbose'] = False
            lm = LearningModel(**kwargs)
            lm.load_state_dict(lm_state)

            # Compute the average integrals
            if set_type == "prediction":
                test_preds = lm.get_intensity_at(
                    time_list=torch.ones(len(time_list), dtype=torch.float), edges=sample_pairs,
                    edge_states=torch.zeros(len(time_list), dtype=torch.long)
                )

            else:
                test_preds = lm.get_intensity_integral_for(
                    time_list=time_list, pairs=sample_pairs, delta_t=delta_t,
                    states=torch.zeros(len(time_list), dtype=torch.long)
                ) / delta_t

            # Compute the roc and pr auc scores
            if score_type == "roc":
                roc_auc = roc_auc_score(y_true=sample_labels, y_score=test_preds)
                roc_auc_scores[seed_idx].append(roc_auc)
            if score_type == "pr":
                pr_auc = average_precision_score(y_true=sample_labels, y_score=test_preds)
                pr_auc_scores[seed_idx].append(pr_auc)

    # Compute the mean and std of the scores
    if score_type == "roc":
        mean_roc_auc = np.mean(roc_auc_scores, axis=0)
        std_roc_auc = np.std(roc_auc_scores, axis=0)

        return mean_roc_auc, std_roc_auc

    if score_type == "pr":
        mean_pr_auc = np.mean(pr_auc_scores, axis=0)
        std_pr_auc = np.std(pr_auc_scores, axis=0)

        return mean_pr_auc, std_pr_auc


if __name__ == '__main__':

    # Define the set parameters
    dataset = "synthetic_n=100_seed=16"

    print(f"Dataset: {dataset}")

    # Define the model parameters
    dim = 2
    epoch_num = 300
    lr = 0.1
    bins = 100
    batch_size = 100
    spe = 10

    # Define the seed and lambda values
    seeds = [10, 20, 30, 40, 50]
    lambda_labels = ["1e10", "1e9", "1e8",  "1e7", "1e6", "1e5", "1e4", "1e3", "1e2", "1e1", ]

    # Define the sample file path
    sample_folder = os.path.join(BASE_FOLDER, f"experiments/samples/{SAMPLE_FILE}/")

    # Load the dataset to get the init time and the last time points for the normalization
    first_half = Dataset(verbose=False)
    first_half.read_edge_list(os.path.join(BASE_FOLDER, f"experiments/network_splits/{SAMPLE_FILE}/{dataset}/residual.edges"))
    init_time, last_time = first_half.get_init_time(), first_half.get_last_time()

    # Compute the validation scores
    valid_mean_roc_auc, valid_std_roc_auc = get_scores(
        sample_folder=sample_folder, dataset=dataset, seeds=seeds,
        bins=bins, dim=dim, epoch_num=epoch_num, spe=spe, batch_size=batch_size, lr=lr,
        set_type="validation", score_type="roc", lambda_values=lambda_labels, init_time=init_time, last_time=last_time,
    )
    valid_mean_pr_auc, valid_std_pr_auc = get_scores(
        sample_folder=sample_folder, dataset=dataset, seeds=seeds,
        bins=bins, dim=dim, epoch_num=epoch_num, spe=spe, batch_size=batch_size, lr=lr,
        set_type="validation", score_type="pr", lambda_values=lambda_labels, init_time=init_time, last_time=last_time,
    )
    max_roc_lambda = lambda_labels[np.argmax(valid_mean_roc_auc)]
    max_pr_lambda = lambda_labels[np.argmax(valid_mean_pr_auc)]
    print(f"Lambda value maximizing the validation ROC-AUC: {max_roc_lambda}")
    print(f"Lambda value maximizing the validation PR-AUC: {max_pr_lambda}")
    print(f"Validation ROC-AUC:: {np.max(valid_mean_roc_auc)} +- {valid_std_roc_auc[np.argmax(valid_mean_roc_auc)]}")
    print(f"Validation PR-AUC:: {np.max(valid_mean_pr_auc)} +- {valid_std_pr_auc[np.argmax(valid_mean_pr_auc)]}")

    # Compute the completion/testing scores
    test_mean_roc_auc, test_std_roc_auc = get_scores(
        sample_folder=sample_folder, dataset=dataset, seeds=seeds,
        bins=bins, dim=dim, epoch_num=epoch_num, spe=spe, batch_size=batch_size, lr=lr,
        set_type="completion", score_type="roc", lambda_values=[max_roc_lambda], init_time=init_time, last_time=last_time,
    )
    test_mean_pr_auc, test_std_pr_auc = get_scores(
        sample_folder=sample_folder, dataset=dataset, seeds=seeds,
        bins=bins, dim=dim, epoch_num=epoch_num, spe=spe, batch_size=batch_size, lr=lr,
        set_type="completion", score_type="pr", lambda_values=[max_roc_lambda], init_time=init_time, last_time=last_time,
    )
    print(f"Testing ROC-AUC: {test_mean_roc_auc[0]} +- {test_std_roc_auc[0]}")
    print(f"Testing PR-AUC: {test_mean_pr_auc[0]} +- {test_std_pr_auc[0]}")

    # Compute the reconstruction scores
    test_mean_roc_auc, test_std_roc_auc = get_scores(
        sample_folder=sample_folder, dataset=dataset, seeds=seeds,
        bins=bins, dim=dim, epoch_num=epoch_num, spe=spe, batch_size=batch_size, lr=lr,
        set_type="reconstruction", score_type="roc", lambda_values=[max_roc_lambda], init_time=init_time, last_time=last_time,
    )
    test_mean_pr_auc, test_std_pr_auc = get_scores(
        sample_folder=sample_folder, dataset=dataset, seeds=seeds,
        bins=bins, dim=dim, epoch_num=epoch_num, spe=spe, batch_size=batch_size, lr=lr,
        set_type="reconstruction", score_type="pr", lambda_values=[max_roc_lambda], init_time=init_time, last_time=last_time,
    )
    print(f"Reconstruction ROC-AUC: {test_mean_roc_auc[0]} +- {test_std_roc_auc[0]}")
    print(f"Reconstruction PR-AUC: {test_mean_pr_auc[0]} +- {test_std_pr_auc[0]}")


    # Compute the prediction scores
    test_mean_roc_auc, test_std_roc_auc = get_scores(
        sample_folder=sample_folder, dataset=dataset, seeds=seeds,
        bins=bins, dim=dim, epoch_num=epoch_num, spe=spe, batch_size=batch_size, lr=lr,
        set_type="prediction", score_type="roc", lambda_values=[max_roc_lambda], init_time=init_time, last_time=last_time,
    )
    test_mean_pr_auc, test_std_pr_auc = get_scores(
        sample_folder=sample_folder, dataset=dataset, seeds=seeds,
        bins=bins, dim=dim, epoch_num=epoch_num, spe=spe, batch_size=batch_size, lr=lr,
        set_type="prediction", score_type="pr", lambda_values=[max_roc_lambda], init_time=init_time, last_time=last_time,
    )
    print(f"Prediction ROC-AUC: {test_mean_roc_auc[0]} +- {test_std_roc_auc[0]}")
    print(f"Prediction PR-AUC: {test_mean_pr_auc[0]} +- {test_std_pr_auc[0]}")

