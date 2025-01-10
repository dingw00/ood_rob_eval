"""
This module calculates the model predictions, model confidence and OoD scores of the seeds 
and the perturbed samples, and saves the results in .csv files.
"""

import os
import yaml
import argparse
import torch
from utils.dataloader import load_dataset
from models.model_utils import InputNormalizer, load_model
from utils.ood_detectors import build_detectors
from utils.attackers import build_attackers
from utils.test_utils import setup_seed, original_data_ood_test, seeds_ood_test, perturbed_samples_ood_test

if __name__ == "__main__":

    # Load configs: benchmarks, model variants, OoD datasets and experiment settings.
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cfg",type=str, default="config.yaml")
    args = argparser.parse_args()
    
    with open(args.cfg, 'r') as f:
        configs = yaml.safe_load(f)

    datadir = configs["datadir"]
    score_functions = configs["score_functions"]
    perturb_functions = configs["perturb_functions"]
    rand_seed = configs["rand_seed"]
    batch_size = configs["batch_size"]
    n_seeds = configs["n_seeds"]
    n_sampling = configs["n_sampling"]
    severity = configs["severity"]

    device = torch.device("cuda" if (configs["device"]=="cuda" and torch.cuda.is_available() ) else "cpu")
    print(f"Device: {device}")

    for benchmark in configs["benchmark"]:
        n_classes = configs["benchmark"][benchmark]["num_classes"]
        ood_datasets = configs["benchmark"][benchmark]["ood_datasets"]
        img_size = configs["benchmark"][benchmark]["img_size"]
        
        for model_name in configs["benchmark"][benchmark]["model"]:
            for variant, weight_name in configs["benchmark"][benchmark]["model"][model_name].items():

                # Load model
                weight_path = os.path.join("models", benchmark.lower(), "state_dicts", weight_name)
                model = load_model(model_name, weight_path, benchmark, device=device)
                model.eval()
                input_normalizer = InputNormalizer(benchmark=benchmark, model_arch=model_name, 
                                                    model_weight=weight_path.split("/")[-1])
                print("Model:", model_name, "\nVariant:", variant, "-", weight_name)
                print("Input Normalizer: Mean:",input_normalizer.mean.view(-1).numpy(), ", Std:", input_normalizer.std.view(-1).numpy())

                # Load ID training dataset
                id_train_data_set, id_train_data_loader = load_dataset(datadir, benchmark, img_size=img_size, benchmark=benchmark, 
                                                                    split="train", batch_size=batch_size, normalize=True,
                                                                    mean=input_normalizer.mean.view(-1), std=input_normalizer.std.view(-1),
                                                                    model_name=model_name)

                # Build OoD detectors
                detectors = build_detectors(configs["score_functions"], model, input_normalizer, id_train_data_loader, 
                                            device=device)

                # Build attackers
                attackers = build_attackers(configs["perturb_functions"], severity_level=severity, img_size=img_size) 

                datasets = [benchmark] + ood_datasets
                for dataset_name in datasets:
                    print("------------------------------------")
                    print("Dataset:", dataset_name)
                    save_dir = os.path.join("results", benchmark.lower(), str(rand_seed), model_name, 
                                            variant, dataset_name, "scores")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # Load dataset
                    test_data_set, test_data_loader = load_dataset(datadir, dataset_name, img_size=img_size, 
                                                                   benchmark=benchmark, split="test", batch_size=1)
                    # Select seeds
                    data_set_ = test_data_set # [data_set[i] for i in range(len(y_all)) if (y_pred_all[i] == y_all[i])]
                    print("Size:", len(data_set_))
                    setup_seed(rand_seed)
                    idx_temp = torch.randperm(len(data_set_))[:n_seeds]
                    
                    ood = dataset_name != benchmark
                    # Test the model and OoD detectors on the whole dataset.
                    original_data_ood_test(model, detectors, test_data_loader, input_normalizer, ood, save_dir, 
                                           device=device)

                    # # Test the model and OoD detectors on the seeds.
                    # seeds_ood_test(model, detectors, data_set_, idx_temp, input_normalizer, ood, save_dir,
                    #                batch_size=batch_size, device=device)
                    
                    # Generate perturbed samples and test the model and OoD detectors on them.
                    # Save the model prediction, confidence & OoD scores of perturbed samples
                    perturbed_samples_ood_test(model, detectors, data_set_, idx_temp, input_normalizer, attackers, 
                                               n_sampling, ood, save_dir, batch_size=batch_size, device=device)
