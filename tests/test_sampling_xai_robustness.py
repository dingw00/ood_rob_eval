"""
This module calculates the similarity between the heatmaps of the seeds and their corresponding 
perturbed samples, and saves the results in .csv files.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import os
import time
from utils.dataloader import load_dataset
from models.model_utils import InputNormalizer, load_model
from utils.attackers import Spatial, Color, Blur, LinfStep, L2Step
from utils.test_utils import setup_seed, cal_image_sim
from utils.visualize import show_images, make_lrp_heatmap

from tqdm import tqdm
import argparse

def cal_heatmap_diff_average_sims(relevance_temp, relevance_perb):
    # evaluate heatmap reliability
    # origianal heatmap
    img_hm_0 = np.transpose(relevance_temp.squeeze(0), (1, 2, 0))
    img_hm_0 = np.mean(img_hm_0, axis=-1)

    # 2) heat map similarity before & after adding the perturbation 
    all_sims = dict()
    for j in range(len(relevance_perb)):
        # perturbed heatmap
        img_hm_1 = np.transpose(relevance_perb[j], (1, 2, 0))
        img_hm_1 = np.mean(img_hm_1, axis=-1)

        hm_sims = cal_image_sim(img_hm_0, img_hm_1)

        # update all_sims dict
        for k, v in hm_sims.items():
            if ("hm_diff_"+k) not in all_sims:
                all_sims["hm_diff_"+k] = []
            all_sims["hm_diff_"+k].append(v)
        
    df_hm_diff = pd.DataFrame(all_sims).transpose()

    return df_hm_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs="+")
    parser.add_argument('--benchmark', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--attackers', type=str, nargs="+")
    parser.add_argument('--checkpoint', type=str)
    
    args = parser.parse_args()

    # Update your existing code to use args.benchmark, args.model_name, etc.

    datasets = args.datasets
    benchmark = args.benchmark
    model_name = args.model
    checkpoint = args.checkpoint
    attacker_names = args.attackers
    
    weight_path=f"./models/{benchmark.lower()}/state_dicts/{checkpoint}"
    if benchmark == "CIFAR10":
        num_classes = 10  
    elif benchmark == "Imagenet100":
        num_classes = 100

    n_seeds = 1000
    n_sampling = 1000
    batch_size = 50

    rand_seed = 0

    parameters = {"rot": 30, "trans": 0.3, "scale": 0.3,
                "hue": 2/3.14, "satu": 0.3, "bright": 0.3, "cont": 0.3,
                "gau_size": 11,"gau_sigma": 1,
                "eps_inf": 0.02, "eps_l2": 1}
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    print("device:", device)

    # Load model
    model = load_model(model_name, weight_path, benchmark, device=device)

    model.to(device)
    model.eval()
    input_normalizer = InputNormalizer(benchmark=benchmark, model_arch=model_name, 
                                    model_weight=weight_path.split("/")[-1])
    print("mean:",input_normalizer.mean.view(-1), "\nstd:", input_normalizer.std.view(-1))

    # build attackers
    attackers_ = {
                "rotation": Spatial(rot=parameters["rot"], trans=0, scale=0),
                "translation": Spatial(rot=0, trans=parameters["trans"], scale=0), 
                "scale": Spatial(rot=0, trans=0, scale=parameters["scale"]), 
                "hue": Color(hue=parameters["hue"], satu=0, bright=0, cont=0), 
                "saturation": Color(hue=0, satu=parameters["satu"], bright=0, cont=0), 
                "bright_contrast": Color(hue=0, satu=0, bright=parameters["bright"], cont=parameters["cont"]), 
                "blur": Blur(gau_size=parameters["gau_size"], gau_sigma=parameters["gau_sigma"]), 
                "Linf": LinfStep(eps=parameters["eps_inf"]), 
                "L2": L2Step(eps=parameters["eps_l2"])
                }
    attackers = {k: v for k, v in attackers_.items() if k in attacker_names}
    print(attackers)
    
    save_folder_path = "results/" + benchmark.lower() + "/" + str(rand_seed) + "/" + \
                    model_name + "/eval/local_lrp_ae_ood_rob_corr/"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    score_functions = ["msp", "odin", "energy"]
    perb_func_params = {"rotation": {"rot": 30.0, "trans": 0.0, "scale": 0.0},
                        "translation": {"rot": 0.0, "trans": 0.3, "scale": 0.0}, 
                        "scale": {"rot": 0.0, "trans": 0.0, "scale": 0.3}, 
                        "hue": {"hue": 2/3.14, "satu": 0.0, "bright": 0.0, "cont": 0.0}, 
                        "saturation": {"hue": 0.0, "satu": 0.3, "bright": 0.0, "cont": 0.0}, 
                        "bright_contrast": {"hue": 0.0, "satu":0.0, "bright": 0.3, "cont": 0.3}, 
                        "blur": {"gau_size": 11,"gau_sigma": 1.0}, 
                        "Linf": {"eps": 0.02}, 
                        "L2": {"eps": 1.0} 
                        }

    df_ood_rob_err = pd.DataFrame()
        
    benchmark_folder_path = "results/" + benchmark.lower() + "/" + str(rand_seed) + \
                        "/" + model_name + "/" + checkpoint.replace(".pt", "") + "/"
    print("benchmark_folder_path:", benchmark_folder_path)

    for dataset_name in datasets:

        # Load dataset
        data_set, data_loader = load_dataset("dataset/", dataset_name, img_size=img_size, benchmark=benchmark, 
                                                split="test", batch_size=1)

        print("-----------------------")
        print("Benchmark:", benchmark, "\nDataset:", dataset_name)
        print("-----------------------")

        if not os.path.exists(benchmark_folder_path + dataset_name + "/heatmap_sims/"):
            os.makedirs(benchmark_folder_path + dataset_name + "/heatmap_sims/")

        df_temp_scores = pd.read_csv(benchmark_folder_path+dataset_name+"/scores/temp_scores.csv")
        indexes = list(set(df_temp_scores["idx"].to_numpy(dtype=int)))
        
        model = load_model(model_name, weight_path, benchmark, device=device)

        model.to(device)
        model.eval()
        input_normalizer = InputNormalizer(benchmark=benchmark, model_arch=model_name, 
                                        model_weight=weight_path.split("/")[-1])
        print("mean:",input_normalizer.mean.view(-1), "\nstd:", input_normalizer.std.view(-1))

        for perb_func, attacker in attackers.items():
            ts = time.time()
            print(f"> {perb_func}")

            df_hm_diff_all = pd.DataFrame()

            for idx in tqdm(indexes):
                
                df_hm_diff = pd.DataFrame()

                x_temp = data_set[idx][0]

                if dataset_name == benchmark:
                    y_true_temp = data_set[idx][1]
                else:
                    y_true_temp = -1
                
                # seed's heatmap
                relevance_temp = make_lrp_heatmap(model, input_normalizer, x_temp, y_true_temp, device=device)

                # generate n_sampling perturbed samples from each seed
                x_perb, param = attacker.random_perturb(x_temp.unsqueeze(0), n_repeat=n_sampling, device=device)
                y_true_perb = torch.tensor([y_true_temp] * n_sampling)

                perb_loader = DataLoader(list(zip(x_perb, y_true_perb)), batch_size=batch_size, shuffle=False, num_workers=4, 
                                        #  pin_memory=True, 
                                            drop_last=False)
                
                idx_perb_suffix = torch.arange(n_sampling)
                # perturbed samples' heatmaps
                for x, y in tqdm(perb_loader):
                    
                    relevance_perb = make_lrp_heatmap(model, input_normalizer, x, y, device=device)

                    # make heatmap similarity table
                    df_hm_diff_ = cal_heatmap_diff_average_sims(relevance_temp, relevance_perb).transpose()
                    df_hm_diff = pd.concat([df_hm_diff, df_hm_diff_], axis=0)

                # df_hm_diff["perturb_function"] = perb_func
                df_hm_diff["idx"] = idx
                df_hm_diff["idx_suffix"] = idx_perb_suffix
                df_hm_diff_all = pd.concat([df_hm_diff_all, df_hm_diff], axis=0)

            save_path = benchmark_folder_path + dataset_name + "/heatmap_sims/" + f"{perb_func}.csv"
            df_hm_diff_all.to_csv(save_path, index=False)

            print(f"time spent: {time.time()-ts}")