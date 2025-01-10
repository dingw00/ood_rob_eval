import yaml
import pandas as pd
import os
import time
import torch
from torch.utils.data import DataLoader
import argparse

from utils.dataloader import load_dataset
from models.model_utils import InputNormalizer, load_model
from utils.ood_detectors import build_detectors
from utils.attackers import build_attackers, build_attacker
from utils.eval import get_thr_tpr
from utils.test_utils import setup_seed

if __name__ == "__main__":

    # Load configs: benchmarks, model variants, OoD datasets and experiment settings.
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cfg",type=str, default="config.yaml")
    args = argparser.parse_args()
    
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)

    datadir = configs["datadir"]
    score_functions = configs["score_functions"]
    perturb_functions = configs["perturb_functions"]
    rand_seed = configs["rand_seed"]
    batch_size = configs["batch_size"]
    n_seeds = configs["n_seeds"]
    n_sampling = configs["n_sampling"]
    n_rs = configs["n_randomized_smoothing"]

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
                                                    mean=input_normalizer.mean.view(-1), std=input_normalizer.std.view(-1))



                # Build OoD detectors
                detectors = build_detectors(configs["score_functions"], model, input_normalizer, id_train_data_loader, 
                                            device=device)

                # Build attackers
                attackers = build_attackers(configs["perturb_functions"], severity_level="all", img_size=img_size)
                attacker_rs = build_attacker("Linf", severity_level="all", benchmark=benchmark)

                # Load OoD scores on the ID test dataset and calculate FPR at TPR=95%
                id_scores_dir = os.path.join("results", benchmark.lower(), str(rand_seed), model_name, 
                                            variant, benchmark, "scores")
                filepath = os.path.join(id_scores_dir, "temp_all_scores.csv")
                thr_dict = get_thr_tpr(filepath, score_functions, tpr=0.95)


                # Randomized Smoothing test 
                for dataset_name in [benchmark]+ood_datasets:
                    print("------------------------------------")
                    print("Dataset:", dataset_name)
                    save_dir = os.path.join("results", benchmark.lower(), str(rand_seed), model_name, 
                                            variant, dataset_name, "rs_scores")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # Load dataset
                    test_data_set, test_data_loader = load_dataset(datadir, dataset_name, img_size=img_size, benchmark=benchmark, 
                                                        split="test", batch_size=1)
                    
                    # Select seeds
                    data_set_ = test_data_set # [data_set[i] for i in range(len(y_all)) if (y_pred_all[i] == y_all[i])]
                    setup_seed(rand_seed)
                    idx_temp = torch.randperm(len(data_set_))[:n_seeds]


                    for attacker_name, attacker in attackers.items():
                        print("> Attacker:", attacker_name)
                        ts = time.time()
                        df_rlt = pd.DataFrame()
                        for idx in idx_temp:
                            x_temp, y_temp = data_set_[idx]
                            x_temp = x_temp.to(device)
                            y_temp = -1 if dataset_name != benchmark else y_temp

                            with torch.no_grad():
                                # Generate perturbed samples
                                x_perb = attacker.random_perturb(x_temp.unsqueeze(0), n_repeat=n_sampling,
                                                                        seed=rand_seed, device=device)
                                # y_perb = torch.tensor([y_temp]*n_sampling)
                                
                                # Randomized smoothing
                                x_perb_rs = attacker.random_perturb(x_perb, n_repeat=n_rs, seed=rand_seed, device=device)
                                shape_ = x_perb_rs.shape
                                x_perb_rs = x_perb_rs.reshape(n_rs, n_sampling, *shape_[1:]).permute(1,0,2,3,4).reshape(shape_)
                                # y_perb_rs = torch.tensor([y_perb]*n_rs)
                                
                                data = torch.concat([x_perb, x_perb_rs], dim=0)
                                data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

                                # DNN model inference
                                y_pred_all = torch.empty(0, dtype=torch.int)
                                conf_all = torch.empty((0, n_classes), dtype=torch.float)
                                for x in data_loader:
                                    x = x.to(device)
                                    x = input_normalizer(x)

                                    out = model(x)
                                    y_pred = torch.argmax(out, dim=1)
                                    conf = torch.softmax(out, dim=1)

                                    y_pred_all = torch.concat([y_pred_all, y_pred.cpu()])
                                    conf_all = torch.concat([conf_all, conf.cpu()], dim=0)
                                    
                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()

                                # Process the results
                                y_pred_perb = y_pred_all[:(n_sampling)]
                                y_pred_rs = y_pred_all[(n_sampling):].reshape(n_sampling, n_rs)
                                y_pred_voting = torch.mode(y_pred_rs, dim=1).values
                                y_pred_voting = y_pred_voting.to(torch.int)

                                conf_rs = conf_all[(n_sampling):].reshape(n_sampling, n_rs, n_classes)
                                conf_avg = torch.mean(conf_rs, dim=1)
                                y_pred_avg = torch.argmax(conf_avg, dim=1)
                                y_pred_avg = y_pred_avg.to(torch.int)

                                df = pd.DataFrame({"idx": idx.repeat(n_sampling), 
                                            "y_pred": y_pred_perb, 
                                            "y_pred_voting": y_pred_voting.tolist(), 
                                            "y_pred_avg": y_pred_avg.tolist()})


                                # OoD detector inference
                                for detector_name, detector in detectors.items():
                                    ood_score_all = torch.empty(0, dtype=torch.float)
                                    for x in data_loader:
                                        x = x.to(device)
                                        x = input_normalizer(x)

                                        ood_score = detector(x)

                                        ood_score_all = torch.concat([ood_score_all, ood_score.cpu()])

                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()
                                                        
                                    # Process the results
                                    ood_score_perb = ood_score_all[:(n_sampling)]
                                    ood_temp_perb = ood_score_perb > thr_dict[f"{detector_name}_score"]

                                    ood_score_rs = ood_score_all[(n_sampling):].reshape(n_sampling, n_rs)
                                    ood_score_avg = torch.mean(ood_score_rs, dim=1)

                                    ood_voting = torch.mode(ood_score_rs > thr_dict[f"{detector_name}_score"], axis=1).values
                                    ood_avg = ood_score_avg > thr_dict[f"{detector_name}_score"]

                                    df_ood = pd.DataFrame({f"{detector_name}_score": ood_score_perb, 
                                                f"{detector_name}_score_avg": ood_score_avg.tolist(),
                                                f"{detector_name}_ood": ood_temp_perb, 
                                                f"{detector_name}_ood_voting": ood_voting.tolist(), 
                                                f"{detector_name}_ood_avg": ood_avg.tolist()})

                                    df = pd.concat([df, df_ood], axis=1).copy()
                                
                                df_rlt = pd.concat([df_rlt, df], axis=0).copy()
                            
                        save_path = os.path.join(save_dir, f"perb_{attacker_name}_scores.csv")
                        df_rlt.to_csv(save_path, index=False)
                        t = time.time() - ts
                        print(f"Experiment results saved to {save_path}, time spent: {t:.2f} sec")