import numpy as np
import random
import torch
import os
import time
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.measure import pearson_corr_coeff
from skimage.registration import phase_cross_correlation

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cal_image_sim(img_0, img_1):
    """
    This function calculates image similarity using 4 similarity metrics, namely mean square error (MSE),
    structure similarity index measure (SSIM), Pearson correlation coefficient (PCC), and normalized 
    cross-correlation coefficient(NCC).
    """
    img_0 = np.array(img_0.cpu())
    img_1 = np.array(img_1.cpu())
    # compare with itself
    mse_ = mean_squared_error(img_0, img_1)

    ssim_ = ssim(img_0, img_1, data_range=max(img_1.max(), img_0.max()) - min(img_1.min(), img_0.min()), 
                 channel_axis=-1,
                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False
                 )

    if img_0.shape[-1] == 3:
        img_0 = np.mean(img_0, axis=-1)
    if img_1.shape[-1] == 3:
        img_1 = np.mean(img_1, axis=-1)

    pcc_ = pearson_corr_coeff(img_0, img_1, mask=None).correlation

    # shift, ncc_err, diffphase = phase_cross_correlation(img_0, img_1,
    #                                                 normalization=None,
    #                                             # reference_mask=mask1,
    #                                             # moving_mask=mask2
    #                                             )
    
    return dict(mse=mse_, ssim=ssim_, pcc=pcc_)

def original_data_ood_test(model, detectors, data_loader, input_normalizer, ood, save_dir, 
                           device=torch.device("cpu")):
    """
    This function calculates the model predictions, model confidence and OoD scores of the 
    original dataset, and saves the test results in csv files.
    """
    filepath = os.path.join(save_dir, "temp_all_scores.csv")
    if os.path.exists(filepath):
        return None
    
    with torch.no_grad():    
        y_pred_all = torch.empty(0, dtype=torch.int)
        y_all = torch.empty(0, dtype=torch.int)
        conf_all = torch.empty(0, dtype=torch.float16)
        ood_scores = {k+"_score": torch.empty(0, dtype=torch.float) for k in detectors.keys()}
        
        time_stamp = time.time()
        for x, y in tqdm(data_loader):
            x = x.to(device)
            x = input_normalizer(x)
            out = model(x)
            y_pred = torch.argmax(out, dim=1)
            conf = torch.max(torch.softmax(out, dim=1), dim=1).values

            y_pred_all = torch.concat([y_pred_all, y_pred.cpu()])
            if ood:
                y = torch.ones_like(y) * (-1)
            y_all = torch.concat([y_all, y])
            conf_all = torch.concat([conf_all, conf.to(dtype=torch.float16).cpu()])

        if device.type == "cuda":
            torch.cuda.empty_cache()

        for detector_name, detector in detectors.items():
            for x, y in tqdm(data_loader):
                x = x.to(device)
                x = input_normalizer(x)
                ood_score = detector(x)
                ood_scores[detector_name+"_score"] = torch.concat([ood_scores[detector_name+"_score"], 
                                                                    ood_score.squeeze().cpu()])

        print(f"Model & OoD detector inference time for all seeds: {time.time()-time_stamp}s")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # save idx, y_true, y_pred, conf, ood_scores
        data_dict = {"idx": range(len(y_all)), "y_true": y_all.squeeze(), "y_pred": y_pred_all.squeeze(), "conf": conf_all.squeeze()}  
        data_dict.update(ood_scores)
        df_all = pd.DataFrame(data_dict)
        df_all.to_csv(filepath, index=False)
        print(f"Saved orginal dataset's OoD scores to csv file.")

def seeds_ood_test(model, detectors, data_set, idx_temp, input_normalizer, ood, save_dir, 
                   batch_size=200, device=torch.device("cpu")):
    """
    This function calculates the model predictions, model confidence and OoD scores of the
    selected seeds, and saves the test results in csv files.
    """
    filepath = os.path.join(save_dir, "temp_scores.csv")
    if os.path.exists(filepath):
        return None
    
    with torch.no_grad():
        # model prediction & ood detection on seeds
        x_temp = torch.vstack([data_set[id][0].unsqueeze(0) for id in idx_temp])
        y_temp = torch.ones_like(idx_temp) * (-1) if ood else torch.tensor([data_set[id][1] for id in idx_temp])

        temp_loader = DataLoader(list(zip(x_temp, y_temp)), batch_size=batch_size, shuffle=False, 
                                drop_last=False)
        y_pred_temp = torch.empty(0, dtype=torch.int)
        conf_temp = torch.empty(0, dtype=torch.float16)
        ood_scores = {k+"_score": torch.empty(0, dtype=torch.float) for k in detectors.keys()}
        
        time_stamp = time.time()
        for x, y in tqdm(temp_loader):
            x = x.to(device)
            # input preprocessor
            x = input_normalizer(x)
            # CNN model
            out = model(x)
            y_pred = torch.argmax(out, dim=1)
            conf = torch.max(torch.softmax(out, dim=1), dim=1).values

            y_pred_temp = torch.concat([y_pred_temp, y_pred.cpu()])
            conf_temp = torch.concat([conf_temp, conf.to(dtype=torch.float16).cpu()])

        if device.type == "cuda":
            torch.cuda.empty_cache()

        for detector_name, detector in detectors.items():
            for x, y in tqdm(temp_loader):
                x = x.to(device)
                x = input_normalizer(x)
                ood_score = detector(x)
                ood_scores[detector_name+"_score"] = torch.concat([ood_scores[detector_name+"_score"], 
                                                                    ood_score.squeeze().cpu()])
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(f"Model & OoD detector inference time for {len(idx_temp)} seeds: {time.time()-time_stamp}s")

        # save seeds' scores to file
        data_dict = {"idx": idx_temp, "y_true": y_temp.squeeze(), "y_pred": y_pred_temp.squeeze(), "conf": conf_temp.squeeze()}  
        data_dict.update(ood_scores)
        df_temp = pd.DataFrame(data_dict)
        df_temp.to_csv(filepath, index=False)
        print(f"Saved the seeds' OoD scores to csv file.")

def perturbed_samples_ood_test(model, detectors, data_set, idx_temp, input_normalizer, attackers, 
                               n_sampling, ood, save_dir, batch_size=200, device=torch.device("cpu")):
    """
    This function calculates the model predictions, model confidence and OoD scores of the perturbed 
    samples, and saves the test results in csv files.
    """
    with torch.no_grad():             
        for perb_func, attacker in attackers.items():
            print("-----------------------")
            print("> Perturbation:", perb_func)

            if attacker.severity == "all":
                score_filepath = os.path.join(save_dir, f"perb_{perb_func}_scores.csv")
            else:
                score_filepath = os.path.join(save_dir, f"perb_{perb_func}_level_{int(attacker.severity)}_scores.csv")
            if os.path.exists(score_filepath):
                continue

            time_stamp = time.time()

            idx_perb = torch.empty(0, dtype=torch.int)
            y_true_perb = torch.empty(0, dtype=torch.int)
            y_pred_perb = torch.empty(0, dtype=torch.int)
            conf_perb = torch.empty(0, dtype=torch.float16)
            ood_scores = {k+"_score": torch.empty(0, dtype=torch.float)for k in detectors.keys()}

            for id in tqdm(idx_temp):
                x_temp = data_set[id][0]
                y_temp = -1 if ood else data_set[id][1]

                # Generate n_sampling perturbed samples from each seed
                x_perb = attacker.random_perturb(x_temp.unsqueeze(0), n_repeat=n_sampling, device=device)
                y_perb = torch.tensor([y_temp]*n_sampling)
                idx_perb = torch.concat([idx_perb, id.repeat(n_sampling)])
                y_true_perb = torch.concat([y_true_perb, y_perb])
                
                perb_loader = DataLoader(list(zip(x_perb, y_perb)), batch_size=batch_size, shuffle=False, 
                                            drop_last=False)
                # DNN model test
                for x, y in perb_loader:
                    x = x.to(device)
                    x = input_normalizer(x)
                    out = model(x)
                    y_pred = torch.argmax(out, dim=1)
                    conf = torch.max(torch.softmax(out, dim=1), dim=1).values
                    # Save the model confidence, ground truth and predicted labels
                    y_pred_perb = torch.concat([y_pred_perb, y_pred.cpu()])
                    conf_perb = torch.concat([conf_perb, conf.to(dtype=torch.float16).cpu()])

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                # OoD detector test
                for detector_name, detector in detectors.items():
                    for x, y in perb_loader:
                        x = x.to(device)
                        x = input_normalizer(x)
                        ood_score = detector(x)
                        # Save the OoD scores
                        ood_scores[detector_name+"_score"] = torch.concat([ood_scores[detector_name+"_score"], 
                                                                        ood_score.squeeze().cpu()])                    
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    
            print(f"Sampling, model & OoD detector inference time for {len(idx_temp)}x{n_sampling} perturbed samples: {time.time()-time_stamp}s")
            # save pertubed samples' scores to file
            data_dict = {"idx": idx_perb,
                        "y_true": y_true_perb.squeeze(), "y_pred": y_pred_perb.squeeze(), 
                        "conf": conf_perb.squeeze()}
            data_dict.update(ood_scores)
            df_perb = pd.DataFrame(data_dict)
            df_perb.to_csv(score_filepath, index=False)
            print(f"Saved perturbed samples' OoD scores to csv file.")


