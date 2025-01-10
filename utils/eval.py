import pandas as pd
import numpy as np
import os
import sklearn.metrics as sk

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):

    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))
    #return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])
    return thresholds[cutoff], fps[cutoff] / (np.sum(np.logical_not(y_true)))


def get_ood_measures(in_scores, out_scores, recall_level=0.95):
    # in_examples: pos, out_examples: neg
    
    in_examples = -in_scores.reshape((-1, 1))
    out_examples = -out_scores.reshape((-1, 1))

    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    examples = np.squeeze(np.vstack((in_examples, out_examples)))
    aupr_in = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)

    threshold, fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples = np.squeeze(-np.vstack((in_examples, out_examples)))
    aupr_out = sk.average_precision_score(labels_rev, examples)
    threshold_out, fpr_out = fpr_and_fdr_at_recall(labels_rev, examples, recall_level)

    return auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out

def percent_to_thr(data, target_percentage, verbose=False):

    sorted_data = sorted(data)
    index = int(len(data) * (1 - target_percentage)) - 1
    index = min(max(0, index), len(data)-1)
    threshold = sorted_data[index]
    if verbose:
        print(f"{str(round(target_percentage*100, ndigits=2))}% above {threshold}.")

    return float(threshold)

def get_thr_tpr(filepath, score_functions, tpr=0.95):
    
    df = pd.read_csv(filepath)
    thr_dict = dict()
    for score_func in score_functions:
        score = score_func + "_score"
        if score in df.columns:
            ood_scores_id = df[score].to_numpy()
            theta = percent_to_thr(ood_scores_id, 1-tpr, verbose=False)
            thr_dict[score] = theta
    return thr_dict

def _get_model_detector_performance(configs, refresh=False): 
    """
    This function processes raw experiment results (model confidence & OoD scores) and 
    evaluates the DNN model & OoD detectors' performance on ID & OoD datasets on the seeds 
    and the perturbed samples, respectively.
    The evaluation results are saved in csv files in "results/eval/performance/" 
    folder, including:
    
    - Model performance metrics: Accuracy.
    - OoD detectors' performance metrics: FPR95, AUROC, AUPR_IN, AUPR_OUT.

    Args:
        configs (dict): A dictionary containing the configurations for the evaluation.
        refresh (bool, optional): If True, refreshes the evaluation results. Defaults to False.
    
    Returns:
        None
    """

    save_dir = os.path.join("results", "eval", "performance")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    score_functions = configs["score_functions"]
    perturb_functions = configs["perturb_functions"]
    rand_seed = configs["rand_seed"]

    df_model_performance = pd.DataFrame()
    df_model_performance_seed = pd.DataFrame()
    df_model_performance_perb = pd.DataFrame()
    df_detector_performance = pd.DataFrame()
    df_detector_performance_seed = pd.DataFrame()
    df_detector_performance_perb = pd.DataFrame()

    for benchmark in configs["benchmark"]:
        
        print("\n=========================")
        print("Evaluating DNN model & OoD detectors' performance.")
        print("=========================")
        print("Benchmark:", benchmark)
        ood_datasets = configs["benchmark"][benchmark]["ood_datasets"]

        for model_name in configs["benchmark"][benchmark]["model"]:
            
            print("-------------------------")
            print("model_name:", model_name)
            weights = configs["benchmark"][benchmark]["model"][model_name]

            for weight_variant in weights:

                print("-------------------------")
                print("weight_variant:", weight_variant)
                scores_dir = os.path.join("results", benchmark.lower(), str(rand_seed), model_name, weight_variant)
                print("scores_dir:", scores_dir)

                print("Loading ID scores.")
                df_id_all_scores = pd.DataFrame()
                file_path = os.path.join(scores_dir, benchmark, "scores", "temp_all_scores.csv")
                if os.path.exists(file_path):
                    df_id_all_scores = pd.read_csv(file_path).copy()
                else:
                    print("File "+file_path+" does not exist!")
                    continue

                # 1) Calculate the DNN model's performance metrics.
                # Load model confidence and OoD scores of seeds & perturbed samples from the ID dataset.
                df_model_performance_0 = pd.DataFrame()
                if os.path.exists(os.path.join(save_dir, "model_performance.csv")):
                    df_model_performance_0 = pd.read_csv(os.path.join(save_dir, "model_performance.csv")).copy()
                    df_model_performance_0.set_index(["benchmark", "model", "variant"], inplace=True)

                if (benchmark, model_name, weight_variant) not in df_model_performance_0.index or refresh:                  
                    # calculate model accuracy
                    acc = (df_id_all_scores["y_true"]==df_id_all_scores["y_pred"]).mean()*100
                    df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], accuracy=[acc]))
                    df_model_performance = pd.concat([df_model_performance, df_perf], axis=0).copy()
                else:
                    df_perf = df_model_performance_0.loc[[(benchmark, model_name, weight_variant)]].reset_index().copy()
                    df_model_performance = pd.concat([df_model_performance, df_perf], axis=0).copy()
                
                df_model_performance_seed_0 = pd.DataFrame()
                if os.path.exists(os.path.join(save_dir, "model_performance_seed.csv")):
                    df_model_performance_seed_0 = pd.read_csv(os.path.join(save_dir, "model_performance_seed.csv")).copy()
                    df_model_performance_seed_0.set_index(["benchmark", "model", "variant"], inplace=True)
                if (benchmark, model_name, weight_variant) not in df_model_performance_seed_0.index or refresh:
                    print("Loading ID seeds' scores.")
                    file_path = os.path.join(scores_dir, benchmark, "scores", "temp_scores.csv")
                    if os.path.exists(file_path):
                        df_id_seed_scores = pd.read_csv(file_path).copy()
                        # calculate model accuracy
                        acc = (df_id_seed_scores["y_true"]==df_id_seed_scores["y_pred"]).mean()*100
                        df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], accuracy=[acc]))
                        df_model_performance_seed = pd.concat([df_model_performance_seed, df_perf], axis=0).copy()
                    else:
                        print("File "+file_path+" does not exist!")
                else:
                    df_perf = df_model_performance_seed_0.loc[[(benchmark, model_name, weight_variant)]].reset_index().copy()
                    df_model_performance_seed = pd.concat([df_model_performance_seed, df_perf], axis=0).copy()
                
                df_model_performance_perb_0 = pd.DataFrame()
                if os.path.exists(os.path.join(save_dir, "model_performance_perb.csv")):
                    df_model_performance_perb_0 = pd.read_csv(os.path.join(save_dir, "model_performance_perb.csv")).copy()
                    df_model_performance_perb_0.set_index(["benchmark", "model", "variant", "perturb_function"], inplace=True)

                print("Loading ID perturbed samples' scores.")
                for perb_func in perturb_functions:
                    if (benchmark, model_name, weight_variant, perb_func) not in df_model_performance_perb_0.index or refresh:
                        print("> Perturbation function:", perb_func)
                        file_path = os.path.join(scores_dir, benchmark, "scores", 
                                                f"perb_{perb_func}_scores.csv")
                        if not os.path.exists(file_path):
                            print("File "+file_path+" does not exist!")
                        else:
                            df = pd.read_csv(file_path).copy()
                            df["perturb_function"] = perb_func
                            # calculate model accuracy
                            acc = (df["y_true"]==df["y_pred"]).mean()*100
                            df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                        perturb_function=[perb_func], accuracy=[acc]))
                            df_model_performance_perb = pd.concat([df_model_performance_perb, df_perf], axis=0).copy()
                    else:
                        df_perf = df_model_performance_perb_0.loc[[(benchmark, model_name, weight_variant, perb_func)]].reset_index().copy()
                        df_model_performance_perb = pd.concat([df_model_performance_perb, df_perf], axis=0).copy()

                # 2) Calculate the OoD detectors' performance metrics.
                df_detector_performance_0 = pd.DataFrame()
                if os.path.exists(os.path.join(save_dir, "detector_performance.csv")):
                    # Load model confidence and OoD scores of seeds & perturbed samples from the OoD datasets.
                    df_detector_performance_0 = pd.read_csv(os.path.join(save_dir, "detector_performance.csv")).copy()
                    df_detector_performance_0.set_index(["benchmark", "model", "variant", "dataset", "score_function"], inplace=True)

                for score_func in score_functions:

                    if (score_func+"_score" in df_id_all_scores.columns):
                        id_scores = df_id_all_scores[score_func+"_score"].to_numpy()
                        ood_scores_all = np.array([])
                        for dataset in ood_datasets:
                            
                            if (benchmark, model_name, weight_variant, dataset, score_func) not in df_detector_performance_0.index or refresh:
                                
                                file_path = os.path.join(scores_dir, dataset, "scores", "temp_all_scores.csv")
                                if os.path.exists(file_path):
                                    df = pd.read_csv(file_path).copy()
                                
                                    if (score_func+"_score" in df.columns):
                                        ood_scores = df[score_func+"_score"].to_numpy()
                                        ood_scores_all = np.concatenate([ood_scores_all, ood_scores], axis=0)
                                        # calculate detector performance metrics: FPR95, AUROC, AUPR_IN, AUPR_OUT
                                        auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out = \
                                            get_ood_measures(id_scores, ood_scores)
                                        
                                        df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                                    dataset=[dataset], score_function=[score_func], FPR95=[fpr*100], 
                                                                    AUROC=[auroc], AUPR_IN=[aupr_in], AUPR_OUT=[aupr_out]), )
                                        df_detector_performance = pd.concat([df_detector_performance, df_perf], axis=0).copy()
                            
                                else:
                                    print("File "+file_path+" does not exist!")

                            else:
                                df_perf = df_detector_performance_0.loc[[(benchmark, model_name, weight_variant, dataset, score_func)]].reset_index().copy()
                                df_detector_performance = pd.concat([df_detector_performance, df_perf], axis=0).copy()
                
                        # Calculate the average detector performance metrics across different OoD datasets.
                        if (benchmark, model_name, weight_variant, "average", score_func) not in df_detector_performance_0.index or refresh:
                            if len(ood_scores_all) > 0:
                                auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out = \
                                    get_ood_measures(id_scores, ood_scores_all)
                                
                                df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                            dataset=["average"], score_function=[score_func], 
                                                            FPR95=[fpr*100], AUROC=[auroc], AUPR_IN=[aupr_in], AUPR_OUT=[aupr_out]))
                                df_detector_performance = pd.concat([df_detector_performance, df_perf], axis=0).copy()
                        else:
                            df_perf = df_detector_performance_0.loc[[(benchmark, model_name, weight_variant, "average", score_func)]].reset_index().copy()
                            df_detector_performance = pd.concat([df_detector_performance, df_perf], axis=0).copy()

                df_detector_performance_seed_0 = pd.DataFrame()
                if os.path.exists(os.path.join(save_dir, "detector_performance_seed.csv")):
                    df_detector_performance_seed_0 = pd.read_csv(os.path.join(save_dir, "detector_performance_seed.csv")).copy()
                    df_detector_performance_seed_0.set_index(["benchmark", "model", "variant", "dataset", "score_function"], inplace=True)

                print("Loading OoD seeds' scores.")
                for score_func in score_functions:
                    if (score_func+"_score" in df_id_all_scores.columns):
                        id_scores = df_id_all_scores[score_func+"_score"].to_numpy()
                        ood_scores_all = np.array([])
                        for dataset in ood_datasets:
                            
                            if (benchmark, model_name, weight_variant, dataset, score_func) not in df_detector_performance_seed_0.index or refresh:

                                file_path = os.path.join(scores_dir, dataset, "scores", "temp_scores.csv")
                                if os.path.exists(file_path):
                                    df = pd.read_csv(file_path).copy()
                                    df = df.drop(["idx", "y_true", "y_pred"], axis=1).copy()
                                
                                    if (score_func+"_score" in df.columns):
                                        ood_scores = df[score_func+"_score"].to_numpy()
                                        ood_scores_all = np.concatenate([ood_scores_all, ood_scores], axis=0)

                                        # calculate detector performance metrics: FPR95, AUROC, AUPR_IN, AUPR_OUT
                                        auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out = \
                                            get_ood_measures(id_scores, ood_scores)
                                        
                                        df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                                    dataset=[dataset], score_function=[score_func], FPR95=[fpr*100], 
                                                                    AUROC=[auroc], AUPR_IN=[aupr_in], AUPR_OUT=[aupr_out]), )
                                        df_detector_performance_seed = pd.concat([df_detector_performance_seed, df_perf], axis=0).copy()
                            
                                else:
                                    print("File "+file_path+" does not exist!")
                                
                            else:
                                df_perf = df_detector_performance_seed_0.loc[[(benchmark, model_name, weight_variant, dataset, score_func)]].reset_index().copy()
                                df_detector_performance_seed = pd.concat([df_detector_performance_seed, df_perf], axis=0).copy()

                    # Calculate the average detector performance metrics across different OoD datasets.
                    if (benchmark, model_name, weight_variant, "average", score_func) not in df_detector_performance_seed_0.index or refresh:
                        if len(ood_scores_all) > 0:
                            auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out = \
                                get_ood_measures(id_scores, ood_scores_all)
                            
                            df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                        dataset=["average"], score_function=[score_func], 
                                                        FPR95=[fpr*100], AUROC=[auroc], AUPR_IN=[aupr_in], AUPR_OUT=[aupr_out]))
                            df_detector_performance_seed = pd.concat([df_detector_performance_seed, df_perf], axis=0).copy()
                    else:
                        df_perf = df_detector_performance_seed_0.loc[[(benchmark, model_name, weight_variant, "average", score_func)]].reset_index().copy()
                        df_detector_performance_seed = pd.concat([df_detector_performance_seed, df_perf], axis=0).copy()

                df_detector_performance_perb_0 = pd.DataFrame()
                if os.path.exists(os.path.join(save_dir, "detector_performance_perb.csv")):
                    df_detector_performance_perb_0 = pd.read_csv(os.path.join(save_dir, "detector_performance_perb.csv")).copy()
                    df_detector_performance_perb_0.set_index(["benchmark", "model", "variant", "dataset", "perturb_function", "score_function"], inplace=True)
        
                print("Loading OoD perturbed samples' scores.")
                for perb_func in perturb_functions:
                    print("> Perturbation function:", perb_func)                        
                    for score_func in score_functions:
                        if (score_func+"_score" in df_id_all_scores.columns):
                            id_scores = df_id_all_scores[score_func+"_score"].to_numpy()
                            ood_scores_all = np.array([])
                            for dataset in ood_datasets:
                                
                                if (benchmark, model_name, weight_variant, dataset, perb_func, score_func) not in df_detector_performance_perb_0.index or refresh:

                                    file_path = os.path.join(scores_dir, dataset, "scores", f"perb_{perb_func}_scores.csv")
                                    if not os.path.exists(file_path):
                                        if score_func == score_functions[0]:
                                            print("File "+file_path+" does not exist!")
                                    else:
                                        df = pd.read_csv(file_path).copy()
                            
                                        if (score_func+"_score" in df.columns):
                                            ood_scores = df[score_func+"_score"].to_numpy()
                                            ood_scores_all = np.concatenate([ood_scores_all, ood_scores], axis=0)
                                            # calculate detector performance metrics: FPR95, AUROC, AUPR_IN, AUPR_OUT
                                            auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out = \
                                                get_ood_measures(id_scores, ood_scores)
                                            
                                            df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                                        dataset=[dataset], perturb_function=[perb_func],
                                                                        score_function=[score_func], FPR95=[fpr*100], 
                                                                        AUROC=[auroc], AUPR_IN=[aupr_in], AUPR_OUT=[aupr_out]))
                                            df_detector_performance_perb = pd.concat([df_detector_performance_perb, df_perf], axis=0).copy()
                                else:
                                    df_perf = df_detector_performance_perb_0.loc[[(benchmark, model_name, weight_variant, dataset, perb_func, score_func)]].reset_index().copy()
                                    df_detector_performance_perb = pd.concat([df_detector_performance_perb, df_perf], axis=0).copy()

                            if (benchmark, model_name, weight_variant, "average", perb_func, score_func) not in df_detector_performance_perb_0.index or refresh:
                                # Calculate the average detector performance metrics across different OoD datasets.
                                if len(ood_scores_all) > 0:
                                    auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out = \
                                            get_ood_measures(id_scores, ood_scores_all)
                                        
                                    df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                                dataset=["average"], perturb_function=[perb_func], 
                                                                score_function=[score_func], 
                                                                FPR95=[fpr*100], AUROC=[auroc], AUPR_IN=[aupr_in], AUPR_OUT=[aupr_out]))
                                    df_detector_performance_perb = pd.concat([df_detector_performance_perb, df_perf], axis=0).copy()
                            else:
                                df_perf = df_detector_performance_perb_0.loc[[(benchmark, model_name, weight_variant, "average", perb_func, score_func)]].reset_index().copy()
                                df_detector_performance_perb = pd.concat([df_detector_performance_perb, df_perf], axis=0).copy()
    
    # Calculate the average model performance metrics across different perturbations and severities.
    if len(df_model_performance_perb) > 0:
        df_model_performance_perb = df_model_performance_perb.sort_values(by=["benchmark", "model", "variant", "perturb_function"],
                                                                          key=lambda x: x.str.lower()).copy()
        df_aver = df_model_performance_perb.groupby(["benchmark", "model", "variant"]).mean().copy().reset_index()
        df_aver["perturb_function"] = "average"
        df_model_performance_perb = pd.concat([df_model_performance_perb, df_aver], axis=0).copy()

    # Calculate the average detector performance metrics under different perturbations and severities.
    if len(df_detector_performance_perb) > 0:
        df_detector_performance_perb = df_detector_performance_perb.sort_values(by=["benchmark", "model", "variant", "dataset", "perturb_function", 
                                                                                    "score_function"],
                                                                                key=lambda x: x.str.lower()).copy()
        df_aver = df_detector_performance_perb.groupby(["benchmark", "model", "variant", "dataset", 
                                                        "score_function"]).mean().copy().reset_index()
        df_aver["perturb_function"] = "average"
        df_detector_performance_perb = pd.concat([df_detector_performance_perb, df_aver], axis=0).copy()

    # Save the performance metrics in csv files
    df_model_performance = df_model_performance.reset_index(drop=True)
    df_model_performance.to_csv(os.path.join(save_dir, "model_performance.csv"), index=False)
    df_model_performance_seed = df_model_performance_seed.reset_index(drop=True)
    df_model_performance_seed.to_csv(os.path.join(save_dir, "model_performance_seed.csv"), index=False)
    df_model_performance_perb = df_model_performance_perb.reset_index(drop=True)
    df_model_performance_perb.to_csv(os.path.join(save_dir, "model_performance_perb.csv"), index=False)
    df_detector_performance = df_detector_performance.reset_index(drop=True)
    df_detector_performance.to_csv(os.path.join(save_dir, "detector_performance.csv"), index=False)
    df_detector_performance_seed = df_detector_performance_seed.reset_index(drop=True)
    df_detector_performance_seed.to_csv(os.path.join(save_dir, "detector_performance_seed.csv"), index=False)
    df_detector_performance_perb =df_detector_performance_perb.reset_index(drop=True)
    df_detector_performance_perb.to_csv(os.path.join(save_dir, "detector_performance_perb.csv"), index=False)

def _get_model_detector_performance_severity(configs, refresh=False): 
    """
    This function processes raw experiment results (model confidence & OoD scores) and 
    evaluates the DNN model & OoD detectors' performance on ID & OoD datasets on the seeds 
    and the perturbed samples, respectively. Specifically, it evaluates the performance under
    different perturbation severities.
    The evaluation results are saved in csv files in "results/eval/performance/severity/" 
    folder, including:
    
    - Model performance metrics: Accuracy.
    - OoD detectors' performance metrics: FPR95, AUROC, AUPR_IN, AUPR_OUT.

    Args:
        configs (dict): A dictionary containing the configurations for the evaluation.
        refresh (bool, optional): If True, refreshes the evaluation results. Defaults to False.
    
    Returns:
        None
    """

    save_dir = os.path.join("results", "eval", "severity_levels", "performance")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    score_functions = configs["score_functions"]
    perturb_functions = configs["perturb_functions"]
    rand_seed = configs["rand_seed"]

    df_model_performance = pd.DataFrame()
    df_model_performance_seed = pd.DataFrame()
    df_model_performance_perb = pd.DataFrame()
    df_detector_performance = pd.DataFrame()
    df_detector_performance_seed = pd.DataFrame()
    df_detector_performance_perb = pd.DataFrame()

    for benchmark in configs["benchmark"]:
        
        print("\n=========================")
        print("Evaluating DNN model & OoD detectors' performance.")
        print("=========================")
        print("Benchmark:", benchmark)
        ood_datasets = configs["benchmark"][benchmark]["ood_datasets"]

        for model_name in configs["benchmark"][benchmark]["model"]:
            
            print("-------------------------")
            print("model_name:", model_name)
            weights = configs["benchmark"][benchmark]["model"][model_name]

            for weight_variant in weights:

                print("-------------------------")
                print("weight_variant:", weight_variant)
                scores_dir = os.path.join("results", benchmark.lower(), str(rand_seed), model_name, weight_variant)
                print("scores_dir:", scores_dir)

                print("Loading ID scores.")
                df_id_all_scores = pd.DataFrame()
                file_path = os.path.join(scores_dir, benchmark, "scores", "temp_all_scores.csv")
                if os.path.exists(file_path):
                    df_id_all_scores = pd.read_csv(file_path).copy()
                else:
                    print("File "+file_path+" does not exist!")
                    continue

                # 1) Calculate the DNN model's performance metrics.
                # Load model confidence and OoD scores of seeds & perturbed samples from the ID dataset.
                df_model_performance_0 = pd.DataFrame()
                if os.path.exists(os.path.join(save_dir, "model_performance.csv")):
                    df_model_performance_0 = pd.read_csv(os.path.join(save_dir, "model_performance.csv")).copy()
                    df_model_performance_0.set_index(["benchmark", "model", "variant"], inplace=True)
                    df_model_performance_0.sort_values(by=["benchmark", "model", "variant"], key=lambda x: x.str.lower(), inplace=True)

                if (benchmark, model_name, weight_variant) not in df_model_performance_0.index or refresh:
                    # calculate model accuracy
                    acc = (df_id_all_scores["y_true"]==df_id_all_scores["y_pred"]).mean()*100
                    df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], accuracy=[acc]))
                    df_model_performance = pd.concat([df_model_performance, df_perf], axis=0).copy()
                else:
                    df_perf = df_model_performance_0.loc[[(benchmark, model_name, weight_variant)]].reset_index().copy()
                    df_model_performance = pd.concat([df_model_performance, df_perf], axis=0).copy()
                
                df_model_performance_seed_0 = pd.DataFrame()
                if os.path.exists(os.path.join(save_dir, "model_performance_seed.csv")):
                    df_model_performance_seed_0 = pd.read_csv(os.path.join(save_dir, "model_performance_seed.csv")).copy()
                    df_model_performance_seed_0.set_index(["benchmark", "model", "variant"], inplace=True)
                    df_model_performance_seed_0.sort_values(by=["benchmark", "model", "variant"], key=lambda x: x.str.lower(), inplace=True)
                if (benchmark, model_name, weight_variant) not in df_model_performance_seed_0.index or refresh:
                    print("Loading ID seeds' scores.")
                    file_path = os.path.join(scores_dir, benchmark, "scores", "temp_scores.csv")
                    if os.path.exists(file_path):
                        df_id_seed_scores = pd.read_csv(file_path).copy()
                        # calculate model accuracy
                        acc = (df_id_seed_scores["y_true"]==df_id_seed_scores["y_pred"]).mean()*100
                        df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], accuracy=[acc]))
                        df_model_performance_seed = pd.concat([df_model_performance_seed, df_perf], axis=0).copy()
                    else:
                        print("File "+file_path+" does not exist!")
                else:
                    df_perf = df_model_performance_seed_0.loc[[(benchmark, model_name, weight_variant)]].reset_index().copy()
                    df_model_performance_seed = pd.concat([df_model_performance_seed, df_perf], axis=0).copy()
                
                df_model_performance_perb_0 = pd.DataFrame()
                if os.path.exists(os.path.join(save_dir, "model_performance_perb.csv")):
                    df_model_performance_perb_0 = pd.read_csv(os.path.join(save_dir, "model_performance_perb.csv")).copy()
                    df_model_performance_perb_0.set_index(["benchmark", "model", "variant", "perturb_function", "severity"], inplace=True)
                    df_model_performance_perb_0.sort_values(by=["benchmark", "model", "variant", "perturb_function", "severity"], 
                                                               inplace=True)

                print("Loading ID perturbed samples' scores.")
                for perb_func in perturb_functions:
                    print("> Perturbation function:", perb_func)
                    for severity in range(1, 6):
                        if (benchmark, model_name, weight_variant, perb_func, str(severity)) not in df_model_performance_perb_0.index or refresh:
                            file_path = os.path.join(scores_dir, benchmark, "scores", 
                                                    f"perb_{perb_func}_level_{severity}_scores.csv")
                            if not os.path.exists(file_path):
                                print("File "+file_path+" does not exist!")
                            else:
                                df = pd.read_csv(file_path).copy()
                                df["perturb_function"] = perb_func
                                # calculate model accuracy
                                acc = (df["y_true"]==df["y_pred"]).mean()*100
                                df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                            perturb_function=[perb_func], severity=[severity], accuracy=[acc]))
                                df_model_performance_perb = pd.concat([df_model_performance_perb, df_perf], axis=0).copy()
                        else:
                            df_perf = df_model_performance_perb_0.loc[[(benchmark, model_name, weight_variant, perb_func, str(severity))]].reset_index().copy()
                            df_model_performance_perb = pd.concat([df_model_performance_perb, df_perf], axis=0).copy()

                
                # 2) Calculate the OoD detectors' performance metrics.
                df_detector_performance_0 = pd.DataFrame()
                if os.path.exists(os.path.join(save_dir, "detector_performance.csv")):
                    # Load model confidence and OoD scores of seeds & perturbed samples from the OoD datasets.
                    df_detector_performance_0 = pd.read_csv(os.path.join(save_dir, "detector_performance.csv")).copy()
                    df_detector_performance_0.set_index(["benchmark", "model", "variant", "dataset", "score_function"], inplace=True)
                    df_detector_performance_0.sort_values(by=["benchmark", "model", "variant", "dataset", "score_function"], 
                                                        key=lambda x: x.str.lower(), inplace=True)
                for score_func in score_functions:

                    if (score_func+"_score" in df_id_all_scores.columns):
                        id_scores = df_id_all_scores[score_func+"_score"].to_numpy()

                        ood_scores_all = np.array([])
                        for dataset in ood_datasets:
                            
                            if (benchmark, model_name, weight_variant, dataset, score_func) not in df_detector_performance_0.index or refresh:
                                
                                file_path = os.path.join(scores_dir, dataset, "scores", "temp_all_scores.csv")
                                if os.path.exists(file_path):
                                    df = pd.read_csv(file_path).copy()
                                
                                    if (score_func+"_score" in df.columns):
                                        ood_scores = df[score_func+"_score"].to_numpy()
                                        ood_scores_all = np.concatenate([ood_scores_all, ood_scores], axis=0)
                                        # calculate detector performance metrics: FPR95, AUROC, AUPR_IN, AUPR_OUT
                                        auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out = \
                                            get_ood_measures(id_scores, ood_scores)
                                        
                                        df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                                    dataset=[dataset], score_function=[score_func], FPR95=[fpr*100], 
                                                                    AUROC=[auroc], AUPR_IN=[aupr_in], AUPR_OUT=[aupr_out]), )
                                        df_detector_performance = pd.concat([df_detector_performance, df_perf], axis=0).copy()
                            
                                else:
                                    print("File "+file_path+" does not exist!")

                            else:
                                df_perf = df_detector_performance_0.loc[[(benchmark, model_name, weight_variant, dataset, score_func)]].reset_index().copy()
                                df_detector_performance = pd.concat([df_detector_performance, df_perf], axis=0).copy()
                
                        # Calculate the average detector performance metrics across different OoD datasets.
                        if (benchmark, model_name, weight_variant, "average", score_func) not in df_detector_performance_0.index or refresh:
                            if len(ood_scores_all) > 0:
                                auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out = \
                                    get_ood_measures(id_scores, ood_scores_all)
                                
                                df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                            dataset=["average"], score_function=[score_func], 
                                                            FPR95=[fpr*100], AUROC=[auroc], AUPR_IN=[aupr_in], AUPR_OUT=[aupr_out]))
                                df_detector_performance = pd.concat([df_detector_performance, df_perf], axis=0).copy()
                        else:
                            df_perf = df_detector_performance_0.loc[[(benchmark, model_name, weight_variant, "average", score_func)]].reset_index().copy()
                            df_detector_performance = pd.concat([df_detector_performance, df_perf], axis=0).copy()

                df_detector_performance_seed_0 = pd.DataFrame()
                if os.path.exists(os.path.join(save_dir, "detector_performance_seed.csv")):
                    df_detector_performance_seed_0 = pd.read_csv(os.path.join(save_dir, "detector_performance_seed.csv")).copy()
                    df_detector_performance_seed_0.set_index(["benchmark", "model", "variant", "dataset", "score_function"], inplace=True)
                    df_detector_performance_seed_0.sort_values(by=["benchmark", "model", "variant", "dataset", "score_function"], 
                                                            key=lambda x: x.str.lower(), inplace=True)
                print("Loading OoD seeds' scores.")
                for score_func in score_functions:
                    if (score_func+"_score" in df_id_all_scores.columns):
                        id_scores = df_id_all_scores[score_func+"_score"].to_numpy()

                        ood_scores_all = np.array([])
                        for dataset in ood_datasets:
                            
                            if (benchmark, model_name, weight_variant, dataset, score_func) not in df_detector_performance_seed_0.index or refresh:

                                file_path = os.path.join(scores_dir, dataset, "scores", "temp_scores.csv")
                                if os.path.exists(file_path):
                                    df = pd.read_csv(file_path).copy()
                                    df = df.drop(["idx", "y_true", "y_pred"], axis=1).copy()
                                
                                    if (score_func+"_score" in df.columns):
                                        ood_scores = df[score_func+"_score"].to_numpy()
                                        ood_scores_all = np.concatenate([ood_scores_all, ood_scores], axis=0)

                                        # calculate detector performance metrics: FPR95, AUROC, AUPR_IN, AUPR_OUT
                                        auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out = \
                                            get_ood_measures(id_scores, ood_scores)
                                        
                                        df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                                    dataset=[dataset], score_function=[score_func], FPR95=[fpr*100], 
                                                                    AUROC=[auroc], AUPR_IN=[aupr_in], AUPR_OUT=[aupr_out]), )
                                        df_detector_performance_seed = pd.concat([df_detector_performance_seed, df_perf], axis=0).copy()
                            
                                else:
                                    print("File "+file_path+" does not exist!")
                                
                            else:
                                df_perf = df_detector_performance_seed_0.loc[[(benchmark, model_name, weight_variant, dataset, score_func)]].reset_index().copy()
                                df_detector_performance_seed = pd.concat([df_detector_performance_seed, df_perf], axis=0).copy()

                    # Calculate the average detector performance metrics across different OoD datasets.
                    if (benchmark, model_name, weight_variant, "average", score_func) not in df_detector_performance_seed_0.index or refresh:
                        if len(ood_scores_all) > 0:
                            auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out = \
                                get_ood_measures(id_scores, ood_scores_all)
                            
                            df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                        dataset=["average"], score_function=[score_func], 
                                                        FPR95=[fpr*100], AUROC=[auroc], AUPR_IN=[aupr_in], AUPR_OUT=[aupr_out]))
                            df_detector_performance_seed = pd.concat([df_detector_performance_seed, df_perf], axis=0).copy()
                    else:
                        df_perf = df_detector_performance_seed_0.loc[[(benchmark, model_name, weight_variant, "average", score_func)]].reset_index().copy()
                        df_detector_performance_seed = pd.concat([df_detector_performance_seed, df_perf], axis=0).copy()

                df_detector_performance_perb_0 = pd.DataFrame()
                if os.path.exists(os.path.join(save_dir, "detector_performance_perb.csv")):
                    df_detector_performance_perb_0 = pd.read_csv(os.path.join(save_dir, "detector_performance_perb.csv")).copy()
                    df_detector_performance_perb_0.set_index(["benchmark", "model", "variant", "dataset", "perturb_function", "severity", "score_function"], inplace=True)
                    df_detector_performance_perb_0.sort_values(by=["benchmark", "model", "variant", "dataset", "perturb_function", "severity", "score_function"],
                                                               inplace=True)
                print("Loading OoD perturbed samples' scores.")
                for perb_func in perturb_functions:
                    print("> Perturbation function:", perb_func)                        
                    for score_func in score_functions:
                        if (score_func+"_score" in df_id_all_scores.columns):
                            id_scores = df_id_all_scores[score_func+"_score"].to_numpy()
                            
                            for severity in range(1, 6):
                                ood_scores_all = np.array([])
                                for dataset in ood_datasets:
                                    
                                    if (benchmark, model_name, weight_variant, dataset, perb_func, str(severity), score_func) not in df_detector_performance_perb_0.index or refresh:

                                        file_path = os.path.join(scores_dir, dataset, "scores", f"perb_{perb_func}_level_{severity}_scores.csv")
                                        if not os.path.exists(file_path):
                                            if score_func == score_functions[0]:
                                                print("File "+file_path+" does not exist!")
                                        else:
                                            df = pd.read_csv(file_path).copy()
                                
                                            if (score_func+"_score" in df.columns):
                                                ood_scores = df[score_func+"_score"].to_numpy()
                                                ood_scores_all = np.concatenate([ood_scores_all, ood_scores], axis=0)
                                                # calculate detector performance metrics: FPR95, AUROC, AUPR_IN, AUPR_OUT
                                                auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out = \
                                                    get_ood_measures(id_scores, ood_scores)
                                                
                                                df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                                            dataset=[dataset], perturb_function=[perb_func], severity=[severity],
                                                                            score_function=[score_func], FPR95=[fpr*100], 
                                                                            AUROC=[auroc], AUPR_IN=[aupr_in], AUPR_OUT=[aupr_out]))
                                                df_detector_performance_perb = pd.concat([df_detector_performance_perb, df_perf], axis=0).copy()
                                    else:
                                        df_perf = df_detector_performance_perb_0.loc[[(benchmark, model_name, weight_variant, dataset, perb_func, str(severity), score_func)]].reset_index().copy()
                                        df_detector_performance_perb = pd.concat([df_detector_performance_perb, df_perf], axis=0).copy()

                                if (benchmark, model_name, weight_variant, "average", perb_func, str(severity), score_func) not in df_detector_performance_perb_0.index or refresh:
                                    # Calculate the average detector performance metrics across different OoD datasets.
                                    if len(ood_scores_all) > 0:
                                        auroc, aupr_in, aupr_out, fpr, threshold, fpr_out, threshold_out = \
                                                get_ood_measures(id_scores, ood_scores_all)
                                            
                                        df_perf = pd.DataFrame(dict(benchmark=[benchmark], model=[model_name], variant=[weight_variant], 
                                                                    dataset=["average"], perturb_function=[perb_func], severity=[severity], 
                                                                    score_function=[score_func], 
                                                                    FPR95=[fpr*100], AUROC=[auroc], AUPR_IN=[aupr_in], AUPR_OUT=[aupr_out]))
                                        df_detector_performance_perb = pd.concat([df_detector_performance_perb, df_perf], axis=0).copy()
                                else:
                                    df_perf = df_detector_performance_perb_0.loc[[(benchmark, model_name, weight_variant, "average", perb_func, str(severity), score_func)]].reset_index().copy()
                                    df_detector_performance_perb = pd.concat([df_detector_performance_perb, df_perf], axis=0).copy()


    # Calculate the average model performance metrics across different perturbations and severities.
    if len(df_model_performance_perb) > 0:
        df_model_performance_perb = df_model_performance_perb.sort_values(by=["benchmark", "model", "variant", "perturb_function", "severity"],
                                                                          ).copy()
        df_aver = df_model_performance_perb.groupby(["benchmark", "model", "variant", "perturb_function"]).mean().copy().reset_index()
        df_aver["severity"] = "average"
        df_model_performance_perb = pd.concat([df_model_performance_perb, df_aver], axis=0).copy()

        df_aver = df_model_performance_perb.groupby(["benchmark", "model", "variant", "severity"]).mean().copy().reset_index()
        df_aver["perturb_function"] = "average"
        df_model_performance_perb = pd.concat([df_model_performance_perb, df_aver], axis=0).copy()

        df_aver = df_model_performance_perb.groupby(["benchmark", "model", "variant"]).mean().copy().reset_index()
        df_aver["perturb_function"] = "average"
        df_aver["severity"] = "average"
        df_model_performance_perb = pd.concat([df_model_performance_perb, df_aver], axis=0).copy()

    # Calculate the average detector performance metrics under different perturbations and severities.
    if len(df_detector_performance_perb) > 0:
        df_detector_performance_perb = df_detector_performance_perb.sort_values(by=["benchmark", "model", "variant", "dataset", "perturb_function", 
                                                                                    "severity", "score_function"],
                                                                                ).copy()
        df_aver = df_detector_performance_perb.groupby(["benchmark", "model", "variant", "dataset", 
                                                        "score_function", "perturb_function"]).mean().copy().reset_index()
        df_aver["severity"] = "average"
        df_detector_performance_perb = pd.concat([df_detector_performance_perb, df_aver], axis=0).copy()

        df_aver = df_detector_performance_perb.groupby(["benchmark", "model", "variant", "dataset", 
                                                        "score_function", "severity"]).mean().copy().reset_index()
        df_aver["perturb_function"] = "average"
        df_detector_performance_perb = pd.concat([df_detector_performance_perb, df_aver], axis=0).copy()

        df_aver = df_detector_performance_perb.groupby(["benchmark", "model", "variant", "dataset", 
                                                        "score_function"]).mean().copy().reset_index()
        df_aver["perturb_function"] = "average"
        df_aver["severity"] = "average"
        df_detector_performance_perb = pd.concat([df_detector_performance_perb, df_aver], axis=0).copy()


    # Save the performance metrics in csv files
    df_model_performance = df_model_performance.reset_index(drop=True)
    df_model_performance.to_csv(os.path.join(save_dir, "model_performance.csv"), index=False)
    df_model_performance_seed = df_model_performance_seed.reset_index(drop=True)
    df_model_performance_seed.to_csv(os.path.join(save_dir, "model_performance_seed.csv"), index=False)
    df_model_performance_perb = df_model_performance_perb.reset_index(drop=True)
    df_model_performance_perb.to_csv(os.path.join(save_dir, "model_performance_perb.csv"), index=False)
    df_detector_performance = df_detector_performance.reset_index(drop=True)
    df_detector_performance.to_csv(os.path.join(save_dir, "detector_performance.csv"), index=False)
    df_detector_performance_seed = df_detector_performance_seed.reset_index(drop=True)
    df_detector_performance_seed.to_csv(os.path.join(save_dir, "detector_performance_seed.csv"), index=False)
    df_detector_performance_perb =df_detector_performance_perb.reset_index(drop=True)
    df_detector_performance_perb.to_csv(os.path.join(save_dir, "detector_performance_perb.csv"), index=False)

def get_model_detector_performance(configs, refresh=False):
    """
    This function processes raw experiment results (model confidence & OoD scores) and 
    evaluates the DNN model & OoD detectors' performance on ID & OoD datasets on the seeds 
    and the perturbed samples, respectively.
    The evaluation results are saved in csv files in "results/eval/performance/" 
    folder, including:
    
    - Model performance metrics: Accuracy.
    - OoD detectors' performance metrics: FPR95, AUROC, AUPR_IN, AUPR_OUT.

    Args:
        configs (dict): A dictionary containing the configurations for the evaluation.
        refresh (bool, optional): If True, refreshes the evaluation results. Defaults to False.
        severity (bool, optional): If True, evaluates the performance under different perturbation 
            severities. Defaults to False.
    Returns:
        None
    """
    severity = configs["eval_severity"]
    if severity:
        _get_model_detector_performance_severity(configs, refresh)
    else:
        _get_model_detector_performance(configs, refresh)

def _count_mae_dae_samples(configs, refresh=False):
    """
    This function processes raw experiment results (model confidence & OoD scores) and 
    finds out MAE and DAE samples for each OoD detector under different perturbations. 
    The outputs are saved in csv files in "results/eval/intermediate_results/" folder.

    Args:
        configs (dict): A dictionary containing the configurations for the evaluation.
        refresh (bool, optional): If True, refreshes the evaluation results. Defaults to False.
    
    Returns:
        None
    """
    
    score_functions = configs["score_functions"]
    perturb_functions = configs["perturb_functions"]
    rand_seed = configs["rand_seed"]
    print("\n=========================")
    print("Analyzing and recording DAE samples.")
    
    for benchmark in configs["benchmark"]:
        print("=========================")
        print("Benchmark:", benchmark)
        ood_datasets = configs["benchmark"][benchmark]["ood_datasets"]
        for model_name in configs["benchmark"][benchmark]["model"]:
            weights = configs["benchmark"][benchmark]["model"][model_name]
            print("model_name:", model_name)
            for weight_variant in weights:
                print("-------------------------")
                print("weight_variant:", weight_variant)
                scores_dir = os.path.join("results", benchmark.lower(), str(rand_seed), model_name, weight_variant)
                print("scores_dir:", scores_dir)
                filepath = os.path.join(scores_dir, benchmark, "scores", "temp_all_scores.csv")

                # OoD detectors' thresholds fixed at ID test dataset TPR=95%
                thr_dict = dict()
                thr_dict.update(get_thr_tpr(filepath, score_functions, tpr=0.95))
                print("thr_dict:", thr_dict)

                # Load model confidence and OoD scores of seeds & perturbed samples from the ID dataset.
                print("Loading ID seeds' scores.")
                file_path = os.path.join(scores_dir, benchmark, "scores", "temp_scores.csv")
                if os.path.exists(file_path):
                    df_id_seed_scores = pd.read_csv(file_path).copy()

                    # We only consider correctly predicted ID samples by the DNN model
                    df_id_seed_scores = df_id_seed_scores[(df_id_seed_scores["y_true"] == df_id_seed_scores["y_pred"])].copy()
                    print("Correct predictions on ID dataset:", len(df_id_seed_scores))

                    save_dir = "results/eval/intermediate_results/" + f"{benchmark.lower()}_{model_name}_{weight_variant}"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    print("Analyzing ID perturbed samples' test outcomes.")
                    if (not os.path.exists(os.path.join(save_dir, f"id_mae_dae_record.csv"))) or refresh: 
                        df_id_outcomes = pd.DataFrame()
                        for perb_func in perturb_functions:
                            print("> Perturbation function:", perb_func)
                            file_path = os.path.join(scores_dir, benchmark, "scores", f"perb_{perb_func}_scores.csv")
                            if not os.path.exists(file_path):
                                print(file_path+" not found!")
                            else:
                                # Load ID perturbed samples' scores.
                                df_perb = pd.read_csv(file_path).copy()
                                # df_perb = df_perb.sample(n=100)        
                                # Merge the model confidence and OoD scores of ID seeds & perturbed samples
                                df_id_scores = pd.merge(df_perb, df_id_seed_scores, on=["idx"], how="inner", 
                                                suffixes=["_perb", "_seed"])
                                
                                df_outcome = df_id_scores[["idx", "conf_seed", "conf_perb"]].copy()
                                # df_outcome["high_conf"] = (df_id_scores["conf_perb"] > (1-alpha))
                                df_outcome["mae"] = ((df_id_scores["y_true_perb"] != df_id_scores["y_pred_perb"]))

                                # Record ID DAE samples of each detector
                                for score_func in score_functions:
                                    if (score_func+"_score_seed" in df_id_scores.columns) and (score_func+"_score_perb" in df_id_scores.columns):
                                        df_outcome["dae_"+score_func] = \
                                            ((df_id_scores[score_func+"_score_perb"] > thr_dict[score_func+"_score"]))
                                        df_outcome.loc[df_id_scores[score_func+"_score_seed"] > thr_dict[score_func+"_score"], 
                                                    "dae_"+score_func] = np.nan
                                    else:
                                        df_outcome["dae_"+score_func] = np.nan
                                df_outcome["perturb_function"] = perb_func
                                df_id_outcomes = pd.concat([df_id_outcomes, df_outcome], axis=0).copy()

                        df_id_outcomes = df_id_outcomes.reset_index(drop=True)
                        print("Saving to "+os.path.join(save_dir, f"id_mae_dae_record.csv"))
                        df_id_outcomes.to_csv(os.path.join(save_dir, "id_mae_dae_record.csv"), index=False)
                else:
                    print("File "+file_path+" does not exist!")

                # Load model confidence and OoD scores of seeds & perturbed samples from the OoD datasets.
                print("Analyzing OoD perturbed samples' test outcomes.")
                
                for dataset in ood_datasets:
                    print("- Dataset:", dataset)
                    if (not os.path.exists(os.path.join(save_dir, f"ood_dae_record_{dataset}.csv"))) or refresh: 
                        df_ood_outcomes = pd.DataFrame()
                        for perb_func in perturb_functions:
                            print("> Perturbation function:", perb_func)

                            file_path = os.path.join(scores_dir, dataset, "scores", "temp_scores.csv")
                            if not os.path.exists(file_path):
                                continue
                            df_seed = pd.read_csv(file_path).copy()

                            file_path = os.path.join(scores_dir, dataset, "scores", f"perb_{perb_func}_scores.csv")
                            if not os.path.exists(file_path):
                                print("File "+file_path+" does not exist!")
                                continue

                            df_perb = pd.read_csv(file_path).copy()
                            # df_perb = df_perb.sample(n=100)
                            
                            # Merge the model confidence and OoD scores of ID seeds & perturbed samples
                            df_ood_scores = pd.merge(df_perb, df_seed, on=["idx"], how="inner", 
                                            suffixes=["_perb", "_seed"])
                            
                            df_outcome = df_ood_scores[["idx", "conf_seed", "conf_perb"]].copy()
                            # df_outcome["high_conf"] = (df_ood_scores["conf_perb"] > (1-alpha))

                            # Record OoD DAE samples of each detector
                            for score_func in score_functions:
                                if (score_func+"_score_seed" in df_ood_scores.columns) and (score_func+"_score_perb" in df_ood_scores.columns):
                                    df_outcome["dae_"+score_func] = \
                                        ((df_ood_scores[score_func+"_score_perb"] <= thr_dict[score_func+"_score"]))
                                    df_outcome.loc[df_ood_scores[score_func+"_score_seed"] <= thr_dict[score_func+"_score"], 
                                                "dae_"+score_func] = np.nan
                                else:
                                    df_outcome["dae_"+score_func] = np.nan
                            df_outcome["perturb_function"] = perb_func
                            df_outcome["dataset"] = dataset
                            df_ood_outcomes = pd.concat([df_ood_outcomes, df_outcome], axis=0).copy()
                        
                        df_ood_outcomes = df_ood_outcomes.reset_index(drop=True)
                        print("Saving to "+os.path.join(save_dir, f"ood_dae_record_{dataset}.csv"))
                        df_ood_outcomes.to_csv(os.path.join(save_dir, f"ood_dae_record_{dataset}.csv"), index=False)
                        
def _count_mae_dae_samples_severity(configs, refresh=False):
    """
    This function processes raw experiment results (model confidence & OoD scores) and 
    finds out MAE and DAE samples for each OoD detector under different perturbations.
    Specifically, it considers different perturbation severities.
    The outputs are saved in csv files in "results/eval/severity_levels/intermediate_results/" folder.

    Args:
        configs (dict): A dictionary containing the configurations for the evaluation.
        refresh (bool, optional): If True, refreshes the evaluation results. Defaults to False.
    
    Returns:
        None
    """
    
    score_functions = configs["score_functions"]
    perturb_functions = configs["perturb_functions"]
    rand_seed = configs["rand_seed"]

    print("\n=========================")
    print("Analyzing and recording DAE samples.")
    for benchmark in configs["benchmark"]:
        print("=========================")
        print("Benchmark:", benchmark)
        ood_datasets = configs["benchmark"][benchmark]["ood_datasets"]
        for model_name in configs["benchmark"][benchmark]["model"]:
            weights = configs["benchmark"][benchmark]["model"][model_name]
        
            print("model_name:", model_name)
            for weight_variant in weights:
                print("-------------------------")
                print("weight_variant:", weight_variant)
                scores_dir = os.path.join("results", benchmark.lower(), str(rand_seed), model_name, weight_variant)
                print("scores_dir:", scores_dir)
                filepath = os.path.join(scores_dir, benchmark, "scores", "temp_all_scores.csv")

                # OoD detectors' thresholds fixed at ID test dataset TPR=95%
                thr_dict = dict()
                thr_dict.update(get_thr_tpr(filepath, score_functions, tpr=0.95))
                print("thr_dict:", thr_dict)

                # Load model confidence and OoD scores of seeds & perturbed samples from the ID dataset.
                print("Loading ID seeds' scores.")
                file_path = os.path.join(scores_dir, benchmark, "scores", "temp_scores.csv")
                if os.path.exists(file_path):
                    df_id_seed_scores = pd.read_csv(file_path).copy()

                    # We only consider correctly predicted ID samples by the DNN model
                    df_id_seed_scores = df_id_seed_scores[(df_id_seed_scores["y_true"] == df_id_seed_scores["y_pred"])].copy()
                    print("Correct predictions on ID dataset:", len(df_id_seed_scores))

                    save_dir = "results/eval/severity_levels/intermediate_results/" + f"{benchmark.lower()}_{model_name}_{weight_variant}"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    print("Analyzing ID perturbed samples' test outcomes.")
                    if (not os.path.exists(os.path.join(save_dir, f"id_mae_dae_record.csv"))) or refresh: 
                        
                        for perb_func in perturb_functions:
                            print("> Perturbation function:", perb_func)
                            df_id_outcomes = pd.DataFrame()
                            for severity in range(1, 6):
                                file_path = os.path.join(scores_dir, benchmark, "scores", f"perb_{perb_func}_level_{severity}_scores.csv")
                                if not os.path.exists(file_path):
                                    print(file_path+" not found!")
                                else:
                                    # Load ID perturbed samples' scores.
                                    df_perb = pd.read_csv(file_path).copy()
                                    # df_perb = df_perb.sample(n=100)        
                                    # Merge the model confidence and OoD scores of ID seeds & perturbed samples
                                    df_id_scores = pd.merge(df_perb, df_id_seed_scores, on=["idx"], how="inner", 
                                                    suffixes=["_perb", "_seed"])
                                    
                                    df_outcome = df_id_scores[["idx", "conf_seed", "conf_perb"]].copy()
                                    # df_outcome["high_conf"] = (df_id_scores["conf_perb"] > (1-alpha))
                                    df_outcome["mae"] = ((df_id_scores["y_true_perb"] != df_id_scores["y_pred_perb"]))

                                    # Record ID DAE samples of each detector
                                    for score_func in score_functions:
                                        if (score_func+"_score_seed" in df_id_scores.columns) and (score_func+"_score_perb" in df_id_scores.columns):
                                            df_outcome["dae_"+score_func] = \
                                                ((df_id_scores[score_func+"_score_perb"] > thr_dict[score_func+"_score"]))
                                            df_outcome.loc[df_id_scores[score_func+"_score_seed"] > thr_dict[score_func+"_score"], 
                                                        "dae_"+score_func] = np.nan
                                        else:
                                            df_outcome["dae_"+score_func] = np.nan
                                    df_outcome["perturb_function"] = perb_func
                                    df_outcome["severity"] = severity
                                    df_id_outcomes = pd.concat([df_id_outcomes, df_outcome], axis=0).copy()

                            df_id_outcomes = df_id_outcomes.reset_index(drop=True)
                            print("Saving to "+os.path.join(save_dir, f"id_mae_dae_record_{perb_func}.csv"))
                            df_id_outcomes.to_csv(os.path.join(save_dir, f"id_mae_dae_record_{perb_func}.csv"), index=False)
                else:
                    print("File "+file_path+" does not exist!")

                # Load model confidence and OoD scores of seeds & perturbed samples from the OoD datasets.
                print("Analyzing OoD perturbed samples' test outcomes.")
                
                for dataset in ood_datasets:
                    print("- Dataset:", dataset)
                    if (not os.path.exists(os.path.join(save_dir, f"ood_dae_record_{dataset}.csv"))) or refresh: 
                        
                        for perb_func in perturb_functions:
                            print("> Perturbation function:", perb_func)
                            
                            file_path = os.path.join(scores_dir, dataset, "scores", "temp_scores.csv")
                            if not os.path.exists(file_path):
                                continue

                            df_ood_outcomes = pd.DataFrame()
                            df_seed = pd.read_csv(file_path).copy()

                            for severity in range(1, 6):
                                file_path = os.path.join(scores_dir, dataset, "scores", f"perb_{perb_func}_level_{severity}_scores.csv")
                                if not os.path.exists(file_path):
                                    print("File "+file_path+" does not exist!")
                                    continue

                                df_perb = pd.read_csv(file_path).copy()
                                # df_perb = df_perb.sample(n=100)
                                
                                # Merge the model confidence and OoD scores of ID seeds & perturbed samples
                                df_ood_scores = pd.merge(df_perb, df_seed, on=["idx"], how="inner", 
                                                suffixes=["_perb", "_seed"])
                                
                                df_outcome = df_ood_scores[["idx", "conf_seed", "conf_perb"]].copy()
                                # df_outcome["high_conf"] = (df_ood_scores["conf_perb"] > (1-alpha))

                                # Record OoD DAE samples of each detector
                                for score_func in score_functions:
                                    if (score_func+"_score_seed" in df_ood_scores.columns) and (score_func+"_score_perb" in df_ood_scores.columns):
                                        df_outcome["dae_"+score_func] = \
                                            ((df_ood_scores[score_func+"_score_perb"] <= thr_dict[score_func+"_score"]))
                                        df_outcome.loc[df_ood_scores[score_func+"_score_seed"] <= thr_dict[score_func+"_score"], 
                                                    "dae_"+score_func] = np.nan
                                    else:
                                        df_outcome["dae_"+score_func] = np.nan
                                df_outcome["severity"] = severity
                                df_ood_outcomes = pd.concat([df_ood_outcomes, df_outcome], axis=0).copy()
                        
                            df_ood_outcomes = df_ood_outcomes.reset_index(drop=True)
                            print("Saving to "+os.path.join(save_dir, f"ood_dae_record_{dataset}_{perb_func}.csv"))
                            df_ood_outcomes.to_csv(os.path.join(save_dir, f"ood_dae_record_{dataset}_{perb_func}.csv"), index=False)
                        
def _get_local_mae_dae_rate(benchmark, ood_datasets, model_name, weights, refresh=False):
    """
    This function calculates the local MAE/DAE rate of each ID/OoD seed, for each OoD 
    detector under different perturbations. 
    The outputs are saved in csv files in the "results/eval/intermediate_results/" folder.

    Args:
        benchmark (str): The benchmark for the evaluation. Can be selected from ["CIFAR10", "Imagenet100"].
        ood_datasets (list): List of OoD datasets.
        model_name (str): The name of the model. Can be selected from ["wrn_40_2", "resnet50"].
        weights (list): List of weight variants. Can be selected from ["NT", "DA", "AT", "PAT"].
        refresh (bool, optional): Whether to refresh the output files. Defaults to False.
        
    Returns:
        None
    """
    
    print("\n=========================")
    print("Calculating local MAE/DAE rates.")
    print("=========================")
    print("Benchmark:", benchmark)
    print("Model_name:", model_name)
    for weight_variant in weights:
        print("-------------------------")
        print("weight_variant:", weight_variant)

        rlt_dir = "results/eval/intermediate_results/" + f"{benchmark.lower()}_{model_name}_{weight_variant}"
        if not os.path.exists(rlt_dir):
            print("  "+rlt_dir+" not exists!")
            continue
        
        print("> Calculating the local MAE/DAE rate on ID dataset.")
        save_path = os.path.join(rlt_dir, f"id_local_mae_dae_rate.csv")
        if (not os.path.exists(save_path)) or refresh: 
            file_path = os.path.join(rlt_dir, f"id_mae_dae_record.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path).copy()
                df = df.groupby(["idx", "perturb_function"]).mean(numeric_only=False).copy().reset_index()
                
                df.to_csv(save_path, index=False)
                print("ID local MAE/DAE rate saved to: "+save_path)
            else:
                print("   "+file_path+" not exists!")
        
        print("> Calculating the local DAE rate on OoD datasets.")
        for dataset in ood_datasets:
            save_path = os.path.join(rlt_dir, f"ood_local_dae_rate_{dataset}.csv")
            if (not os.path.exists(save_path)) or refresh: 
                print(" - Dataset:", dataset)
                file_path = os.path.join(rlt_dir, f"ood_dae_record_{dataset}.csv")
                if not os.path.exists(file_path):
                    print("   "+file_path+" not exists!")
                    continue
                df = pd.read_csv(file_path).copy()
                df.sort_values(by=["idx", "perturb_function"], ignore_index=True, inplace=True)
                df = df.groupby(["idx", "perturb_function"]).mean(numeric_only=False).reset_index()
                df.to_csv(save_path, index=False)
                print("OoD local DAE rate saved to: "+save_path)

def _get_local_mae_dae_rate_severity(benchmark, ood_datasets, model_name, weights, 
                                     perturb_functions, refresh=False):
    """
    This function calculates the local MAE/DAE rate of each ID/OoD seed, for each OoD 
    detector under different perturbations. 
    Specifically, it considers different perturbation severities.
    The outputs are saved in csv files in the "results/eval/severity_levels/intermediate_results/" folder.

    Args:
        benchmark (str): The benchmark for the evaluation. Can be selected from ["CIFAR10", "Imagenet100"].
        ood_datasets (list): List of OoD datasets.
        model_name (str): The name of the model. Can be selected from ["wrn_40_2", "resnet50"].
        weights (list): List of weight variants. Can be selected from ["NT", "DA", "AT", "PAT"].
        refresh (bool, optional): Whether to refresh the output files. Defaults to False.
        
    Returns:
        None
    
    """
    print("\n=========================")
    print("Calculating local MAE/DAE rates.")
    print("=========================")
    print("Benchmark:", benchmark)
    print("Model_name:", model_name)
    for weight_variant in weights:
        print("-------------------------")
        print("weight_variant:", weight_variant)

        rlt_dir = "results/eval/severity_levels/intermediate_results/" + f"{benchmark.lower()}_{model_name}_{weight_variant}"
        if not os.path.exists(rlt_dir):
            print("  "+rlt_dir+" not exists!")
            continue
        
        print("> Calculating the local MAE/DAE rate on ID dataset.")
        save_path = os.path.join(rlt_dir, f"id_local_mae_dae_rate.csv")
        if (not os.path.exists(save_path)) or refresh:
            df_mae_dae = pd.DataFrame()
            for perb_func in perturb_functions:
                file_path = os.path.join(rlt_dir, f"id_mae_dae_record_{perb_func}.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path).copy()
                    df = df.groupby(["idx", "severity"]).mean(numeric_only=False).copy().reset_index()
                    df["perturb_function"] = perb_func
                    df_mae_dae = pd.concat([df_mae_dae, df], axis=0).copy()
                else:
                    print("   "+file_path+" not exists!")
            df_mae_dae.to_csv(save_path, index=False)
            print("ID local MAE/DAE rate saved to: "+save_path)
        
        print("> Calculating the local DAE rate on OoD datasets.")
        for dataset in ood_datasets:
            save_path = os.path.join(rlt_dir, f"ood_local_dae_rate_{dataset}.csv")
            df_dae = pd.DataFrame()
            if (not os.path.exists(save_path)) or refresh: 
                print(" - Dataset:", dataset)
                for perb_func in perturb_functions:
                    file_path = os.path.join(rlt_dir, f"ood_dae_record_{dataset}_{perb_func}.csv")
                    if not os.path.exists(file_path):
                        print("   "+file_path+" not exists!")
                        continue

                    df = pd.read_csv(file_path).copy()
                    df = df.sort_values(by=["idx", "severity"], ignore_index=True)
                    df = df.groupby(["idx", "severity"]).mean(numeric_only=False).reset_index()
                    df["perturb_function"] = perb_func
                    df_dae = pd.concat([df_dae, df], axis=0).copy()
                
                df_dae.to_csv(save_path, index=False)
                print("OoD local DAE rate saved to: "+save_path)

def _get_mae_dae_statistics(benchmark, ood_datasets, model_name, weights, refresh=False): 
    """
    This function calculates the statistics of local MAE/DAE rate (mean & std) for each OoD 
    detector under different perturbations. The outputs are saved in csv files in the
    "results/eval/intermediate_results/" folder.

    Args:
        benchmark (str): The benchmark for the evaluation. Can be selected from ["CIFAR10", "Imagenet100"].
        ood_datasets (list): List of OoD datasets.
        model_name (str): The name of the model. Can be selected from ["wrn_40_2", "resnet50"].
        weights (list): List of weight variants. Can be selected from ["NT", "DA", "AT", "PAT"].
        refresh (bool, optional): Whether to refresh the output files. Defaults to False.
        
    Returns:
        None
    """
    print("\n=========================")
    print("Calculating MAE/DAE rate statistics.")
    print("=========================")
    print("Benchmark:", benchmark)
    print("model_name:", model_name)
    for weight_variant in weights:
        rlt_dir = "results/eval/intermediate_results/" + f"{benchmark.lower()}_{model_name}_{weight_variant}"
        if not os.path.exists(rlt_dir):
            print(rlt_dir+" not exists!")
            continue

        print("-------------------------")
        print("weight_variant:", weight_variant)

        # Statistics of local MAE/DAE rate on ID dataset.
        save_path = os.path.join(rlt_dir, f"id_local_mae_dae_rate_mean_std.csv")
        if (not os.path.exists(save_path)) or refresh: 
            file_path = os.path.join(rlt_dir, f"id_local_mae_dae_rate.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path).copy()
                df.pop("idx")
                df_mean_perb_func = df.groupby("perturb_function").mean().reset_index()
                df_std_perb_func = df.groupby("perturb_function").std().reset_index()
                df_mean = df.mean().to_frame().T
                df_std = df.std().to_frame().T

                df_mean["perturb_function"] = "average"
                df_std["perturb_function"] = "average"
                df_mean = pd.concat([df_mean, df_mean_perb_func], axis=0)
                df_std = pd.concat([df_std, df_std_perb_func], axis=0)
                df = pd.merge(df_mean, df_std, on=["perturb_function"], how="inner", suffixes=["_mean", "_std"]).set_index(["perturb_function"])
                df = df.mul(100).round(3)
                df = df.reindex(sorted(df.columns), axis=1)
                df.to_csv(save_path, index=True)
                print("ID local MAE/DAE statistics saved to:", save_path)

            else:
                print("  ", file_path+" not exists!")

        # Statistics of local DAE rate on OoD datasets.
        save_path = os.path.join(rlt_dir, f"ood_local_dae_rate_mean_std.csv")
        if (not os.path.exists(save_path)) or refresh: 
            df_ood = pd.DataFrame()
            for dataset in ood_datasets:
                print("> OoD dataset:", dataset)
                file_path = os.path.join(rlt_dir, f"ood_local_dae_rate_{dataset}.csv")
                if not os.path.exists(file_path):
                    print("  ", file_path+" not exists!")
                    continue
                df = pd.read_csv(file_path).copy()
                df.pop("idx")
                df["dataset"] = dataset
                df_ood = pd.concat([df_ood, df], axis=0)

            if len(df_ood) == 0:
                continue
            
            df_ood = df_ood.sort_values(by=["perturb_function", "dataset"], 
                                        key=lambda col: col.str.lower(), ignore_index=True)
            
            df_mean_perb_func_dataset = df_ood.groupby(["perturb_function", "dataset"]).mean().reset_index()
            df_std_perb_func_dataset = df_ood.groupby(["perturb_function", "dataset"]).std().reset_index()
            df_mean_perb_func = df_ood.groupby("perturb_function").mean().reset_index()
            df_std_perb_func = df_ood.groupby("perturb_function").std().reset_index()
            df_mean_dataset = df_ood.groupby("dataset").mean().reset_index()
            df_std_dataset = df_ood.groupby("dataset").std().reset_index()
            df_mean = df_ood.mean().to_frame().T
            df_std = df_ood.std().to_frame().T

            df_mean_perb_func["dataset"] = "average"
            df_std_perb_func["dataset"] = "average"
            df_mean_dataset["perturb_function"] = "average"
            df_std_dataset["perturb_function"] = "average"
            df_mean[["perturb_function", "dataset"]] = "average"
            df_std[["perturb_function", "dataset"]] = "average"

            df_mean = pd.concat([df_mean, df_mean_perb_func, df_mean_dataset, df_mean_perb_func_dataset], axis=0)
            df_std = pd.concat([df_std, df_std_perb_func, df_std_dataset, df_std_perb_func_dataset], axis=0)

            df = pd.merge(df_mean, df_std, on=["perturb_function", "dataset"], how="inner", suffixes=["_mean", "_std"]).set_index(["perturb_function", "dataset"])
            df = df.mul(100).round(3)
            df = df.reindex(sorted(df.columns), axis=1)
            df.to_csv(save_path, index=True)
            print("OoD local DAE statistics saved to:", save_path)

def _get_mae_dae_statistics_severity(benchmark, ood_datasets, model_name, weights, refresh=False): 
    """
    This function calculates the statistics of local MAE/DAE rate (mean & std) for each OoD 
    detector under different perturbations. 
    Specifically, it considers different perturbation severities.
    The outputs are saved in csv files in the "results/eval/severity_levels/intermediate_results/" folder.

    Args:
        benchmark (str): The benchmark for the evaluation. Can be selected from ["CIFAR10", "Imagenet100"].
        ood_datasets (list): List of OoD datasets.
        model_name (str): The name of the model. Can be selected from ["wrn_40_2", "resnet50"].
        weights (list): List of weight variants. Can be selected from ["NT", "DA", "AT", "PAT"].
        refresh (bool, optional): Whether to refresh the output files. Defaults to False.
        
    Returns:
        None
    """
    print("\n=========================")
    print("Calculating MAE/DAE rate statistics.")
    print("=========================")
    print("Benchmark:", benchmark)
    print("model_name:", model_name)
    for weight_variant in weights:
        rlt_dir = "results/eval/severity_levels/intermediate_results/" + f"{benchmark.lower()}_{model_name}_{weight_variant}"
        if not os.path.exists(rlt_dir):
            print(rlt_dir+" not exists!")
            continue

        print("-------------------------")
        print("weight_variant:", weight_variant)

        # Statistics of MAE/DAE rate on ID dataset.
        save_path = os.path.join(rlt_dir, f"id_mae_dae_rate_mean_std.csv")
        if (not os.path.exists(save_path)) or refresh: 
            file_path = os.path.join(rlt_dir, f"id_local_mae_dae_rate.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path).copy()
                df.pop("idx")
                
                df_mean = df.mean().to_frame().T
                df_std = df.std().to_frame().T
                df_mean["perturb_function"] = "average"
                df_std["perturb_function"] = "average"
                df_mean["severity"] = "average"
                df_std["severity"] = "average"

                df_mean_perb_func_severity = df.groupby(["perturb_function", "severity"]).mean().reset_index()
                df_std_perb_func_severity = df.groupby(["perturb_function", "severity"]).std().reset_index()

                df_mean_perb_func = df.groupby("perturb_function").mean().reset_index()
                df_std_perb_func = df.groupby("perturb_function").std().reset_index()
                df_mean_perb_func["severity"] = "average"
                df_std_perb_func["severity"] = "average"

                df_mean_severity = df.groupby("severity").mean().reset_index()
                df_std_severity = df.groupby("severity").std().reset_index()
                df_mean_severity["perturb_function"] = "average"
                df_std_severity["perturb_function"] = "average"

                df_mean = pd.concat([df_mean, df_mean_perb_func_severity, df_mean_perb_func, df_mean_severity], axis=0).copy()
                df_std = pd.concat([df_std, df_std_perb_func_severity, df_std_perb_func, df_std_severity], axis=0).copy()

                df = pd.merge(df_mean, df_std, on=["perturb_function", "severity"], how="inner", 
                              suffixes=["_mean", "_std"]).set_index(["perturb_function", "severity"])
                df = df.mul(100).round(3)
                df = df.reindex(sorted(df.columns), axis=1)
                df.to_csv(save_path, index=True)
                print("ID local MAE/DAE statistics saved to:", save_path)

            else:
                print("  ", file_path+" not exists!")

        # Statistics of local DAE rate on OoD datasets.
        save_path = os.path.join(rlt_dir, f"ood_dae_rate_mean_std.csv")
        if (not os.path.exists(save_path)) or refresh: 
            df_ood = pd.DataFrame()
            for dataset in ood_datasets:
                print("> OoD dataset:", dataset)
                file_path = os.path.join(rlt_dir, f"ood_local_dae_rate_{dataset}.csv")
                if not os.path.exists(file_path):
                    print("  ", file_path+" not exists!")
                    continue
                df = pd.read_csv(file_path).copy()
                df.pop("idx")
                df["dataset"] = dataset
                df_ood = pd.concat([df_ood, df], axis=0)

            if len(df_ood) == 0:
                continue
            
            df_ood = df_ood.sort_values(by=["perturb_function", "severity", "dataset"], 
                                        ignore_index=True)
            df_mean = df_ood.mean().to_frame().T
            df_std = df_ood.std().to_frame().T
            df_mean[["perturb_function", "severity", "dataset"]] = "average"
            df_std[["perturb_function", "severity", "dataset"]] = "average"

            df_mean_perb_func_severity_dataset = df_ood.groupby(["perturb_function", "severity", "dataset"]).mean().reset_index()
            df_std_perb_func_severity_dataset = df_ood.groupby(["perturb_function", "severity", "dataset"]).std().reset_index()

            df_mean_perb_func_dataset = df_ood.groupby(["perturb_function", "dataset"]).mean().reset_index()
            df_std_perb_func_dataset = df_ood.groupby(["perturb_function", "dataset"]).std().reset_index()
            df_mean_perb_func_dataset["severity"] = "average"
            df_std_perb_func_dataset["severity"] = "average"

            df_mean_perb_func_severity = df_ood.groupby(["perturb_function", "severity"]).mean().reset_index()
            df_std_perb_func_severity = df_ood.groupby(["perturb_function", "severity"]).std().reset_index()
            df_mean_perb_func_severity["dataset"] = "average"
            df_std_perb_func_severity["dataset"] = "average"

            df_mean_severity_dataset = df_ood.groupby(["severity", "dataset"]).mean().reset_index()
            df_std_severity_dataset = df_ood.groupby(["severity", "dataset"]).std().reset_index()
            df_mean_severity_dataset["perturb_function"] = "average"
            df_std_severity_dataset["perturb_function"] = "average"

            df_mean_perb_func = df_ood.groupby("perturb_function").mean().reset_index()
            df_std_perb_func = df_ood.groupby("perturb_function").std().reset_index()
            df_mean_perb_func[["severity", "dataset"]] = "average"
            df_std_perb_func[["severity", "dataset"]] = "average"

            df_mean_dataset = df_ood.groupby("dataset").mean().reset_index()
            df_std_dataset = df_ood.groupby("dataset").std().reset_index()
            df_mean_dataset[["severity", "perturb_function"]] = "average"
            df_std_dataset[["severity", "perturb_function"]] = "average"

            df_mean_severity = df_ood.groupby("severity").mean().reset_index()
            df_std_severity = df_ood.groupby("severity").std().reset_index()
            df_mean_severity[["perturb_function", "dataset"]] = "average"
            df_std_severity[["perturb_function", "dataset"]] = "average"

            df_mean = pd.concat([df_mean, df_mean_dataset, df_mean_perb_func, df_mean_severity, 
                                df_mean_perb_func_severity, df_mean_perb_func_dataset, df_mean_severity_dataset, 
                                df_mean_perb_func_severity_dataset], axis=0).copy()
            df_std = pd.concat([df_std, df_std_dataset, df_std_perb_func, df_std_severity, 
                                df_std_perb_func_severity, df_std_perb_func_dataset, df_std_severity_dataset, 
                                df_std_perb_func_severity_dataset], axis=0).copy()

            df = pd.merge(df_mean, df_std, on=["perturb_function", "severity", "dataset"], how="inner", 
                          suffixes=["_mean", "_std"]).set_index(["perturb_function", "severity", "dataset"])
            df = df.mul(100).round(3)
            df = df.reindex(sorted(df.columns), axis=1)
            df.to_csv(save_path, index=True)
            print("OoD local DAE statistics saved to:", save_path)

def _get_mae_dae_rate(configs, refresh=False):
    """
    This function calculates the statistics of local MAE/DAE rate (mean & std) for each OoD 
    detector under different perturbations. It assemblies the robustness evaluation results 
    across all benchmarks and test datasets, and the evaluation results are saved in csv files 
    in "results/eval/robustness/" folder.

    Args:
        configs (dict): A dictionary containing the configurations for the evaluation.
        refresh (bool, optional): If True, refreshes the evaluation results. Defaults to False.
    
    Returns:
        None
    """
    # Get the statistics of local MAE/DAE rate for each OoD detector under different perturbation functions.
    save_dir = os.path.join("results", "eval", "robustness")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Count MAE & DAE samples for each seed.
    _count_mae_dae_samples(configs, refresh=refresh)

    for benchmark in configs["benchmark"]:

        ood_datasets = configs["benchmark"][benchmark]["ood_datasets"]

        df_id_mae_dae_mean_std = pd.DataFrame()
        df_ood_dae_mean_std = pd.DataFrame()
        for model_name in configs["benchmark"][benchmark]["model"]:
        
            weights = configs["benchmark"][benchmark]["model"][model_name]

            # Calculate the local MAE/DAE rate for each OoD detector under different perturbation functions.
            _get_local_mae_dae_rate(benchmark, ood_datasets, model_name, weights, refresh=refresh)
            # Get the statistics of local MAE/DAE rate for each OoD detector under different perturbation functions.
            _get_mae_dae_statistics(benchmark, ood_datasets, model_name, weights, refresh=refresh)

            # Assembly the local MAE/DAE rate statistics accross different models and weight variants.
            print("Assemblying MAE/DAE statistics accross different models and weight variants.")
            for weight_variant in weights:
                rlt_dir = "results/eval/intermediate_results/" + f"{benchmark.lower()}_{model_name}_{weight_variant}"
                if not os.path.exists(rlt_dir):
                    print(rlt_dir+" not exists!")
                    continue
                
                file_path = os.path.join(rlt_dir, f"id_local_mae_dae_rate_mean_std.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path).copy()
                    df[["model", "variant"]] = model_name, weight_variant
                    df_id_mae_dae_mean_std = pd.concat([df_id_mae_dae_mean_std, df], axis=0)
                else:
                    print(file_path+" not exists!")

                file_path = os.path.join(rlt_dir, f"ood_local_dae_rate_mean_std.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path).copy()
                    df[["model", "variant"]] = model_name, weight_variant
                    df_ood_dae_mean_std = pd.concat([df_ood_dae_mean_std, df], axis=0)
                else:
                    print(file_path+" not exists!")
            
        if len(df_id_mae_dae_mean_std) > 0:
            df_id_mae_dae_mean_std = df_id_mae_dae_mean_std.sort_values(by=["perturb_function"], 
                                                                        key=lambda col: col.str.lower(), ignore_index=True)
            cols = df_id_mae_dae_mean_std.columns.tolist()
            cols = cols[-2:] + cols[:-2]
            df_id_mae_dae_mean_std = df_id_mae_dae_mean_std[cols]

            save_path = os.path.join(save_dir, f"{benchmark.lower()}_id_local_mae_dae_rate_mean_std.csv")
            df_id_mae_dae_mean_std.to_csv(save_path, index=False)
            print("Saved local ID MAE/DAE rate statistics to"+save_path+".")
            # display(df_id_mae_dae_mean_std) 

        if len(df_ood_dae_mean_std) > 0:
            df_ood_dae_mean_std = df_ood_dae_mean_std.sort_values(by=["perturb_function", "dataset"],
                                                                key=lambda col: col.str.lower(), ignore_index=True)
            cols = df_ood_dae_mean_std.columns.tolist()
            cols = cols[-2:] + cols[:-2]
            df_ood_dae_mean_std = df_ood_dae_mean_std[cols]

            save_path = os.path.join(save_dir, f"{benchmark.lower()}_ood_local_dae_rate_mean_std.csv")
            df_ood_dae_mean_std.to_csv(save_path, index=False)
            print("Saved local OoD DAE rate statistics to"+save_path+".")
            # display(df_ood_dae_mean_std)

def _get_dae_summary(configs):

    save_dir = os.path.join("results", "eval", "robustness")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_dae_all = pd.DataFrame()
    for benchmark in configs["benchmark"]:

        sorters = dict(model=configs["benchmark"][benchmark]["model"],
                    variant=["NT", "DA", "AT", "PAT"], detector=configs["score_functions"],
                    data=["ID", "OoD"], dataset=[benchmark, "average"]+configs["benchmark"][benchmark]["ood_datasets"], 
                    perturb_function=["average"]+configs["perturb_functions"])

        id_path = os.path.join(save_dir, f"{benchmark.lower()}_id_local_mae_dae_rate_mean_std.csv")
        ood_path = os.path.join(save_dir, f"{benchmark.lower()}_ood_local_dae_rate_mean_std.csv")
        if (not os.path.exists(id_path)) or (not os.path.exists(ood_path)):
            continue
        df_id = pd.read_csv(id_path).copy()
        df_ood = pd.read_csv(ood_path).copy()

        df_id["dataset"] = benchmark
        df_id["data"] = "ID"
        df_ood["data"] = "OoD"

        df = pd.concat([df_id, df_ood], ignore_index=True)
        
        dae_mean_cols = [col for col in df.columns if "dae" in col and "mean" in col]
        dae_std_cols = [col for col in df.columns if "dae" in col and "std" in col]
        df_dae_mean = df.melt(id_vars=["model", "variant", "data", "dataset", "perturb_function", "mae_mean", "mae_std"], 
                            value_vars=dae_mean_cols, var_name="detector", value_name="dae_mean")
        df_dae_std = df.melt(id_vars=["model", "variant", "data", "dataset", "perturb_function", "mae_mean", "mae_std"], 
                            value_vars=dae_std_cols, var_name="detector", value_name="dae_std")

        df_dae_mean["detector"] = df_dae_mean["detector"].apply(lambda x: x.split("_")[1])
        df_dae_std["detector"] = df_dae_std["detector"].apply(lambda x: x.split("_")[1])

        df_dae = pd.merge(df_dae_mean, df_dae_std, on=["model", "variant", "detector", "data", "dataset", "perturb_function", "mae_mean", "mae_std"])
        df_dae["benchmark"] = benchmark

        for sort_col in ["model", "variant", "detector", "data", "dataset", "perturb_function"]:
            df_dae[sort_col] = df_dae[sort_col].astype("category")
            df_dae[sort_col] = df_dae[sort_col].cat.set_categories(sorters[sort_col], ordered=True)
        df_dae.sort_values(by=["model", "variant", "detector", "data", "dataset", "perturb_function"], inplace=True)
        df_dae_all = pd.concat([df_dae_all, df_dae], ignore_index=True)

    df_dae_all = df_dae_all[["benchmark", "model", "variant", "detector", "data", "dataset", "perturb_function", "dae_mean", "dae_std", "mae_mean", "mae_std"]].copy()
    df_dae_all.to_csv(os.path.join(save_dir, "dae_summary.csv"), index=False)

def _get_mae_dae_rate_severity(configs, refresh=False):
    """
    This function calculates the statistics of local MAE/DAE rate (mean & std) for each OoD 
    detector under different perturbations. It assemblies the robustness evaluation results 
    across all benchmarks and test datasets considering different perturbation severities.
    The evaluation results are saved in csv files in the "results/eval/severity_levels/robustness/" 
    folder.

    Args:
        configs (dict): A dictionary containing the configurations for the evaluation.
        refresh (bool, optional): If True, refreshes the evaluation results. Defaults to False.
    
    Returns:
        None
    """
    # Get the statistics of local MAE/DAE rate for each OoD detector under different perturbation functions.
    save_dir = os.path.join("results", "eval", "severity_levels", "robustness")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  
    perturb_functions = configs["perturb_functions"]
    # Count MAE & DAE samples for each seed.
    _count_mae_dae_samples_severity(configs, refresh=refresh)
    for benchmark in configs["benchmark"]:
        ood_datasets = configs["benchmark"][benchmark]["ood_datasets"]
        df_id_mae_dae_mean_std = pd.DataFrame()
        df_ood_dae_mean_std = pd.DataFrame()
        for model_name in configs["benchmark"][benchmark]["model"]:
        
            weights = configs["benchmark"][benchmark]["model"][model_name]

            # Calculate the local MAE/DAE rate for each OoD detector under different perturbation functions.
            _get_local_mae_dae_rate_severity(benchmark, ood_datasets, model_name, weights, perturb_functions, refresh=refresh)
            # Get the statistics of local MAE/DAE rate for each OoD detector under different perturbation functions.
            _get_mae_dae_statistics_severity(benchmark, ood_datasets, model_name, weights, refresh=refresh)

            # Assembly the local MAE/DAE rate statistics accross different models and weight variants.
            print("Assemblying MAE/DAE statistics accross different models and weight variants.")
            for weight_variant in weights:
                rlt_dir = "results/eval/severity_levels/intermediate_results/" + f"{benchmark.lower()}_{model_name}_{weight_variant}"
                if not os.path.exists(rlt_dir):
                    print(rlt_dir+" not exists!")
                    continue
                
                file_path = os.path.join(rlt_dir, f"id_mae_dae_rate_mean_std.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path).copy()
                    df[["model", "variant"]] = model_name, weight_variant
                    df_id_mae_dae_mean_std = pd.concat([df_id_mae_dae_mean_std, df], axis=0)
                else:
                    print(file_path+" not exists!")

                file_path = os.path.join(rlt_dir, f"ood_dae_rate_mean_std.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path).copy()
                    df[["model", "variant"]] = model_name, weight_variant
                    df_ood_dae_mean_std = pd.concat([df_ood_dae_mean_std, df], axis=0)
                else:
                    print(file_path+" not exists!")
            
        if len(df_id_mae_dae_mean_std) > 0:
            df_id_mae_dae_mean_std = df_id_mae_dae_mean_std.sort_values(by=["perturb_function", "severity"], 
                                                                        ignore_index=True)
            cols = df_id_mae_dae_mean_std.columns.tolist()
            cols = cols[-2:] + cols[:-2]
            df_id_mae_dae_mean_std = df_id_mae_dae_mean_std[cols]

            save_path = os.path.join(save_dir, f"{benchmark.lower()}_id_local_mae_dae_rate_mean_std.csv")
            df_id_mae_dae_mean_std.to_csv(save_path, index=False)
            print("Saved local ID MAE/DAE rate statistics to"+save_path+".")

        if len(df_ood_dae_mean_std) > 0:
            df_ood_dae_mean_std = df_ood_dae_mean_std.sort_values(by=["perturb_function", "severity", "dataset"],
                                                                  ignore_index=True)
            cols = df_ood_dae_mean_std.columns.tolist()
            cols = cols[-2:] + cols[:-2]
            df_ood_dae_mean_std = df_ood_dae_mean_std[cols]

            save_path = os.path.join(save_dir, f"{benchmark.lower()}_ood_local_dae_rate_mean_std.csv")
            df_ood_dae_mean_std.to_csv(save_path, index=False)
            print("Saved local OoD DAE rate statistics to"+save_path+".")

def _get_dae_summary_severity(configs):
    save_dir = os.path.join("results", "eval", "severity_levels", "robustness")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_dae_all = pd.DataFrame()
    for benchmark in configs["benchmark"]:
        sorters = dict(model=configs["benchmark"][benchmark]["model"],
                    variant=["NT", "DA", "AT", "PAT"], detector=configs["score_functions"],
                    data=["ID", "OoD"], dataset=[benchmark, "average"]+configs["benchmark"][benchmark]["ood_datasets"], 
                    perturb_function=["average"]+configs["perturb_functions"], severity=["average", "1", "2", "3", "4", "5"])


        id_path = os.path.join(save_dir, f"{benchmark.lower()}_id_local_mae_dae_rate_mean_std.csv")
        ood_path = os.path.join(save_dir, f"{benchmark.lower()}_ood_local_dae_rate_mean_std.csv")
        if (not os.path.exists(id_path)) or (not os.path.exists(ood_path)):
            continue
        df_id = pd.read_csv(id_path).copy()
        df_ood = pd.read_csv(ood_path).copy()

        df_id["dataset"] = benchmark
        df_id["data"] = "ID"
        df_ood["data"] = "OoD"

        df = pd.concat([df_id, df_ood], ignore_index=True)
        
        dae_mean_cols = [col for col in df.columns if "dae" in col and "mean" in col]
        dae_std_cols = [col for col in df.columns if "dae" in col and "std" in col]
        df_dae_mean = df.melt(id_vars=["model", "variant", "data", "dataset", "perturb_function", "severity", "mae_mean", "mae_std"], 
                            value_vars=dae_mean_cols, var_name="detector", value_name="dae_mean")
        df_dae_std = df.melt(id_vars=["model", "variant", "data", "dataset", "perturb_function", "severity", "mae_mean", "mae_std"], 
                            value_vars=dae_std_cols, var_name="detector", value_name="dae_std")

        df_dae_mean["detector"] = df_dae_mean["detector"].apply(lambda x: x.split("_")[1])
        df_dae_std["detector"] = df_dae_std["detector"].apply(lambda x: x.split("_")[1])

        df_dae = pd.merge(df_dae_mean, df_dae_std, on=["model", "variant", "detector", "data", "dataset", "perturb_function", "severity", "mae_mean", "mae_std"])
        df_dae["benchmark"] = benchmark

        for sort_col in ["model", "variant", "detector", "data", "dataset", "perturb_function", "severity"]:
            df_dae[sort_col] = df_dae[sort_col].astype("category")
            df_dae[sort_col] = df_dae[sort_col].cat.set_categories(sorters[sort_col], ordered=True)
        df_dae.sort_values(by=["model", "variant", "detector", "data", "dataset", "perturb_function", "severity"], inplace=True)
        df_dae_all = pd.concat([df_dae_all, df_dae], ignore_index=True)

    df_dae_all = df_dae_all[["benchmark", "model", "variant", "detector", "data", "dataset", "perturb_function", "severity", "dae_mean", "dae_std", "mae_mean", "mae_std"]].copy()
    df_dae_all.to_csv(os.path.join(save_dir, "dae_summary.csv"), index=False)
def get_mae_dae_rate(configs, refresh=False):
    """
    This function calculates the statistics of local MAE/DAE rate (mean & std) for each OoD 
    detector under different perturbations. It assemblies the robustness evaluation results 
    across all benchmarks and test datasets, and the evaluation results are saved in csv files 
    in "results/eval/robustness/" folder.

    Args:
        configs (dict): A dictionary containing the configurations for the evaluation.
        refresh (bool, optional): If True, refreshes the evaluation results. Defaults to False.
        severity (bool, optional): If True, considers different perturbation severities. 
            Defaults to False.
    Returns:
        None
    """
    severity = configs["eval_severity"]
    if severity:
        _get_mae_dae_rate_severity(configs, refresh=refresh)
        _get_dae_summary_severity(configs)
    else:
        _get_mae_dae_rate(configs, refresh=refresh)
        _get_dae_summary(configs)