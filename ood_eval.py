"""
We process the raw experiment outcomes (model confidence and OoD scores) and provide a primary analysis 
on the performance and robustness metrics of DNN model and OoD detectors.

1. Evaluate the DNN model & OoD detectors' performance.
    - Model performance metrics: Accuracy.
    - OoD detectors' performance metrics: FPR95, AUROC, AUPR_IN, AUPR_OUT.
    
2. Evaluate the DNN model & OoD detectors' robustness.
    - Model robustness metrics: MAE (Model Adversarial Example) rate.
    - OoD detectors' robustness metrics: DAE (Detector Adversarial Example) rate.

    The OoD detectors are connected to different model architectures (WRN-40-2, ResNet50) and variants 
    (NT, DA, AT, PAT), considering:
        - ID and OoD seeds. 
        - Different perturbation functions.
        - OoD seeds from different OoD datasets.

The evaluation summaries are saved in the `results/eval/performance/` and `results/eval/robustness/` directories. 
"""

import yaml
import argparse
from utils.eval import get_model_detector_performance, get_mae_dae_rate

if __name__ == "__main__":

    # Load configs: benchmarks, model variants, OoD datasets and experiment settings.
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cfg",type=str, default="config.yaml")
    argparser.add_argument("--refresh", action='store_true')
    args = argparser.parse_args()
    
    with open(args.cfg, 'r') as f:
        configs = yaml.safe_load(f)

    refresh = args.refresh

    # 1. Evaluate the DNN model & OoD detectors' performance.
    #     - Model performance metrics: Accuracy.
    #     - OoD detectors' performance metrics: FPR95, AUROC, AUPR_IN, AUPR_OUT.
    get_model_detector_performance(configs, refresh=refresh)

    # 2. Evaluate the DNN model & OoD detectors' robustness.
    #     - Model robustness metrics: MAE (Model Adversarial Example) rate.
    #     - OoD detectors' robustness metrics: DAE (Detector Adversarial Example) rate.
    get_mae_dae_rate(configs, refresh=refresh)
