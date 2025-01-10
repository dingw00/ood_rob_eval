from pytorch_ood.api import Detector
from pytorch_ood.detector import *

def build_detectors(detector_names, model, input_normalizer, id_train_data_loader, device="cpu"):
    detectors = {}
    model.eval()
    if "Entropy"in detector_names:
        detectors["Entropy"] = Entropy(model)
    if "Mahalanobis+ODIN" in detector_names:
        if hasattr(model, 'feature'):
            detectors["Mahalanobis+ODIN"] = Mahalanobis(
                model.feature, norm_std=list(input_normalizer.std.view(-1).numpy()), eps=0.0014)
        elif hasattr(model, 'forward_features'):
            detectors["Mahalanobis+ODIN"] = Mahalanobis(
                model.forward_features, norm_std=list(input_normalizer.std.view(-1).numpy()), eps=0.0014)
    if "Mahalanobis" in detector_names:
        detectors["Mahalanobis"] = Mahalanobis(model.forward_features)
    if "RMD" in detector_names:
        detectors["RMD"] = RMD(model.forward_features)
    if "MSP" in detector_names:
        detectors["MSP"] = MaxSoftmax(model)
    if "EnergyBased" in detector_names:
        detectors["EnergyBased"] = EnergyBased(model)
    if "MaxLogit" in detector_names:
        detectors["MaxLogit"] = MaxLogit(model)
    if "ODIN" in detector_names:
        detectors["ODIN"] = ODIN(model, norm_std=list(input_normalizer.std.view(-1).numpy()), eps=0.0014)
    if "KLMatching" in detector_names:
        detectors["KLMatching"] = KLMatching(model)
    if "ViM" in detector_names:
        if hasattr(model, 'fc'):
            detectors["ViM"] = ViM(model.forward_features, d=64, w=model.fc.weight, b=model.fc.bias)
        elif hasattr(model, 'linear'):
            detectors["ViM"] = ViM(model.forward_features, d=64, w=model.linear.weight, b=model.linear.bias)
        elif hasattr(model, 'head'):
            detectors["ViM"] = ViM(model.forward_features, d=64, w=model.head.weight, b=model.head.bias)
    if "SHE" in detector_names:
        if hasattr(model, 'head'):
            detectors["SHE"] = SHE(model.forward_features, model.head)
        else:
            detectors["SHE"] = SHE(model.forward_features, model.forward_head)
    if "DICE" in detector_names:
        if hasattr(model, 'fc'):  
            detectors["DICE"] = DICE(
                model=model.forward_features, w=model.fc.weight, b=model.fc.bias, p=0.65)
        elif hasattr(model, 'linear'):
            detectors["DICE"] = DICE(
                model=model.forward_features, w=model.linear.weight, b=model.linear.bias, p=0.65)
        elif hasattr(model, 'head'):
            detectors["DICE"] = DICE(
                model=model.forward_features, w=model.head.weight, b=model.head.bias, p=0.65)
    if "MCD" in detector_names:
        detectors["MCD"] = MCD(model)
    if "KNN" in detector_names:
        detectors["KNN"] = KNN(model)
    if "ASH" in detector_names:
        detectors["ASH"] = ASH(backbone=model.forward_features, head=model.forward_head, 
                                   detector = EnergyBased.score)
    if "ReAct" in detector_names:
        detectors["ReAct"] = ReAct(backbone=model.forward_features, head=model.forward_head, 
                                   detector = EnergyBased.score)
    print(f"> Fitting {len(detectors)} detectors")
    for name, detector in detectors.items():
        print(f"--> Fitting {name}")

        detector.fit(id_train_data_loader, device=device)

        # # Workaround when GPU memory is not enough
        # model.to("cpu")
        # detector.fit(id_train_data_loader, device="cpu")
        # model.to(device)

    return detectors

