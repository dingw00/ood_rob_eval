import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import subprocess

OOD_DATASETS_DICT = {
    "CIFAR10": ["Textures", "SVHN", "LSUN-C", "LSUN-R", "iSUN", "Places365"],
    "Imagenet100": ["NINCO", "Textures", "iNaturalist", "SUN", "Places"],
    "Imagenet1k": ["NINCO", "Textures", "iNaturalist"]
}
MODELS = {"CIFAR10": {"wrn_40_2": {"NT": "wrn_40_2.pt", 
                                   "DA": "wrn_40_2_default_aug.pt",
                                   "AT": "wrn_40_2_Hendrycks2020AugMix.pt", 
                                   "PAT": "wrn_40_2_pat_0.25_epoch_199.pt"}, 
                      "resnet50":{"NT": "resnet50.pt", 
                                  "DA": "resnet50_da_all_199.pt",
                                  "AT": "resnet50_pgd_linf_8_pgd_l2_1_stadv_0.05_recoloradv_0.06_average.pt", 
                                  "PAT": "resnet50_pat_self_0.5.pt"
                                  }},
          "Imagenet100": {"resnet50": {"NT": "resnet50.pt", 
                                       "DA": "resnet50_da_all_115.pt",
                                       "AT": "resnet50_pgd_linf_4_pgd_l2_1200_jpeg_linf_0.125_stadv_0.05_recoloradv_0.06_random.pt", 
                                       "PAT": "resnet50_pat_alexnet_0.5.pt"}},
          "Imagenet1k": {"swin": {"NT": "swin_base_patch4_window7_224"}, 
                         "deit": {"NT": "deit_base_patch16_224"}, 
                         "vit": {"NT": "vit_b_16"}}}

DATASET_LIST = ["Textures", "SVHN", "LSUN-C", "LSUN-R", "iSUN", "Places365",
                   "NINCO", "iNaturalist", "SUN", "Places", 
                   "CIFAR100", "Food101", "Flowers102", "GTSRB", "EuroSAT",
                   "other"]
OOD_DETECTORS = ["Entropy", "ViM", "Mahalanobis+ODIN", "Mahalanobis", "KLMatching", "SHE", "MSP", 
                   "EnergyBased", "MaxLogit", "ODIN", "DICE", "RMD", "MCD", "KNN", "ASH", "ReAct"]

# Create the main window
root = tk.Tk()
root.title("OoD Robustness Test Settings")

# Benchmark Configuration
benchmark_frame = ttk.Frame(root, padding=(5, 5, 5, 5), border=1)
benchmark_frame.grid(row=0, column=0, sticky=tk.W+tk.E)
ttk.Label(benchmark_frame, text="Benchmark:", width=20).grid(row=0, column=0, sticky=tk.W)
benchmark_combobox = ttk.Combobox(benchmark_frame, values=["CIFAR10", "Imagenet100", "Imagenet1k"], width=11)
benchmark_combobox.set("CIFAR10")
benchmark_combobox.grid(row=0, column=1, sticky=tk.W)

ttk.Label(benchmark_frame, text="Number of Classes:", width=20).grid(row=1, column=0, sticky=tk.W)
num_classes_spinbox = ttk.Spinbox(benchmark_frame, from_=1, to=1000, width=4)
num_classes_spinbox.grid(row=1, column=1, sticky=tk.W,)

ttk.Label(benchmark_frame, text="Image Size:", width=20).grid(row=2, column=0, sticky=tk.W)
img_size_spinbox = ttk.Spinbox(benchmark_frame, from_=1, to=1000, width=4)
img_size_spinbox.grid(row=2, column=1, sticky=tk.W)

ttk.Separator(root, orient="horizontal").grid(row=3, column=0, sticky=tk.W+tk.E, pady=5)
ood_dataset_frame = ttk.Frame(root)
ood_dataset_frame.grid(row=4, column=0, sticky=tk.W+tk.E)
ttk.Label(ood_dataset_frame, text="Data Directory:", width=20).grid(row=0, column=0, sticky=tk.W)
datadir = tk.StringVar(value="dataset/")
datadir_entry = ttk.Entry(ood_dataset_frame, textvariable=datadir, width=15)
datadir_entry.grid(row=0, column=1, sticky=tk.W)

ttk.Label(ood_dataset_frame, text="OoD Datasets:", width=20).grid(row=1, column=0, sticky=tk.W)
ood_dataset_scrollbar = tk.Scrollbar(ood_dataset_frame)
ood_dataset_listbox = tk.Listbox(ood_dataset_frame, yscrollcommand=ood_dataset_scrollbar.set, 
                                height=5, width=15)
ood_dataset_listbox.grid(row=1, column=1, sticky=tk.W)
ood_dataset_scrollbar.grid(row=1, column=2, sticky=tk.W+tk.N+tk.S)
ood_dataset_scrollbar.config(command=ood_dataset_listbox.yview)

add_dataset_frame = ttk.Frame(ood_dataset_frame, padding=(5, 5, 5, 5), border=1, relief="groove")
add_dataset_frame.grid(row=1, column=3, sticky=tk.W+tk.S)
dataset_entry = ttk.Entry(add_dataset_frame, width=16)

ttk.Separator(root, orient="horizontal").grid(row=5, column=0, sticky=tk.W+tk.E, pady=5)
model_frame = ttk.Frame(root)
model_frame.grid(row=6, column=0, sticky=tk.W+tk.E)
model_backbone_label = ttk.Label(model_frame, text="Model Backbone:", width=20)
model_config_var = tk.IntVar(value=0)

weight_frame = ttk.Frame(root)
weight_frame.grid(row=7, column=0, sticky=tk.W+tk.E)
ttk.Label(weight_frame, text="Model Variants:", width=20).grid(row=0, column=0, sticky=tk.W)
model_variant_scrollbar = tk.Scrollbar(weight_frame)
model_variant_listbox = tk.Listbox(weight_frame, yscrollcommand=model_variant_scrollbar.set, height=4, width=20)
model_variant_listbox.grid(row=0, column=1, sticky=tk.W)
model_variant_scrollbar.grid(row=0, column=2, sticky=tk.W+tk.N+tk.S, pady=5)
model_variant_scrollbar.config(command=model_variant_listbox.yview)

add_weight_frame = ttk.Frame(weight_frame, padding=(5, 5, 5, 5), border=1, relief="groove")
add_weight_frame.grid(row=0, column=2, sticky=tk.W, pady=5)
ttk.Label(add_weight_frame, text="Variant:", width=15).grid(row=0, column=0, sticky=tk.W)
weight_combobox = ttk.Combobox(add_weight_frame, width=5)
weight_combobox.grid(row=0, column=1, sticky=tk.W)
weight_name_entry = ttk.Entry(add_weight_frame, width=5)
ttk.Label(add_weight_frame, text="Weight Path:", width=10).grid(row=1, column=0, sticky=tk.W)
weight_path_var = tk.StringVar(value="")
weight_path = ttk.Entry(add_weight_frame, textvariable=weight_path_var, width=15)
weight_path.grid(row=1, column=1, sticky=tk.W)

add_model_variant_btn = ttk.Button(add_weight_frame, text="Add Model Variant")
add_model_variant_btn.grid(row=2, column=0, sticky=tk.W)
model_variant_dict = dict()
remove_model_variant_btn = ttk.Button(add_weight_frame, text="Remove Model Variant")
remove_model_variant_btn.grid(row=2, column=1, sticky=tk.W)


def update_benchmark(event):
    
    benchmark_selected = benchmark_combobox.get()

    # Update number of classes
    if benchmark_selected == "CIFAR10":
        num_classes_spinbox.set(10)
        img_size_spinbox.set(32)
    elif benchmark_selected == "Imagenet100":
        num_classes_spinbox.set(100)
        img_size_spinbox.set(224)
    elif benchmark_selected == "Imagenet1k":
        num_classes_spinbox.set(1000)
        img_size_spinbox.set(224)

    # Update OOD Datasets
    ood_dataset_listbox.delete(0, tk.END)
    for i, ds in enumerate(OOD_DATASETS_DICT.get(benchmark_selected, [])):
        ood_dataset_listbox.insert(tk.END, ds)
    
    dataset_options = [dataset for dataset in DATASET_LIST if dataset not in ood_dataset_listbox.get(0, tk.END)]
    dataset_combobox = ttk.Combobox(add_dataset_frame, values=dataset_options, width=10)
    dataset_combobox.set(dataset_options[0]) 
    dataset_combobox.grid(row=0, column=0, sticky=tk.W+tk.S)

    # update the position and contents of user-defined dataset entry
    def on_dataset_selected(event):
        if dataset_combobox.get() == "other":          
            dataset_entry.grid(row=0, column=1, sticky=tk.W+tk.S)
        else:
            dataset_entry.grid_forget()

    # update OoD datasets
    def add_dataset():
        ds = dataset_combobox.get()
        if ds == "other":
            ds = dataset_entry.get()
            dataset_entry.delete(0, tk.END)

        if ds == "" or (ds in ood_dataset_listbox.get(0, tk.END)):
            return
        ood_dataset_listbox.insert(tk.END, ds)
        
        # update the position and contents of the dataset combobox
        dataset_options = [dataset for dataset in DATASET_LIST if dataset not in ood_dataset_listbox.get(0, tk.END)]
        dataset_combobox['values'] = dataset_options
        dataset_combobox.set(dataset_options[0])       

    dataset_combobox.bind("<<ComboboxSelected>>", on_dataset_selected)
    add_dataset_button = ttk.Button(add_dataset_frame, text="Add Dataset", 
                                    command=add_dataset, width=15)
    add_dataset_button.grid(row=1, column=0, sticky=tk.W+tk.S)

    def remove_dataset():
        selected_indices = ood_dataset_listbox.curselection()
        if len(selected_indices) == 0:
            return
        index = selected_indices[-1]
        ood_dataset_listbox.delete(index)
        
        dataset_options = [dataset for dataset in DATASET_LIST if dataset not in ood_dataset_listbox.get(0, tk.END)]
        dataset_combobox['values'] = dataset_options
        if dataset_options:
            dataset_combobox.set(dataset_options[0])
        else:
            dataset_combobox.set("")

    remove_dataset_btn = ttk.Button(add_dataset_frame, text="Remove Dataset", 
                                       command=remove_dataset, width=15)
    remove_dataset_btn.grid(row=1, column=1, sticky=tk.W+tk.S)

    # Update Model Configuration
    def update_weight(event):
        weight_name = weight_combobox.get()
        if weight_name == "other":
            weight_name_entry.grid(row=0, column=2, sticky=tk.W)
            weight_path_var.set("")
        else:
            weight_name_entry.grid_forget()
            weight_path_var.set(model_variant_dict[weight_name])      

    def update_weight_list(var, idx, mode):
        global weight_path_var
        global model_variant_dict
        model_names = list(MODELS[benchmark_selected].keys())
        model_name_selected = model_names[model_config_var.get()]
        model_variant_dict = MODELS[benchmark_selected][model_name_selected]
        weight_names = list(model_variant_dict.keys())+["other"]
        weight_combobox["value"] = weight_names
        weight_combobox.set(weight_names[0])
        update_weight(None)
        model_variant_listbox.delete(0, tk.END) 
    
    def add_weight(event):
        weight_name = weight_combobox.get()
        if weight_name == "other":
            weight_name = weight_name_entry.get()
            weight_name_entry.delete(0, tk.END)
        weight_path = weight_path_var.get()
        if weight_name == "" or weight_path == "":
            return
        model_variant_str = f"{weight_name}: {weight_path}"
        # Check if model_variant_str already exists in model_variant_listbox
        if model_variant_str not in model_variant_listbox.get(0, tk.END):
            model_variant_listbox.insert(tk.END, model_variant_str)

        if weight_name in model_variant_dict:
            model_variant_dict.pop(weight_name)
        weight_names = list(model_variant_dict.keys())+["other"]
        weight_combobox["value"] = weight_names
        weight_combobox.set(weight_names[0])
        update_weight(None)
    
    def remove_model_variant(event):
        selected_indices = model_variant_listbox.curselection()
        if len(selected_indices) == 0:
            return
        index = selected_indices[-1]
        
        model_variant_str = model_variant_listbox.get(index)
        weight_name, weight_path = model_variant_str.split(": ")
        model_variant_listbox.delete(index)
        model_variant_dict[weight_name] = weight_path
        
        weight_names = list(model_variant_dict.keys())+["other"]
        weight_combobox["value"] = weight_names
        weight_combobox.set(weight_names[0])
        update_weight(None)

    # reset the selection of model config
    for widget in model_frame.grid_slaves(row=0):
        widget.grid_forget()
    model_backbone_label.grid(row=0, column=0, sticky=tk.W)
    model_names = list(MODELS[benchmark_selected].keys())
    # update radiobuttons for model config selection
    for i, model_name in enumerate(model_names):
        model_config_radiobtn = ttk.Radiobutton(model_frame, text=model_name, variable=model_config_var, value=i)
        model_config_radiobtn.grid(row=0, column=i+1, sticky=tk.W, padx=5)
    
    update_weight_list(None, None, None)
    add_weight(None)
    model_config_var.trace_add("write", update_weight_list)
    weight_combobox.bind("<<ComboboxSelected>>", update_weight)
    add_model_variant_btn.bind("<Button-1>", add_weight)
    remove_model_variant_btn.bind("<Button-1>", remove_model_variant)

update_benchmark(None)  # Trigger the function on first initialization
benchmark_combobox.bind("<<ComboboxSelected>>", update_benchmark)

# OoD Detectors
detector_frame = ttk.Frame(root, relief="groove", padding=(5,5,5,5))
detector_frame.grid(row=8, column=0, columnspan=5, sticky=tk.W+tk.E)
ttk.Label(detector_frame, text="OoD Detectors:", width=20).grid(row=0, column=0, sticky=tk.W)
detector_scrollbar = tk.Scrollbar(detector_frame)
detector_listbox = tk.Listbox(detector_frame, yscrollcommand=detector_scrollbar.set, height=5, width=15)
for detector in OOD_DETECTORS[6:8]:
    detector_listbox.insert(tk.END, detector)
detector_listbox.grid(row=0, column=1, sticky=tk.W)
detector_scrollbar.grid(row=0, column=1, sticky=tk.E+tk.N+tk.S)
detector_scrollbar.config(command=detector_listbox.yview)

modify_detector_frame = ttk.Frame(detector_frame)
modify_detector_frame.grid(row=0, column=2, sticky=tk.W)
detector_button_frame = ttk.Frame(modify_detector_frame)
detector_button_frame.grid(row=0, column=0)
add_detector_btn = ttk.Button(detector_button_frame, text="\u21E6", width=1.5)
add_detector_btn.grid(row=0, column=0)
remove_detector_btn = ttk.Button(detector_button_frame, text="\u21E8", width=1.5)
remove_detector_btn.grid(row=1, column=0)

detector_scrollbar1 = tk.Scrollbar(modify_detector_frame)
detector_listbox1 = tk.Listbox(modify_detector_frame, yscrollcommand=detector_scrollbar1.set, height=5, width=15)
for detector in OOD_DETECTORS:
    if detector not in detector_listbox.get(0, tk.END):
        detector_listbox1.insert(tk.END, detector)
detector_listbox1.grid(row=0, column=1, sticky=tk.W)
detector_scrollbar1.grid(row=0, column=2, sticky=tk.E+tk.N+tk.S)
detector_scrollbar1.config(command=detector_listbox1.yview)

def add_detector(event):
    detector = detector_listbox1.get(tk.ACTIVE)
    if detector == "":
        return
    detector_listbox1.delete(tk.ACTIVE)
    detector_listbox.insert(tk.END, detector)

def remove_detector(event):
    detector = detector_listbox.get(tk.ACTIVE)
    if detector == "":
        return
    detector_listbox.delete(tk.ACTIVE)
    detector_listbox1.insert(tk.END, detector)

add_detector_btn.bind("<Button-1>", add_detector)
remove_detector_btn.bind("<Button-1>", remove_detector)


# Perturbation Functions Checkboxes
perturb_func_frame = ttk.Frame(root, relief="groove", padding=(5,5,5,5))
perturb_func_frame.grid(row=9, column=0, columnspan=5, sticky=tk.W+tk.E)
PERTURB_FUNCTIONS = ["rotation", "translation", "scale", "hue", "saturation", "bright_contrast", "blur", "Linf", "L2"]
ttk.Label(perturb_func_frame, text="Perturb Functions:", width=20).grid(row=0, column=0, sticky=tk.W)
perturb_func_scrollbar = tk.Scrollbar(perturb_func_frame)
perturb_func_listbox = tk.Listbox(perturb_func_frame, yscrollcommand=perturb_func_scrollbar.set, height=5, width=15)
for perturb_func in PERTURB_FUNCTIONS:
    perturb_func_listbox.insert(tk.END, perturb_func)
perturb_func_listbox.grid(row=0, column=1, sticky=tk.W)
perturb_func_scrollbar.grid(row=0, column=1, sticky=tk.E+tk.N+tk.S)
perturb_func_scrollbar.config(command=perturb_func_listbox.yview)

modify_perturb_func_frame = ttk.Frame(perturb_func_frame)
modify_perturb_func_frame.grid(row=0, column=2, sticky=tk.W)
button_frame = ttk.Frame(modify_perturb_func_frame, relief="groove")
button_frame.grid(row=0, column=0)
add_perturb_func_btn = ttk.Button(button_frame, text="\u21E6", width=1.5)
add_perturb_func_btn.grid(row=0, column=0)
remove_perturb_func_btn = ttk.Button(button_frame, text="\u21E8", width=1.5)
remove_perturb_func_btn.grid(row=1, column=0)

perturb_func_scrollbar1 = tk.Scrollbar(modify_perturb_func_frame)
perturb_func_listbox1 = tk.Listbox(modify_perturb_func_frame, yscrollcommand=perturb_func_scrollbar1.set, height=5, width=15)
perturb_func_listbox1.grid(row=0, column=1, sticky=tk.W)
perturb_func_scrollbar1.grid(row=0, column=2, sticky=tk.E+tk.N+tk.S)
perturb_func_scrollbar1.config(command=perturb_func_listbox1.yview)

ttk.Label(perturb_func_frame, text="Severity:", width=20).grid(row=1, column=0, sticky=tk.W)
severity_combobox = ttk.Combobox(perturb_func_frame, values=[1,2,3,4,5,"avg"], width=5)
severity_combobox.set("avg")
severity_combobox.grid(row=1, column=1, sticky=tk.W)

def add_perturb_func(event):
    perturb_func = perturb_func_listbox1.get(tk.ACTIVE)
    if perturb_func == "":
        return
    perturb_func_listbox1.delete(tk.ACTIVE)
    perturb_func_listbox.insert(tk.END, perturb_func)

def remove_perturb_func(event):
    perturb_func = perturb_func_listbox.get(tk.ACTIVE)
    if perturb_func == "":
        return
    perturb_func_listbox.delete(tk.ACTIVE)
    perturb_func_listbox1.insert(tk.END, perturb_func)

add_perturb_func_btn.bind("<Button-1>", add_perturb_func)
remove_perturb_func_btn.bind("<Button-1>", remove_perturb_func)

# Other Configurations
other_config_frame = tk.LabelFrame(root, text="Other Configurations", padx=5, pady=5)
other_config_frame.grid(row=10, column=0, columnspan=5, sticky=tk.W+tk.E)

ttk.Label(other_config_frame, text="Random Seed:", width=20).grid(row=0, column=0, sticky=tk.W)
rand_seed_spinbox = ttk.Spinbox(other_config_frame, from_=0, to=1000, width=4)
rand_seed_spinbox.set(0)
rand_seed_spinbox.grid(row=0, column=1, sticky=tk.W)

ttk.Label(other_config_frame, text="Batch Size:", width=20).grid(row=1, column=0, sticky=tk.W)
batch_size_spinbox = ttk.Spinbox(other_config_frame, from_=1, to=1000, width=4)
batch_size_spinbox.set(128)
batch_size_spinbox.grid(row=1, column=1, sticky=tk.W)

ttk.Label(other_config_frame, text="Number of Seeds:", width=20).grid(row=2, column=0, sticky=tk.W)
n_seeds_spinbox = ttk.Spinbox(other_config_frame, from_=1, to=20000, width=4)
n_seeds_spinbox.set(1000)
n_seeds_spinbox.grid(row=2, column=1, sticky=tk.W)

ttk.Label(other_config_frame, text="Number of Sampling:", width=20).grid(row=3, column=0, sticky=tk.W)
n_sampling_spinbox = ttk.Spinbox(other_config_frame, from_=1, to=1000, width=4)
n_sampling_spinbox.set(50)
n_sampling_spinbox.grid(row=3, column=1, sticky=tk.W)

ttk.Label(other_config_frame, text="Evaluate Severity:", width=20).grid(row=4, column=0, sticky=tk.W)
eval_severity_combobox = ttk.Combobox(other_config_frame, values=["False", "True"], width=5)
eval_severity_combobox.set("False")
eval_severity_combobox.grid(row=4, column=1, sticky=tk.W)

ttk.Label(other_config_frame, text="Device:", width=20).grid(row=6, column=0, sticky=tk.W)
device_var = tk.IntVar(value=1)
device_cpu = ttk.Radiobutton(other_config_frame, text="cuda", value=1, variable=device_var)
device_cuda = ttk.Radiobutton(other_config_frame, text="cpu", value=0, variable=device_var)
device_cpu.grid(row=6, column=1, sticky=tk.W)
device_cuda.grid(row=6, column=2, sticky=tk.W)

ttk.Label(other_config_frame, text="Config File:", width=20).grid(row=7, column=0, sticky=tk.W)
config_file_var = tk.StringVar(value="config.yaml")
config_file_entry = ttk.Entry(other_config_frame, width=12, textvariable=config_file_var)
config_file_entry.grid(row=7, column=1, sticky=tk.W)

# Save Button
command_frame = ttk.Frame(root, padding=(5, 5, 5, 5))
command_frame.grid(row=11, column=0, sticky=tk.W+tk.E)

# Function to collect data from the GUI
def collect_data():
    benchmark_selected = benchmark_combobox.get()
    model_names = list(MODELS[benchmark_selected].keys())
    model_name_selected = model_names[model_config_var.get()]
    severity = severity_combobox.get()
    if severity.isnumeric():
        severity = int(severity)

    data = {
        "benchmark": {
            benchmark_selected: {
                "num_classes": int(num_classes_spinbox.get()),
                "img_size": int(img_size_spinbox.get()),
                "ood_datasets": list(ood_dataset_listbox.get(0, tk.END)),
                "model": {
                    model_name_selected: {kv.split(": ")[0]: kv.split(": ")[1] for kv in model_variant_listbox.get(0, tk.END)}
                }
            }
        },
        "score_functions": list(detector_listbox.get(0, tk.END)),
        "perturb_functions": list(perturb_func_listbox.get(0, tk.END)),
        "rand_seed": int(rand_seed_spinbox.get()),
        "batch_size": int(batch_size_spinbox.get()),
        "n_seeds": int(n_seeds_spinbox.get()),
        "n_sampling": int(n_sampling_spinbox.get()),
        "severity": severity,
        "eval_severity": eval_severity_combobox.get(),
        "datadir": datadir.get(),
        "device": ["cpu", "cuda"][device_var.get()],
    }
    return data

# Function to save data to a YAML file
def save_to_yaml(verbose=True):
    data = collect_data()
    config_file = config_file_var.get()
    if config_file.endswith(".yaml"):
        with open(config_file, "w") as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        if verbose:
            messagebox.showinfo("Save Successful", f"Data saved to {config_file}")

def run_test():
    save_to_yaml(verbose=False)
    config_file = config_file_var.get()
    subprocess.run(["python", "ood_test.py", "--cfg", config_file])

def run_eval():
    save_to_yaml(verbose=False)
    config_file = config_file_var.get()
    subprocess.run(["python", "ood_eval.py", "--cfg", config_file])

ttk.Button(command_frame, text="Save Config to YAML", command=save_to_yaml, width=21).grid(row=0, column=0, sticky=tk.W, padx=5)
ttk.Button(command_frame, text="Save & Run Test", command=run_test, width=21).grid(row=0, column=2, sticky=tk.W, padx=5)
ttk.Button(command_frame, text="Save & Run Eval", command=run_eval, width=21).grid(row=0, column=3, sticky=tk.W, padx=5)
ttk.Button(command_frame, text="Exit", command=root.quit, width=5).grid(row=0, column=4, sticky=tk.W, padx=5)

root.mainloop()
