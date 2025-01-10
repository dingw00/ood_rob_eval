import torch as torch
import kornia
from . import spatial
from .test_utils import setup_seed

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, severity="all"):

        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (torch.tensor): the original input
        '''
        self.severity = severity

    def random_perturb(self, x, seed=0):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

### Instantiations of the AttackerStep class

def transform(x, rotation, translation, scale):

    with torch.no_grad():
        transformed = spatial.transform(x, rotation, translation, scale)
 
    return transformed

def transform_kornia(x, hue, saturation, bright, contrast):
    # assert x.shape[1] == 3

    with torch.no_grad():
        x = kornia.enhance.adjust_hue(x, hue)
        x = kornia.enhance.adjust_saturation(x, saturation)
        x = kornia.enhance.adjust_brightness(x, bright)
        x = kornia.enhance.adjust_contrast(x, contrast)
        transformed = x
    return transformed


def blur_kornia(x, gau_size, gau_sigma1, gau_sigma2):

    with torch.no_grad():
        bs = x.shape[0] 
        if bs == 1:
            transformed = kornia.filters.gaussian_blur2d(x, (gau_size, gau_size), (gau_sigma1, gau_sigma2))
        else:
            transformed = x
            for i in range(bs):
                transformed[i,:,:,:] = kornia.filters.gaussian_blur2d(x[i, :, :,:].unsqueeze(0), 
                                                                      (gau_size, gau_size), 
                                                                      (gau_sigma1[i], gau_sigma2[i]))
    return transformed


class Spatial(AttackerStep):

    rot_list = [0,6,12,18,24,30]
    trans_list = [0,0.04,0.08,0.12,0.16,0.20]
    scale_list = [0,0.06,0.12,0.18,0.24,0.30]
    
    def __init__(self, severity=5, perturb_type="rotation"):

        super().__init__(severity=severity)
        
        rot_max, rot_min = 0, 0
        trans_max, trans_min = 0, 0
        scale_max, scale_min = 0, 0

        assert severity in [1, 2, 3, 4, 5, "all"]
        if severity == "all":
            severity1 = 5
            severity0 = 1
        else:
            severity1 = severity
            severity0 = severity

        if perturb_type == "rotation":
            rot_max = self.rot_list[severity1]
            rot_min = self.rot_list[severity0-1]
        elif perturb_type == "translation":
            trans_max = self.trans_list[severity1]
            trans_min = self.trans_list[severity0-1]
        elif perturb_type == "scale":
            scale_max = self.scale_list[severity1]
            scale_min = self.scale_list[severity0-1]

        self.rot_max = float(rot_max)
        self.rot_min = float(rot_min)
        self.trans_max = float(trans_max)
        self.trans_min = float(trans_min)
        self.scale_max = float(scale_max)
        self.scale_min = float(scale_min)

    def random_perturb(self, x, n_repeat=1, seed=0, device="cpu"):

        assert x.shape[2] == x.shape[3]
        x = x.to(device)        
        x = x.repeat(n_repeat, 1, 1, 1)
        bs = x.shape[0]

        setup_seed(seed)
        # (-rot_max, -rot_min), (+rot_min, +rot_max)
        flip_sign = (torch.rand(size=(bs,), device=device) > 0.5) * 2 - 1
        rots = torch.rand(size=(bs,), device=device) * (self.rot_max - self.rot_min) + self.rot_min 
        rots = rots * flip_sign

        setup_seed(seed)
        # (-trans_max, -trans_min), (+trans_min, +trans_max)
        flip_sign = (torch.rand(size=(bs, 2), device=device) > 0.5) * 2 - 1
        txs = torch.rand(size=(bs, 2), device=device) * (self.trans_max - self.trans_min) + self.trans_min
        txs = txs * flip_sign

        setup_seed(seed)
        # (1-trans_max, 1-trans_min), (1+trans_min, 1+trans_max)
        flip_sign = (torch.rand(size=(bs, 2), device=device) > 0.5) * 2 - 1
        scales = torch.rand(size=(bs, 2), device=device) * (self.scale_max - self.scale_min) + self.scale_min
        scales = scales * flip_sign + 1

        transformed = transform(x, rots, txs, scales)
        transformed = torch.clamp(transformed, 0, 1)

        return transformed.cpu()
        

class Color(AttackerStep):

    hue_list = [0,0.06,0.09,0.12,0.15,0.18]
    satu_list = [0,0.16,0.32,0.48,0.64,0.8]
    
    bright_list = [0,0.06,0.12,0.18,0.24,0.30]
    cont_list = [0,0.06,0.12,0.18,0.24,0.30]
    
    def __init__(self, severity=5, perturb_type="hue"):

        super().__init__(severity=severity)
        hue_max, hue_min = 0, 0
        satu_max, satu_min = 0, 0
        bright_max, bright_min = 0, 0
        cont_max, cont_min = 0, 0

        assert severity in [1, 2, 3, 4, 5, "all"]
        
        if severity == "all":
            severity1 = 5
            severity0 = 1
        else:
            severity1 = severity
            severity0 = severity

        if perturb_type == "hue":
            hue_max = self.hue_list[severity1]
            hue_min = self.hue_list[severity0-1]
        elif perturb_type == "saturation":
            satu_max = self.satu_list[severity1]
            satu_min = self.satu_list[severity0-1]
        elif perturb_type == "bright_contrast":
            bright_max = self.bright_list[severity1]
            bright_min = self.bright_list[severity0-1]
            cont_max = self.cont_list[severity1]
            cont_min = self.cont_list[severity0-1]

        self.hue_max= float(hue_max)
        self.hue_min = float(hue_min)
        self.satu_max = float(satu_max)
        self.satu_min = float(satu_min)
        self.bright_max = float(bright_max)
        self.bright_min = float(bright_min)
        self.cont_max = float(cont_max)
        self.cont_min = float(cont_min)

    def random_perturb(self, x, n_repeat=1, seed=0, device="cpu"):

        assert x.shape[2] == x.shape[3]
        x = x.to(device)        
        x = x.repeat(n_repeat, 1, 1, 1)
        bs = x.shape[0]
        
        setup_seed(seed)
        # (-hue_max, -hue_min)*pi, (+hue_min, +hue_max)*pi
        flip_sign = (torch.rand(size=(bs,), device=device) > 0.5) * 2 - 1
        hues = torch.rand(size=(bs,), device=device) * (self.hue_max - self.hue_min) + self.hue_min
        hues = hues * flip_sign * torch.pi

        setup_seed(seed)
        # (1-satu_max, 1-satu_min), (1+satu_min, 1+satu_max)
        flip_sign = (torch.rand(size=(bs,), device=device) > 0.5) * 2 - 1
        satus = torch.rand(size=(bs,), device=device) * (self.satu_max - self.satu_min) + self.satu_min
        satus = satus * flip_sign + 1

        setup_seed(seed)
        # (-bright_max, -bright_min), (+bright_min, +bright_max)
        flip_sign = (torch.rand(size=(bs,), device=device) > 0.5) * 2 - 1
        brights = torch.rand(size=(bs,), device=device) * (self.bright_max - self.bright_min) + self.bright_min 
        brights = brights * flip_sign

        setup_seed(seed)
        # (1-cont_max, 1-cont_min), (1+cont_min, 1+cont_max)
        flip_sign = (torch.rand(size=(bs,), device=device) > 0.5) * 2 - 1
        conts = torch.rand(size=(bs,), device=device) * (self.cont_max - self.cont_min) + self.cont_min
        conts = conts * flip_sign + 1

        transformed = transform_kornia(x, hues, satus, brights, conts)
        transformed = torch.clamp(transformed, 0, 1)

        return transformed.cpu()
        

class Blur(AttackerStep):

    gau_size_dict = {"CIFAR10": [5, 5, 7, 7, 9], "Imagenet100": [9, 17, 25, 33, 49], "Imagenet1k": [9, 17, 25, 33, 49]}
    gau_sigma_dict = {"CIFAR10": [0, 0.4, 0.6, 0.7, 0.8, 1.0], "Imagenet100": [0, 1, 2, 3, 4, 6], "Imagenet1k": [0, 1, 2, 3, 4, 6]}
    
    def __init__(self, severity=5, benchmark="CIFAR10"):
        
        assert severity in [1, 2, 3, 4, 5, "all"]
        super().__init__(severity=severity)
        if severity == "all":
            severity1 = 5
            severity0 = 1
        else:
            severity1 = severity
            severity0 = severity

        gau_sigma_max = self.gau_sigma_dict[benchmark][severity1]
        gau_sigma_min = self.gau_sigma_dict[benchmark][severity0-1]

        self.gau_size = int(self.gau_size_dict[benchmark][severity1-1])
        
        self.gau_sigma_max = float(gau_sigma_max)
        self.gau_sigma_min = float(gau_sigma_min)

    def random_perturb(self, x, n_repeat=1, seed=0, device="cpu"):

        assert x.shape[2] == x.shape[3]
        x = x.to(device)        
        x = x.repeat(n_repeat, 1, 1, 1)

        bs = x.shape[0]

        setup_seed(seed)
        gau_sigma1 = torch.rand(size=(bs,), device=device) * (self.gau_sigma_max - self.gau_sigma_min) + self.gau_sigma_min
        gau_sigma2 = torch.rand(size=(bs,), device=device) * (self.gau_sigma_max - self.gau_sigma_min) + self.gau_sigma_min

        transformed = blur_kornia(x, self.gau_size, gau_sigma1, gau_sigma2)
        transformed = torch.clamp(transformed, 0, 1)

        return transformed.cpu()
    
# L-infinity threat model
class LinfStep(AttackerStep):

    eps_dict = {"CIFAR10": [0,0.016,0.032,0.048,0.064,0.08], "Imagenet100": [0,0.04,0.08,0.12,0.16,0.2], "Imagenet1k": [0,0.04,0.08,0.12,0.16,0.2]}

    def __init__(self, severity=5, benchmark="CIFAR10"):

        assert severity in [1, 2, 3, 4, 5, "all"]
        super().__init__(severity=severity)
        if severity == "all":
            severity1 = 5
            severity0 = 1
        else:
            severity1 = severity
            severity0 = severity

        eps_max = self.eps_dict[benchmark][severity1]
        eps_min = self.eps_dict[benchmark][severity0-1]

        self.eps_max = float(eps_max)
        self.eps_min = float(eps_min)

    def random_perturb(self, x, n_repeat=1, seed=0, device="cpu"):

        x = x.to(device)
        x = x.repeat(n_repeat, 1, 1, 1)

        setup_seed(seed)
        # (-eps_max, eps_min), (eps_min, eps_max)
        flip_sign = (torch.rand(x.shape, device=device) > 0.5) * 2 - 1
        noise = torch.rand(x.shape, device=device) * (self.eps_max - self.eps_min) + self.eps_min
        noise = noise * flip_sign
        transformed = torch.clamp(x+noise, 0, 1)

        return transformed.cpu()

# L2 threat model
class L2Step(AttackerStep):

    eps_dict = {"CIFAR10": [0,0.8,1.6,2.4,3.2,4.0], "Imagenet100": [0,16,32,48,64,80], "Imagenet1k": [0,16,32,48,64,80]}

    def __init__(self, severity=5, benchmark="CIFAR10"):
        
        assert severity in [1, 2, 3, 4, 5, "all"]
        super().__init__(severity=severity)
        if severity == "all":
            severity1 = 3
        else:
            severity1 = severity

        eps = self.eps_dict[benchmark][severity1]

        self.eps = float(eps)

    def random_perturb(self, x, n_repeat=1, seed=0, device="cpu"):

        x = x.to(device)
        x = x.repeat(n_repeat, 1, 1, 1)

        setup_seed(seed)
        noise = (torch.rand(x.shape, device=device) - 0.5).renorm(p=2, dim=0, maxnorm=self.eps)

        transformed = torch.clamp(x+noise, 0, 1)

        return transformed.cpu()


def build_attacker(perb_func, severity_level, benchmark):
    if perb_func == "rotation":  
        attacker = Spatial(severity=severity_level, perturb_type="rotation")
    elif perb_func == "translation":
        attacker = Spatial(severity=severity_level, perturb_type="translation")
    elif perb_func == "scale":
        attacker = Spatial(severity=severity_level, perturb_type="scale")

    elif perb_func == "hue":
        attacker = Color(severity=severity_level, perturb_type="hue")
    elif perb_func == "saturation":
        attacker = Color(severity=severity_level, perturb_type="saturation")
    elif perb_func == "bright_contrast":
        attacker = Color(severity=severity_level, perturb_type="bright_contrast")
    
    elif perb_func == "blur":
        attacker = Blur(severity=severity_level, benchmark=benchmark)
    elif perb_func == "Linf":
        attacker = LinfStep(severity=severity_level, benchmark=benchmark)
    elif perb_func == "L2":
        attacker = L2Step(severity=severity_level, benchmark=benchmark)
    else:
        raise NotImplementedError(f"Attack {perb_func} not implemented.")
        
    return attacker

def build_attackers(perturb_functions, severity_level, benchmark):
    attackers = dict()
    for perb_func in perturb_functions:
        attacker = build_attacker(perb_func, severity_level, benchmark)
        attackers[perb_func] = attacker
    return attackers