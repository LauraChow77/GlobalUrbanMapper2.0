import torch
import copy
from .unet_res_multi_enc import UNetResMultiEnc


def _strip_kw_head(state_dict):
    """Remove kw-branch decode-head weights from a teacher checkpoint.

    Teacher models may have been trained with kw_branch=True, but in ST mode
    we instantiate the frozen teachers with kw_branch=False to save memory.
    Filtering lets us load only the shared backbone/decoder weights.
    """
    return {k: v for k, v in state_dict.items() if not k.startswith('kw_decode_head')}


def build_model(cfg, ema=False):
    model_name, head_name = cfg.MODEL.NAME.split('_')
    if model_name == 'UNetResMultiEnc':
        model = UNetResMultiEnc(kw_branch=True)
    else:
        raise NotImplementedError(f"Model {model_name} is not supported")
    if cfg.resume != "":
        print("Resuming model from", cfg.resume)
        state = torch.load(cfg.resume)
        model.load_state_dict(state["state_dict"])
    if cfg.MODEL.WEIGHTS != "":
        print("Loading model from", cfg.MODEL.WEIGHTS)
        checkpoint = torch.load(cfg.MODEL.WEIGHTS, map_location=lambda storage, loc: storage)
        checkpoint_state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key in checkpoint_state_dict:
            new_state_dict[key] = checkpoint_state_dict[key]
        model.load_state_dict(new_state_dict, strict=False)

    if cfg.MODE == 'ST':
        no_kw_model = UNetResMultiEnc(kw_branch=False)
        t_model = copy.deepcopy(no_kw_model)
        s1_topo_model = copy.deepcopy(no_kw_model)
        s2_topo_model = copy.deepcopy(no_kw_model)

        t_checkpoint = torch.load(cfg.MODEL.TEACHER_FULL, map_location=lambda storage, loc: storage)
        t_checkpoint_state_dict = _strip_kw_head(t_checkpoint['state_dict'])
        t_model.load_state_dict(t_checkpoint_state_dict)

        s1_topo_checkpoint = torch.load(cfg.MODEL.TEACHER_S1_TOPO, map_location=lambda storage, loc: storage)
        s1_topo_checkpoint_state_dict = _strip_kw_head(s1_topo_checkpoint['state_dict'])
        s1_topo_model.load_state_dict(s1_topo_checkpoint_state_dict)

        s2_topo_checkpoint = torch.load(cfg.MODEL.TEACHER_S2_TOPO, map_location=lambda storage, loc: storage)
        s2_topo_checkpoint_state_dict = _strip_kw_head(s2_topo_checkpoint['state_dict'])
        s2_topo_model.load_state_dict(s2_topo_checkpoint_state_dict)

        return model, t_model, s1_topo_model, s2_topo_model

    return model
