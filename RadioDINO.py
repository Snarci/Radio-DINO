from vit import vit_tiny, vit_small, vit_base
import torch
import torch.nn as nn
DIRECT_LINK_TINY = 'https://drive.google.com/u/0/uc?id=1Z-tczdk9SFFZQpD982chH3VUTEjTdaIf&export=download'
DIRECT_LINK_SMALL = 'https://drive.google.com/u/0/uc?id=1NTAxWKFAGPPyEXyUzCSw2FwDAc0--zNU&export=download'
DIRECT_LINK_BASE = 'https://drive.google.com/u/0/uc?id=1GKbSsvJkNI7Vl3l2yC9xipVuz2CT55Pn&export=download'

def RadioDINO(num_classes, model_name, pretrained=False,drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.):
    if model_name == 'Radio DINO tiny':
        model= vit_tiny(
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
            )
    elif model_name == 'Radio DINO small':
        model= vit_small(
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
            )
    elif model_name == 'Radio DINO base':
        model= vit_base(
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
            )
    else:
        raise ValueError(f"model {model_name} not supported, please choose from ['Radio DINO tiny', 'Radio DINO small', 'Radio DINO base']")
    if pretrained:
        strict = False if num_classes != 1000 else True
        print(f"Loading pretrained weights for {model_name}")
        if model_name == 'Radio DINO tiny':
            checkpoint = torch.hub.load_state_dict_from_url(DIRECT_LINK_TINY, progress=True,model_dir ='checkpoints',file_name='RadioDINO_tiny.pth')
        elif model_name == 'Radio DINO small':
            checkpoint = torch.hub.load_state_dict_from_url(DIRECT_LINK_SMALL, progress=True,model_dir ='checkpoints',file_name='RadioDINO_small.pth')
        elif model_name == 'Radio DINO base':
            checkpoint = torch.hub.load_state_dict_from_url(DIRECT_LINK_BASE, progress=True,model_dir ='checkpoints',file_name='RadioDINO_base.pth')
        model.load_state_dict(checkpoint, strict=strict)
            
    return model

def full_to_backbone_local(model, model_name, checkpoint):
    # load finetuned weights
    pretrained = torch.load(checkpoint, map_location=torch.device('cpu'))
    # make correct state dict for loading
    new_state_dict = {}
    pretrained=pretrained['teacher']
    for key, value in pretrained.items():
        if 'dino_head' in key or "ibot_head" in key or"head"in key or  "student" in key or 'loss' in key:
            pass
        else:
            new_key = key.replace('backbone.', '').replace('teacher.', '')
            new_state_dict[new_key] = value
            head_weights = model.head.weight
            head_bias = model.head.bias
            new_state_dict['head.weight'] = head_weights
            new_state_dict['head.bias'] = head_bias
    model.load_state_dict(new_state_dict, strict=True)
    return model


if __name__ == '__main__':
    model = RadioDINO(num_classes=1000, model_name='Radio DINO small', pretrained=True)
    #load 
    #model = full_to_backbone_local(model, 'Radio DINO small', 'Radio DINO small.pth')
    #save
    #torch.save(model.state_dict(), 'Radio DINO small backbone.pth')
    print(model)