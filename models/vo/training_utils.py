import os
import torch
import datetime
import yaml

def optim_to(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    return optim

def print_losses(losses_dict, state_dict, log_path, write=False):
    now = datetime.datetime.now()
    print_line = '[%s: epoch:%d, iter:%d]' % (now.strftime('%Y-%m-%d %H:%M:%S'), state_dict['epoch'], state_dict['itr'])
    for k, v in losses_dict.items():
        print_line = print_line + ' | %s: %.3f' % (k, v.item())
    print(print_line)
    
    if write:
        with open(os.path.join(log_path, 'log.txt'), 'a') as f:
            f.write(print_line + '\n')

def print_config(config, config_path, write=False):
    lines = ''
    for _key, _value in config.items():
        lines = lines + _key + ': ' + str(_value) + '\n'
    print(lines)

    if write:
        with open(os.path.join(config_path, 'config.yaml'),'w') as f:
            f.write(yaml.dump(config))

def save_ckpt(model_params, state, optim_params, save_path):
    save_params = {
        "state": state,
        "model_params": model_params,
        "optim_params": optim_params,
    }
    torch.save(save_params, os.path.join(save_path, 'ckpt.%d.pth' % (state["itr"])))