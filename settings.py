params = {

    'batch_size': 16, 
    # optimizer settings
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'max_epoches': 40,
    # data settings
    'root_dir': '~/workspace/TCN_VRP_pytorch',
    'dataset': 'vg200', # vrd, vg200
    # other settings
    'print_freq': 10,
    'num_epoches': 40,
    'save_freq': 10,
    'model_name': 'TCN',
    'momentum': 0.9,
    'epsilon': [60, 60, 20],
    'zero_shot': True,
    'weighted_loss': 'weighted_label',
    'prefix': '',
    'lr_decay': 0.5,
    'hidden_units': 8192,
    'feature': 'fl', # fl, fa, ff
    'fc2': True,
    'lr_freq': 10,
    'shareAB': True,
    'dropout_prob': 0.3
}

def fix_settings(args=None):
    if args is not None:
        for k, v in args.items():
            params[k] = v
    params['model_file_name'] = '{}_{}.pth'.format(params['model_name'], params['dataset'])
