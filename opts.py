opts = {
    'dataset': {
        'name': 'dummy',
        'n_classes': 2,
        'dir': "./datasets/dummy",
        'train_test_ratio': 0.5,
    },
    'dataloader': {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 1,
        'drop_last': True,
    },
    'experiment': {
        'folder': 'results/dummy',
    },
    'optim': {
        'type': 'sgd',
        'n_epoch': 300,
        'lr_init': 0.1,
        'lr_schedule': [100, 180, 240, 280],
        'lr_decay': 0.2,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'network': {
        'name': 'simplenet',
        'pretrained': False,
    },
    'criterion': {
        'train': 'cross_entropy',
        'test': 'cross_entropy',
    },
}
