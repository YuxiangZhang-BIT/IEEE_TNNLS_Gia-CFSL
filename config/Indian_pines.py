from collections import OrderedDict

config = OrderedDict()
config['data_path'] = '/datasets'
config['save_path'] = '/results'
config['source_data'] = 'Chikusei_imdb_128.pickle'
config['target_data'] = 'IP/indian_pines_corrected.mat'
config['target_data_gt'] = 'IP/indian_pines_gt.mat'
config['num_generation'] = 1
config['gpu'] = 0
config['point_distance_metric'] = 'l2'
config['distribution_distance_metric'] = 'l2'


train_opt = OrderedDict()
train_opt['patch_size'] = 9
train_opt['batch_task'] = 1
train_opt['num_ways'] = 5
train_opt['num_shots'] = 1
train_opt['episode'] = 10000
train_opt['lr'] = 1e-2
train_opt['weight_decay'] = 1e-4
train_opt['dropout'] = 0.1
train_opt['lambda_1'] = 1
train_opt['lambda_2'] = 0.1
train_opt['d_emb'] = 128
train_opt['src_input_dim'] = 128
train_opt['tar_input_dim'] = 200
train_opt['n_dim'] = 100
train_opt['class_num'] = 16
train_opt['shot_num_per_class'] = 1
train_opt['query_num_per_class'] = 19

train_opt['test_class_num'] = 16
train_opt['test_lsample_num_per_class'] = 5

config['train_config'] = train_opt
