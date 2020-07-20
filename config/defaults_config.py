from yacs.config import CfgNode as CN

cfg = CN()

# data
cfg.data = CN()
cfg.data.mean = [0.485, 0.456, 0.406]
cfg.data.std = [0.229, 0.224, 0.225]
cfg.data.resize = (256, 192)

cfg.data.split = 'trainval'
cfg.data.partition_idx = 0
# model
cfg.model = CN()
cfg.model.num_att = 51
cfg.model.last_conv_stride = 2
cfg.model.drop_pool5 = True
cfg.model.drop_pool5_rate = 0.5
cfg.model.pretrained = True
# train_model
# pre_train
cfg.train = CN()
cfg.train.batch_size = 48
cfg.train.num_workers = 4
cfg.train.none_classifier_params = 0.01
cfg.train.classifier_params = 0.1
cfg.train.momentum = 0.9
cfg.train.weight_decay = 0.0005
cfg.train.decay_at_epochs = (10, 20)
cfg.train.decay_at_epochs_factor = 0.1
# for training
cfg.train.device_ids = (0, )
cfg.train.set_seed = False
cfg.train.dataset = 'peta'
cfg.train.total_epochs = 45
# for tools
cfg.train.epoch_per_save = 1
cfg.train.save_model_path = '/data1/zhengxiaoqiang/model_save/zxq_pedestrain-attribute-recongnitoin-baseline'
cfg.train.epoch_per_val = 1
cfg.train.epoch_per_print = 20
cfg.train.weight_entropy = True
cfg.train.load_model_weight = True
cfg.train.weight_path = '/data1/zhengxiaoqiang/model_save/zxq_pedestrain-attribute-recongnitoin-baseline/focal_loss_with_tircks_baseline/epoch_10.pth'
cfg.train.resume = False
# test_step
cfg.test = CN()
cfg.test.test_split = 'test'
cfg.test.batch_size = 64
cfg.test.num_workers = 4

# loss function hyper-parameters
cfg.gamma_pos = 2
cfg.gamma_neg = 2
cfg.loss_p_w = 1.7
cfg.loss_n_w = 1
cfg.FocalLoss_pos_w = 1.7
cfg.FocalLoss_neg_w = 0.8





