from isegm.utils.exp_imports.default import *
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss

MODEL_NAME = 'cocolvis_multigaussianvector_decoder_vpuformer_pcl_prompt012_errormaskbox_scratch_base448_0225'.format(decoder_tpye)

def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (448, 448)
    model_cfg.num_max_points = 24
    model_cfg.with_aux_output=True

    backbone_params = dict(
        img_size=model_cfg.crop_size,
        patch_size=(16,16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4, 
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim = 768,
        out_dims = [128, 256, 512, 1024],
        img_size=model_cfg.crop_size,
    )

    head_params = dict(
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        loss_decode=CrossEntropyLoss(),
        align_corners=False,
        upsample=cfg.upsample,
        ed_loss=True,
        channels={'x1':256, 'x2': 128, 'x4': 64}[cfg.upsample],
    )

    model = VitMultiGaussianVector_ed_Model(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=cfg.random_split,
        residual=True,
        with_aux_output=model_cfg.with_aux_output,
    )
    
    model.backbone.init_weights_from_pretrained(cfg.IMAGENET_PRETRAINED_MODELS.MAE_BASE)
    model.to(cfg.device)

    return model, model_cfg


def train(model, cfg, model_cfg):
    print("-------------model_cfg.with_aux_output: {} -------------".format(model_cfg.with_aux_output))
    
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.epochs = 5 if cfg.epochs < 1 else cfg.epochs
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2, penalty_loss=False)
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.instance_aux_loss = DiceLoss(use_sigmoid=True, activate=True, naive_dice=True, loss_weight=1.0)
    loss_cfg.instance_aux_loss_weight = 1.0
    
    if model_cfg.with_aux_output:
        loss_cfg.instance_aux3_loss = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        loss_cfg.instance_aux3_loss_weight = 2.0

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    trainset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000,
        # epoch_len=-1,
        stuff_prob=0.30,
        copy_paste_prob=0.5,
        image_mix_prob=0.5,
        word_length=model_cfg.num_max_points*2,
        cfg = cfg,
    )

    valset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000,
        word_length=model_cfg.num_max_points*2,
        cfg = cfg,
    )
    
    # trainset = SBDDataset(
    #     cfg.SBD_PATH,
    #     split='train',
    #     augmentator=train_augmentator,
    #     min_object_area=20,
    #     keep_background_prob=0.01,
    #     points_sampler=points_sampler,
    #     samples_scores_path='./assets/sbd_samples_weights.pkl',
    #     samples_scores_gamma=1.25,
    #     copy_paste_prob=0.5,
    #     image_mix_prob=0.5
    # )

    # valset = SBDDataset(
    #     cfg.SBD_PATH,
    #     split='val',
    #     augmentator=val_augmentator,
    #     min_object_area=20,
    #     points_sampler=points_sampler,
    #     epoch_len=500
    # )

    optimizer_params = {
        'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8,
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[190, 210], gamma=0.1)
    
    # optimizer_params = {
    #     'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8
    # }

    # lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
    #                        milestones=[50, 55], gamma=0.1)
    
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        layerwise_decay=cfg.layerwise_decay,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 5), (190, 1)],
                        image_dump_interval=300,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3,
                        use_iterloss=True,
                        iterloss_weights=[1, 2, 3],
                        use_random_clicks=True,
                        ed_loss=True,
                        as_multi_prompts_ed_loss=True,
                        as_allmask=False)
    trainer.run(num_epochs=230 if not cfg.debug else 1, validation=False)
