#!/bin/bash

# python scripts/evaluate_model.py NoBRS \
#     --gpus=1 \
#     --checkpoint=weights/cocolvis_icl_vit_huge.pth \
#     --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD \
#     --cf-n=4 \
#     --acf

# cf-n: CFR steps
# acf: adaptive CFR

# cf-4: 1.34   |  1.42   |  1.70
# cf-1: 1.32   |  1.34   |  1.40   |  1.70
# experiments/iSegNet/cocolvis_gaussianvector_fpnformer_base448/003/checkpoints/051.pth

# |    NoBRS    | Berkeley  |  1.35   |  1.41   |  1.78   |
# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_base448/007/checkpoints/064.pth

# |    NoBRS    |  GrabCut  |  1.26   |  1.30   |  1.38   |  1.74   |
# experiments/iSegNet/cocolvis_gaussianvector_detrdecoder_fpnformer_base448/002/checkpoints/002.pth

# python scripts/evaluate_model.py NoBRS \
#     --gpus=2 \
#     --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_fpnformer_base448/003/checkpoints/051.pth \
#     --datasets=GrabCut,Berkeley,SBD,DAVIS \
#     --cf-n=4 \
#     --acf \
#     --iou-analysis \
#     --save-ious \
#     --print-ious 



# python scripts/evaluate_model.py NoBRS \
#     --gpus=0 \
#     --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_base448/007/checkpoints/064.pth \
#     --datasets=DAVIS,GrabCut,Berkeley,DAVIS,PascalVOC,SBD \
#     --cf-n=0 \
#     --acf


# python scripts/evaluate_model.py NoBRS \
#     --gpus=2 \
#     --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_base448/007/checkpoints/014.pth \
#     --datasets=Berkeley \
#     --cf-n=1 \
#     --acf \
#     --vis-preds

# python scripts/evaluate_model.py NoBRS \
#     --gpus=2 \
#     --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_base448/007/checkpoints/004.pth \
#     --datasets=GrabCut,Berkeley,SBD,DAVIS,PascalVOC \
#     --cf-n=1 \
#     --acf 
#     --save-ious \
#     --vis-preds

# # desc : for loop
# for MODEL_PATH in 09 15
# do
#     python scripts/evaluate_model.py NoBRS \
#         --gpus=0 \
#         --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_base448/023/checkpoints/${MODEL_PATH}0.pth \
#         --datasets=GrabCut,Berkeley,SBD,DAVIS \
#         --cf-n=1 \
#         --acf
#         echo ${MODEL_PATH}
# done 

# experiments/backbone_compare/cocolvis_hrnet_w18_small_gv_samdecoder_size256_noginit/007/checkpoints/055.pth
# experiments/backbone_compare/cocolvis_hrnet_w18_small_gv_samdecoder_size256_noginit/007/checkpoints/070.pth
# experiments/backbone_compare/cocolvis_hrnet_w18_small_gv_samdecoder_size256_noginit/007/checkpoints/225.pth


# # exp -> edloss 
# # experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_base448/070/checkpoints/038.pth
# # experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_base448/070/checkpoints/072.pth
# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_base448/070/checkpoints/080.pth

# python scripts/evaluate_model.py NoBRS \
#     --gpus=0 \
#     --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_base448/070/checkpoints/080.pth \
#     --datasets=DAVIS,SBD \
#     --cf-n=1 \
#     --acf

# for i in {0..4..1}
# do
#     echo ${i}
#     python scripts/evaluate_model.py NoBRS \
#         --gpus=4 \
#         --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_base448/070/checkpoints/072.pth \
#         --datasets=DAVIS \
#         --cf-n=${i} \
#         --acf
# done 

# # desc : for loop
# for i in {10..80..2}
# do
#     echo ${i}
#     python scripts/evaluate_model.py NoBRS \
#         --gpus=0 \
#         --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_base448/070/checkpoints/0${i}.pth \
#         --datasets=GrabCut,Berkeley \
#         --cf-n=1 \
#         --acf
# done 

# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_pcl_a1b5c1_base448/000/checkpoints/009.pth

# # desc : for loop
# for i in {0..9..1}
# do
#     echo ${i}
#     python scripts/evaluate_model.py NoBRS \
#         --gpus=0 \
#         --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_pcl_a1b5c1_base448/000/checkpoints/00${i}.pth \
#         --datasets=GrabCut,Berkeley,DAVIS \
#         --cf-n=1 \
#         --acf
# done 

# # exp -> vqu_edloss 
# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_base448/112/checkpoints/014.pth
# experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_fpnformer_pcl_lr5e5_base448/000/checkpoints/010.pth

# # experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_fpnformer_pcl_jitterbox_lr5e5_base448/004/checkpoints/002.pth

# experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_fpnformer_pcl_jitterbox_lr5e5_base448/003/checkpoints/002.pth  # good
# experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_pcl_prompt01_errormaskbox_scratch_base448/000/checkpoints/023.pth
# experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_nopcl_prompt01_errormaskbox_scratch_base448/002/checkpoints/027.pth
# experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_nopcl_prompt02_errormaskbox_scratch_base448/000/checkpoints/021.pth
# BraTS,ssTEM,OAIZIB

# MODEL_PATH=experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_nopcl_prompt02_errormaskbox_scratch_base448/000/checkpoints/021.pth
# echo "e21-prompt0-cfn0->"${MODEL_PATH}
# python scripts/evaluate_pclmodel.py NoBRS \
#     --gpus=2 \
#     --checkpoint=${MODEL_PATH} \
#     --datasets=GrabCut,Berkeley,DAVIS \
#     --cf-n=0 \
#     --acf \
#     --iou-analysis \
#     --save-ious \
#     --print-ious 
#     # --vis-preds

# # experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_base448/070/checkpoints/080.pth
# # experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_pcl_a1b1c2_base448/000/checkpoints/004.pth
# MODEL_PATH=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_base448/070/checkpoints/080.pth
# echo "080-prompt0-cfn0->"${MODEL_PATH}
# python scripts/evaluate_pclmodel.py NoBRS \
#     --gpus=2 \
#     --checkpoint=${MODEL_PATH} \
#     --datasets=PascalVOC \
#     --iou-analysis \
#     --save-ious \
#     --print-ious \
#     --cf-n=0 \
#     --acf 

# Epoch 229 lr 5.000000000000001e-07, training loss 5.5561, Train-Metrics/AdaptiveIoU: 0.87591: 100%|9
# experiments/iSegNet/cocolvis_multigaussianvector_only_samdecoder_vpuformer_pcl_prompt012_errormaskbox_scratch_base448_0225/001/checkpoints/229.pth
# Epoch 229 lr 5.000000000000001e-07, training loss 5.4818, Train-Metrics/AdaptiveIoU: 0.88653: 100%|9
# experiments/iSegNet/cocolvis_multigaussianvector_only_samdecoder_vpuformer_pcl_prompt0_errormaskbox_scratch_base448_0226/000/checkpoints/229.pth
# Epoch 229 lr 5.000000000000001e-07, training loss 5.3717, Train-Metrics/AdaptiveIoU: 0.88787: 100%|9
# experiments/iSegNet/cocolvis_multigaussianvector_only_samdecoder_vpuformer_pcl_prompt01_errormaskbox_scratch_base448_0227/000/checkpoints/229.pth

# MODEL_PATH=experiments/iSegNet/cocolvis_multigaussianvector_only_samdecoder_vpuformer_pcl_prompt01_errormaskbox_scratch_base448_0227/000/checkpoints/229.pth
# echo "e28-prompt02-cf-0->"${MODEL_PATH}
# python scripts/evaluate_pclmodel.py NoBRS \
#     --gpus=3 \
#     --checkpoint=${MODEL_PATH} \
#     --datasets=GrabCut,Berkeley,SBD,DAVIS \
#     --cf-n=0 \
#     --acf \
#     --iou-analysis \
#     --save-ious \
#     --print-ious
#     # --vis-preds

# experiments/iSegNet/cocolvis_gaussianvector_fpnformer_base448/003/checkpoints/054.pth
# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_base448/023/checkpoints/150.pth # 

# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_base448/070/checkpoints/084.pth
# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_pcl_a1b1c1_base448/000/checkpoints/009.pth


# # experiments/iSegNet/sbd_gaussianvector_samdecoder_fpnformer_edloss_a1b5c1_base448/000/checkpoints/037.pth
# # # experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_nopcl_prompt01_errormaskbox_scratch_base448/002/checkpoints/031.pth
# # experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_pcl_prompt012_errormaskbox_scratch_base448/038/checkpoints/229.pth

# MODEL_PATH=experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_pcl_prompt012_errormaskbox_pretrain_base448/002/checkpoints/028.pth
MODEL_PATH=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_base448/070/checkpoints/080.pth
echo "nopcl-cf-0->"${MODEL_PATH}
python scripts/evaluate_pclmodel.py NoBRS \
    --gpus=2 \
    --checkpoint=${MODEL_PATH} \
    --datasets=GrabCut \
    --cf-n=0 \
    --acf \
    --iou-analysis \
    --save-ious \
    --print-ious
    # --vis-preds
    # --target-iou=0.9 \

# MODEL_PATH=experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_pcl_prompt012_errormaskbox_pretrain_base448/002/checkpoints/028.pth
# echo "e28-prompt02-cf-0->"${MODEL_PATH}
# python scripts/evaluate_pclmodel.py NoBRS \
#     --gpus=3 \
#     --checkpoint=${MODEL_PATH} \
#     --datasets=ADE20K \
#     --cf-n=0 \
#     --acf 
    
    # --iou-analysis \
    # --save-ious \
    # --print-ious
    # --vis-preds

# COCO_MVal, ADE20K, PascalVOC

# # exp-> lambda-123: 
# # experiments/iSegNet/sbd_gaussianvector_samdecoder_fpnformer_edloss_a1b5c1_base448/000/checkpoints/036.pth          # sbd-a1b5c1
# # experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_a1b1c1_base448/003/checkpoints/008.pth     # cl-a1b1c1
# # experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_a1b5c1_base448/001/checkpoints/009.pth     # cl-a1b5c1
# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_pcl_a1b1c2_base448/000/checkpoints/008.pth          # a1b1c2
# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_pcl_a1b1c5_base448/000/checkpoints/008.pth          # a1b1c5
# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_pcl_a1b1c10_base448/000/checkpoints/002.pth         # a1b1c10
# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_pcl_a5b1c2_base448/000/checkpoints/007.pth          # a5b1c2
# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_pcl_a3b1c1_base448/000/checkpoints/003.pth          # a3b1c1
# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_pcl_a5b5c1_base448/000/checkpoints/004.pth          # a5b5c1
# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_pcl_a1b1c0.5_base448/002/checkpoints/009.pth        # a1b1c0.5

# # desc : for loop
# for i in {0..9..1}
# do
#     echo a1b1c0.5-${i}
#     python scripts/evaluate_model.py NoBRS \
#         --gpus=1 \
#         --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_pcl_a1b1c0.5_base448/002/checkpoints/00${i}.pth \
#         --datasets=Berkeley,DAVIS \
#         --cf-n=1 \
#         --acf \
#         --iou-analysis \
#         --save-ious \
#         --print-ious
# done 

# # desc : for loop
# for i in {0..10..1}
# do
#     echo ${i}
#     python scripts/evaluate_model.py NoBRS \
#         --gpus=0 \
#         --checkpoint=experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_fpnformer_pcl_lr5e5_base448/000/checkpoints/00${i}.pth \
#         --datasets=Berkeley,DAVIS \
#         --cf-n=1 \
#         --acf
# done 


# # exp-> lambda-123: 
# # experiments/iSegNet/sbd_gaussianvector_samdecoder_fpnformer_edloss_a1b5c1_base448/000/checkpoints/036.pth          # sbd-a1b5c1
# # experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_a1b1c1_base448/003/checkpoints/008.pth     # cl-a1b1c1
# # experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_edloss_a1b5c1_base448/001/checkpoints/009.pth     # cl-a1b5c1
# experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_nopcl_prompt01_errormaskbox_scratch_base448/002/checkpoints/001.pth
# experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_pcl_prompt01_errormaskbox_scratch_base448/000/checkpoints/015.pth
# experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_nopcl_prompt01_errormaskbox_scratch_base448/002/checkpoints/017.pth

# MODEL_PATH=experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_nopcl_prompt01_errormaskbox_scratch_base448/002/checkpoints/017.pth
# echo ${MODEL_PATH}
# python scripts/evaluate_pclmodel.py NoBRS \
#     --gpus=0 \
#     --checkpoint=${MODEL_PATH} \
#     --datasets=Berkeley,DAVIS \
#     --cf-n=1 \
#     --acf 
#     # --iou-analysis \
#     # --save-ious \
#     # --print-ious \
#     # --vis-preds

# # # hr18
# # experiments/backbone_compare/cocolvis_hrnet_w18_small_gv_samdecoder_size256_noginit/007/checkpoints/190.pth
# # # desc : for loop
# for MODEL_PATH in {80..85..5}
# do
#     echo ${MODEL_PATH}
#     python scripts/evaluate_model.py NoBRS \
#         --gpus=0 \
#         --checkpoint=experiments/backbone_compare/cocolvis_hrnet_w18_small_gv_samdecoder_size256_noginit/007/checkpoints/${MODEL_PATH}.pth \
#         --datasets=DAVIS \
#         --cf-n=1 \
#         --acf
# done 

# python scripts/evaluate_model.py NoBRS \
#     --gpus=0 \
#     --checkpoint=experiments/backbone_compare/cocolvis_hrnet_w18_small_gv_samdecoder_size256_noginit/007/checkpoints/080.pth \
#     --datasets=DAVIS \
#     --cf-n=1 \
#     --acf

# experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_base448/023/checkpoints/070.pth

# python scripts/evaluate_model.py NoBRS \
#     --gpus=0 \
#     --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_base448/023/checkpoints/070.pth \
#     --datasets=SBD,DAVIS,GrabCut,Berkeley \
#     --cf-n=0 \
#     --acf 

# python scripts/evaluate_model.py NoBRS \
#     --gpus=0 \
#     --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_base448/007/checkpoints/014.pth \
#     --datasets=BraTS,OAIZIB,ssTEM \
#     --cf-n=0 \
#     --acf \
#     --iou-analysis \
#     --save-ious \
#     --print-ious \
#     --vis-preds

# python scripts/evaluate_model.py NoBRS \
#     --gpus=0 \
#     --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_base448/007/checkpoints/014.pth \
#     --datasets=GrabCut \
#     --cf-n=1 \
#     --acf \
#     --iou-analysis \
#     --save-ious \
#     --print-ious \
#     --vis-preds

# python scripts/evaluate_model.py NoBRS \
#     --gpus=0 \
#     --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_base448/007/checkpoints/064.pth \
#     --datasets=BraTS,ssTEM,OAIZIB \
#     --cf-n=1 \
#     --acf \
#     --iou-analysis \
#     --save-ious \
#     --print-ious \
#     --vis-preds

# python scripts/evaluate_model.py NoBRS \
#     --gpus=2 \
#     --checkpoint=experiments/backbone_compare/sbd_resnet_gv_samdecoder_size256/011/checkpoints/225.pth \
#     --datasets=GrabCut,Berkeley,SBD,DAVIS \
#     --cf-n=1 \
#     --acf \
#     --iou-analysis \
#     --save-ious \
#     --print-ious 

# python scripts/evaluate_model.py NoBRS \
#     --gpus=2 \
#     --checkpoint=experiments/iSegNet/cocolvis_gaussianvector_samdecoder_fpnformer_base448/022/checkpoints/010.pth \
#     --datasets=GrabCut,Berkeley,SBD,DAVIS \
#     --cf-n=1 \
#     --acf 

