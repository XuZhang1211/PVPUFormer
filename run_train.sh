# export CUDA_VISIBLE_DEVICES=1,3

# python train.py models/plainvit_huge448_cocolvis.py \
#     --batch-size=2 \
#     --gpus=1

# python train.py models/plainvit_huge448_cocolvis.py \
#     --batch-size=32 \
#     --ngpus=4


# python train.py models/plainvit_base448_cocolvis.py \
#     --batch-size=40 \
#     --gpus=0,1

# python train.py models/plainvit_base448_cocolvis.py \
#     --batch-size=40 \
#     --gpus=1

# export CUDA_VISIBLE_DEVICES=4
# python train.py models/iSegNet/guassianvector_huge448_cocolvis.py \
#     --batch-size=2 \
#     --workers=2 \
#     --ngpus=1

# export CUDA_VISIBLE_DEVICES=4
# python train.py models/iSegNet/mmvector_base448_cocolvis.py \
#     --is-model-path=isegm/model/is_vitdetr_mmvector_model.py \
#     --batch-size=16 \
#     --workers=8 \
#     --ngpus=1 \
#     --epochs=55

# MODEL_TYPE=gaussian
# echo models/iSegNet/${MODEL_TYPE}vector_base448_cocolvis.py
# echo isegm/model/is_vitdetr_${MODEL_TYPE}vector_model.py

# export CUDA_VISIBLE_DEVICES=3,4
# python train.py models/iSegNet/${MODEL_TYPE}vector_base448_cocolvis.py \
#     --is-model-path=isegm/model/is_vitdetr_${MODEL_TYPE}vector_model.py \
#     --batch-size=32 \
#     --workers=16 \
#     --ngpus=2


MODEL_PATH=experiments/iSegNet/cocolvis_multigaussianvector_samdecoder_vpuformer_pcl_prompt012_errormaskbox_pretrain_base448/002/checkpoints/028.pth
echo "e28-prompt0-cf-0->"${MODEL_PATH}
python scripts/evaluate_pclmodel.py NoBRS \
    --gpus=3 \
    --checkpoint=${MODEL_PATH} \
    --datasets=COCO_MVal \
    --cf-n=0 \
    --acf

# COCO_MVal, ADE20K, PascalVOC

# export CUDA_VISIBLE_DEVICES=2
# python train.py models/iSegNet/multigaussianvector_only_edloss_base448_cocolvis.py \
#     --is-model-path=isegm/model/is_vitdetr_multigaussianvector_only_edloss_model.py \
#     --batch-size=12 \
#     --workers=4 \
#     --ngpus=1

# export CUDA_VISIBLE_DEVICES=3
# python train.py models/iSegNet/gaussianvector_edloss_base448_cocolvis.py \
#     --is-model-path=isegm/model/is_vitdetr_gaussianvector_edloss_model.py \
#     --batch-size=12 \
#     --workers=12 \
#     --ngpus=1

# MODEL_TYPE=gaussian
# echo models/iSegNet/${MODEL_TYPE}vector_base448_cocolvis.py
# echo isegm/model/is_vitdetr_${MODEL_TYPE}vector_model.py

# export CUDA_VISIBLE_DEVICES=1,2,0
# python train.py models/iSegNet/${MODEL_TYPE}vector_base448_sbd.py \
#     --is-model-path=isegm/model/is_vitdetr_${MODEL_TYPE}vector_model.py \
#     --batch-size=48 \
#     --workers=16 \
#     --ngpus=3 \
#     --resume-exp=007 \
#     --resume-prefix=064

# MODEL_TYPE=gaussian
# echo models/iSegNet/${MODEL_TYPE}vector_base448_cocolvis.py
# echo isegm/model/is_vitdetr_${MODEL_TYPE}vector_model.py

# export CUDA_VISIBLE_DEVICES=1,2,0
# python train.py models/iSegNet/${MODEL_TYPE}vector_base448_cocolvis.py \
#     --is-model-path=isegm/model/is_vitdetr_${MODEL_TYPE}vector_model.py \
#     --batch-size=45 \
#     --workers=15 \
#     --ngpus=3 \
#     --resume-exp=007 \
#     --resume-prefix=014

# MODEL_TYPE=gaussian
# echo models/iSegNet/${MODEL_TYPE}vector_huge448_cocolvis.py
# echo isegm/model/is_vitdetr_${MODEL_TYPE}vector_model.py

# export CUDA_VISIBLE_DEVICES=3,4,5
# python train.py models/iSegNet/${MODEL_TYPE}vector_huge448_cocolvis.py \
#     --is-model-path=isegm/model/is_vitdetr_${MODEL_TYPE}vector_model.py \
#     --batch-size=12 \
#     --workers=6 \
#     --ngpus=3


