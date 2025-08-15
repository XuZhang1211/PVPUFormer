import os
import sys
from glob import glob


model_path = sys.argv[1]
gpu_id = sys.argv[2]
exp_name = sys.argv[3]
exp_prefix = model_path.split('/')[1]

batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 35
accu_grad = int(sys.argv[5]) if len(sys.argv) > 5 else 140 // batch_size

# run train.py
TRAIN_CMD = f"""
python train.py {model_path} --gpus={gpu_id} \\
    --batch-size={batch_size} --accumulate-grad={accu_grad} --workers=12 --amp \\
    --exp-name={exp_name}
"""
# print(TRAIN_CMD)
os.system(TRAIN_CMD)


exp_path = glob(f'experiments/{exp_prefix}/*/*_{exp_name}')[0]
checkpoint = os.path.join(exp_path, 'checkpoints', '*.pth')
checkpoint = sorted(glob(checkpoint))[-1]
checkpoint = os.path.basename(checkpoint).split('.')[0]
exp_path = exp_path.replace('experiments/', '')

EVAL_CMD = f"""
python scripts/evaluate_model.py NoBRS \\
    --exp-path={exp_path}:{checkpoint} --gpus={gpu_id}
"""
# print(EVAL_CMD)
os.system(EVAL_CMD)
