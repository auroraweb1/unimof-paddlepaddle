#!/bin/bash

# ✅ 设置路径（根据你自己的目录修改）
DATA_PATH="/root/Uni-Mof-Main-PaddlePaddle/single-system_gas_adsorption_property_prediction"
USER_DIR="/root/Uni-Mof-Main-PaddlePaddle/unimof"
MODEL_PATH="/root/Uni-Mof-Main-PaddlePaddle/save_finetune_single_gpu/CoRE_LCD_bset.pdparams"
RESULTS_PATH="/root/Uni-Mof-Main-PaddlePaddle/results"
SCRIPT_PATH="/root/Uni-Mof-Main-PaddlePaddle/unimof/infer.py"

# ✅ 设置推理参数
TASK_NAME="CoRE_LCD"         # 任务名：结构特征预测
TASK_TYPE="unimof_v1"        # Uni-MOF 的任务定义
LOSS="mof_v1_mse"
ARCH="unimat_base"           # 模型架构
SUBSET="test"                
BATCH_SIZE=16
NUM_WORKERS=4
DEVICE_ID=0


# ✅ 执行推理
/root/miniconda3/envs/unimof-paddle/bin/python $SCRIPT_PATH $DATA_PATH \
  --user-dir $USER_DIR \
  --task $TASK_TYPE \
  --task-name $TASK_NAME \
  --arch $ARCH \
  --valid-subset $SUBSET \
  --path $MODEL_PATH \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --results-path $RESULTS_PATH \
  --fp16 \
  --device-id $DEVICE_ID \
  --loss $LOSS
  
