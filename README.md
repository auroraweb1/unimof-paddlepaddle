# Uni-MOF-based-on-PaddlePaddle

**Uni-MOF: A comprehensive transformer-based approach for high-accuracy gas adsorption predictions in metal-organic frameworks**

*Nature Communications* [[paper](https://www.nature.com/articles/s41467-024-46276-x)][[arXiv](https://chemrxiv.org/engage/chemrxiv/article-details/6447d756e4bbbe4bbf3afeaa)]<a href="https://bohrium.dp.tech/notebook/cca98b584a624753981dfd5f8bb79674" target="_parent"><img src="https://cdn.dp.tech/bohrium/web/static/images/open-in-bohrium.svg" alt="Open In Bohrium"/></a>

<p align="center"><img src="unimof/figure/overview_new.jpg" width=60%></p>
<p align="center"><b>Uni-MOF框架示意图</b></p>

## 论文概述

作者提出了一个用于**金属有机框架（MOF）材料气体吸附性能预测的通用机器学习框架——Uni-MOF**，可预测 MOF 材料在不同气体、温度和压力条件下的吸附性能，目标是作为一个“**气体吸附检测器**”。

### 背景与挑战

- MOFs 是用于气体分离的优良材料，因其可调孔径和多样化结构。
- 传统模拟方法（如分子动力学、蒙特卡洛方法）虽然准确，但计算开销极大，不适用于大规模筛选。
- 早期的机器学习方法依赖特征工程，容易过拟合，且通常只支持单一气体、固定条件预测。

### 方法创新

1. **三维结构预训练（Self-supervised learning）**：
   - 使用超 63 万个 MOF/COF 结构数据进行预训练。
   - 类似 BERT，使用原子掩码预测与三维坐标去噪任务，学习材料的空间结构特征。
2. **统一预测模型（Fine-tuning）**：
   - 结合气体种类、温度、压力进行跨系统预测。
   - 输入仅需结构的 `.cif` 文件和操作条件。
3. **数据生成与增强**：
   - 使用 ToBaCCo 3.0 程序和 GCMC 模拟生成高质量、多样化的吸附数据。

### 实验结果

- 在多个大型数据库上表现出色：
  - hMOF_MOFX_DB（R²=0.98）
  - CoRE_MOFX_DB（R²=0.92）
  - CoRE_MAP_DB（R²=0.83）
- 能准确预测不同 MOF 中的不同气体吸附量，甚至在**高压下的吸附性能**也可仅通过**低压数据预测**。
- 实验结果表明，Uni-MOF 的预测与真实实验高度一致。
- 在预测未知气体的吸附时也显示出较强的泛化能力（如 Kr 的 R²=0.85）。

### 结构特征预测与可视化

- Uni-MOF 还可预测 MOF 的结构特征，如孔径、比表面积、孔隙率等（R²>0.99）。
- 使用 t-SNE 可视化展示其学习到的结构嵌入与吸附行为的显著相关性。

## 环境依赖

 - [Uni-Core](https://github.com/dptech-corp/Uni-Core)，可以查看它的[下载文档](https://github.com/dptech-corp/Uni-Core#installation)，可能会花一定时间。

可以使用以下命令直接拉取docker镜像：

```bash
docker pull dptechnology/unimol:latest-pytorch1.11.0-cuda11.3
```

 - rdkit==2021.09.5, 通过`conda install -y -c conda-forge rdkit==2021.09.5`安装。

## 训练

### 材料预训练脚本

```
#!/bin/bash

data_path="./examples/mof" # replace to your data path
save_dir="./save/" # replace to your save path
n_gpu=8
MASTER_PORT=$1
lr=3e-4
wd=1e-4
batch_size=8
update_freq=2
masked_token_loss=1
masked_coord_loss=1
masked_dist_loss=1
dist_threshold=5.0
minkowski_p=2.0
lattice_loss=1
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.15
noise_type='uniform'
noise=1.0
seed=1
warmup_steps=10000
max_steps=100000
global_batch_size=`expr $batch_size \* $n_gpu \* $update_freq`

# PaddlePaddle 分布式训练环境变量
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.8
export PADDLE_NCCL_FORCE_SYNC=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# PaddlePaddle 分布式启动命令
nohup python -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    --log_dir=${save_dir}/logs \
    train.py \
    --data_path=$data_path \
    --train_subset train \
    --valid_subset valid \
    --num_workers 8 \
    --task unimat \
    --loss unimat \
    --arch unimat_base \
    --optimizer adam \
    --beta1 0.9 \
    --beta2 0.99 \
    --epsilon 1e-6 \
    --weight_decay $wd \
    --lr_scheduler polynomial_decay \
    --learning_rate $lr \
    --warmup_steps $warmup_steps \
    --max_steps $max_steps \
    --update_freq $update_freq \
    --seed $seed \
    --fp16 \
    --fp16_opt_level O2 \
    --max_update $max_steps \
    --log_interval 1000 \
    --save_interval_updates 1000 \
    --validate_interval_updates 1000 \
    --keep_interval_updates 10 \
    --no_epoch_checkpoints \
    --masked_token_loss $masked_token_loss \
    --masked_coord_loss $masked_coord_loss \
    --masked_dist_loss $masked_dist_loss \
    --x_norm_loss $x_norm_loss \
    --delta_pair_repr_norm_loss $delta_pair_repr_norm_loss \
    --lattice_loss $lattice_loss \
    --mask_prob $mask_prob \
    --noise_type $noise_type \
    --noise $noise \
    --batch_size $batch_size \
    --dist_threshold $dist_threshold \
    --minkowski_p $minkowski_p \
    --remove_hydrogen \
    --save_dir $save_dir \
    >> "./logs/${save_dir}.log" 2>&1 &
```

脚本迁移的主要变更点：

1. 分布式启动命令：
   - 从 `torch.distributed.launch` 改为 `paddle.distributed.launch`
   - `--nproc_per_node` 参数改为 `--gpus` 指定GPU列表
2. 环境变量：
   - 移除了 `NCCL_ASYNC_ERROR_HANDLING` 和 `OMP_NUM_THREADS`
   - 添加了 PaddlePaddle 特定的环境变量
3. 参数命名风格：
   - 将 PyTorch 的下划线风格参数名改为 PaddlePaddle 更常用的形式（如 `--lr` 改为 `--learning_rate`）
4. FP16 配置：
   - 从 `--fp16-init-scale` 和 `--fp16-scale-window` 改为 Paddle 的 `--fp16_opt_level`
5. 日志和保存：
   - 保留了相同的日志和模型保存逻辑

### 跨系统气体吸附性质微调

```
#!/bin/bash

data_path="./cross-system_gas_adsorption_property_prediction"  # 数据路径
save_dir="./save_finetune"  # 保存路径
n_gpu=8
MASTER_PORT=10086
task_name="CoRE"  # 属性预测任务名称
num_classes=1
exp_name="mof_v2"
weight_path="./weights/checkpoint.pdparams"  # 改为Paddle权重格式
lr=3e-4
batch_size=8
epoch=50
dropout=0.2
warmup=0.06
update_freq=2
global_batch_size=`expr $batch_size \* $n_gpu \* $update_freq`
ckpt_dir="${exp_name}_${task_name}_trial"

# PaddlePaddle分布式环境配置
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.8
export PADDLE_NCCL_FORCE_SYNC=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# PaddlePaddle分布式启动命令
nohup python -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    --log_dir=${save_dir}/logs \
    train.py \
    --data_path=$data_path \
    --task_name=$task_name \
    --train_subset train \
    --valid_subset valid,test \
    --num_workers 8 \
    --task unimof_v2 \
    --loss mof_v2_mse \
    --arch unimof_v2 \
    --optimizer adam \
    --beta1 0.9 \
    --beta2 0.99 \
    --epsilon 1e-6 \
    --clip_norm 1.0 \
    --lr_scheduler polynomial_decay \
    --learning_rate $lr \
    --warmup_ratio $warmup \
    --max_epoch $epoch \
    --batch_size $batch_size \
    --update_freq $update_freq \
    --seed 1 \
    --use_amp \
    --amp_level O2 \
    --num_classes $num_classes \
    --pooler_dropout $dropout \
    --finetune_mol_model $weight_path \
    --log_interval 500 \
    --validate_interval_updates 500 \
    --remove_hydrogen \
    --save_interval_updates 1000 \
    --keep_interval_updates 10 \
    --no_epoch_checkpoints \
    --keep_best_checkpoints 1 \
    --save_dir ./logs_finetune/$save_dir \
    --best_checkpoint_metric valid_r2 \
    --maximize_best_checkpoint_metric \
    > ./logs_finetune/$save_dir.log 2>&1 &
```

### 单系统气体吸附性质微调

以MOF结构特征的预测为例，可能需要4个小时才能获得测试集上最佳的模型性能[判定系数(R2)]。数据集下载链接为https://github.com/auroraweb1/unimof-paddlepaddle/releases/download/v2.1.0-beta/mof_database.zip。

```
#!/bin/bash

data_path="./single-system_gas_adsorption_property_prediction"  # replace to your data path
save_dir="./save_finetune"  # replace to your save path
n_gpu=8
MASTER_PORT=10086
task_name="CoRE_PLD"  # property prediction task name
num_classes=1
exp_name='mof_v1'
weight_path="./weights/checkpoint.pdparams"  # changed to PaddlePaddle weight format
lr=3e-4
batch_size=8
epoch=50
dropout=0.2
warmup=0.06
update_freq=2
global_batch_size=`expr $batch_size \* $n_gpu \* $update_freq`

# PaddlePaddle 分布式训练环境变量
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.8
export PADDLE_NCCL_FORCE_SYNC=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# PaddlePaddle 分布式启动命令
nohup python -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    --log_dir=${save_dir}/logs \
    train.py \
    --data_path=$data_path \
    --task_name=$task_name \
    --train_subset train \
    --valid_subset valid,test \
    --num_workers 8 \
    --task unimof_v1 \
    --loss mof_v1_mse \
    --arch unimat_base \
    --optimizer adam \
    --beta1 0.9 \
    --beta2 0.99 \
    --epsilon 1e-6 \
    --weight_decay 0.0 \
    --clip_norm 1.0 \
    --lr_scheduler polynomial_decay \
    --learning_rate $lr \
    --warmup_ratio $warmup \
    --max_epoch $epoch \
    --batch_size $batch_size \
    --update_freq $update_freq \
    --seed 1 \
    --fp16 \
    --fp16_opt_level O2 \
    --num_classes $num_classes \
    --pooler_dropout $dropout \
    --finetune_from_model $weight_path \
    --log_interval 100 \
    --validate_interval 1 \
    --remove_hydrogen \
    --save_interval_updates 1000 \
    --keep_interval_updates 10 \
    --no_epoch_checkpoints \
    --keep_best_checkpoints 1 \
    --save_dir ./logs_finetune/$save_dir \
    --best_checkpoint_metric valid_r2 \
    --maximize_best_checkpoint_metric \
    > ./logs_finetune/$save_dir.log 2>&1 &
```

## 验证

