#!/bin/bash

TARGET_PID=2710

# 检查目标进程是否存在
while kill -0 "$TARGET_PID" 2>/dev/null; do
    echo "进程 $TARGET_PID 仍在运行，等待中..."
    sleep 10
done

echo "进程 $TARGET_PID 已结束，开始运行 main.py"

# 正确执行命令并将输出写入 nohup1.out
nohup python main.py \
  --task_name=mtl \
  --seed=100 \
  --model_name=ple \
  --dataset_path='Tenrec/ctr_data_0.1M.csv' \
  --train_batch_size=512 \
  --val_batch_size=512 \
  --test_batch_size=512 \
  --epochs=20 \
  --lr=0.000005 \
  --embedding_size=32 \
  --mtl_task_num=2 > nohup1.out 2>&1 &