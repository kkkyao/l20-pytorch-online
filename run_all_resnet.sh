#!/usr/bin/env bash
# set -e   

# ===============================
# ResNet18 — Baseline 
# ===============================
for opt in sgd adam rmsprop; do
  for seed in 0 1 2 3 4; do
    python cifar10_resnet18_baseline.py \
      --opt ${opt} \
      --seed ${seed} \
      --data_seed 42 \
      --epochs 50 \
      --bs 128 \
      --preprocess_dir artifacts/cifar10_conv2d_preprocess \
      --wandb \
      --wandb_group cifar10_resnet18_baseline_${opt} \
      --wandb_run_name cifar10_resnet18_baseline_${opt}_seed${seed}
  done
done

# ===============================
# ResNet18 — F3 
# ===============================
for seed in 0 1 2 3 4; do
  python cifar10_resnet18_train_f3.py \
    --seed ${seed} \
    --data_seed 42 \
    --epochs 50 \
    --bs 128 \
    --preprocess_dir artifacts/cifar10_conv2d_preprocess \
    --c_base 0.5 \
    --eta_max 0.2 \
    --theta_lr 3e-4 \
    --beta 0.9 \
    --val_meta_batches 2 \
    --eta_change_ratio 0.08 \
    --wd 5e-4 \
    --wandb \
    --wandb_project "l2o-online(new1)" \
    --wandb_group cifar10_resnet18_f3 \
    --wandb_run_name cifar10_resnet18_f3_seed${seed}
done

# ===============================
# ResNet18 — F3 + alpha
# ===============================
for seed in 0 1 2 3 4; do
  python cifar10_resnet18_train_f3_alpha.py \
    --seed ${seed} \
    --data_seed 42 \
    --epochs 50 \
    --bs 128 \
    --preprocess_dir artifacts/cifar10_conv2d_preprocess \
    --c_base 0.5 \
    --eta_max 0.2 \
    --theta_lr 3e-4 \
    --beta 0.9 \
    --alpha_mix 1.0 \
    --val_meta_batches 2 \
    --eta_change_ratio 0.08 \
    --wd 5e-4 \
    --wandb \
    --wandb_project "l2o-online(new1)" \
    --wandb_group cifar10_resnet18_f3_alpha_a1 \
    --wandb_run_name cifar10_resnet18_f3_alpha_a1_seed${seed}
done

# ===============================
# ResNet18 — F1 
# ===============================
for seed in 0 1 2 3 4; do
  python cifar10_resnet18_train_f1.py \
    --seed ${seed} \
    --data_seed 42 \
    --epochs 50 \
    --bs 128 \
    --preprocess_dir artifacts/cifar10_conv2d_preprocess \
    --c_base 0.5 \
    --eta_max 0.2 \
    --theta_lr 3e-4 \
    --beta 0.9 \
    --val_meta_batches 2 \
    --eta_change_ratio 0.08 \
    --wd 5e-4 \
    --wandb \
    --wandb_project "l2o-online(new1)" \
    --wandb_group cifar10_resnet18_f1 \
    --wandb_run_name cifar10_resnet18_f1_seed${seed}
done
