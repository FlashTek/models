#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Roland Zimmermann & Julien Siems
#
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012 using MobileNet-v2.
# Users could also modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   ps ./local_test_mobilenetv2.ps
#
#

# Exit immediately if a command exits with a non-zero status.
$ErrorActionPreference = "Stop"

function mkdirs($path)
{
    If(!(test-path $path))
    {
          mkdir -p $path
    }
}

# Update PYTHONPATH.
$env:PYTHONPATH = "$env:PYTHONPATH;$pwd;$pwd/slim"

# Set up the working environment.
$CURRENT_DIR = $pwd
$WORK_DIR = "$CURRENT_DIR/deeplab"

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
$DATASET_DIR="datasets"
cd "$WORK_DIR/$DATASET_DIR"
# ps download_and_convert_voc2012.ps1

cd $CURRENT_DIR

# Set up the working directories.
$PASCAL_FOLDER="pascal_voc_seg"
$EXP_FOLDER="exp/train_on_trainval_set_mobilenetv2_no_edge_loss"
$INIT_FOLDER="$WORK_DIR/$DATASET_DIR/$PASCAL_FOLDER/init_models"
$TRAIN_LOGDIR="$WORK_DIR/$DATASET_DIR/$PASCAL_FOLDER/$EXP_FOLDER/train"
$EVAL_LOGDIR="$WORK_DIR/$DATASET_DIR/$PASCAL_FOLDER/$EXP_FOLDER/eval"
$VIS_LOGDIR="$WORK_DIR/$DATASET_DIR/$PASCAL_FOLDER/$EXP_FOLDER/vis"
$EXPORT_DIR="$WORK_DIR/$DATASET_DIR/$PASCAL_FOLDER/$EXP_FOLDER/export"
mkdirs $INIT_FOLDER
mkdirs $TRAIN_LOGDIR
mkdirs $EVAL_LOGDIR
mkdirs $VIS_LOGDIR
mkdirs $EXPORT_DIR

# Copy locally the trained checkpoint as the initial checkpoint.
#$TF_INIT_ROOT="http://download.tensorflow.org/models"
$TF_INIT_ROOT="https://storage.googleapis.com/mobilenet_v2/checkpoints"
$CKPT_NAME="mobilenet_v2_1.0_224"
$TF_INIT_CKPT=$CKPT_NAME + ".tgz"
cd "$INIT_FOLDER"
# wget "$TF_INIT_ROOT/$TF_INIT_CKPT" -OutFile $TF_INIT_CKPT
tar -xf "$TF_INIT_CKPT"
cd "$CURRENT_DIR"

$PASCAL_DATASET="$WORK_DIR/$DATASET_DIR/$PASCAL_FOLDER/tfrecord"

# Train 10 iterations.
$NUM_ITERATIONS=30000
python "$WORK_DIR/train.py" `
  --logtostderr `
  --train_split="trainval" `
  --model_variant="mobilenet_v2" `
  --output_stride=16 `
  --train_crop_size=513 `
  --train_crop_size=513 `
  --train_batch_size=4 `
  --training_number_of_steps="$NUM_ITERATIONS" `
  --train_logdir="$TRAIN_LOGDIR" `
  --dataset_dir="$PASCAL_DATASET" `
  --tf_initial_checkpoint="$INIT_FOLDER/$CKPT_NAME/mobilenet_v2_1.0_224.ckpt" `
  --fine_tune_batch_norm=true

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=75.34%.
python "$WORK_DIR/eval.py" `
  --logtostderr `
  --eval_split="val" `
  --model_variant="mobilenet_v2" `
  --eval_crop_size=513 `
  --eval_crop_size=513 `
  --checkpoint_dir="$TRAIN_LOGDIR" `
  --eval_logdir="$EVAL_LOGDIR" `
  --dataset_dir="$PASCAL_DATASET" `
  --max_number_of_evaluations=1

# Visualize the results.
python "$WORK_DIR/vis.py" `
  --logtostderr `
  --vis_split="val" `
  --model_variant="mobilenet_v2" `
  --vis_crop_size=513 `
  --vis_crop_size=513 `
  --checkpoint_dir="$TRAIN_LOGDIR" `
  --vis_logdir="$VIS_LOGDIR" `
  --dataset_dir="$PASCAL_DATASET" `
  --max_number_of_iterations=1

# Export the trained checkpoint.
$CKPT_PATH="$TRAIN_LOGDIR/model.ckpt-$NUM_ITERATIONS"
$EXPORT_PATH="$EXPORT_DIR/frozen_inference_graph.pb"

python "$WORK_DIR/export_model.py" `
  --logtostderr `
  --checkpoint_path="$CKPT_PATH" `
  --export_path="$EXPORT_PATH" `
  --model_variant="mobilenet_v2" `
  --num_classes=21 `
  --crop_size=513 `
  --crop_size=513 `
  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
