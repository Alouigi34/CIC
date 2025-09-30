---
library_name: peft
license: cc-by-nc-4.0
base_model: facebook/musicgen-melody
tags:
- base_model:adapter:facebook/musicgen-melody
- lora
- transformers
datasets:
- audiofolder
model-index:
- name: musicgen-melody-lora-punk
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# musicgen-melody-lora-punk

This model is a fine-tuned version of [facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody) on the /HOME/ALELS_STAR/DESKTOP/AI_COMPOSITION_ASSISTANT/V0.01/MUSICGEN_WAV_GENERATOR/RETRAIN_MODELS/MUSICGEN-DREAMBOOTHING/PUNK_DATASET - NA dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 1
- seed: 456
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.99) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 20.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.16.0
- Transformers 4.54.0.dev0
- Pytorch 2.7.1+cu126
- Datasets 4.0.0
- Tokenizers 0.21.2