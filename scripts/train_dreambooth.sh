# Modify pretrained_model_name_or_path, instance_data_dir, output_dir, instance_prompt, class_data_dir and class_prompt before start training
# use_given_prompts: use captions labeled by VLM (metadata.jsonl)
accelerate launch examples/dreambooth/train_dreambooth.py \
    --pretrained_model_name_or_path='/path/to/stable-diffusion-v1-5' \
    --instance_data_dir='/path/to/dataset/' \
    --output_dir='/path/to/save/outputs' \
    --instance_prompt="photo of sks <superclass>" \
    --class_data_dir="/path/to/superclass/data" \
    --class_prompt="a photo of <superclass>" \
    --with_prior_preservation --prior_loss_weight=1.0 --num_class_images=50 \
    --use_given_prompts \
    --train_batch_size=1 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --checkpointing_steps=100 \
    --validation_steps=100 \
    --seed 0 \
    --validation_scheduler="DDIMScheduler" \
    --offset_noise \
    --snr_gamma=5.0 \
    --ddloss_weight=0.2 \
	--reweight_dd_loss \
	--detach_non_subject
