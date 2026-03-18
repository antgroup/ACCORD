# Modify pretrained_model_name_or_path, instance_data_dir, output_dir, instance_prompt, modifier_token and vision_encoder_path before start training
# modifier_token: trainable subject embedding
# use_given_prompts: use captions labeled by VLM (metadata.jsonl)
# vision_encoder_path: necessary for pdloss
accelerate launch examples/custom_diffusion/train_custom_diffusion.py \
    --pretrained_model_name_or_path='/path/to/stable-diffusion-v1-5' \
    --instance_data_dir='/path/to/dataset/' \
    --output_dir='/path/to/save/outputs' \
    --instance_prompt="photo of <new1> <superclass>" \
    --modifier_token="<new1>" \
    --use_given_prompts \
    --train_batch_size=8 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --checkpointing_steps=100 \
    --validation_steps=100 \
    --seed 0 \
    --validation_scheduler="DDIMScheduler" \
    --hflip \
    --num_class_images=50 \
    --ddloss_weight=0.1 \
	--reweight_dd_loss \
	--detach_non_subject \
    --vision_encoder_path="/path/to/clip-vit-large-patch14" \
	--pdloss_weight=0.003 \
	--desim_refer_father
