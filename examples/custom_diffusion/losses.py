import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity


def exclude_subject_tokens(input_ids, first_place_token_id, num_vectors):
    start_mask = (input_ids == first_place_token_id)
    general_input_ids = []
    for i in range(len(input_ids)):
        start_pos = torch.nonzero(start_mask[i])[0].item()
        general_input_ids.append(torch.cat([input_ids[i, :start_pos], input_ids[i, start_pos+num_vectors:]], dim=0))

    general_input_ids = torch.stack(general_input_ids, dim=0)
    return general_input_ids


class DDLoss(nn.Module):
    def __init__(self, args, placeholder_token_ids, noise_scheduler, empty_token, subject_token):
        super().__init__()
        self.placeholder_token_ids = placeholder_token_ids
        self.noise_scheduler = noise_scheduler
        self.empty_token = empty_token
        self.subject_token = subject_token
        self.empty_subject_tokens = torch.cat([empty_token, subject_token], dim=0)
        self.detach_non_subject = args.detach_non_subject
        self.reweight_dd_loss = args.reweight_dd_loss

    def forward(self, text_encoder, weight_dtype, unet, xt, input_ids, timesteps, allc_pred):
        # Get the denoise result of general prompt
        general_input_ids = exclude_subject_tokens(input_ids, self.placeholder_token_ids[0], len(self.placeholder_token_ids))
        general_encoder_hidden_states = text_encoder(general_input_ids)[0].to(dtype=weight_dtype)
        general_model_pred = unet(xt, timesteps, general_encoder_hidden_states, return_dict=False)[0]
        # Get the denoise result of subject prompt and empty prompt
        encoder_hidden_states = text_encoder(self.empty_subject_tokens)[0].to(dtype=weight_dtype)
        empty_hidden_states = encoder_hidden_states[:1].repeat(xt.size(0), 1, 1)
        subject_hidden_states = encoder_hidden_states[1:].repeat(xt.size(0), 1, 1)
        empty_model_pred = unet(xt, timesteps, empty_hidden_states, return_dict=False)[0]
        subject_model_pred = unet(xt, timesteps, subject_hidden_states, return_dict=False)[0]
        # Calculate L_correlation
        if self.detach_non_subject:
            general_model_pred = general_model_pred.detach()
            empty_model_pred = empty_model_pred.detach()
        full_to_general_dist = F.mse_loss(allc_pred, general_model_pred, reduction='none').mean([1, 2, 3])
        full_to_subject_dist = F.mse_loss(allc_pred, subject_model_pred, reduction='none').mean([1, 2, 3])
        full_to_empty_dist = F.mse_loss(allc_pred, empty_model_pred, reduction='none').mean([1, 2, 3])
        L_dd = torch.abs(full_to_general_dist + full_to_subject_dist - full_to_empty_dist)

        if self.reweight_dd_loss:
            reweight_coeffs = timesteps / self.noise_scheduler.num_train_timesteps
            L_dd = L_dd * reweight_coeffs

        L_dd = L_dd.mean()
        return L_dd, {"general_model_pred": general_model_pred, "empty_model_pred": empty_model_pred, "subject_model_pred": subject_model_pred}


class PDLoss(nn.Module):
    def __init__(self, subject_token_id, clip_image_projector, clip_text_projector, father_cls_embeds=None):
        super().__init__()
        self.image_projector = clip_image_projector
        self.text_projector = clip_text_projector
        self.subject_token_id = subject_token_id
        self.father_cls_embeds = father_cls_embeds

    def forward(self, input_ids, text_encoder, visual_embeds):
        general_input_ids = exclude_subject_tokens(input_ids, self.subject_token_id[0], len(self.subject_token_id))
        general_pooled_output = text_encoder(general_input_ids)[1]
        text_cls_embeds = self.text_projector(general_pooled_output) # (B, D)
        B, N, D = visual_embeds.size()
        visual_embeds = visual_embeds.view(-1, D)
        image_cls_embeds = self.text_projector(visual_embeds)
        image_cls_embeds = image_cls_embeds.view(B, N, D)
        # Calculate abs cosine similarity
        similarities = cosine_similarity(image_cls_embeds, text_cls_embeds.unsqueeze(1), dim=-1) # (B, N)
        if self.father_cls_embeds is not None:
            refer_similarities = cosine_similarity(text_cls_embeds, self.father_cls_embeds, dim=-1).unsqueeze(1)
            similarity = torch.abs(similarities - refer_similarities).mean()
        else:
            similarity = torch.abs(similarities).mean()

        return similarity

