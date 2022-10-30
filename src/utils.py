import torch.nn as nn
# from dataloader.default_split import DEFAULT_SPLIT
import json
import os
import torch

def label_smoothed_nll_loss(lprobs, target, epsilon=0.1, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for n, par in model.named_parameters():
        if 'prompt' in n:
            continue
        par.requires_grad = False

def freeze_prompt_blend_embeds(args, model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    freeze_params(model.model)
    # freeze_params(model.all_prompt_parameters)
    if not args.do_tune_bert:
        freeze_params(model.blender.word_encoder)

def freeze_blend_intrinsic_embeds(args, model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    freeze_params(model.model)
    # freeze_params(model.all_prompt_parameters)
    if not args.do_tune_bert:
        freeze_params(model.blender.word_encoder)
    for n, par in model.named_parameters():
        if 'prompt_W' in n and 'mapping' not in n:
            par.requires_grad = False
        else:
            continue

def freeze_intrinsic_mlp_embeds(args, model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    freeze_params(model.model)
    # freeze_params(model.all_prompt_parameters)
    if not args.do_tune_bert:
        freeze_params(model.blender)
    for n, par in model.named_parameters():
        if 'prompt_W' in n and 'mapping' not in n:
            par.requires_grad = False
        else:
            continue

def freeze_bert_of_blend(model):
    freeze_params(model.word_encoder)

# def freeze_embeds(model):
#     """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
#     model_type = model.config.model_type

#     if model_type == "t5":
#         freeze_params(model.shared)
#         for d in [model.encoder, model.decoder]:
#             freeze_params(d.embed_tokens)
#     elif model_type == "fsmt":
#         for d in [model.model.encoder, model.model.decoder]:
#             freeze_params(d.embed_positions)
#             freeze_params(d.embed_tokens)
#     else:
#         freeze_params(model.model.shared)
#         for d in [model.model.encoder, model.model.decoder]:
#             freeze_params(d.embed_positions)
#             freeze_params(d.embed_tokens)

def freeze_embeds(model, AE_recover=False, AE_recover_stage_two=False):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    if AE_recover:
        for n, p in model.named_parameters():
            if not AE_recover_stage_two:
                if 'prompt_task' not in n:
                    p.requires_grad = False
            else:
                if 'prompt_embeddings' not in n:
                    p.requires_grad = False
    else:
        freeze_params(model.model)

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

def get_tasks_list(filename, split_name):
    with open(filename, "r") as fin:
        split_dict = json.load(fin)
    return sorted(split_dict[split_name])

def load_prompt_parameters(args, prompt_path, train_tasks):
    train_tasks_split = sorted(train_tasks.split(" "))
    prompt_weight = torch.tensor([])
    for task in train_tasks_split: 
        task_dir = os.path.join(args.data_dir, task)
        files = sorted(os.listdir(task_dir))
        prefixes = []
        for filename in files:
            if not filename.endswith(".tsv"):
                continue
            prefix = "_".join(filename.split("_")[:-1])
            if prefix not in prefixes:
                prefixes.append(prefix)

        # 固定seed100
        prefix = prefixes[0]
        task_prompt_weight_path = os.path.join(prompt_path, "singletask-"+task, "prompt_weight")
        if prefix+"_best.pt" in os.listdir(task_prompt_weight_path):
            task_prompt_weight = torch.load(os.path.join(task_prompt_weight_path, prefix+"_best.pt"))
        elif prefix+"_lr_1e-05_bsz_2.pt" in os.listdir(task_prompt_weight_path):
            task_prompt_weight = torch.load(os.path.join(task_prompt_weight_path, prefix+"_lr_1e-05_bsz_2.pt"))
        else:
            task_prompt_weight = torch.rand(100, 768)
        prompt_weight = torch.cat((prompt_weight, task_prompt_weight.unsqueeze(0)),0)
    
    return prompt_weight

def load_intrinsic_parameters(args, intrinsic_path, train_tasks):
    train_tasks_split = sorted(train_tasks.split(" "))
    intrinsic_weight = torch.tensor([])
    for task in train_tasks_split: 
        task_dir = os.path.join(args.data_dir, task)
        files = sorted(os.listdir(task_dir))
        prefixes = []
        for filename in files:
            if not filename.endswith(".tsv"):
                continue
            prefix = "_".join(filename.split("_")[:-1])
            if prefix not in prefixes:
                prefixes.append(prefix)

        # 固定seed100
        prefix = prefixes[0]
        task_intrinsic_weight_path = os.path.join(intrinsic_path, task, '100/1e-05_4/best-ckpt.pt')
        if os.path.exists(task_intrinsic_weight_path):
            task_intrinsic_weight = torch.load(task_intrinsic_weight_path)['model']['prompt_task.weight']
        else:
            print('Intrinsic weight of {} doesnot exist', task)
        intrinsic_weight = torch.cat((intrinsic_weight, task_intrinsic_weight.unsqueeze(0)),0) # [100,1,3]
    
    return intrinsic_weight


def get_prefixes(task_dir, task):
    task_dir = os.path.join(task_dir, task)
    files = sorted(os.listdir(task_dir))
    prefixes = []
    for filename in files:
        if not filename.endswith(".tsv"):
            continue
        prefix = "_".join(filename.split("_")[:-1])
        if prefix not in prefixes:
            prefixes.append(prefix)

    return prefixes