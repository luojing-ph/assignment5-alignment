import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase



def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    prompt_input_ids = []
    output_input_ids = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_input_ids.append(torch.tensor(prompt_tokens))
        output_tokens = tokenizer.encode(output, add_special_tokens=False)
        output_input_ids.append(torch.tensor(output_tokens))

    seq_len = [len(p_ids) + len(o_ids) for p_ids, o_ids in zip(prompt_input_ids, output_input_ids)]
    max_length = max(seq_len)

    concatenated_input_ids = []
    concatenated_labels = []
    response_masks = []

    for (p_ids, o_ids) in zip(prompt_input_ids, output_input_ids):
        input_ids = torch.cat([p_ids, o_ids], dim=0)
        pad_length = max_length - input_ids.shape[0]
        padded_input_ids = F.pad(input_ids, (0, pad_length), value=pad_token_id)

        response_mask = torch.cat([torch.zeros_like(p_ids).bool(), torch.ones_like(o_ids).bool()], dim=0)
        padded_response_mask = F.pad(response_mask, (0, pad_length), value=False)

        concatenated_input_ids.append(padded_input_ids[:-1])
        concatenated_labels.append(padded_input_ids[1:])
        response_masks.append(padded_response_mask[1:])

    input_ids_tensor = torch.stack(concatenated_input_ids)
    label_tensor = torch.stack(concatenated_labels)
    response_mask_tensor = torch.stack(response_masks)

    return {"input_ids": input_ids_tensor, "labels": label_tensor, "response_mask": response_mask_tensor}
