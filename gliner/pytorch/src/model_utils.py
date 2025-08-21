# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


def pre_processing(model, texts, labels):
    model_input, raw_batch = model.prepare_model_inputs(texts, labels)
    input_ids = model_input["input_ids"]
    token_type_ids = model_input["token_type_ids"]
    attention_mask = model_input["attention_mask"]
    words_mask = model_input["words_mask"]
    span_idx = model_input["span_idx"]
    span_mask = model_input["span_mask"]
    text_lengths = model_input["text_lengths"]
    inputs = [
        input_ids,
        token_type_ids,
        attention_mask,
        words_mask,
        span_idx,
        span_mask,
        text_lengths,
    ]

    return inputs, raw_batch


def post_processing(model, model_output, texts, raw_batch):
    if not isinstance(model_output, torch.Tensor):
        model_output = torch.from_numpy(model_output)

    outputs = model.decoder.decode(
        raw_batch["tokens"],
        raw_batch["id_to_classes"],
        model_output,
        flat_ner=True,
        threshold=0.5,
        multi_label=False,
    )

    all_entities = []
    for i, output in enumerate(outputs):
        start_token_idx_to_text_idx = raw_batch["all_start_token_idx_to_text_idx"][i]
        end_token_idx_to_text_idx = raw_batch["all_end_token_idx_to_text_idx"][i]
        entities = []
        for start_token_idx, end_token_idx, ent_type, ent_score in output:
            start_text_idx = start_token_idx_to_text_idx[start_token_idx]
            end_text_idx = end_token_idx_to_text_idx[end_token_idx]
            entities.append(
                {
                    "start": start_token_idx_to_text_idx[start_token_idx],
                    "end": end_token_idx_to_text_idx[end_token_idx],
                    "text": texts[i][start_text_idx:end_text_idx],
                    "label": ent_type,
                    "score": ent_score,
                }
            )
        all_entities.append(entities)

    return all_entities[0]
