"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

from common import load_questions, str_to_torch_dtype, reorg_answer_file, construct_turn_question
from conversation import construct_conv_template


def load_vllm_model(
    model_path: str,
    num_gpus: int = 1,
    max_gpu_memory: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    quantization: str = None,
    revision: str = "main",
    enforce_eager: bool = False,
    load_format: str = "auto"
):
    """Load a model from vllm."""
    
    model = LLM(model=model_path,
                trust_remote_code = True,
                tensor_parallel_size=num_gpus,
                dtype = "auto" if dtype is None else dtype,
                quantization = quantization,
                revision = revision,
                gpu_memory_utilization=0.9, # if max_gpu_memory is None else float(max_gpu_memory),
                enforce_eager=enforce_eager,
                load_format=load_format
                )
    tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    return model, tokenizer


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    conv_template,
    model_sys_mesg = None,
    force_temperature = None,
    load_format = "auto",
    repetition_penalty = None
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                conv_template=conv_template,
                model_sys_mesg=model_sys_mesg,
                force_temperature=force_temperature,
                load_format=load_format,
                repetition_penalty=repetition_penalty
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    conv_template,
    model_sys_mesg = None,
    force_temperature = None,
    load_format = "auto",
    repetition_penalty = None
):
    
    model, tokenizer = load_vllm_model(
        model_path,
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        quantization=None,
        revision=revision,
        enforce_eager=False,
        load_format=load_format
    )

    for question in tqdm(questions):
        if force_temperature is not None:
            temperature = force_temperature
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)

            turns = []
            questions = []
            answers = []
            for j in range(len(question["turns"])):
                conv = construct_conv_template(conv_template)
                conv.set_system_message("")
                if model_sys_mesg is not None:
                    # Use user defined model system message
                    conv.set_system_message(model_sys_mesg)
                
                usr_q = question["turns"][j]
                
                for hist_q, hist_a in zip(questions, answers):
                    conv.append_message(conv.roles[0], hist_q)
                    conv.append_message(conv.roles[1], hist_a)
                
                input_prompt = construct_turn_question(question, j)
                conv.append_message(conv.roles[0], input_prompt)
                conv.append_message(conv.roles[1], None)

                prompt = conv.get_prompt()
                print(f"{question}; \n=====\nprompt = {prompt}\n====\n")

                # some models may error out when generating long outputs
                try:
                    sampling_params = SamplingParams(temperature=temperature, 
                                                     max_tokens=max_new_token)
                    if repetition_penalty is not None:
                        sampling_params.repetition_penalty = repetition_penalty
                    outputs = model.generate([prompt], sampling_params, use_tqdm=False)
                    output_ids = outputs[0].outputs[0].token_ids

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and isinstance(conv.stop_str, list):
                        stop_str_indices = sorted(
                            [
                                output.find(stop_str)
                                for stop_str in conv.stop_str
                                if output.find(stop_str) > 0
                            ]
                        )
                        if len(stop_str_indices) > 0:
                            output = output[: stop_str_indices[0]]
                    elif conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)
                # append quetsions and answers
                questions.append(usr_q)
                answers.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench_tw",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--question-file", type=str, default=None, help="The path to question jsonl file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--model-sys-mesg",
        type=str,
        default=None,
        help="The overwriting system prompt to the conv template",
    )
    parser.add_argument(
        "--force-temperature",
        type=float,
        default=None,
        help="Temperature for generation. Use temperature config by default",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition Penalty setting in vllm",
    )
    parser.add_argument(
        "--conv-template",
        type=str,
        default=None,
        help="Conversation template for chat completion. Use One-shot template by default.",
    )
    parser.add_argument(
        "--load-format",
        type=str,
        default="auto",
        help="vllm load_format {auto, pt, safetensors}",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init(address="auto", _temp_dir="/tmp")

    question_file = f"data/questions.jsonl"
    if args.question_file:
        question_file = args.question_file

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        conv_template=args.conv_template,
        model_sys_mesg=args.model_sys_mesg,
        force_temperature=args.force_temperature,
        load_format=args.load_format,
        repetition_penalty=args.repetition_penalty
    )

    reorg_answer_file(answer_file)
