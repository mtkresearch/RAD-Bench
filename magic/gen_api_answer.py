"""Generate answers with GPT-4

# NOTE: code adapted from fastchat/llm_judge/gen_api_answer.py

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures
from typing import Optional, List
import sys

import openai
import shortuuid
import tqdm

from common import (
    load_questions,
    load_questions_with_idx,
    load_model_answers,
    chat_completion_openai,
    construct_turn_question,
    reorg_answer_file
)

from conversation import construct_conv_template


def get_answer(
    question: dict, 
    model: str, 
    num_choices: int, 
    max_tokens: int, 
    answer_file: str, 
    sys_mesg: str = None
):
    assert (
        args.force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if args.force_temperature is not None:
        temperature = args.force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    else:
        temperature = 0.7
    print(f".... temperature = {temperature}")
    choices = []
    for i in range(num_choices):

        turns = []
        questions = []
        answers = []
        for j in range(len(question["turns"])):
            # Apply reference to question
            usr_q = question["turns"][j]

            # set up conv
            conv = construct_conv_template(model)
            conv.set_system_message("")
            if sys_mesg is not None:
                conv.set_system_message(sys_mesg)

            for hist_q, hist_a in zip(questions, answers):
                conv.append_message(conv.roles[0], hist_q)
                conv.append_message(conv.roles[1], hist_a)
            input_prompt = construct_turn_question(question, j)

            conv.append_message(conv.roles[0], input_prompt)
            conv.append_message(conv.roles[1], None)

            # Assume the api are uniform. All follows openai's api
            output = chat_completion_openai(model, conv, temperature, max_tokens, {
                    'api_base': os.environ['api_base'],
                    'api_key': os.environ['api_key']
                })

            conv.update_last_message(output)
            turns.append(output)
            # append quetsions and answers
            questions.append(usr_q)
            answers.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="magic",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="gpt-35-turbo-16k-4k")
    parser.add_argument("--model-sys-mesg", type=str, default=None, help="System mesg for the model")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--retry", nargs="+", type=int, default=None, help="Whether to inference again for error questions"
    )
    parser.add_argument(
        "--skip-inferenced", action="store_true", help="Whetehr to skip the ones that are inferenced."
    )
    parser.add_argument("--openai-api-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    args = parser.parse_args()

    if args.openai_api_base is not None:
        openai.api_base = args.openai_api_base

    question_file = f"question.jsonl"
    if args.question_file is not None:
        print(f".... found question file. Use {args.question_file} instead of default {question_file}")
        question_file = args.question_file
    
    questions = load_questions(question_file, args.question_begin, args.question_end)
    if args.retry is not None:
        questions = load_questions_with_idx(question_file, args.retry)

    model_name = args.model.split('/')[-1]
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"model_answer/{model_name}"
        answer_file += ".jsonl"
    print(f"Output to {answer_file}")
    
    model_answers = load_model_answers("data/model_answer", model_name)
    skip_qids = set()
    if args.skip_inferenced:
        skip_qids = model_answers[model_name].keys()
        print(f"....Skipping inferenced. {model_name} already contains {len(skip_qids)} inferenced questions")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            qid = question['question_id']
            if qid in skip_qids:
                print(f".... Skipping {qid}")
                continue
            future = executor.submit(
                get_answer,
                question,
                args.model,
                args.num_choices,
                args.max_tokens,
                answer_file,
                args.model_sys_mesg
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
