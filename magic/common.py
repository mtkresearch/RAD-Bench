"""
Common data structures and utilities.
"""

import ast
import dataclasses
import glob
import json
import os
import re
import time
from typing import Optional
import openai
import anthropic

from conversation import construct_conv_template

# Judge mode keys
JUDGE_MODE_KEYS = ["KI-1-turn", "KI-2-turn", "KI-3-turn",
                    "KR-1-turn", "KR-2-turn", "KR-3-turn",
                    "TR-1-turn", "TR-2-turn", "TR-3-turn"]

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 500
API_ERROR_OUTPUT = "$ERROR$"

TIE_DELTA = 0.1

# Categories that need reference answers
NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]

# Extract scores from judgments
two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")


reverse_model_map = {
    "model_1": "model_2",
    "model_2": "model_1",
}


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: dict
    ref_based: bool = True
    multi_turn: bool = True
    ref_ans_model: str = None


@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    judge: Judge
    turn: int # turn 1, 2, or 3
    type: str # KI, KR, TR
    ref_answer: dict = None
    # multi_turn: bool = False


def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r", encoding="utf-8") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

def load_questions_with_idx(question_file: str, ids: list):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r", encoding="utf-8") as ques_file:
        for idx, line in enumerate(ques_file):
            if idx in ids:
                if line:
                    questions.append(json.loads(line))
    return questions

def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename, encoding="utf-8") as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def load_judge_prompts(prompt_file: str):
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    prompts = {}
    with open(prompt_file, encoding="utf-8") as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def run_judge_single(question, answer, judge, ref_answer, turn, type):
    """
    type should be KI, KR, or TR
    """
    kwargs = {}
    model = judge.model_name

    if type == "TR":
        kwargs["context"] = question["contexts"][turn-1]
        if turn > 1:
            kwargs["question_1"] = question["turns"][0]
            if turn == 3:
                kwargs["question_2"] = question["turns"][1]
                kwargs["reference"] = ref_answer["choices"][0]["turns"][2]
    else:
        kwargs["reference"] = ref_answer["choices"][0]["turns"][turn-1]
        if type == "KR":
            # context is needed
            context = ""
            if isinstance(question["contexts"][turn-1], list):
                context = "\n".join(question["contexts"][turn-1])
            elif isinstance(question["contexts"][turn-1], str):
                context = question["contexts"][turn-1]
            kwargs["context"] = context
        if turn > 1: # 2nd or 3rd turn
            kwargs["question_1"] = question["turns"][0]
            kwargs["reference_1"] = ref_answer["choices"][0]["turns"][0]
            if turn == 3:
                kwargs["question_2"] = question["turns"][1]
                kwargs["reference_2"] = ref_answer["choices"][0]["turns"][1]
    user_prompt = judge.prompt_template["prompt_template"].format(
        question=question["turns"][turn-1], # 1-index to 0-index
        answer=answer["choices"][0]["turns"][turn-1],
        **kwargs
    )

    rating = -1

    system_prompt = judge.prompt_template["system_prompt"]
    conv = construct_conv_template(model)
    conv.set_system_message(system_prompt)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
        judgment = chat_completion_openai(model, conv, temperature=0, max_tokens=2048, 
                                          api_dict={'api_base': os.environ['api_base'],
                                                    'api_key': os.environ['api_key']})
    else:
        raise ValueError(f"Invalid judge model name: {model}")

    if judge.prompt_template["output_format"] == "[[rating]]":
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return rating, user_prompt, judgment


def play_a_match_single(match: MatchSingle, output_file: str):
    question, model, answer, judge, ref_answer, turn, type = (
        match.question,
        match.model,
        match.answer,
        match.judge,
        match.ref_answer,
        match.turn,
        match.type
    )

    if judge.prompt_template["type"] == "single":
        score, user_prompt, judgment = run_judge_single(
            question, answer, judge, ref_answer, turn=turn, type=type
        )

        question_id = question["question_id"]
        result = {
            "question_id": question_id,
            "turn": turn,
            "type": type,
            "score": score,
            "model": model,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            # "user_prompt": user_prompt,
            "judgment": judgment,
            "tstamp": time.time(),
        }
        print(
            f"question: {question_id}, turn: {turn}, model: {model}, "
            f"score: {score}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


def chat_completion_openai(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            # print(messages)
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict=None):
    openai.api_type = "azure"
    openai.api_version = "2023-07-01-preview"
    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
    else:
        openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
        openai.api_key = os.environ["AZURE_OPENAI_KEY"]

    if "azure-" in model:
        model = model[6:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                engine=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.error.InvalidRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(response)
            break

    return output


def chat_completion_anthropic(model, conv, temperature, max_tokens):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            prompt = conv.get_prompt()
            response = c.completions.create(
                model=model,
                prompt=prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
            )
            output = response.completion
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output.strip()


def load_single_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    for line in open(filename, encoding="utf-8"):
        obj = json.loads(line)
        judge = tuple(obj["judge"])
        qid, model = obj["question_id"], obj["model"]

        if judge not in judge_dict:
            judge_dict[judge] = {}

        gamekey = (qid, model)

        judge_dict[judge][gamekey] = {
            "score": obj["score"],
            "judgment": obj["judgment"],
        }
    return judge_dict


def resolve_single_judgment_dict(
    question, model_judgments_normal, model_judgments_math, multi_turn=False
):
    """Return the correct single answer grading judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[("gpt-4", "single-math-v1-multi-turn")]
        return model_judgments_normal[("gpt-4", "single-v1-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[("gpt-4", "single-math-v1")]
    else:
        return model_judgments_normal[("gpt-4", "single-v1")]


def get_single_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        qid, model = gamekey

        res = judgment_dict[gamekey]

        g1_judgment = res["judgment"]
        g1_score = res["score"]

        return (
            f"**Game 1**. **A**: {model}, **Score**: {g1_score}\n\n"
            f"**Judgment**: {g1_judgment}"
        )
    except KeyError:
        return "N/A"


def check_data(questions, model_answers, ref_answers, models, judges, bench_name = "mt_bench_tc"):
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert (
                q["question_id"] in m_answer
            ), f"Missing model {m}'s answer to Question {q['question_id']}"
    # check ref answers
    for jg in judges.values():
        if not jg.ref_based:
            continue
        for q in questions:
            if bench_name == "mt_bench_tc":
                if "reference" not in q:
                    continue
                assert (
                    q["question_id"] in ref_answers[jg.ref_ans_model]
                ), f"Missing reference answer to Question {q['question_id']} for judge {jg.ref_ans_model}"
            else:
                if q["category"] not in NEED_REF_CATS:
                    continue
                assert (
                    q["question_id"] in ref_answers[jg.model_name]
                ), f"Missing reference answer to Question {q['question_id']} for judge {jg.model_name}"
        


def get_model_list(answer_dir):
    file_paths = glob.glob(f"{answer_dir}/*.jsonl")
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    return file_names

# Adapted from fastchat.llm_judge.gen_model_answer.reorg_answer_file
def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as fin:
        for l in fin:
            data = json.loads(l)
            qid = data["question_id"]
            answers[qid] = data

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w", encoding="utf-8") as fout:
        for qid in qids:
            fout.write(json.dumps(answers[qid], ensure_ascii=False) + "\n")


def str_to_torch_dtype(dtype: str):
    import torch

    if dtype is None:
        return None
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unrecognized dtype: {dtype}")


def construct_ctx_doc(contexts):
    if isinstance(contexts, str):
        return contexts
    
    ctx_toids = []
    for ctx in contexts:
        ctx_toid = f"<context>\n{ctx}\n</context>"
        ctx_toids.append(ctx_toid)
    ctx_doc = '\n'.join(ctx_toids)
    return ctx_doc


def construct_turn_question(question, turn_id):
    prefix_1st_turn = "Use the following pieces of context (provided within <context></context> XML tags) to answer the question at the end. Please strictly follow the given context to answer the question. Do not make up your answer."
    prefix_nth_turn = "Use the following pieces of context (provided within <context></context> XML tags) to answer the question at the end. You are also provided with conversation history. Please strictly follow the given context and follow instructions in the conversation history to answer the question at the end. Do not make up your answer."
    
    prefix = ""
    if turn_id == 0:
        prefix = prefix_1st_turn
    else:
        prefix = prefix_nth_turn

    ctx_doc = construct_ctx_doc(question['contexts'][turn_id])
    return f"{prefix}\n\n{ctx_doc}\n\nQuestion: {question['turns'][turn_id]}"
