"""
Usage:
python3 qa_browser.py --share
"""
import os

import argparse
from collections import defaultdict
import re

import gradio as gr

from common import (
    load_questions,
    load_model_answers,
    load_single_model_judgments,
    get_single_judge_explanation,
    construct_ctx_doc,
    JUDGE_MODE_KEYS
)

_CUR_DIR=os.path.dirname(os.path.abspath(__file__))

questions = []
model_answers = {}

answer_dir = ""
question_file = ""

judge_model_name = "gpt-4o"

model_judgments_normal_single = {}

question_selector_map = {}
category_selector_map = defaultdict(list)

single_model_judgment_file = ""


def display_question(category_selector, request: gr.Request):
    print(f"in display_question: {category_selector}")
    choices = category_selector_map[category_selector]
    # return gr.Dropdown.update(value=choices[0],choices=choices,)
    return gr.Dropdown(value=choices[0],choices=choices,)


def display_pairwise_answer(
    question_selector, model_selector1, model_selector2, request: gr.Request
):
    q = question_selector_map[question_selector]
    qid = q["question_id"]

    ans1 = model_answers[model_selector1][qid]
    ans2 = model_answers[model_selector2][qid]

    chat_mds = pairwise_to_gradio_chat_mds(q, ans1, ans2)
    print(f".... pairwise blocks text = {len(chat_mds)}")
    gamekey = (qid, model_selector1, model_selector2)

    for turn_id in ["1", "2", "3"]:    
        judgment_dict = {}
        for jk in [f"CS-{turn_id}-turn", f"CR-{turn_id}-turn", f"TR-{turn_id}-turn"]:
            judgment_dict.update(model_judgments_normal_single.get((judge_model_name, jk), {}) )
        title = f"##### Model Judgment ({turn_id} turn)\n"
        jug_a_exp = get_single_judge_explanation((qid, model_selector1), judgment_dict)
        jug_b_exp = get_single_judge_explanation((qid, model_selector2), judgment_dict)
        chat_mds.extend(pairwise_jug_to_gradio_chat_mds(title, jug_a_exp, jug_b_exp))

    return chat_mds


def display_single_answer(question_selector, model_selector1, request: gr.Request):
    q = question_selector_map[question_selector]
    qid = q["question_id"]

    ans1 = model_answers[model_selector1][qid]

    chat_mds = single_to_gradio_chat_mds(q, ans1)
    
    print(f".... single blocks text = {len(chat_mds)}")
    gamekey = (qid, model_selector1)

    # BRUTE FORCE merging judgment
    for turn_id in ["1", "2", "3"]:    
        judgment_dict = {}
        for jk in [f"CS-{turn_id}-turn", f"CR-{turn_id}-turn", f"TR-{turn_id}-turn"]:
            judgment_dict.update(model_judgments_normal_single.get((judge_model_name, jk), {}) )
    
        explanation = f"##### Model Judgment ({turn_id} turn)\n\n" + get_single_judge_explanation(
            gamekey, judgment_dict
        )
        chat_mds.append(explanation)

    return chat_mds

newline_pattern1 = re.compile("\n\n(\d+\. )")
newline_pattern2 = re.compile("\n\n(- )")


def post_process_answer(x):
    """Fix Markdown rendering problems."""
    x = x.replace("\u2022", "- ")
    x = re.sub(newline_pattern1, "\n\g<1>", x)
    x = re.sub(newline_pattern2, "\n\g<1>", x)
    return x


def pairwise_to_gradio_chat_mds(question, ans_a, ans_b):
    end = len(question["turns"])

    # user Q1, assistant A1, user Q2, assistant A2, user Q3, assistant A3, ref1, ref2, ref3, judge1, judge2, judge3
    mds = [""] * (4 * end + end)
    for i in range(end):
        base = i * 4
        mds[base + 0] = f"##### ({i})User\n" + question["turns"][i]
        mds[base + 1] = f"##### ({i}) Context\n" + construct_ctx_doc(question["contexts"][i])
        mds[base + 2] = f"##### ({i}) Assistant A\n" + post_process_answer(
            ans_a["choices"][0]["turns"][i].strip()
        )
        mds[base + 3] = f"##### ({i}) Assistant B\n" + post_process_answer(
            ans_b["choices"][0]["turns"][i].strip()
        )

    refs = question.get("references", [""] * end)

    mds[-end] = ""
    for i in range(end):
        ref = refs[i]
        ref_str = ref
        if isinstance(ref, list):
            ref_str = "\n\n".join(ref)
        elif isinstance(ref, str):
            ref_str = ref
        else:
            raise NotImplemented(f"Reference type not recognized. ref = {ref}")
        prefix = f"##### Reference Solution\n"
        mds[-end + i] = prefix + f"Q-{i} ans:\n{ref_str}\n"
    return mds


def pairwise_jug_to_gradio_chat_mds(title, jug_a, jug_b):
    mds = [title, "", ""]
    base = 0
    mds[base + 1] = "##### Assistant A\n" + post_process_answer(jug_a.strip()
    )
    mds[base + 2] = "##### Assistant B\n" + post_process_answer(jug_b.strip())
    return mds


def single_to_gradio_chat_mds(question, ans):
    end = len(question["turns"])
    
    # user Q1, ctx1, assistant A1, user Q2, ctx2, assistant A2, reference solution
    # mds = ["", "", "", "", ""]
    mds = [""] * (end * 3 + end)
    # User questions
    for i in range(end):
        base = i * 3
        mds[base + 0] = f"##### ({i}) User\n" + question["turns"][i]
        mds[base + 1] = f"##### ({i}) Context\n" + construct_ctx_doc(question["contexts"][i])
        mds[base + 2] = f"##### ({i}) Assistant\n" + post_process_answer(
            ans["choices"][0]["turns"][i].strip()
        )

    refs = question.get("references", [""] * end)
    mds[-end] = ""
    for i in range(end):
        ref = refs[i]
        ref_str = ref
        if isinstance(ref, list):
            ref_str = "\n\n".join(ref)
        elif isinstance(ref, str):
            ref_str = ref
        else:
            raise NotImplemented(f"Reference type not recognized. ref = {ref}")
        prefix = f"##### Reference Solution\n"
        mds[-end + i] = prefix + f"Q-{i} ans:\n\n{ref_str}\n"
    return mds


def build_question_selector_map():
    global question_selector_map, category_selector_map

    # Build question selector map
    for q in questions:
        preview = f"{q['question_id']}: " + q["turns"][0][:128] + "..."
        question_selector_map[preview] = q
        category_selector_map[q["category"]].append(preview)


def build_pairwise_browser_tab():
    global question_selector_map, category_selector_map

    models = list(model_answers.keys())
    num_sides = 2
    num_turns = 3
    side_names = ["A", "B"]

    question_selector_choices = list(question_selector_map.keys())
    category_selector_choices = list(category_selector_map.keys())

    # Selectors
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            category_selector = gr.Dropdown(
                choices=category_selector_choices, label="Category", container=False
            )
        with gr.Column(scale=100):
            question_selector = gr.Dropdown(
                choices=question_selector_choices, label="Question", container=False
            )

    model_selectors = [None] * num_sides
    with gr.Row():
        for i in range(num_sides):
            with gr.Column():
                if i == 0:
                    value = models[0]
                else:
                    value = models[-1]
                model_selectors[i] = gr.Dropdown(
                    choices=models,
                    value=value,
                    label=f"Model {side_names[i]}",
                    container=False,
                )

    # Conversation
    chat_mds = []
    for i in range(num_turns):
        chat_mds.append(gr.Markdown(elem_id=f"turn_{i}_user_question"))
        # Context
        with gr.Row():
            with gr.Column(scale=100):
                chat_mds.append(gr.Textbox(lines=20))
        with gr.Row():
            for j in range(num_sides):
                with gr.Column(scale=100):
                    chat_mds.append(gr.Markdown())

                if j == 0:
                    with gr.Column(scale=1, min_width=8):
                        gr.Markdown()
    print(f".... pair tab len(chat_mds) = {len(chat_mds)}")
    # reference = gr.Markdown(elem_id=f"reference")
    # chat_mds.append(reference)
    references = [gr.Textbox(lines=20, elem_id=f"turn_{t}_reference") for t in range(num_turns)]
    chat_mds.extend(references)
    print(f".... pair tab len(chat_mds) = {len(chat_mds)}")
    # Model explanations
    for i in range(num_turns):
        chat_mds.append(gr.Markdown(elem_id=f"turn_{i}_model_judgment"))
        with gr.Row():
            for j in range(num_sides):
                with gr.Column(scale=100):
                    chat_mds.append(gr.Textbox(lines=20, elem_id=f"turn_{i}_model_{j}_explanation"))

                if j == 0:
                    with gr.Column(scale=1, min_width=8):
                        gr.Markdown()

    print(f".... pair tab len(chat_mds) = {len(chat_mds)}")
    # Callbacks
    category_selector.change(display_question, [category_selector], [question_selector])
    question_selector.change(
        display_pairwise_answer,
        [question_selector] + model_selectors,
        chat_mds
    )

    for i in range(num_sides):
        model_selectors[i].change(
            display_pairwise_answer,
            [question_selector] + model_selectors,
            chat_mds
            # chat_mds + [model_explanation] + [model_explanation2],
        )

    return (category_selector,)


def build_single_answer_browser_tab():
    global question_selector_map, category_selector_map

    models = list(model_answers.keys())
    num_sides = 1
    num_turns = 3
    side_names = ["A"]

    question_selector_choices = list(question_selector_map.keys())
    category_selector_choices = list(category_selector_map.keys())

    # Selectors
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            category_selector = gr.Dropdown(
                choices=category_selector_choices, label="Category", container=False
            )
        with gr.Column(scale=100):
            question_selector = gr.Dropdown(
                choices=question_selector_choices, label="Question", container=False
            )

    model_selectors = [None] * num_sides
    with gr.Row():
        for i in range(num_sides):
            with gr.Column():
                model_selectors[i] = gr.Dropdown(
                    choices=models,
                    value=models[i] if len(models) > i else "",
                    label=f"Model {side_names[i]}",
                    container=False,
                )

    # Conversation
    chat_mds = []
    for i in range(num_turns):
        chat_mds.append(gr.Markdown(elem_id=f"turn_{i}_user_question"))
        with gr.Row():
            with gr.Column(scale=100):
                chat_mds.append(gr.Textbox(lines=20))
        with gr.Row():
            for j in range(num_sides):
                with gr.Column(scale=100):
                    chat_mds.append(gr.Markdown())

                if j == 0:
                    with gr.Column(scale=1, min_width=8):
                        gr.Markdown()

    print(f".... single tab len(chat_mds) = {len(chat_mds)}")
    references = [gr.Textbox(lines=20, elem_id=f"turn_{i}_reference") for i in range(num_turns)]
    chat_mds.extend(references)
    print(f".... single tab len(chat_mds) = {len(chat_mds)}")

    model_explanations = [gr.Textbox(lines=20, elem_id=f"turn_{i}_model_explanation") for i in range(num_turns)]
    chat_mds.extend(model_explanations)
    
    print(f".... single tab len(chat_mds) = {len(chat_mds)}")
    # Callbacks
    category_selector.change(display_question, [category_selector], [question_selector])
    question_selector.change(
        display_single_answer,
        [question_selector] + model_selectors,
        chat_mds,
    )

    for i in range(num_sides):
        model_selectors[i].change(
            display_single_answer,
            [question_selector] + model_selectors,
            chat_mds
        )

    return (category_selector,)


color_ops = ["#DEEBF7", "#E2F0D9", "#FFF2CC", "#FBE5D6"]
block_css_basis = [f"""
#turn_{i}_user_question {{
    background-color: {color_ops[i]};
}}
#turn_{i}_reference {{
    background-color: {color_ops[i]};
}}
#turn_{i}_model_explanation {{
    background-color: {color_ops[i]};
}}
#turn_{i}_model_judgment {{
    background-color: {color_ops[i]};
}}"""
    for i in range(3)
]

block_css = "".join(block_css_basis)

def load_demo():
    # dropdown_update = gr.Dropdown.update(value=list(category_selector_map.keys())[0])
    dropdown_update = gr.Dropdown(value=list(category_selector_map.keys())[0])
    return dropdown_update, dropdown_update


def build_demo():
    build_question_selector_map()
    _reload_info()
    with gr.Blocks(
        title="MT RAG Bench Browser",
        theme=gr.themes.Base(text_size=gr.themes.sizes.text_lg),
        css=block_css,
    ) as demo:
        gr.Markdown(
            """
# MAGIC-Bench Browser
The code to generate answers and judgments is at [MAGIC-Bench](https://github.com/mtkresearch/MAGIC-Bench).
"""
        )
        with gr.Tab("Single Answer Grading"):
            (category_selector,) = build_single_answer_browser_tab()
        with gr.Tab("Pairwise Comparison"):
            (category_selector2,) = build_pairwise_browser_tab()
        demo.load(load_demo, [], [category_selector, category_selector2])

    return demo

def _reload_info():
    global questions, question_file
    global model_answers, answer_dir
    global model_judgments_normal_single
    
    questions = []
    model_answers = {}
    model_judgments_normal_single = {}

    # Load questions
    assert os.path.exists(question_file), f".... question file not exists: {question_file}"
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    
    # Load judgments
    if os.path.exists(single_model_judgment_file):
        model_judgments_normal_single = load_single_model_judgments(single_model_judgment_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--bench-name", type=str, default="magic")
    parser.add_argument("--judge-model-name", default="gpt-4o", type=str)
    args = parser.parse_args()
    print(args)
    
    judge_model_name = args.judge_model_name
    
    question_file = f"{_CUR_DIR}/data/questions.jsonl"
    answer_dir = f"{_CUR_DIR}/data/model_answer"
    single_model_judgment_file = (
        f"{_CUR_DIR}/data/model_judgment/{judge_model_name}.jsonl"
    )
    print(f".... question file = {question_file}")
    print(f".... answer_dir = {answer_dir}")
    print(f".... judge file = {single_model_judgment_file}")

    _reload_info()

    demo = build_demo()
    demo.queue(status_update_rate=10, api_open=False).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )