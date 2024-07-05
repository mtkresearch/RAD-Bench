"""
Usage:
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
from tqdm import tqdm


from common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchSingle,
    JUDGE_MODE_KEYS
)


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    turn,
    type,
    baseline_model=None,
    ref_answers=None,
    
):
    """
    judge: judge prompt
    ans_model: which file to reference
    """
    matches = []
    for q in questions:
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.ref_ans_model][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, turn=turn, type=type
                    )
                )
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, turn=turn, type=type))
    return matches


def make_judge_single(judge_model, judge_prompts, ref_ans_model = None):
    judges = {jk: Judge(judge_model, judge_prompts[jk], ref_ans_model=ref_ans_model) for jk in JUDGE_MODE_KEYS}
    return judges


def reorg_judgment_file(judgment_file):
    """Sort by question id and de-duplication"""
    judgments = {}
    with open(judgment_file, "r", encoding="utf-8") as fin:
        for l in fin:
            data = json.loads(l)
            uid = f"{data['question_id']}_{data['model']}_{'_'.join(data['judge'])}"
            judgments[uid] = l

    uids = sorted(list(judgments.keys()))
    with open(judgment_file, "w", encoding="utf-8") as fout:
        for uid in uids:
            fout.write(judgments[uid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="magic",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--baseline-model", type=str, default="gpt-35-turbo-16k-4k")
    parser.add_argument("--ref-ans-model", type=str, default="reference_ans_model")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single"],
        help=(
            "Evaluation mode. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
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
        "--question-file", type=str, default="data/questions.jsonl"
    )
    parser.add_argument(
        "--answer-dir", type=str, default="data/model_answer"
    )
    parser.add_argument(
        "--reference-answer-dir", type=str, default="data/reference_answer"
    )
    parser.add_argument(
        "--output-file", type=str, default="model_judgment/breexe_single.jsonl"
    )

    args = parser.parse_args()

    question_file = args.question_file
    answer_dir = args.answer_dir
    ref_answer_dir = args.reference_answer_dir

    # Load questions
    q_beg = args.question_begin
    q_end = args.question_end
    questions = load_questions(question_file, q_beg, q_end)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)
    print(f".... Available model answers = {list(model_answers.keys())}")
    print(f".... Available reference model answers = {list(ref_answers.keys())}")
    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.first_n:
        questions = questions[: args.first_n]

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts, args.ref_ans_model)
        play_a_match_func = play_a_match_single
        output_file = (
            args.output_file
        )
        make_match_func = make_match_single
        baseline_model = None
    else:
        raise NotImplementedError()
        
    # check_data(questions, model_answers, ref_answers, models, judges, args.bench_name)
    
    # make questions
    ki_q = [q for q in questions if q['category'] in ['news', 'education', 'academic']]
    kr_q = [q for q in questions if q['category'] in ['finance', 'customer']] # finance and customer
    travel_q = [q for q in questions if q['category'] == 'travel']

    # assert len(ki_q) + len(kr_q) + len(travel_q) == len(questions) # assert the amount of questions in different categories are correct

    # Make matches
    matches = []
    for turn in range(1,4): # turn 1,2,3
        matches += make_match_func(
            ki_q,
            models,
            model_answers,
            judges[f"CS-{turn}-turn"],
            turn,
            "CS",
            baseline_model,
            ref_answers
        )
        matches += make_match_func(
            kr_q,
            models,
            model_answers,
            judges[f"CR-{turn}-turn"],
            turn,
            "CR",
            baseline_model,
            ref_answers,
        )
        matches += make_match_func(
            travel_q,
            models,
            model_answers,
            judges[f"TR-{turn}-turn"],
            turn,
            "TR",
            baseline_model,
            ref_answers,
        )
    match_stat = {}
    match_stat["bench_name"] = args.bench_name
    match_stat["mode"] = args.mode
    match_stat["judge"] = args.judge_model
    match_stat["judge_file"] = args.judge_file
    match_stat["baseline"] = baseline_model
    match_stat["ref_ans_model"] = args.ref_ans_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file
    match_stat["question_file"] = question_file
    match_stat["model_ans_dir"] = answer_dir
    match_stat["reference_dir"] = ref_answer_dir
    match_stat["num_accounts"] = len(_MR_EMP_IDS)

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    input("Press Enter to confirm...")

    # Play matches
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass
    
    reorg_judgment_file(output_file)