export api_base="<YOUR_MODEL_API_BASE>"
export api_key="<YOUR_MODEL_API_KEY>"

BENCH_NAME="magic"
LLM_JUG_DIR=..
JUDGE_FILE="./data/judge_prompts.jsonl"
JUDGE_MODEL="gpt-4o"

REF_ANS_MODEL="reference_ans_model" # decide which reference answer to use (so we can create a reference answer file and list it here)
MODE="single"
MODEL_LIST="gpt-35-turbo-16k-4k gpt-4o breexe-8x7b-instruct-v01 breeze-7B-32k-instruct-v10 llama3-70b-instruct mixtral-8x22b-instruct llama3-8b-instruct"

NUM_PROC=16
Q_BEG=0
Q_END=89
QUESTION_FILE="data/questions.jsonl"
OUTPUT_FILE="model_judgment/${JUDGE_MODEL}_${Q_BEG}-${Q_END}.jsonl"

cd $LLM_JUG_DIR
python3 gen_judgment.py\
    --bench-name $BENCH_NAME \
    --judge-file $JUDGE_FILE \
    --judge-model $JUDGE_MODEL \
    --ref-ans-model $REF_ANS_MODEL \
    --mode $MODE \
    --model-list $MODEL_LIST \
    --parallel $NUM_PROC \
    --question-file $QUESTION_FILE \
    --question-begin $Q_BEG \
    --question-end $Q_END \
    --output-file $OUTPUT_FILE
cd -