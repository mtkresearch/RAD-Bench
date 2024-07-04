export api_base="<YOUR_MODEL_API_BASE>"
export api_key="<YOUR_MODEL_API_KEY>"

bench_name="magic"
model_name="breeze-7B-32k-instruct-v10"

answer_file="data/model_answer/${model_name}.jsonl"
num_fewshots=0
q_beg=0
q_end=89
parallel=16
question_file="data/questions.jsonl"
retry=0 # if specified, it inferenes problems in RETRY_IDS (specified in gen_api_answer.py)

exec > run_${model_name}_inference_${q_beg}-${q_end}.log 2>&1

cd ..
python3 gen_api_answer.py \
    --bench-name $bench_name \
    --model $model_name \
    --model-sys-mesg "$SYS_MESG" \
    --question-begin $q_beg \
    --question-end $q_end \
    --answer-file $answer_file \
    --num-fewshots $num_fewshots \
    --question-file $question_file \
    --retry $retry \
    --parallel $parallel
cd -