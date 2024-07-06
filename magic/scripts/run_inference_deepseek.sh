export api_base="https://api.deepseek.com"
export api_key="sk-73e8a37f2f1a4e08b404bb0abd0dd3fd"

bench_name="magic"
model_name="deepseek-chat"

answer_file="data/model_answer/${model_name}.jsonl"
q_beg=0
q_end=88
parallel=16
question_file="data/questions.jsonl"

exec > run_${model_name}_inference_${q_beg}-${q_end}.log 2>&1

cd ..
python3 gen_api_answer.py \
    --bench-name $bench_name \
    --model $model_name \
    --model-sys-mesg "$SYS_MESG" \
    --question-begin $q_beg \
    --question-end $q_end \
    --answer-file $answer_file \
    --question-file $question_file \
    --parallel $parallel
cd -