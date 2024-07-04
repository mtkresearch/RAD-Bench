# MAGIC-Bench

**MAGIC-Bench: A Benchmark for Evaluating Large Language Models with
Context Utilization in Multi-Turn Dialogues**

Tzu-Lin Kuo*, Mu-Wei, Hsieh,
Fu-Chieh Chang, Po-Chun Hsu, Da-Shan Shiu, Feng-Ting Liao*

*core contributors

.... paper coming soon.

## Benchmark results

![model_radar](./asset/model_radar.png)

## Installation
```
pip install -r requirements.txt
```

## Inference
Simply do,
```
cd magic/script
sh run_inference.sh
```

## Evaluation
Simply do,
```
cd magic/script
sh run_evaluation.sh
```

## Visulization
To see the results, do
```
cd magic/script
sh run_qa_browser.sh
```
then open `localhost:1234` in your browser



## Citation
Please cite this repo with
```
@misc{magic-kuo2024,
      title={MAGIC-Bench: A Benchmark for Evaluating Large Language Models with
Context Utilization in Multi-Turn Dialogues},
      author={Tzu-Lin Kuo, Mu-Wei, Hsieh, Fu-Chieh Chang, Po-Chun Hsu, Da-Shan Shiu, Feng-Ting Liao},
      year={2024}
}
```
