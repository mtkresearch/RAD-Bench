# MAGIC-Bench

**MAGIC-Bench: A Benchmark for Evaluating Large Language Models with
Context Utilization in Multi-Turn Dialogues**

Tzu-Lin Kuo*, Mu-Wei Hsieh,
Fu-Chieh Chang, Po-Chun Hsu, Da-Shan Shiu, Feng-Ting Liao*

*core contributors

.... paper coming soon.

## Benchmark results

![model_radar](./asset/model_radar.png)
|               |   academic |    news |   education |   finance |   customer |   travel |   average |
|:--------------|-----------:|--------:|------------:|----------:|-----------:|---------:|----------:|
| GPT-4o        |    8.4 | 8.6 |     9.1     |   9.0 |    9.0 |  7.8 |   8.69 |
| GPT-35-Turbo  |    5.2 | 5.1 |     7.0     |   8.2 |    8.2 |  5.9 |   6.65 |
| Llama3-70B    |    8.1     | 7.4 |     7.9     |   8.7 |    7.6 |  4.3 |   7.38 |
| Mixtral-8x22B |    7.9 | 7.4 |     7.8 |   8.1 |    8.4     |  5.6 |   7.59  |
| BreeXe-8x7B   |    8.1 | 8.1 |     8.3    |   7.6 |    7.7     |  5.7 |   7.61 |
| Llama3-8B     |    6.5     | 6.9 |     7.3     |   7.2 |    7.1 |  3.6 |   6.47 |
| Breeze-7B     |    7.4     | 7.3 |     7.3 |   6.9 |    7.1 |  5.0 |   6.86 |

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
      author={Tzu-Lin Kuo, Mu-Wei Hsieh, Fu-Chieh Chang, Po-Chun Hsu, Da-Shan Shiu, Feng-Ting Liao},
      year={2024}
}
```
