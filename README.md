# MARPLE: A Benchmark for Long-Horizon Inference
[**Project Page**](https://marple-benchmark.github.io/) | [**Dataset**](https://drive.google.com/drive/folders/1zXsErNVOMYjBMWzTnmZS4e4aIljWlRce?usp=sharing)

This repository contains the code for the simulator and experiments in the paper: **MARPLE: A Benchmark for Long-Horizon Inference**.

Emily Jin\*, Zhuoyi Huang\*, Jan-Philipp Fränken, Weiyu Liu, Hannah Cha, Erik Brockbank, Sarah Wu, Ruohan Zhang, Jiajun Wu, Tobias Gerstenberg.

Under Review.

<!-- ```
@inproceedings{,
  title = {},
  booktitle = {},
  author = {},
  year = {2024},
}
``` -->

## Overview
MARPLE (in reference to Agatha Christie's Miss Marple) is a benchmark for long-horizon inference based on multimodal evidence. The main goal of MARPLE is to test a model's ability to answer “whodunit”-style questions in daily household scenarios, such as “who turned on the laundry?” The inference problem requires choosing the correct agent from two potential suspects, given knowledge about their prior behaviors and the state of the environment.

**Inference Scenario Setup.** Two agents, A and B, each perform a mission, such as “do laundry” and "change clothes." To complete their mission, each agent must interact with the environment, causing changes in the world and leaving evidence of its activity. A “whodunit” question is constructed by selecting a state that is unique to one agent’s trajectory. For example, a state that is unique to agent A is “laundry is on,” so we pose the question: "Which agent turned on the laundry?" To answer “whodunit” questions, models must leverage evidence in the form of multimodal observations from each agent’s activity history. 

![](https://github.com/marple-benchmark/marple/blob/main/figures/main.png)

## Contents
- [Installation](#installation)
- [Repository Structure](#folder-structure)
- [Learning Agent Models](#learning-agent-models) 
- [Inference Experiments](#inference-experiments)
  - [Mental Simulation with Learned Agent Models](#mental-simulation-with-learned-agent-models)
  - [GPT-4](#gpt-4)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation
```bash
git clone https://github.com/marple-benchmark/marple.git
conda create -n marple python==3.10
conda activate marple
pip install -r requirements.txt
pip install -e .
```

## Folder Structure
Below is the folder structure of our project:

- `/gpt`: contains all of the code for experiments using GPT-4 experiments, including prompts and scripts.
- `/mcts`: contains all of the code for experiments using the Mental Simulation method, including scripts for training, evaluation, and inference.
- `/src`
  - `/benchmark`: contains files that define the missions and inference states for the benchmark.
  - `/mini_behavior`: forked from [Mini-BEHAVIOR](https://github.com/StanfordVL/mini_behavior) repo.
  - `/models`: contains all of the code for the agent policy models and inference models used for the mental simulation method.
  - `/simulator`: contains all of the code for the multimodal environment simulator and hierarchical agent planner.
- `/figures`: contains figures used in the README
- `experiments.json`: specifies the 5 inference scenarios, inference questions and answers, and trajectories that were used in experiments.
- `setup.py`
- `requirements.txt`
- `datasheet.md`
- `README.md`

## Data
The MARPLE Benchmark features 10 diverse, long-horizon missions, which are paired to create 5 challenging inference scenarios that offer a balanced representation of complexity and diversity. Each mission is accompanied by both train and test datasets: two train datasets, each containing 5000 agent trajectories (one for evaluating in-distribution performance and the other for out-of-distribution performance), and a test dataset with 500 diverse agent trajectories.

### Downloading Data
The data is uploaded [here](https://drive.google.com/drive/folders/1zXsErNVOMYjBMWzTnmZS4e4aIljWlRce?usp=sharing) as zip files. There is one folder for each set of training data, and each folder contains a zip file for each inference scenario. The test data for all scenarios is in one zip file. You can download and add them to the `data` directory, following the same folder structure. The scripts will assume that the data is located there.
<!-- ### Generating Data -->

## Learning Agent Models
### Train Agent Models
You can train a policy model on any dataset of agent trajectories. All of the policy models can be trained using the following command:
```bash
python mcts/train.py experiment.mission_1=${mission_1} experiment.mission_1_pref=${mission_1_pref} experiment.mission_2=${mission_2} experiment.mission_2_pref=${mission_2_pref} model.model_name=${model_name} data.generalization=${generalization}
```
The parameters `experiment.mission_1`, `experiment.mission_1_pref` and `experiment.mission_2`, `experiment.mission_2_pref` specify the mission and corresponding mission preference for agents A and B, respectively. 

The parameter `model.model_name` specifies the type of policy model. It can take two values: `low_policy` (for the vision-only and audio-augmented models) and `subgoal_low_policy` (for the language-conditioned and audio-augmented language-conditioned policy models). 

The parameter `data.generalization` specifies whether dataset of trajectories to train the model on. Use `in` to train in-distribution and `ood` for out-of-distribution.

The model checkpoints will be saved to the `mcts/checkpoints` directory under the subdirectory `model.generalization/{mission_1}-{mission_2}/{mission_1}-{mission_1_pref}-{mission_2}-{mission_2_pref}/model.model_name)` directory. Feel free to change this by setting the `model.dirpath` parameter. 

For example, to train a vision-only policy model on data in-distribution for an agent that performs the mission 'feed dog' 60% of the time and 'do laundry' 40% of the time, run the following command:
```bash
python mcts/train.py experiment.mission_1=feed_dog experiment.mission_1_pref=0.6 experiment.mission_2=take_shower experiment.mission_2_pref=0.4 model.model_name=low_policy data.generalization=in
```

### Evaluate Agent Models
To evaluate the model checkpoints, use the following command: 
```bash
python mcts/eval.py experiment.mission_1=${mission_1} experiment.mission_1_pref=${mission_1_pref} experiment.mission_2=${mission_2} experiment.mission_2_pref=${mission_2_pref} model.model_name=${model_name} model.checkpoint_name=${checkpoint_name} data.generalization=${generalization} data.split=${split}
```

The parameter `model.checkpoint_name` specifies the filename of the checkpoint that you want to evaluate, `data.split` specifies whether to evaluate on the `train` or `test` trajectories. If you set `model.dirpath` during training, you should set it here as well.

The results will be saved at `mcts/results`, but feel free to change this by setting `experiment.results_dir`.

## Inference Experiments
The json file [`experiments.json`](https://github.com/marple-benchmark/marple/blob/main/experiments.json) specifies the inference scenarios, inference answer, and trajectories used in our experiments. The inference scenarios (formatted as {mission_1}-{mission_2}) are the keys, with a corresponding dictionary that specifies the `inference_answer` (agent `A` or `B`), agent `A_mission`, agent `B_mission`, `inference_question`, and list of test trajectories `trajs` (tuples that indicate the `room` and `traj_name`).

### Mental Simulation with Learned Agent Models
Inference for a single trajectory takes around 1.5 hours to run with 1 NVIDIA TITAN RTX GPU and 8 CPU. To run an inference experiment using a trained policy model for a specified inference scenario and trajectory, use the following command:
```bash
python mcts/main.py rollout.inference_scenario=${inference_scenario} rollout.inference_answer=${inference_answer} rollout.policy_model_checkpoint=${policy_model_checkpoint} rollout.language=${language} rollout.audio=${audio} rollout.room_config=${room_config} rollout.traj_name=${traj_name} rollout.a_mission=${A_mission} rollout.a_pref=${A_mission_pref} rollout.b_mission=${B_mission} rollout.b_pref=${B_mission_pref} 
```
The parameter `rollout.inference_scenario` specifies the inference scenario, `rollout.inference_answer` is the agent that is the answer to the inference question, and `rollout.room_config` and `rollout.traj_name` specify which trajectory in `data/test` to perform inference on. 

`rollout.policy_model_checkpoint` specifies the full path of the policy model, `rollout.language` is a `bool` that specifies whether to use the language modality (`False` if the policy model is `low_model` and `True` if `subgoal_low_model`), `rollout.audio` is a `bool` that specifies whether to use the audio modality. 

### GPT-4 Inference Experiments
To run inference experiments for all 5 inference scenarios using GPT-4, use the following command:
```bash
python gpt/main.py 
```
This relies on the environment variable `OPENAI_API_KEY`.

## License
The project is licensed under the [MIT License](https://github.com/marple-benchmark/marple/blob/main/LICENSE).

## Acknowledgements
This work was in part supported by a grant from the Stanford Institute for Human-Centered Artificial Intelligence (HAI), NSF CCRI #2120095, and ONR MURI N00014-22-1-2740.

## CRediT author statement
*[What is a CRediT author statement?](https://www.elsevier.com/researcher/author/policies-and-guidelines/credit-author-statement)*

- **Emily Jin\***: 
- **Zhuoyi Huang\***: 
- **Philipp Jan-Fränken**: 
- **Weiyu Liu**:
- **Hannah Cha**:
- **Erik Brockbank**:
- **Sarah Wu**:  
- **Ruohan Zhang**:  
- **Jiajun Wu**: 
- **Tobias Gerstenberg**: 
