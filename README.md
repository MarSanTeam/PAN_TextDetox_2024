# Marsan at PAN 2024 TextDetox: ToxiCleanse RL, Paving the Way for Toxicity-Free Online Discourse

 <div align="center">
<b>Maryam Najafi</b><sup>1, 2</sup>
<b>Ehsan Tavan</b><sup>2</sup>,
<b>Simon Colreavy</b><sup>1</sup>,


</div>
<div align="center">
<sup>1</sup>Department of Computer Science and Information Systems, University of Limerick, Ireland
</div>
<div align="center">
<sup>2</sup>NLP Department, Part AI Research Center, Tehran, Iran
</div>

 

## Overview

Addressing the pervasive issue of toxicity in online communication requires innovative solutions beyond mere identification and removal of harmful content. In this paper, we propose a novel approach termed ToxiCleanse RL, which employs Reinforcement Learning (RL), specifically Proximal Policy Optimization (PPO), in tandem with Large Language Models (LLMs), for detoxification through text style transfer (TST). Our method aims to automatically rewrite toxic messages while preserving their original meaning. By utilizing a toxicity-based reward model, we guide the RL fine-tuning process to effectively reduce the generation of toxic language. Empirical evaluation on English and Russian datasets demonstrates the superior performance of our approach compared to existing detoxification techniques, achieving a manual evaluation score of 0.89 (ranked 2nd) for English and 0.70 (ranked 7th) for Russian. These results underscore the potential of RL-based approaches in mitigating toxicity in online discourse, paving the way for safer and more inclusive digital environments.   


This repository contains the code and data for the **ToxiCleanse RL** approach presented at the Multilingual Text Detoxification (TextDetox) shared task at PAN 2024. The MarSan_AI team developed this innovative solution, which leverages **Reinforcement Learning (RL)**, specifically **Proximal Policy Optimization (PPO)**, in combination with **Large Language Models (LLMs)**, to detoxify text through **Text Style Transfer (TST)**.

**ToxiCleanse RL** is an RL-based approach designed to automatically rewrite toxic messages while preserving their original meaning. By utilizing a toxicity-based reward model, the method guides the RL fine-tuning process to effectively reduce toxic language generation. Our empirical evaluation on English and Russian datasets demonstrates superior performance compared to existing detoxification techniques. The method achieved a manual evaluation score of 0.89 (ranked 2nd) for English and 0.70 (ranked 7th) for Russian.

For more details, you can read our full paper [here](https://ceur-ws.org/Vol-3740/paper-269.pdf).  


## Contents

- [Task & Data Description](#task--data-description)
- [Approach Overview](#approach-overview)
  - [Base LLM and Parameters](#base-llm-and-parameters)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
  - [Reinforcement Learning with Proximal Policy Optimization (PPO)](#reinforcement-learning-with-proximal-policy-optimization-ppo)
  - [Reward Model](#reward-model)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
  - [Reinforcement Learning (PPO)](#reinforcement-learning-ppo)
  - [Inference and Evaluation](#inference-and-evaluation)

## Task & Data Description

The task involves detoxifying text messages while preserving their meaning and maintaining fluency. The datasets provided for English and Russian contain toxic and non-toxic text pairs. The task is to develop a model that transforms toxic messages into non-toxic ones.

## Approach Overview

Our approach, **ToxiCleanse RL**, consists of two main phases: Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) fine-tuning.

### Base LLM and Parameters

We use a Large Language Model (LLM) as the base model. The fine-tuning process begins with Supervised Fine-Tuning (SFT) to learn the text detoxification task.

### Supervised Fine-Tuning (SFT)

During the SFT phase, the model is trained on parallel datasets of toxic and non-toxic texts using supervised learning. The objective is to teach the model to generate non-toxic outputs for toxic inputs.

### Reinforcement Learning with Proximal Policy Optimization (PPO)

Following SFT, we employ **Proximal Policy Optimization (PPO)** to further fine-tune the model. The PPO algorithm optimizes the policy to maximize rewards, which are based on detoxification quality and semantic similarity to the original text.

### Reward Model

A pre-trained reward model is used to provide feedback on the generated outputs during the PPO fine-tuning phase. The rewards are computed based on toxicity reduction, fluency, and semantic similarity.

## Setup and Installation

To set up the environment and install the necessary dependencies, run the following commands:
```bash
git clone https://github.com/MarSanAI/ToxiCleanse-RL.git
cd ToxiCleanse-RL
pip install -r requirements.txt

```

## Usage

### Supervised Fine-Tuning (SFT)

To run **Supervised Fine-Tuning (SFT)**, execute the following script:

```bash
python runner.py 
```
### Reinforcement Learning (PPO)

To fine-tune the model using Proximal Policy Optimization (PPO), run:
```bash
python runnerPPO.py
```

### Inference and Evaluation

For inference and evaluation, the following scripts are available:
- Inference: Run inferecer.py to generate detoxified outputs from the trained model.
- Evaluation: Use evaluater.py to evaluate the model's performance on detoxification tasks.
```bash
python inferecer.py
python evaluater.py
```
