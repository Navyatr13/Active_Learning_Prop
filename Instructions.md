# Entalpic: take-home

Hey! Thanks a lot for applying to Entalpic.

This readme contains the instructions for the take-home exercise we will use as a foundation to measure your technical skills and code philosophy.

## About this exercise

You should not spend too much time on this exercise. We typically ask that you spend not more than 3-4h actively working on it (not counting installs, setups etc.).

We will ask you to report the time spent, to attempt and adjust our expectations across candidates. If you decide to spend more than 3-4h on this problem, there is no penalty but you must be transparent about it. We have decided to be trusting and not time explicitely this take-home exercise so we count on your integrity in reporting the time spent on it. (Scientific) integrity is a value of paramount importance at Entalpic.

We encourage you to use Git as if you were working in the team. In particular, you _may_ choose to code in a private repository and open PRs to your own code to show us how you would do things with us. Doing so will likely slow you down. We'll let you be the judge of what balance between code quality, robust collaboration and task completion enables you to best showcase your strengths. There is no good answer, just good motivations for any answer!

We do realize there will likely be some learning involved here. This means we don't expect great performance, but rather great thinking and great software engineering. You will likely not have the time to do everything ; but again, your choices and their motivation matter more than the end result.

## Problem

In this problem you will implement an active learning (AL) loop to most efficiently improve the performance of a custom graph neural network.

It is intended to be too long to give different candidates the opportunity to shine where they are the brightest. This means you are not expected to both get to the end _and_ write great, robust and reproducible code in such a short time frame. Again, you're the judge of that balance.

**Note**: If you are already quite familiar with AL, you may choose to jump to multi-fidelity AL from the start (i.e. not go through single-fidelity first; you should still do the rest).

### Steps

We will use the `Atomization enthalpy at 298.15K` property of molecules in the QM9 data set.

1. As a sanity check for your install and as a warm-up into today's problem: refactor and improve the code quality of `sanity_check.py`.
2. Implement your own custom graph neural network using low-level layers in `torch` & `torch_geometric`
   1. "Low-level" = "don't just import a model" != "re-implement everything as if you only had access to matrix multiplications".
   2. Your GNN should be rather "small" (i.e. <1M parameters).
   3. 🔥 In your report, describe how you would perform hyper-parameter optimization for it.
3. Integrate a pre-trained model of your choice (from anywhere online, including the one used in `sanity_check.py`).
4. Implement an active learning loop using the pre-trained model as a labeler for your custom model.
   1. Use an initial data set of 1000 randomly chosen data points.
   2. If you don't know anything about active learning, a vanilla approach is to label points where your model has the most uncertainty. A simple uncertainty estimate can be obtained either through _MC Dropout_ or _Deep Ensemble_.
5. 🔥 Report on the efficacy/performance of your approach*.
6. Re-run your approach, but use the ground-truth target as new labels, not the pre-trained model's predictions and 🔥 comment.
7. Compare multiple active learning approaches (batch sizes, surrogates, aquisition functions, etc.) and 🔥 report comparison.
8. Implement a multi-fidelity active learning loop using 2 pre-trained models of your choice and 🔥 present your results including the detailed of your implemented / future evaluation pipeline.

_**Optional**_: (constructive) feedback is very welcome, feel free to tell us during the interview or in an email what you think about this take-home exercise.

\* Reporting on performance does not mean optimizing for it. It means critically reflecting on it in ways that allow informed decisions (such as "this is not a good aquisition function, we should use another one"). Negative results are welcome, as long as they are identified as such.

## Instructions

Some notions here and in the sample code are a bit rough, some instructions may seem a little bit unclear and the code you'll find online will often be sub-optimal. But that's a good dive into coding at the frontier of science! Enjoy the ride.

1. Provide a Readme with detailed instructions on how to use your code and the conclusions of question in the previous section (report-related questions are marked by a 🔥).
   1. You may provide any complementary format / support as long as it is listed and described in the readme.
2. Package your code such that it is as robust and qualitative as possible from another Machine Learning Engineer's perspective but also usable by a Data Scientist or Computational Chemist with more superficial Python skills.
3. You may use any resource at your disposal, as you would in the real world.
4. 🔥 You should include next steps to production in the readme so we understand your thought process beyond what is actually doable in such a "short" exercise.
5. Your code should be runnable on your machine but handle GPU logic as if you had one.
   1. 🔥 You may also descibe necessary steps to multi-GPU training and associated points of attention, if and only if this is something you are familiar with.
6. Your results should be reproducible.

## Getting Started

### Basic installations

The following installation instructions should allow you to run `sanity_check.py`. You're free to install anything else. You're in charge of library compatibilities etc.

This was tested on macOS with an Intel CPU and Python 3.11.9

_Warning_: There have been reported segmentation faults with 3.11.0

_Warning_: on Apple M1 CPUs, installation steps may be a bit more tricky. [This blog post](https://medium.com/@jgbrasier/installing-pytorch-geometric-on-mac-m1-with-accelerated-gpu-support-2e7118535c50) should help.

```bash
pip install -r requirements.txt
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

`torch_scatter` and `torch_sparse` are expected to take a long time to build, don't worry.

**Note**: The above libraries are notoriously peaky about their environment, including lower level libraries in your system. If you spend more than 20-30mins looking for other versions of `torch`, `torch_geometric`, `torch_scatter`, etc. that let you run the `sanity_check.py` script, you should stop and get back to us, this is not an exercise in building annoying libraries.

**Note 2**: you may also use Google Colab (or other) to run your code _as a package_ if you dont want to / cannot use your own machine. We recommend against using Notebooks for code development.

### Pointers

* QM9 Data Set https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html#torch-geometric-datasets-qm9
* Understanding the QM9 Data Structure https://colab.research.google.com/github/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb#scrollTo=byQrx71Udlv5&line=1&uniqifier=1
* Torch Geometric https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers
* Bayesian Optimization in PyTorch https://botorch.org/
* An invariant GNN model: [SchNet: A continuous-filter convolutional neural network for modeling quantum interactions](https://arxiv.org/pdf/1706.08566.pdf)
* On Geometric Graph Neural Networks: [A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems](https://arxiv.org/pdf/2312.07511.pdf)

Remember: just do your thing, you have 100% freedom to make this project yours.
