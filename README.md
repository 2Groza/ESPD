# ESPD
code for the [paper](https://arxiv.org/abs/2004.12909) Evolutionary Stochastic Policy Distillation

The paper results can be reproduced by running the jupyter notebooks, or running the corresponding files directly by running, e.g.,

```
python main.py
```

To change the environment, users can just change the ENV_NAME from one of 'FetchPush-v1','FetchSlide-v1', and 'FetchPickAndPlace-v1'.

The result of [PCHID](https://sites.google.com/view/neurips2019pchid) can be reproduced based on this codebase by progressively increase the hyper-parameter Horizon from 1 to K. As our ablation studies have shown in the paper, using Horizon = 8 can always achieve relatively high performance. Therefore PCHID can be interpreted as a special case for ESPD, and its performance should be upper bounded by ESPD.

### Bibtex

```
@article{sun2020evolutionary,
  title={Evolutionary Stochastic Policy Distillation},
  author={Sun, Hao and Pan, Xinyu and Dai, Bo and Lin, Dahua and Zhou, Bolei},
  journal={arXiv preprint arXiv:2004.12909},
  year={2020}
}
```
