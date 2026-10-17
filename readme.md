# Spatial-Forcing Eval：评估可靠性扩展

[![Evaluation regression tests](https://github.com/ton242678-glitch/spatial-forcing-eval/actions/workflows/eval-tests.yml/badge.svg)](https://github.com/ton242678-glitch/spatial-forcing-eval/actions/workflows/eval-tests.yml)

**这是 Spatial-Forcing 的衍生评估改进仓库。原论文、模型和原始实现属于原作者，本仓库新增的是动作窗口修复、配置检查、结构化结果和回归测试。**

Forked from [yifanwu-victor/Spatial-Forcing](https://github.com/yifanwu-victor/Spatial-Forcing), originally developed by [OpenHelix-Team](https://github.com/OpenHelix-Team/Spatial-Forcing). Baseline: `372819b31ba9a2c8fc5989edc7d525cb187cecd5`.

## 本次实现的三个切入点

| 切入点 | 源码中发现的问题 | 本仓库的改进 |
| --- | --- | --- |
| 动作执行窗口 | 当 `num_open_loop_steps=2`，把 8 个预测动作加入容量为 2 的 `deque`，会留下最后两个动作 | 显式取前两个动作，保持时间顺序；默认 8 步行为保持一致 |
| 无效参数尽早报错 | 空 checkpoint、非正执行步数等参数缺少检查，部分断言会被 `python -O` 关闭 | 加载模型权重前抛出明确的 `ValueError`，校验冲突选项和数值范围 |
| 实验结果便于比较 | 文本日志难以按任务比较，异常与普通失败混在一起 | 每轮原子写入 JSON，记录任务、seed、配置、成功/失败/异常/跳过；支持多报告 CSV 导出 |

这些改进提高评估行为的可解释性和结果可追溯性。**尚未进行完整模型仿真或真机实验，不报告成功率或速度提升。**

## 不需要 GPU 的快速体验

在仓库根目录运行以下命令。新增工具只用 Python 标准库，Python 3.10+ 即可，无需安装 PyTorch 或下载模型。

```bash
python -m unittest discover -s tests -v
python tools/demo_eval_report.py --output reports/synthetic-demo.json
python tools/summarize_eval.py reports/synthetic-demo.json --csv reports/synthetic-summary.csv
```

演示只生成明确标为 `synthetic_example` 的人工样例，不是机器人评估成绩。CSV 保留这一标记，避免与真实评估混淆。

## 接入真实 LIBERO 评估

先按 [原始仿真说明](openvla-SF/readme.md) 安装依赖、LIBERO 和 checkpoint。进入 `openvla-SF` 后运行：

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint ckpts/spatial-forcing-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_open_loop_steps 2
```

新增 JSON 会保存在与原 `.txt` 日志相同的目录，文件名相同、后缀为 `.json`。回到仓库根目录后可比较多个报告：

```bash
python tools/summarize_eval.py path/to/run-1.json path/to/run-2.json --csv reports/comparison.csv
```

完整推理仍依赖原项目的模型、仿真环境和 GPU 配置。工具在 Windows 上可用，不代表原项目的完整训练环境已在 Windows 验证。

- [实现依据、测试范围与统计口径](docs/evaluation_extensions_zh.md)
- [进一步研究的四个切入方向](docs/research_directions_zh.md)
- [上游原始项目说明](#upstream-project)

## 来源与许可

原始提交历史、作者列表和许可文件均保留。根目录 MIT 许可见 [LICENSE](LICENSE)；`openpi-SF` 子项目许可见 [openpi-SF/LICENSE](openpi-SF/LICENSE)。新增代码沿用根目录 MIT 许可。下方为保留的上游项目介绍，其中性能数据均是原作者报告，与本仓库新增改进的验证范围不同。

---

<a id="upstream-project"></a>

# Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model
<div align="center">

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2510.12276) [![Page](https://img.shields.io/badge/Project--Page-blue?style=for-the-badge&logo=homepage&logoColor=white)](https://spatial-forcing.github.io/) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/haofuly/spatial-forcing-68ea1bf0f1ac2c60e2ec6caa) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](https://github.com/OpenHelix-Team/Spatial-Forcing/issues/1)

</div>

## :loudspeaker: News!
- **[2026/07/07]** We got the first place on the generalization dimension and second place on the average of all 5 dimensions of the comprehensive embodied leaderboard [RoboDojo](https://robodojo-benchmark.com/).
- **[2026/01/27]** Spatial Forcing is accepted by ICLR 2026!🎉
- **[2025/10/28]** We released our codes based on Pi_0 in real world! Everyone is welcome to use it!🎉
- **[2025/10/24]** 🏆 Congradulations to Jialong! He and our SF got the **second place** in **Agibot World Challenge** as well as 5000$ prize💰!
- **[2025/10/18]** Our paper won the 🥇**first place** in the [daily list](https://huggingface.co/papers/2510.12276) and 🥉**third place** in the [weekly list](https://huggingface.co/papers/week/2025-W42) in HF! ⭐
- **[2025/10/12]** We released our paper on [ArXiv](http://arxiv.org/abs/2510.12276).

## 🌟 Key Features of Spatial-Forcing (SF)

1. **Universality**: SF is a **plug-and-play** 3D finetune strategy that can be seamlessly integrated with any VLA training process, requiring only **30 lines** of code modifications. It substantially enhances spatial reasoning and manipulation capabilities. We provide implementations based on **OpenVLA** and **Pi0**, along with a **quick-start guide** for adapting SF to other VLA models.

2. **Strong Performance**: SF achieves **state-of-the-art (SOTA)** results on both **LIBERO** and **RoboTwin** benchmarks.  
In real-world experiments involving complex spatial structures, SF improves task success rates by **up to 50%**.

3. **Efficient Training**: SF requires only **3% of the training steps** or **5% of the training data** to reach a 66% success rate on LIBERO-Long. Moreover, it achieves strong real-world performance with as few as **20 demonstrations**.


## 📃 Overview
![teaser](./figs/teaser.png)

Our Spatial-Forcing (SF) model aligns the intermediate visual embeddings of VLAs with geometric representations produced by pretrained 3D foundation models. This alignment demonstrates enhanced performance, training efficiency, and data effificency.


<!-- ## Table of Contents
- [🚀 Get Started](#-get-started)
  - [Environment Setup](#environment-setup)
  - [Data Preparation](#data-preparation)
  - [Inference](#inference)
  - [Training](#training) -->


## 🚀 Get Started
- To reproduce our simulation results, ref to our [openvla-SF folder](openvla-SF).

- To deploy policy in real-world robots, ref to our [openpi-SF folder](openpi-SF).

- To integrate Spatial-Forcing strategy into your VLA model, ref to [Simulation Training Scripts Line373-Line400](./openvla-SF/vla-scripts/finetune_align.py#L373-L399).


## 🔥 TODO List
✅ Training and inference code on LIBERO (Base model: OpenVLA)<br>
✅ Checkpoints on LIBERO (Base model: OpenVLA)<br>
✅ Deployment code in real world (Base model: Pi_0 torch version)<br>


## 🌏 Contact
For further discussion and collaboration, please feel free to contact us via Email and WeChat:

| Author | Email | WeChat |
|:---:|:---:|:---:|
| Fuhao Li | lfh23@mails.tsinghua.edu.cn | haofuly |
| Wenxuan Song | songwenxuan0115@gmail.com | swx0757 |
<!-- > WeChat Communication Group is at [here](https://github.com/OpenHelix-Team/Spatial-Forcing/issues/1) -->


## ❤️ Acknowledgement
We thank these great works and open-source codebases: [OpenVLA-OFT](https://github.com/moojink/openvla-oft) & [OpenPI](https://github.com/Physical-Intelligence/openpi) & [VGGT](https://github.com/facebookresearch/vggt) & [REPA](https://github.com/sihyun-yu/REPA)


## 🖊 Citation
If you find this work useful, please cite:

```bibtex
@article{spatialforcing2025,
  author    = {Li Fuhao, Song Wenxuan, Zhao Han, Wang Jingbo, Ding Pengxiang, Wang Donglin, Zeng Long, Li Haoang},
  title     = {Spatial Forcing: Implicit Spatial Representation Alignment For Vision-Language-Action Model},
  journal   = {arXiv preprint arXiv:2510.12276},
  year      = {2025},
}
```
