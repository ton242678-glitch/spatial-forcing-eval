# 评估扩展：实现依据与验证范围

## 1. 动作窗口修复

源码位置：`openvla-SF/experiments/robot/libero/run_libero_eval.py` 的 `run_episode`。

策略一次预测一组连续动作。上游使用 `deque(maxlen=num_open_loop_steps)` 保存动作，然后直接 `extend(actions)`。当预测长度为 8、执行窗口设为 2 时，Python 队列会丢弃最早的 6 个动作，实际执行序号 6、7。

本仓库在入队前调用 `select_action_chunk`，使用最早的 N 个动作。序号 `0,1,2,3,4,5,6,7` 在窗口为 2 时得到 `0,1`，然后重新查询模型。窗口为 8 时保持原有顺序。空动作列表转为明确的评估异常，避免含糊的空队列报错。

测试运行真实 `run_episode` 函数，替换外部模型与物理环境，验证实际送到环境的动作及再次查询时机。该测试证明调度逻辑，不证明机器人任务成功率。

## 2. 参数检查

`validate_evaluation_config` 不导入模型库，接受本地 checkpoint 路径或 Hugging Face ID，并在模型权重加载前检查：

- 模型与任务集名称，非空 checkpoint；
- 执行窗口在 1 到模型动作块大小之间；
- 试验次数、图像尺寸、输入图像数为正整数；
- 等待步数非负，seed 在 NumPy 支持的 32 位范围；
- 两种量化方式不能同时开启；L1 和 diffusion 动作头不能同时开启；
- image_aug checkpoint 的中心裁剪，以及 diffusion 步数。

错误用 `ValueError` 表达，不受 Python 优化模式影响。不在本地检查 Hugging Face ID 是否存在，也不验证 checkpoint 内容是否完整；这些检查仍由原模型加载逻辑完成。

## 3. JSON 的统计规则

新增 `eval_support.py` 在每个已记录 episode 后，先写同目录临时文件，再用 `os.replace` 替换 JSON。视频编码在记录之后进行，所以编码失败不会抹去已经保存的该轮结果。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 当前报告结构版本为 1 |
| `data_kind` | `evaluation` 为评估入口生成；`synthetic_example` 为人工演示 |
| `status` | `running`、`completed` 或 `aborted` |
| `config` | checkpoint、seed、任务集和评估参数快照 |
| `episodes` | 每轮任务编号、原始 episode 索引、任务描述及 outcome |
| `summary` | 整体计数与成功率 |
| `tasks` | 各任务的计数与成功率 |

`successes` 为成功；`failures` 为正常结束但未成功；`errors` 为 episode 执行循环捕获到的异常；`skipped` 为未通过专家示范检查而跳过的条目。

`completed = successes + failures + errors`，`success_rate = successes / completed`。异常属于已经尝试的失败，不从分母移除；跳过项不计入分母。没有完成任何尝试时，成功率为 JSON `null`，CSV 留空，终端显示 `N/A`。

`completed` 状态表示评估流程正常结束，并不表示所有任务成功。发生模型执行异常的轮次仍可按原有行为继续评估。外层异常或 Ctrl+C 使报告标记为 `aborted`，保留此前已落盘的轮次；当前尚未结束的轮次不计入分母。进程被强制结束或断电时无法执行收尾逻辑，最近一次有效 JSON 可能仍为 `running`，并不具备自动断点续评功能。

CSV 工具从 `episodes` 重新计算，避免错误的缓存总数影响导出。不同 checkpoint、seed、任务集、试验次数、裁剪配置的结果需要分别检查，不能只看整体百分比。

## 4. 验证与边界

执行：`python -m unittest discover -s tests -v`。

本地 22 项回归测试覆盖动作顺序、再次查询、无效参数、`python -O`、异常记录、跳过示范、零分母、重复 episode、原子替换失败、Ctrl+C、日志关闭、视频编码失败、JSON/CSV 命令行流程和错误输入。集成测试通过替换模型、仿真等外部依赖，运行本仓库实际评估函数。

GitHub Actions 配置为 Ubuntu / Windows × Python 3.10 / 3.12，并执行同一组测试、语法检查、人工样例生成与 CSV 导出。运行结果以仓库 Actions 实际状态为准。

没有下载大型模型或数据集，没有进行训练、完整 LIBERO 仿真、真机实验或性能对比。本仓库不据此宣称论文指标提升。
