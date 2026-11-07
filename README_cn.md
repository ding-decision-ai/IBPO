# IBPO：比较与纠正的强化学习代码

以下命令均从本目录根部（能看到 `setup.py`、`verl/`、`recipe/` 的目录）执行，路径相对此工作目录。不要进入 `verl/` 子目录后运行命令。

## 1. 安装环境

### 新建环境

先安装可用的 NVIDIA 驱动和 uv，然后执行：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --only-binary vllm,torch,xformers -r recipe/ibpo/requirements-reproduction.txt
uv pip install --no-deps -e .
```

### 复用已安装的 GPU 环境

如果已经安装了上述兼容的 CUDA PyTorch 和 vLLM，先激活该环境，再执行下面一条安装当前代码，无须重新安装或编译 vLLM：

```bash
uv pip install --no-deps -e .
```

新建或复用环境后，均执行：

```bash
export PYTHONPATH="$PWD"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

uv pip check
python -c "import torch, vllm, verl; print(torch.__version__, vllm.__version__, verl.__file__); assert torch.cuda.is_available()"
```

## 2. 模型

训练主响应和纠正响应使用同一冻结旧策略，所有纠正采样完成后进行一次联合 optimizer 更新。

## 3. 准备训练数据

论文使用 DeepMath-103K 和 OpenCodeReasoning。本包不附原训练集和原始混合划分；读者须准备有使用权限的数据并转换为以下 verl Parquet 格式。只需问题和用于终止验证的参考答案/测试用例，不把原数据集的长回复直接当 RL 训练目标。

数学记录示例：

```json
{"prompt":[{"role":"user","content":"Compute 1+1. Put the final answer in \\boxed{}."}],"data_source":"DigitalLearningGmbH/MATH-lighteval","reward_model":{"style":"rule","ground_truth":"2"},"extra_info":{"index":0}}
```

这里 `data_source` 用来选择数学终止验证器，不代表训练数据一定来自该同名数据集。该验证器提取最终 boxed 答案；应按实际数据规范统一模型提示与 ground_truth。原始 DeepMath 数据源名不能直接交给尚未注册的 verifier。

把上述结构的 JSONL 转换为 Parquet（正式运行应提供足够记录以组成完整原题批量）：

```bash
mkdir -p data
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

for split in ("train", "val"):
    rows = [json.loads(line) for line in Path(f"data/{split}.jsonl").read_text().splitlines() if line.strip()]
    assert rows and all({"prompt", "data_source", "reward_model"} <= row.keys() for row in rows)
    pd.DataFrame(rows).to_parquet(f"data/{split}.parquet", index=False)
PY
```

`val` 应为训练来源的独立留出数据。代码任务需提供可执行的完整测试用例及对应 verifier；只有参考程序字符串不足以执行正确性验证。可通过 `reward.custom_reward_function.path/name` 接入自定义二元奖励函数，返回 `{"score": 0或1, "acc": 0或1}`；未知来源会报错而不会自动猜测评分规则。

## 4. 运行

先准备模型到 `models/Qwen3-32B`，训练和留出集到上述 `data/`。下面是 32 卡集群例子；4 节点×8 卡示例。多机应在每个节点安装相同环境，使用相同代码、模型和数据路径。

```bash
# 主节点执行，把 IP 替换为机器间可达的实际地址。
ray start --head --node-ip-address=主节点IP --port=6379 --num-gpus=8

# 其余三个节点分别执行。
ray start --address=主节点IP:6379 --num-gpus=8
```

在主节点的公开代码根目录执行：

```bash
export RAY_ADDRESS=auto
python -m recipe.ibpo.main_ibpo --config-name=ibpo_trainer \
  actor_rollout_ref.model.path=models/Qwen3-32B \
  data.train_files=data/train.parquet \
  data.val_files=data/val.parquet \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  trainer.nnodes=4 trainer.n_gpus_per_node=8 \
  trainer.total_epochs=1 trainer.val_before_train=true \
  trainer.test_freq=100 trainer.save_freq=100 \
  trainer.resume_mode=disable trainer.logger='[console,file]' \
  trainer.default_local_dir=checkpoints/ibpo-fixed \
  actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra,hf_model]'
```
训练可恢复的分片和 HF 模型保存在 `checkpoints/ibpo-fixed/global_step_*/actor/`。

Next 模型使用 `--config-name=ibpo_qwen3next`，同时把模型路径换为自己的 Next 模型目录。并行布局须根据实际集群确定。

变体：ratio 增加 `ibpo.variant=ratio ibpo.shaping_weight=0.4`；mask 增加 `ibpo.variant=mask ibpo.mask_fallback=unmasked`，或明确选 `skip`。辅助训练可用 `ibpo.aux_weight=0` 关闭；Gc=1 时组内辅助优势为零。

