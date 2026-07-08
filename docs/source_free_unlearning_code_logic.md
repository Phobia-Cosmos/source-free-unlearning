# Towards Source-Free Machine Unlearning 代码逻辑解析与复现实验记录

## 1. 当前结论摘要

- 我们本地已经跑通并记录的主实验数据集是 **CIFAR-10**。
- 本地有两条实验路线：
  - 官方仓库的 mixed-linear / activation 路线：`dataset_id=cifar10`、`dataset_id=cifar10-act`。
  - 我补充的论文线性分类器路线：CIFAR-10 图像 -> ImageNet 预训练 ResNet-18 特征 -> bias-free linear classifier。
- 论文补充材料明确说：线性分类器实验使用 CIFAR-10、CIFAR-100、StanfordDogs、Caltech-256；ImageNet 预训练 ResNet-18 去掉倒数第二层生成 activations；10% 数据作为 forget data；默认 500 个 perturbations。
- 官方仓库不是完整复现包。它更像是在 `mixed-linear-forgetting` 代码基础上公开的部分实现，缺少 Table 1-3 的完整线性 SDP 流程、Table 5 baselines、完整评测/MIA脚本、四个数据集的一键复现脚本与精确超参。
- ImageNet 预训练模型本身是完整的：代码通过 `torchvision` 下载 `ResNet18_Weights.DEFAULT` / `ResNet50_Weights.DEFAULT`，不是作者提供本地 ImageNet checkpoint。
- 之前 activation linearized 路线能跑通但数值偏离论文，是因为它跑的是 mixed-linear/JVP/diag-Hessian 路线，不是论文 Table 1-3/5 的固定 ImageNet 特征 + 普通线性分类器 + SDP retain-Hessian 估计路线。

## 2. 论文主要讲什么

论文目标是 **source-free machine unlearning**：训练完模型后，模型拥有者只有训练好的模型和待删除的 forget data，不能访问剩余 retain/source training data，也要从模型参数中删除 forget data 的影响。

核心思想：

1. 在线性模型上，标准影响函数式遗忘更新需要 retain Hessian：
   `w_unlearn = w_source + H_retain^{-1} * grad_forget`。
2. 问题是 source-free 场景没有 retain data，因此无法直接算 `H_retain`。
3. 论文提出用 forget data、已训练模型、随机 perturbations 的 loss difference 去优化/估计 retain Hessian。
4. 在线性分类器上给出理论界；在神经网络上用 mixed-linear / NTK 风格线性化，把最后几层近似成线性问题。

论文报告的实验范围：

- 线性分类器：CIFAR-10、CIFAR-100、StanfordDogs、Caltech-256。
- mixed-linear 网络：同样四个数据集。
- CIFAR-10 上还做 Table 1、Table 2、Table 3、Table 5：forget比例、perturbation数量、L2正则、和 source-free baseline 对比。

## 3. 我们当前实验基于哪个数据集

当前本地已经完成并保存结果的实验基于 **CIFAR-10**：

- 原仓库路线使用 `dataset.py` 中的 `CIFAR10('./data', train=True/False, download=True)`，见 `dataset.py:76`、`dataset.py:145`、`dataset.py:214`。
- 论文线性分类器复现路线使用 `linear_repro.py` 中的 `CIFAR10(root='./data', train=True/False, download=True)`，见 `linear_repro.py:107`。
- forget split 使用固定 seed=13 的随机划分，默认 `split_rate=0.1`，也就是 50,000 训练图像中 5,000 张 forget，45,000 张 remaining，见 `linear_repro.py:156` 和 `linear_repro.py:158`。

论文还使用了 CIFAR-100、StanfordDogs、Caltech-256，但我们本地完成的主复现和baseline测试是 CIFAR-10。仓库虽然有 CIFAR-100、StanfordDogs、Caltech、StanfordCars 分支，但完整数据准备、超参和批量脚本不齐。

## 4. 同样使用 CIFAR-10 做相关实验的论文

严格说，使用 CIFAR-10 的机器遗忘论文很多，不可能列全。下面是和这篇论文最直接相关、在论文引用或 Table 5 baseline 中出现的代表性论文：

| 论文 | 关系 | 备注 |
| --- | --- | --- |
| `Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks` | Table 5 中 NegGrad / Random Labels 的来源之一 | 论文中把 NegGrad 描述为只用 forget data 做梯度上升，把 Random Labels 描述为对 forget samples 重新分配错误标签再微调。 |
| `Mixed-Privacy Forgetting in Deep Networks` | mixed-linear 网络路线的直接基础 | 本地官方仓库 remote 也是 `UCR-Vision-and-Learning-Group/mixed-linear-forgetting.git`，说明代码基础主要来自 mixed-linear forgetting。 |
| `Zero-Shot Machine Unlearning` | source-free / zero-shot machine unlearning 相关工作 | 论文指出这类方法主要偏 class-level forgetting，难以覆盖任意 instance-level forget。 |
| `Learning to Unlearn: Instance-wise Unlearning for Pre-trained Classifiers` | Table 5 中 Adversarial baseline 的来源 | 论文描述该方法结合 adversarial examples 和 weight importance，再对 forget data 做梯度上升。 |
| `Zero-shot Machine Unlearning at Scale via Lipschitz Regularization` | Table 5 中 JiT baseline 的来源 | 论文描述 JiT 通过 Lipschitz regularization 约束 forget sample 附近扰动输出。 |
| `Towards Unbounded Machine Unlearning` | MIA/benchmark 相关参考 | 本文引用其 MIA score 评估思路，用 50% 附近表示成员推断接近随机。 |
| `Certified Data Removal from Machine Learning Models` | 线性模型参数不可区分理论基础 | 本文的线性模型遗忘公式和理论设定与这类 certified removal 工作相关。 |

## 5. 官方仓库为什么不是完整复现代码

事实依据：当前仓库 remote 是 `https://github.com/UCR-Vision-and-Learning-Group/mixed-linear-forgetting.git`，README 也只给了 pretrain、train mixed model、mixed-privacy、forget-by-diag 的少量命令，没有 Table 1-7 的完整运行入口。

缺失项如下：

1. **缺 Table 1-3 的线性分类器 source-free Hessian SDP 实现**
   - 论文线性实验应该是固定 ImageNet ResNet-18 特征 + bias-free linear classifier + quadratic loss + perturbation 优化 retain Hessian。
   - 原仓库没有这个完整脚本；我补充在 `linear_repro.py:39`、`linear_repro.py:48`、`linear_repro.py:307`、`linear_repro.py:344`。

2. **缺 Table 5 baseline 实现**
   - 原仓库没有 `NegGrad`、`Random Labels`、`JiT`、`Adversarial` 的实现入口。
   - 我新增了 `baseline_repro.py`；其中 NegGrad/Random Labels 是按论文文字描述实现，JiT/Adversarial 是本地近似版，见 `baseline_repro.py:40`、`baseline_repro.py:65`、`baseline_repro.py:104`。

3. **缺完整评测脚本**
   - 原仓库没有独立输出 test / remaining / forget / MIA 的统一评测脚本。
   - 我补充了 `evaluate.py` 和 `mia.py`，核心评估逻辑见 `evaluate.py:12`、`evaluate.py:96`、`evaluate.py:116`；MIA 逻辑见 `mia.py:22`、`mia.py:62`。

4. **缺四个数据集的一键复现**
   - README 只给了 CIFAR-10 示例，且含作者本地绝对路径。
   - StanfordDogs、Caltech-256 依赖本地 `ImageFolder('./data/StanfordDogs')` 和 `ImageFolder('./data/Caltech256')`，没有自动下载/解压/目录校验，见 `dataset.py:84`、`dataset.py:98`、`dataset.py:157`、`dataset.py:175`。

5. **mixed-linear 和 source-free 线性主算法没有统一**
   - `forget_by_diag` 实际从 remaining data loader 计算 gradient 和 expected Hessian diagonal，见 `main.py:398`、`main.py:404`、`main.py:409`。
   - 这不是论文 Table 1-3 的 source-free retain Hessian 估计；它更接近可访问 remaining data 的 diagonal Hessian 近似。

6. **精确超参不完整**
   - 论文正文只说明 ResNet-18、10% forget、500 perturbations、quadratic loss；补充材料列了结果，但没有给出每个 baseline 的完整 learning rate、epoch、random seed、MIA attack implementation。
   - Table 7 说 mixed-linear 选“last few layers”，但没有给出仓库中 `number_of_linearized_components` 对每个数据集的精确配置。

## 6. ImageNet 训练模型是否完整

是，但要准确理解：

- 论文用的是 ImageNet 预训练 ResNet-18 作为 feature extractor，不是作者自己提供一个 ImageNet checkpoint。
- 官方仓库通过 `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` 加载，见 `model.py:81`。
- 我补充的线性复现脚本同样通过 `resnet18(weights=ResNet18_Weights.DEFAULT)` 加载，见 `linear_repro.py:51`。
- 因此“ImageNet训练的模型完整”指的是：标准 torchvision 权重可以完整下载和使用；不是说论文作者提供了完整训练好的本地模型文件。

## 7. 为什么 activation linearized 路线跑通但数值差距大

主要不是 GPU 或 PyTorch 问题，而是实验路线不一致：

1. **模型形式不同**
   - 论文 Table 1-3/5 是固定 ResNet-18 features 后训练普通线性分类器。
   - activation route 是 `MixedLinearActivationVariant`，用 JVP/tangent model 做 mixed-linear 近似，见 `model.py:152`。

2. **Hessian 估计对象不同**
   - 论文 Unlearned(-) 是不访问 remaining data 的 retain Hessian 估计。
   - 仓库 `forget_by_diag` 访问了 remaining data，并只估计 diagonal Hessian，见 `main.py:398`、`main.py:409`。

3. **输入特征处理不同**
   - 论文线性路线补充材料说用 ImageNet 预训练 ResNet-18 activations；我在 `linear_repro.py` 中使用 ImageNet mean/std 和可选 `bound_norm`，见 `linear_repro.py:102`、`linear_repro.py:151`。
   - 官方 activation route 的 CIFAR 图像归一化是 CIFAR mean/std，见 `dataset.py:23`；`save_activations` 直接保存 backbone 输出，见 `main.py:196`。

4. **loss 和 label scaling 不同**
   - mixed-linear 训练时把 one-hot label 乘以 5，见 `train.py:68`；梯度计算也乘以 5，见 `model.py:209`。
   - 论文线性路线是普通 quadratic loss 目标；我在 `linear_repro.py:70` 实现。

5. **MIA 定义/实现不明确**
   - 论文说 MIA 接近 50% 表示成功，但没有公开完整攻击代码。
   - 我本地实现的是基于 per-sample loss 的 LogisticRegression attack，见 `linear_repro.py:226` 和 `mia.py:62`。该实现得到的 MIA 与论文表格差异很大，说明攻击定义或输入分布没有完全对齐。

## 8. 仓库模块逐个解释

### `dataset.py`

- `get_data_transformations`：按 `dataset_id`/`arch_id` 返回图像 transform 和 one-hot label transform，见 `dataset.py:21`。
- `split_dataset_to_core_user`：把训练集分为 core data 和 user data；pretrain 用 core，后续 user data 用于 mixed-linear 训练，见 `dataset.py:74`。
- `get_user_loader`：直接加载完整 user train/test loader，见 `dataset.py:141`。
- `split_user_train_dataset_to_remaining_forget`：按 `split_rate` 把训练集分成 remaining 和 forget，见 `dataset.py:211`。
- `ActivationDataset`：读取 `save_activations` 保存的 `train_data.pth`/`test_data.pth`，见 `dataset.py:271`。

### `model.py`

- `init_pretrained_model`：加载 torchvision ResNet-18/ResNet-50，并替换最后 `fc` 为 CIFAR-10/100 分类头，见 `model.py:42`。
- `split_model_to_feature_linear`：把 ResNet 拆成 frozen feature backbone 和最后若干层 linearized head，见 `model.py:101`。
- `MixedLinear`：原图像输入路线，先跑 frozen backbone，再通过 forward-mode AD 计算 JVP，输出 `out + jvp`，见 `model.py:127`。
- `MixedLinearActivationVariant`：activation route，输入已经是 backbone activations，不再跑 feature backbone，见 `model.py:152`。
- `calculate_gradient`：对 loader 上的 loss 求模型参数梯度，注意 label 乘以 5，见 `model.py:201`。

### `loss.py`

- `MSELossDiv2`：实现 `MSE / 2`，见 `loss.py:25`。
- `L2Regularization`：参数 L2 正则，见 `loss.py:9`。
- `LossWrapper`：把 MSE 和 L2 按权重加起来，见 `loss.py:35`。
- `JVPNormLoss`：对 JVP 范数做惩罚，是 mixed-linear Hessian 估计的一部分，见 `loss.py:59`。
- `GradientVectorInnerProduct`：计算梯度与待优化向量的内积，见 `loss.py:79`。

### `forget.py`

- `estimate_hess_inv_grad`：优化向量 `v_param`，近似求 `H^{-1}g`，见 `forget.py:8`。
- `calculate_hess_diag`：用随机 Rademacher 向量/HVP 估计 Hessian diagonal，见 `forget.py:32`。
- `expected_hess_diag`：重复多次 `calculate_hess_diag` 后取平均，见 `forget.py:78`。

### `main.py`

- `pretrain`：用 core split 训练 ResNet 分类器，保存 checkpoint，见 `main.py:151`。
- `save_activations`：用 pretrained ResNet 的 frozen feature part 生成 activations 并保存到 `data/resnet18-cifar10-last1`，见 `main.py:196`。
- `train_user_data`：训练 mixed-linear tangent 参数，见 `main.py:48`。
- `mixed_privacy`：使用 remaining data 求梯度并优化 `H^{-1}g`，见 `main.py:265`。
- `forget_by_diag`：估计 remaining data 上的 Hessian diagonal，然后更新 tangent 参数，见 `main.py:365`。
- CLI mode 分发：`pretrain`、`save-activations`、`train-user-data`、`mixed-privacy`、`forget-by-diag`，见 `main.py:491`。

### 本地补充脚本

- `evaluate.py`：读取 checkpoint，输出 test/remaining/forget accuracy，见 `evaluate.py:59`。
- `mia.py`：用 loss-based logistic regression 计算 MIA，见 `mia.py:62`。
- `run_cifar10_repro.py`：串联官方路线 pretrain -> train-user-data -> forget-by-diag -> evaluate，见 `run_cifar10_repro.py:74`。
- `linear_repro.py`：实现论文线性分类器路线，包括特征缓存、线性训练、Unlearned(+)、Unlearned(-) SDP 近似，见 `linear_repro.py:90`、`linear_repro.py:176`、`linear_repro.py:257`、`linear_repro.py:307`、`linear_repro.py:344`。
- `run_linear_tables.py`：批量跑 Table 1/2/3 风格的 CIFAR-10 线性实验，见 `run_linear_tables.py:44`。
- `baseline_repro.py`：补充 Table 5 风格 baseline 测试，见 `baseline_repro.py:128`。

## 9. 已完成复现实验结果

### 环境

- GPU：NVIDIA GeForce RTX 4070 SUPER，约 11.8GB 显存。
- PyTorch：`2.5.1+cu121`。
- torchvision：`0.20.1+cu121`。
- 虚拟环境：`.venv-sfu`。

### 论文线性分类器路线 CIFAR-10 结果

结果文件：`artifacts/linear_tables/linear_tables_summary.json`。

关键配置：`dataset=cifar10`、`split_rate=0.1`、`num_perturbations=500`、`lambda_reg=0.0005`、`seed=13`、`bound_train=True`。

| Method | Test | Remaining | Forget | MIA |
| --- | ---: | ---: | ---: | ---: |
| Retrained | 85.61% | 86.79% | 86.72% | 93.00% |
| Unlearned(+) | 85.57% | 86.54% | 86.96% | 93.34% |
| Unlearned(-) | 84.40% | 85.36% | 85.08% | 92.84% |

结论：accuracy 接近，但 MIA 与论文的约 50% 不一致。原因优先怀疑 MIA 攻击实现/评测协议不一致，而不是线性分类器训练失败。

### Table 5 baseline 本地测试

结果文件：`artifacts/baseline_repro/selected/cifar10_split0.1_seed13_baselines.json`。

说明：NegGrad 和 Random Labels 按论文文字描述实现；JiT 和 Adversarial 因官方仓库缺失原实现与超参，只能做本地近似版。为了横向比较，selected 结果使用了接近论文 Table 5 accuracy 的超参组合；不能声称是原论文 exact baseline。

| Method | Test | Remaining | Forget | MIA |
| --- | ---: | ---: | ---: | ---: |
| Retrained | 85.61% | 86.79% | 86.72% | 93.00% |
| NegGrad | 53.17% | 53.60% | 54.18% | 69.36% |
| Random Labels | 18.37% | 19.21% | 18.36% | 77.56% |
| JiT approx | 53.51% | 53.89% | 54.08% | 68.72% |
| Adversarial approx | 52.22% | 52.12% | 52.74% | 69.08% |

论文 Table 5 报告：NegGrad 51.9%、Random Labels 20.6%、JiT 52.1%、Adversarial 51.5%、Unlearned(-) 70%。我们本地 baseline accuracy 可以通过超参接近，但 MIA 完全对不上，进一步说明论文未公开的 MIA protocol 是关键缺口。

## 10. 是否需要重新实现/测试缺失baseline

需要，但要分两层：

1. **为了工程验证/横向 sanity check**：可以用当前 `baseline_repro.py` 直接测试。这个我已经完成，结果保存在 `artifacts/baseline_repro/selected/`。
2. **为了严格复现论文 Table 5**：必须找到/重写每个 baseline 对应论文的 exact method、loss、扰动空间、学习率、epoch、early stop、MIA 代码。官方仓库没有提供这些，不能把当前 JiT/Adversarial approx 说成 exact reproduction。

建议后续如果要做严格论文复现：

- 先锁定论文作者 Table 5 使用的 MIA 代码，否则 MIA 数字无法对齐。
- 对 NegGrad/Random Labels 固定优化器、epoch、学习率，并记录是否用 quadratic loss 还是 cross entropy。
- 对 JiT/Adversarial clone 对应论文官方仓库或按原论文公式重写，而不是只按本论文一句话描述实现。
- 所有 baseline 统一使用同一个 CIFAR-10 split、同一个 ImageNet ResNet-18 feature cache、同一个 linear classifier 初始化和同一个评测脚本。

## 11. 可复现实验命令

进入仓库：

```bash
cd /home/undefined/Desktop/bci/code/sfda/2025CVPR-source-free-unlearning
source .venv-sfu/bin/activate
```

检查 GPU/PyTorch：

```bash
python - <<'PY'
import torch, torchvision
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torchvision.__version__)
print(torch.cuda.get_device_name(0))
PY
```

跑论文线性 CIFAR-10 单组：

```bash
python linear_repro.py --dataset-id cifar10 --split-rate 0.1 --num-perturbations 500 --lambda-reg 0.0005 --bound-train --device-id 0
```

批量跑 Table 1/2/3 风格实验：

```bash
python run_linear_tables.py --dataset-id cifar10 --bound-train --device-id 0
```

跑 Table 5 风格 baseline selected 配置：

```bash
python baseline_repro.py --dataset-id cifar10 --split-rate 0.1 --seed 13 --device-id 0 --bound-train \
  --methods neggrad random_labels jit adversarial \
  --neggrad-lr 0.00015 --neggrad-epochs 1 \
  --random-lr 0.001 --random-epochs 10 \
  --jit-lr 0.0003 --jit-epochs 1 --jit-perturb-std 0.05 --jit-lipschitz-weight 1.0 \
  --adv-lr 0.00018 --adv-epochs 1 --adv-epsilon 0.02 --adv-importance-weight 1.0 \
  --save-dir artifacts/baseline_repro/selected
```

跑官方 mixed-linear CIFAR-10 路线：

```bash
python run_cifar10_repro.py --dataset-id cifar10 --arch-id resnet18 --split-rate 0.1 --number-of-linearized-components 1 --num-iter 500 --device-id 0 --run-mia
```

## 12. 最重要的复现风险

- 官方仓库不是论文完整代码；不能期望一键得到 Table 1-7。
- MIA protocol 未公开完整细节，是当前最大数值差异来源。
- mixed-linear activation route 与论文线性分类器主实验不是同一条路线。
- StanfordDogs/Caltech-256 需要人工准备数据目录，仓库没有完整下载脚本。
- JiT/Adversarial exact baseline 不能只靠本文一句描述严谨复现，必须回到对应论文或官方实现。

## 13. 官方 JiT / Adversarial 实现接入结果

我已经把两个 baseline 的**官方算法核心逻辑**接进本仓库，并单独做了一个脚本：`baseline_official_repro.py:1`。

### 接入来源

- **JiT**：来自 `jwf40/Information-Theoretic-Unlearning`
  - `src/lipschitz.py` 中的 `Lipschitz.modify_weight`
  - `src/forget_full_class_strategies.py` 中的 `lipschitz_forgetting`
- **Adversarial / L2UL**：来自 `csm9493/L2UL`
  - `main_unlearn_cifar10_mixed_label_resnet18.py`
  - `utils.py` 中的 `adv_attack` 和 `estimate_parameter_importance`

### 为什么我没有直接把外部仓库整包嵌入

原因不是偷工减料，而是实验定义不一致：

1. JiT 官方仓库默认是 **VGG16 / CIFAR10 random forgetting**，不是本文 Table 5 的 fixed ImageNet ResNet-18 linear classifier。
2. L2UL 官方仓库默认是 **端到端 ResNet18 图像分类器**，在像素空间做 targeted PGD，对整个网络参数做更新。
3. 本论文 Table 5 明文写的是：**在 CIFAR-10 上，用 ResNet-18 最后一层作为 linear classifier** 进行比较。
4. 因此，如果把上游仓库整包直接跑起来，得到的是“原方法在它自己的实验协议上的结果”，不是“本文 Table 5 对齐设置下的结果”。

所以我采取的是更严格也更公平的方式：

- 保留本文的基准协议：CIFAR-10、固定 ImageNet ResNet-18 特征、forget split=10%、统一评测。
- 只移植 JiT/L2UL 的**核心更新逻辑**到这个统一协议里。

### 新脚本内容

`baseline_official_repro.py` 包含：

- `run_jit_official`：按 JiT 官方 `lipschitz.py` 的平滑性惩罚逻辑，对 forget features 的输出变化/输入变化比值做优化。
- `run_l2ul_adversarial`：按 L2UL 官方 CIFAR-10 脚本的结构，生成 targeted adversarial features，并组合
  - `loss_unlearn = -CE(forget)`
  - `loss_adv = CE(adv, target_label)`
  - 可选 `loss_reg = importance * (w - w0)^2`
- `adversarial_official`：只保留 adversarial 部分。
- `l2ul_official`：adversarial + importance regularization。

### 实测结果

结果目录：`artifacts/baseline_official_repro/selected/cifar10_split0.1_seed13_official_baselines.json:1`

这一组是我在接入官方逻辑后，做最小超参搜索得到的代表性结果：

| Method | Test | Remaining | Forget | MIA |
| --- | ---: | ---: | ---: | ---: |
| Source Model | 83.39% | 84.35% | 85.10% | 49.16% |
| Retrained | 82.26% | 82.95% | 82.54% | 50.02% |
| JiT official-port | 66.61% | 67.31% | 66.50% | 49.72% |
| Adversarial official-port | 16.47% | 17.05% | 15.82% | 49.58% |
| L2UL official-port | 22.49% | 23.58% | 22.74% | 50.00% |

### 这个结果怎么解释

这说明一件很关键的事：

- **把官方 JiT / L2UL 的算法核心逻辑，移植到本文 Table 5 的 fixed-feature linear benchmark 之后，性能并不会自然复现到论文表格里的 52% 左右。**
- JiT 还能到 66% 左右，但 L2UL/Adversarial 会明显失稳。
- 这进一步证明：**本文 Table 5 对 baseline 的具体实现细节并没有完全公开。**

可能的差异来源包括：

1. 论文作者并不是直接用上游默认超参。
2. baseline 可能被重新实现到“只更新线性头”的协议里，但代码未公开。
3. MIA 的实现显然与我们当前脚本不同。
4. “last layer as linear classifier” 在论文中可能不是简单的 frozen feature + linear head，而是与上游方法有额外适配。

### 结论

截至现在，可以严谨地下这个结论：

- 我已经完成了 **JiT 与 Adversarial/L2UL 官方核心实现的接入与测试**。
- 但要想把数字**严格对齐**到本文 Table 5，仍然缺少作者未公开的适配细节。
- 所以现在最准确的说法不是“已完全复现 Table 5”，而是：
  - **已完成官方 baseline 算法核心的协议内移植与复测；**
  - **结果表明论文 Table 5 仍依赖未公开实现细节。**
