# 筛选样本分类 probe：2026-09-17 远端实验

代码：`screened_classifier.py`。复用已有 hidden states 和 main labels，不新增 rollout。
所有拟合和 PCA 在 CPU 执行。原 MLP、线性回归模型与原始标签保持不变。

## 固定的筛选规则

- 正样本：`utility_raw > 1e-12`、候选成功率极差 `max(q)-min(q) >= 0.5`，
  且至少一个候选有至少两次成功续写。
- 背景：`utility_raw <= 1e-12` 且成功率极差不超过 0.25。
- 其他位置标为 uncertain，在主实验中排除，但在验证和测试中仍保留。
- 主实验保留全部正样本和背景，通过 loss 权重让两类总权重各为 50%，
  并非丢弃绝大多数背景。小样本全错依然只是背景代理标签，不是可靠的真实负例。

训练集：940 个位置 = 112 positive + 760 background + 68 uncertain。
验证集：120 个位置，8 个符合新规则的 positive。
测试集：120 个位置，6 个符合新规则的 positive。
新规则比 U>0 严格，因此 positive 数目不等于之前报告中的正 U 数目。

## 模型与对照

每个 hidden state 先做自身 RMS 归一化，再用仅在训练集拟合的 PCA 投影和尺度归一化。
分类器为零初始化的 logistic regression，LBFGS 优化显式 L2 正则的二元交叉熵。

有界实验网格：PCA 32/128 维，L2=0.1/1.0，三种训练模式共 12 个配置：

1. screened_balanced：排除 uncertain，正/背景 loss 总权重平衡。
2. all_balanced：保留全部位置，非 positive 都作为代理负例，类别权重平衡。
3. all_natural：保留全部位置，保持自然类别比例。

只在 screened_balanced 的四个配置中，根据验证集 proxy AP 选取主模型。
输出为排序 logit，不应当作校准后的“值得分叉概率”。
验证集 80% 分位数保存为固定阈值，另外报告离线同题选 3/12 个位置的相同预算指标。
后者不是实际在线 tree decoding 实验。

## 实际结果

选中 `screened_balanced_pca128_l21.0`，分类头 129 个可训练参数，PCA/归一化为固定 buffer。

| 指标 | 验证 probe | 验证 entropy | 测试 probe | 测试 entropy |
|---|---:|---:|---:|---:|
| Proxy AP | 0.39134 | 0.04550 | 0.07133 | 0.05063 |
| 同题同预算平均 U | 0.00855 | -0.00184 | 0.00285 | 0.00189 |
| 同题 U Spearman | 0.04917 | -0.01042 | 0.09126 | -0.15046 |

测试集相对 entropy 的同题平均 U 差值为 0.000958，题目 bootstrap 95% CI 为 [0, 0.002167]。
10 道题中 3 道改善、7 道持平；这些数值没有证实统计上稳定的优势。
固定验证阈值在测试集选出 22/120 个位置，平均 U=0.003982。

验证集筛选组与不筛选对照的同预算收益接近，因此不能把相对旧模型的改善
单独归因于筛选；PCA、正则化、分类目标也一起发生了变化。
验证到测试 AP 明显下降；测试只有 6 个代理正例，仍需独立新题目确认泛化。
这里的测试集此前已用于旧模型评估，不是全新的盲测集。

## 产物与复现

远端结果目录：
`/inspire/hdd/global_user/weilongxuan-253108120168/probe_runs/math7b-topk-pilot/models/screened_classifier_v1/`

- `probe.pt`：选中的投影、归一化、分类器、阈值及实验配置。
- `screening.json`：筛选计数。
- `results.json`：所有配置的训练/验证指标、对照及主模型选择结果。
- `test.json`：冻结模型后的一次测试评估。
- `validation_predictions.csv` / `test_predictions.csv`：逐位置结果。
- 本地指标副本：`tree/probe/experiments/screened_classifier_v1/`，不包含 checkpoint。

```bash
cd /inspire/hdd/global_user/weilongxuan-253108120168/verl0.6.0
python3 tree/probe/screened_classifier.py fit \
  --work-dir /inspire/hdd/global_user/weilongxuan-253108120168/probe_runs/math7b-topk-pilot \
  --name screened_classifier_v1 --threads 4
python3 tree/probe/screened_classifier.py evaluate \
  --work-dir /inspire/hdd/global_user/weilongxuan-253108120168/probe_runs/math7b-topk-pilot \
  --name screened_classifier_v1 --threads 4
```

远端 Python 编译检查、12 配置训练/验证、checkpoint 加载及测试评估均成功完成。
未进行独立续写重标注、GRPO 训练或在线分叉接入。
