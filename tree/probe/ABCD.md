# 单 seed ABCD 协议

四组各拟合一次，seed=42，不做超参数搜索。采用原 pilot 的同一 100 题、
80/10/10 划分、轨迹和位置，共 1180 个位置。所有组的候选都是本地 vLLM 一致的
Gumbel-top-k 无放回 sample，k=4、branch_temperature=1。

初始化复用冻结模型产生的 hidden states，通过同一 checkpoint 的 LM head 重新计算候选分布。
不能复用旧 top-k 的整组 U；但相同前缀、相同候选 token 和相同续写配置下，
旧的独立续写结果仍可作为该候选的初始观测使用（保存 reused_from_topk 和原始 seed）。
新候选重新生成。独立确认不复用任何初始观测。

| 组 | 训练数据 | 目标 | 输入 |
|---|---|---|---|
| A | 初始观测的 screened positive/background | 类别平衡 BCE + L2 | RMS hidden → PCA128 |
| B | 初始与独立确认类别一致的位置 | 同 A | 同 A |
| C | 同 B | B + 同题 pairwise logistic 排序 | 同 A |
| D | 同 C | 同 C | PCA hidden + 概率与长度特征 |

初始观测每候选训练 4 次、val/test 8 次，与旧 pilot 保持一致。
训练集每个位置再对相同候选独立续写 8 次作为确认，不增加训练题目，不改 val/test 标签。
正例规则：U_raw>1e-12、max(q)-min(q)>=0.5、至少一个候选成功至少两次。
背景规则：U_raw<=1e-12 且 max(q)-min(q)<=0.25。其他位置暂不参与训练。
B 只保留两批都判为 positive 或两批都判为 background 的位置；不把确认数据与初始数据合并后再确认。
这个条件可以减小偶然差异，但也可能漏掉低概率的有效位置，不代表严格统计显著性。

排序对只来自训练题：confirmed positive 对 confirmed background，确认 U 差至少 0.01；
优先同轨迹，每题最多 32 对，每题排序 loss 总权重相同，排序权重 1。
如果没有符合条件的 pair，会明确记录 rank_pairs=0，此时不能声称排序训练生效。

所有组共享仅在训练集拟合的 PCA 与标准化；固定 L2=1.0、零初始化线性 head、LBFGS 至多 200 次。
类别通过 loss 权重平衡，不做多 seed 或多次训练。
D 额外输入 entropy、top1 log-prob、top1/top2 log-prob 间隔、四个候选排序后的 log-prob、
候选总概率质量、已生成比例和剩余预算比例。没有引入结果或未来 token 信息。

验证集保持所有位置，报告指标并校准各组自己的 80% 分位阈值。
固定所有 checkpoint 后，各组只对既有测试集评估一次；它不是新的盲测题库。
比较相同每题 3/12 选择预算下的 U、proxy precision/recall、同题 Spearman，
同时报告 entropy、一个 seed42 随机排序、随机选择的精确期望。
bootstrap 用于区间估计，不是多 seed 训练。

入口：`bash tree/scripts/run_probe_abcd.sh`。输出目录默认
`/inspire/hdd/global_user/weilongxuan-253108120168/probe_runs/math7b-sample-abcd-seed42`。
`status.json` 保存当前阶段。GPU 标注按题目断点恢复；四组产物位于 `abcd/`。
不会改动旧 top-k pilot 或其 checkpoint，不运行 GRPO。
