# 幼蜱分配与生物响应决定感染若蜱产出

论文 *Larval allocation and biological responses shape infected-nymph production* 的分析代码，作者为丁煦东、孟新柱。

## 安装

使用 Python 3.10 或更新版本，在本目录运行：

```bash
python -m pip install '.[full]'
```

计算所用的依赖版本见 `requirements.txt`。

以下命令使用精确分数计算 A/B 例子，不需要观测数据：

```bash
python -m lyme_paper1 demo
```

## 数据与分析

将配套数据解压为 `paper1-inputs`，放在本代码目录旁。数据来源和变量定义见[数据说明](docs/DATA.md)。

```bash
python -m lyme_paper1 primary --data ../paper1-inputs --output outputs/primary
python -m lyme_paper1 reproduce shapes --inputs ../paper1-inputs --output outputs/shapes
python -m lyme_paper1 reproduce paired-observations --inputs ../paper1-inputs --output outputs/paired
python -m lyme_paper1 reproduce infection-forecasts --inputs ../paper1-inputs --output outputs/infection
python -m lyme_paper1 reproduce measurement --inputs ../paper1-inputs --output outputs/measurement
python -m lyme_paper1 reproduce anchor-sensitivity --inputs ../paper1-inputs --output outputs/anchors
```

这些命令分别计算主预测、宿主负担分布、配对观测汇总、感染预测和测量条件下的产出范围。每项分析使用单独的输出目录；`reproduce` 生成的表格和图位于 `outputs/<分析名称>/workspace/lyme/` 下。其余分析的命令见[分析命令表](docs/COVERAGE.md)。

感染预测根据观测计数拟合感染模型，再与数据中提供的若蜱丰度预测分布结合，计算感染若蜱密度。文献模型已拟合的预测值也包含在数据中，用于比较。

从两份 Cary 来源表生成主预测数据的方法见[英文说明](README.md)，主要数值见[预期结果](docs/VALIDATION.md)。
