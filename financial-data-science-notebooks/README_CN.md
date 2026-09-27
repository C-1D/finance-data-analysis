# Financial Data Science 项目中文学习指南

## 1. 项目要解决什么问题

这个项目把金融市场数据和宏观经济数据转化为可分析的统计结果。主要训练四种能力：

- 获取和整理真实金融数据
- 用统计学描述收益率、波动率和相关性
- 用计量经济学模型解释和预测时间序列
- 用机器学习、自然语言处理和风险模型扩展分析

## 2. 目录如何阅读

建议按以下顺序打开原项目 Notebook：

1. `notebooks/1.1_stock_prices.ipynb`：股票价格与收益率的统计特征
2. `notebooks/2.2_regression_diagnostics.ipynb`：回归模型与诊断
3. `notebooks/2.3_time_series.ipynb`：通胀、工业生产等经济时间序列
4. `notebooks/3.4_value_at_risk.ipynb`：风险价值 VaR
5. `notebooks/1.3_fama_french.ipynb`：因子模型与线性回归

原始 Notebook 仓库：
https://github.com/terence-lim/financial-data-science-notebooks

## 3. 第一个实操作业：股票收益率分析

### 步骤一：准备环境

安装 Python 3.11 或更高版本，然后在终端运行：

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install jupyterlab pandas numpy matplotlib yfinance statsmodels scikit-learn
jupyter lab
```

### 步骤二：建立数据表

选择两只股票和一个市场指数，下载日收盘价。不要直接比较价格，要先计算对数收益率：

```python
import numpy as np
import yfinance as yf

tickers = ["AAPL", "MSFT", "^GSPC"]
prices = yf.download(tickers, start="2020-01-01", end="2025-01-01", auto_adjust=True)["Close"]
returns = np.log(prices / prices.shift(1)).dropna()
```

### 步骤三：做描述统计

```python
summary = returns.describe().T
summary["annualized_return"] = returns.mean() * 252
summary["annualized_volatility"] = returns.std() * np.sqrt(252)
summary
```

解释：

- 年化收益率反映平均增长速度
- 年化波动率反映风险
- 最大值和最小值帮助识别极端波动
- 偏度和峰度可以进一步判断收益分布是否偏斜、厚尾

### 步骤四：画图

```python
import matplotlib.pyplot as plt

(returns.cumsum()).plot(figsize=(12, 6), title="Cumulative Log Returns")
plt.ylabel("Cumulative log return")
plt.show()
```

### 步骤五：做相关性分析

```python
import seaborn as sns

sns.heatmap(returns.corr(), annot=True, cmap="coolwarm")
plt.title("Return Correlations")
plt.show()
```

相关性高，说明资产可能同时受到相似的市场因素影响；相关性低，可能有助于分散风险。相关性不是因果关系。

## 4. 第二个实操作业：回归与因子解释

选择一只股票作为被解释变量，把市场指数作为解释变量：

```python
import statsmodels.api as sm

y = returns["AAPL"]
X = sm.add_constant(returns["^GSPC"])
model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print(model.summary())
```

重点观察：

- 市场 beta
- 截距项 alpha
- 置信区间
- R-squared
- HAC 稳健标准误

## 5. 第三个实操作业：风险价值 VaR

先使用历史模拟法计算 95% VaR：

```python
var_95 = returns["AAPL"].quantile(0.05)
print("Daily 95% historical VaR:", var_95)
```

这个数字表示，在历史分布假设下，最差 5% 的日收益率大约低于该值。它不是保证，也不能替代压力测试。

## 6. 每次学习必须提交的结果

每完成一个 Notebook，在自己的仓库中记录：

1. 数据来源和时间范围
2. 使用的统计方法
3. 一张关键图表
4. 两条主要发现
5. 一个局限性
6. 下一步改进计划

## 7. 你的第一个作品集版本

完成后，把标题改成：

**Financial Data Science: Stock Returns, Regression and Risk Analysis**

README 中用英文写摘要，分析正文可以保留中文。这样既能展示英文项目能力，也方便你理解统计方法。
