# FARSim 阵列特性评估

本仓库提供 FARSim 阵列效率与事例率计算的 Python 源码、配置和中文使用手册。默认参考库采用质子和铁核等权的峰值电场及 Xmax 几何。

完整运行包包含模型、阵列布局、通量和系统响应，请从 [Releases](https://github.com/JuliusErv1ng/FARSim-array-assessment/releases) 下载。解压后进入 `02_阵列特性评估/FARSim`，用 Python 3.11 或 3.12 执行：

```text
python -m pip install -r requirements.txt
python verify_bundle.py
python run_rate.py configs/smoke.json
```

参数设置、正式运行与代码对应关系见[中文使用手册](FARSim/使用手册.txt)。
