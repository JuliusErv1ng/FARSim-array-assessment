FARSim 阵列特性评估

输入 data、响应 responses/default、模型 farsim_waveform_py/models 均位于本目录。
配置中的路径相对于本目录解析，整体移动后无需修改。

在验收文件夹执行 python 运行.py 阵列，可计算单方向示例。
在本目录执行完整角度计算：
python run_rate.py configs/gp80_rate.json
仅检查配置与输入：
python run_rate.py configs/gp80_rate.json --check-only
按相同输入续算：
python run_rate.py configs/gp80_rate.json --resume

完整参数与输出字段见“使用手册.txt”。
