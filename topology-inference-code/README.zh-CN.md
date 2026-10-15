# 设备模型拓扑关系推理代码包

英文说明见 [README](README.md)。本包只包含算法源码、公开参数说明、依赖及
完全程序生成的模拟示例；不包含真实厂站清单、连接关系、参考库、训练权重、
原始数据划分或预测日志。

## 最快运行

建议使用 Python 3.12，在本目录建立并激活虚拟环境后执行：

```sh
python -m pip install -r requirements.txt
python -m topology_inference demo --output outputs/synthetic-demo --epochs 2
python -m unittest discover -s tests -v
```

示例会生成虚构厂站，实际执行基础预测、字符及分层网络、规则与模板、母线候选
预测网络和历史匹配。两轮训练只用于检查流程，输出指标不是论文实验结果。
运行目录必须是尚不存在的新目录。

## 本地数据与保密边界

输入格式见 [字段说明](docs/data_schema.md)。训练和验证需要连接标签，推断只
保留设备属性；历史参考中的已知连接仅来自训练集合。同一物理厂站的不同导出
须使用相同的 `group_id`，不得跨训练、验证和评价集合。


本包提供算法实现与运行接口；真实数据及其具体划分不公开，因此外部读者不能
仅靠此包复现论文中的精确指标。论文配置记录与便携运行入口的区别已在
[实现说明](docs/implementation_notes.md) 和 [实验协议](docs/paper_protocol.md)
中说明。原论文结果没有嵌入模拟程序，也不会被模拟结果替换。


