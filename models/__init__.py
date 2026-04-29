# models/__init__.py
from .basic_template import TrainTask

# 延迟导入避免循环依赖
def get_model_dict():
    # -------------------------------------------------
    # 临时注释掉 try/except，让错误直接爆出来！
    # -------------------------------------------------
    # try:
    from .LPSDiff.LPSDiff import LPSDiff
    return {
        'LPSDiff': LPSDiff,
    }
    # except ImportError as e:
    #     print(f"!!! 致命错误: 导入 LPSDiff 失败 !!!")
    #     print(f"错误详情: {e}")
    #     # 再次抛出异常，不要吞掉它，否则 main.py 无法知道具体原因
    #     raise e

model_dict = get_model_dict()