# backtest/rqalpha_runner.py
from rqalpha import run_func
from rqalpha.api import order_percent, update_universe

def init(context):
    # 初始化，定义标的
    context.stock = "000300.XSHG"
    update_universe([context.stock])
    context.has_position = False

def handle_bar(context, bar_dict):
    # 简单策略：首次没有持仓时全部买入
    if not context.has_position:
        order_percent(context.stock, 1.0)
        context.has_position = True

def run_backtest():
    config = {
        "base": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "benchmark": "000300.XSHG",
            "accounts": {"stock": 100000},
        },
        "extra": {
            "log_level": "verbose"
        },
        "mod": {
            "sys_analyser": {"enabled": True, "plot": True}
        }
    }
    # 使用 run_func 运行策略
    run_func(init=init, handle_bar=handle_bar, config=config)

if __name__ == "__main__":
    run_backtest()
