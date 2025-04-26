# batch_update.py
from apscheduler.schedulers.blocking import BlockingScheduler
from data_update import update_daily_data

# 示例：常用股票池（可根据需要扩展）
stock_list = [
    "000001",  # 平安银行
    "600519",  # 贵州茅台
    "000333",  # 美的集团
    "002594",  # 比亚迪
    "300750",  # 宁德时代
]

def update_all():
    print("📈 批量更新启动...")
    for symbol in stock_list:
        try:
            update_daily_data(symbol)
        except Exception as e:
            print(f"❌ {symbol} 更新失败：{e}")
    print("✅ 所有股票更新完成")

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    # 每天 18:00 运行
    scheduler.add_job(update_all, "cron", hour=18, minute=0)
    print("🕒 调度器已启动（每天18:00更新）...")
    update_all()  # 启动时先执行一次
    scheduler.start()
