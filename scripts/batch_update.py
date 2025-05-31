# quant_platform/scripts/batch_update.py (示意代码)
import subprocess
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime

# (日志配置...)
logger = logging.getLogger(__name__)
# ... (logging setup) ...

# 定义脚本路径 (相对于项目根目录)
SCRIPTS_DIR = "scripts"  # 或者 os.path.join(PROJECT_ROOT, "scripts")
SYNC_DAILY_PATH = os.path.join(SCRIPTS_DIR, "sync_daily_data.py")
UPDATE_AGGREGATED_PATH = os.path.join(SCRIPTS_DIR, "update_aggregated_data.py")
SYNC_FINANCIAL_PATH = os.path.join(SCRIPTS_DIR, "sync_financial_data.py")
SYNC_DISCLOSURE_META_PATH = os.path.join(SCRIPTS_DIR, "sync_disclosure_data.py")
POPULATE_CONTENT_PATH = os.path.join(SCRIPTS_DIR, "polulate_selected_raw_content.py")  # 使用您的脚本名


def run_script_step(script_name: str, script_path: str, script_args: list = None) -> bool:
    logger.info(f"--- 开始执行步骤: {script_name} ---")
    # ... (使用 subprocess.run 调用脚本，包含错误处理和日志记录，如上一轮所示) ...
    # 返回 True 表示成功, False 表示失败
    # (参照上一轮的 run_script 函数)
    if script_args is None: script_args = []
    command = ['python', script_path] + script_args
    logger.info(f"执行命令: {' '.join(command)}")
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        logger.info(f"{script_path} STDOUT:\n{process.stdout}")
        if process.stderr: logger.warning(f"{script_path} STDERR:\n{process.stderr}")
        logger.info(f"--- 步骤: {script_name} 执行成功 ---")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(
            f"运行 {script_path} 失败 (步骤: {script_name}), 返回码: {e.returncode}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"找不到脚本: {script_path} (步骤: {script_name})。请检查路径。")
        return False
    except Exception as e:
        logger.error(f"运行 {script_path} (步骤: {script_name}) 时发生未知错误: {e}", exc_info=True)
        return False


def full_data_pipeline():
    logger.info("🚀🚀🚀 启动完整数据处理流程 🚀🚀🚀")

    # 步骤 0: (可选) 同步股票列表，如果它不是静态的或由其他方式维护
    # if not run_script_step("同步股票列表", "scripts/sync_stock_list.py"): return

    if not run_script_step("同步日交易数据", SYNC_DAILY_PATH):
        logger.error("关键步骤失败：日交易数据同步。流程中止。")
        return

    if not run_script_step("更新周/月聚合数据", UPDATE_AGGREGATED_PATH):  # 默认模式，非full-rebuild
        logger.error("步骤失败：周/月聚合数据更新。后续步骤可能基于不完整数据。")
        # 可以选择中止或继续，这里选择继续但警告

    if not run_script_step("同步财务数据", SYNC_FINANCIAL_PATH):
        logger.warning("步骤警告：财务数据同步失败。")
        # 根据重要性决定是否中止

    if not run_script_step("同步公告元数据", SYNC_DISCLOSURE_META_PATH):
        logger.warning("步骤警告：公告元数据同步失败。")
        # 根据重要性决定是否中止

    # 只有在公告元数据同步成功后，才尝试填充内容
    # （或者您可以将这个判断逻辑放在 run_script_step 的调用处）
    # 这里我们假设，即使元数据同步部分失败，仍尝试处理已有的元数据
    if not run_script_step("填充公告正文", POPULATE_CONTENT_PATH):
        logger.warning("步骤警告：填充公告正文失败。")

    logger.info("🎉🎉🎉 完整数据处理流程执行完毕 🎉🎉🎉")


if __name__ == "__main__":
    # (APScheduler 的调度逻辑，如上一轮所示，调用 full_data_pipeline)
    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    scheduler.add_job(full_data_pipeline, "cron", day_of_week='mon-fri', hour=18, minute=0, id='full_data_pipeline_job',
                      replace_existing=True)
    logger.info(
        f"🕒 调度器已启动。完整数据处理流程将在周一至周五的 {datetime.now().replace(hour=18, minute=0, second=0, microsecond=0).strftime('%H:%M')} 执行。")
    try:
        logger.info("服务启动，立即执行一次完整数据处理流程...")
        full_data_pipeline()
    except Exception as e:
        logger.error(f"启动时执行 full_data_pipeline 失败: {e}", exc_info=True)
    scheduler.start()