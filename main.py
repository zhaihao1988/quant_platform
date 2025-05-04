# main.py (示例，结合了数据处理和报告生成)
import argparse
import logging
import time
from sqlalchemy.orm import Session

# --- 项目模块导入 (根据你的最终目录结构调整) ---
from db.database import Session, engine # 假设包含 SessionLocal 和 engine
from db import models # 导入模型以创建表（如果需要）
from db.crud import get_stock_list_info, retrieve_relevant_disclosures # 添加 retrieve 用于获取价格上下文相关公告
from data_processing.loader import load_announcements_to_scrape
from data_processing.scraper import scrape_and_store_announcements, embed_existing_content # 导入嵌入现有内容的函数
from core.prompting import generate_stock_report
from integrations.email_sender import send_email
from config import settings # 导入配置，可能需要接收邮件地址

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('urllib3').setLevel(logging.WARNING) # 减少 requests 的日志噪音
logging.getLogger('sentence_transformers').setLevel(logging.WARNING) # 减少 embedding 模型的日志噪音
logger = logging.getLogger(__name__)

def create_database_tables():
    """创建数据库表 (如果尚不存在)"""
    logger.info("Creating database tables if they don't exist...")
    try:
        models.Base.metadata.create_all(bind=engine)
        logger.info("Database tables checked/created.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        # 根据情况决定是否退出
        # exit(1)

def get_recent_price_context(db: Session, symbol: str, days: int = 5) -> str:
     """从 StockDaily (如果数据已填充) 获取最近N日价格变动信息"""
     from database.models import StockDaily # 在函数内导入避免循环依赖或放在crud中
     from sqlalchemy import desc
     import pandas as pd

     context = f"最近 {days} 个交易日股价变动及归因：\n"
     try:
          recent_data = db.query(StockDaily).filter(
               StockDaily.symbol == symbol
          ).order_by(
               desc(StockDaily.date)
          ).limit(days).all()

          if not recent_data:
               context += "[数据库中未找到最近股价数据]\n"
               return context

          # 将数据逆序，按时间从早到晚排列
          recent_data.reverse()

          # 使用 Pandas 方便计算和展示 (可选，也可以手动格式化)
          # 需要安装 pandas: pip install pandas
          df = pd.DataFrame([(d.date, d.close, d.pct_change) for d in recent_data], columns=['Date', 'Close', 'PctChange'])
          df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d') # 格式化日期

          first_day = df.iloc[0]
          last_day = df.iloc[-1]
          overall_change = ((last_day['Close'] - first_day['Close']) / first_day['Close']) * 100 if first_day['Close'] else 0

          context += f"  从 {first_day['Date']} (收盘价: {first_day['Close']:.2f}) 到 {last_day['Date']} (收盘价: {last_day['Close']:.2f})，"
          context += f"整体涨跌幅约为 {overall_change:.2f}%。\n"
          context += "  每日简况:\n"
          for _, row in df.iterrows():
               change_str = f"{row['PctChange']:.2f}%" if row['PctChange'] is not None else "N/A"
               context += f"    - {row['Date']}: 收盘价 {row['Close']:.2f}, 涨跌幅 {change_str}\n"

          # 添加一个提示让 LLM 进行归因分析
          context += "  请结合其他信息分析近期股价变动原因。\n"

     except Exception as e:
          logger.error(f"Error retrieving recent price data for {symbol}: {e}")
          context += f"[获取股价数据时出错: {e}]\n"

     return context


def process_stock_data(db: Session, symbol: str):
    """处理单个股票的数据：爬取、存储、嵌入"""
    logger.info(f"--- Processing data for symbol: {symbol} ---")

    # 1. 加载需要爬取的公告
    announcements_to_scrape = load_announcements_to_scrape(db, symbol)
    if announcements_to_scrape:
         logger.info(f"Found {len(announcements_to_scrape)} new announcements to scrape for {symbol}.")
         # 2. 爬取、存储、嵌入新公告
         scrape_and_store_announcements(db, announcements_to_scrape)
    else:
         logger.info(f"No new announcements found requiring scraping for {symbol}.")

    # 3. (可选) 为只有内容没有向量的旧数据生成向量
    # 这个查询可能比较慢，可以考虑单独运行或优化
    # logger.info(f"Checking for existing content needing embedding for {symbol}...")
    # announcements_needing_vector = db.query(models.StockDisclosure).filter(
    #      models.StockDisclosure.symbol == symbol,
    #      models.StockDisclosure.raw_content != None,
    #      models.StockDisclosure.content_vector == None
    # ).all()
    # if announcements_needing_vector:
    #      logger.info(f"Found {len(announcements_needing_vector)} announcements with existing content needing embedding for {symbol}.")
    #      for ann in announcements_needing_vector:
    #           embed_existing_content(db, ann)
    # else:
    #      logger.info(f"No existing content found needing embedding for {symbol}.")


def run_analysis_and_email(db: Session, symbol: str, recipient_email: str | None, no_email: bool = False):
     """为单个股票生成报告并通过邮件发送"""
     logger.info(f"--- Generating report for symbol: {symbol} ---")

     # --- 获取价格背景信息 ---
     # 注意：这需要 StockDaily 表中有数据
     price_context = get_recent_price_context(db, symbol, days=5)
     logger.info(f"Generated price context for {symbol}:\n{price_context}")
     # --- 如何将 price_context 传入 LLM？ ---
     # 方案 A: 修改 generate_stock_report 接收 price_context 参数 (推荐)
     # report = generate_stock_report(db, symbol, price_context=price_context)
     # 方案 B: 在这里将 price_context 加入到 web_context 或 kb_context (不太好)

     # 这里我们先用方案 A 的思路，需要修改 generate_stock_report 函数签名和内部 prompt 构建
     # 暂时注释掉调用，因为 generate_stock_report 未修改
     # report = generate_stock_report(db, symbol) # 旧调用方式
     # logger.warning("Price context generated but not passed to generate_stock_report yet. Modify function signature.")
     # 假设 generate_stock_report 已修改
     report = generate_stock_report(db, symbol) # 使用已更新的 generate_stock_report
     # 注意：目前的 generate_stock_report 内部会自行构建 price_context 的占位符
     # 如果要传入真实的 price_context，需要修改 generate_stock_report

     if report and not report.startswith("报告生成失败"):
          logger.info(f"Report generated successfully for {symbol}.")
          print(f"\n--- Report for {symbol} ---\n{report}\n--- End Report ---") # 打印到控制台

          if not no_email and recipient_email:
               subject = f"股票分析报告: {symbol}"
               # body = report # 或者可以添加一些头部信息
               email_body = f"这是为您生成的关于股票 {symbol} 的分析报告：\n\n{report}"
               logger.info(f"Attempting to send report for {symbol} to {recipient_email}")
               success = send_email(subject, email_body, recipient_email)
               if success:
                    logger.info(f"Report for {symbol} sent successfully to {recipient_email}.")
               else:
                    logger.error(f"Failed to send report email for {symbol} to {recipient_email}.")
          elif no_email:
               logger.info("Email sending skipped due to --no-email flag.")
          else:
               logger.warning("No recipient email provided, skipping email.")

     else:
          logger.error(f"Failed to generate report for {symbol}. Reason: {report}")


def main():
    parser = argparse.ArgumentParser(description="Quant Platform: Stock Analysis Report Generator")
    parser.add_argument("-s", "--symbol", type=str, help="Specify a single stock symbol (e.g., 600519).")
    parser.add_argument("-a", "--all", action="store_true", help="Process all stocks in the StockList table.")
    parser.add_argument("--skip-data", action="store_true", help="Skip data processing (scraping/embedding).")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation and emailing.")
    parser.add_argument("--no-email", action="store_true", help="Generate report but do not send email.")
    parser.add_argument("-r", "--recipient", type=str, default=settings.EMAIL_USER, help="Email address to send the report to.") # 默认发送给自己

    args = parser.parse_args()

    # --- 初始化 ---
    start_time = time.time()
    create_database_tables() # 确保表存在
    db: Session = SessionLocal()

    target_symbols = []
    if args.symbol:
        target_symbols.append(args.symbol)
    elif args.all:
        logger.info("Fetching all stock symbols from StockList...")
        try:
            all_stocks = db.query(models.StockList.code).all()
            target_symbols = [s[0] for s in all_stocks]
            logger.info(f"Found {len(target_symbols)} stocks to process.")
        except Exception as e:
            logger.error(f"Error fetching stock list: {e}")
            target_symbols = [] # 出错则不处理
    else:
        # 默认行为：处理一个示例股票或提示用户
        logger.warning("No specific symbol provided and --all flag not set. Processing example '600519'. Use -s SYMBOL or -a.")
        target_symbols.append("600519") # 示例

    if not target_symbols:
         logger.error("No target symbols to process. Exiting.")
         db.close()
         return

    # --- 执行流程 ---
    try:
        # 1. 数据处理 (除非跳过)
        if not args.skip_data:
            logger.info("--- Starting Data Processing Phase ---")
            for symbol in target_symbols:
                process_stock_data(db, symbol)
                time.sleep(1) # 短暂休眠，避免数据库压力过大（如果处理很快）
            logger.info("--- Data Processing Phase Complete ---")
        else:
            logger.info("Skipping data processing phase.")

        # 2. 报告生成与邮件发送 (除非跳过)
        if not args.skip_report:
            logger.info("--- Starting Report Generation Phase ---")
            for symbol in target_symbols:
                 run_analysis_and_email(db, symbol, args.recipient, args.no_email)
                 time.sleep(5) # LLM 调用和邮件发送可能较慢，增加间隔
            logger.info("--- Report Generation Phase Complete ---")
        else:
            logger.info("Skipping report generation phase.")

    except Exception as e:
        logger.exception(f"An error occurred during the main process: {e}") # 使用 exception 记录堆栈跟踪
    finally:
        db.close()
        end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds.")
        logger.info("Quant Platform finished.")

if __name__ == "__main__":
    main()