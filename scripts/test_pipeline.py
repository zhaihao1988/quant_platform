# scripts/test_pipeline.py
import sys
import os
import logging
import json
import io
from datetime import date

# --- 动态路径修复：确保能找到项目模块 ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 项目模块导入 ---
import pdfplumber
from data_processing.scraper import (
    extract_section_from_text,
    remove_tables,
    extract_qa_with_ai
)
# [V2] 新增导入，用于在测试脚本中模拟分块并保存结果
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings
# from scripts.to_chunks import embed_and_store_disclosure_chunks, embedder_instance # <-- DB操作，禁用
# from db.database import SessionLocal # <-- DB操作，禁用
# from db.models import StockDisclosure, StockDisclosureChunk # <-- DB操作，禁用

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipelineTest")

# --- 测试配置 ---
# [V2] 创建专门的测试输出目录
TEST_OUTPUT_DIR = os.path.join(project_root, "tests", "output_test")
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
logger.info(f"Test outputs will be saved to: {TEST_OUTPUT_DIR}")

# 测试PDF文件位于 data_processing/reports/ 目录下
PDF_BASE_PATH = os.path.join(project_root, "data_processing", "reports")
TEST_NARRATIVE_PDF_PATH = os.path.join(PDF_BASE_PATH, "test.pdf")
TEST_QA_PDF_PATH = os.path.join(PDF_BASE_PATH, "test2.pdf")

def read_pdf_text(file_path: str) -> str:
    """从本地PDF文件路径读取所有文本。"""
    if not os.path.exists(file_path):
        logger.error(f"测试文件不存在: {file_path}")
        return ""
    try:
        with open(file_path, 'rb') as f:
            with pdfplumber.open(f) as pdf:
                return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        logger.error(f"读取或解析PDF时出错 {file_path}: {e}", exc_info=True)
        return ""


def run_data_processing_and_logging_test():
    """
    【V3 - 纯日志版】
    执行数据提取和分块流程，但跳过所有数据库操作，仅将结果保存为文件。
    """
    logger.info("=" * 80)
    logger.info("🚀 开始数据处理与日志记录测试 (数据库操作已禁用) 🚀")
    logger.info("=" * 80)

    try:
        # --- 1. 处理年报/半年报 (test.pdf) ---
        logger.info("\n---【任务1: 处理年报/半年报】---")
        logger.info(f"读取文件: {TEST_NARRATIVE_PDF_PATH}")
        narrative_full_text = read_pdf_text(TEST_NARRATIVE_PDF_PATH)
        if narrative_full_text:
            logger.info("步骤 1/2: [Scraper] 提取'管理层讨论与分析'章节并清理...")
            narrative_section = extract_section_from_text(narrative_full_text, "管理层讨论与分析")
            raw_content_narrative = remove_tables(narrative_section) if narrative_section else ""

            if raw_content_narrative:
                output_path = os.path.join(TEST_OUTPUT_DIR, "1_narrative_cleaned_content.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(raw_content_narrative)
                logger.info(f"✅ 已保存清理后的年报内容到: {output_path}")

                logger.info("步骤 2/2: [Chunking] 模拟分块...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP,
                    separators=settings.TEXT_SPLITTER_SEPARATORS
                )
                narrative_chunks = text_splitter.split_text(raw_content_narrative)
                chunk_output_path = os.path.join(TEST_OUTPUT_DIR, "2_narrative_chunks.txt")
                with open(chunk_output_path, "w", encoding="utf-8") as f:
                    f.write(f"--- Document: {os.path.basename(TEST_NARRATIVE_PDF_PATH)} ---\n")
                    f.write(f"--- Total Chunks: {len(narrative_chunks)} ---\n\n")
                    for i, chunk in enumerate(narrative_chunks):
                        f.write(f"--- CHUNK {i+1}/{len(narrative_chunks)} (Length: {len(chunk)}) ---\n")
                        f.write(chunk.strip())
                        f.write("\n\n")
                logger.info(f"✅ 已保存年报分块结果到: {chunk_output_path}")
            else:
                logger.warning("年报未能提取到有效内容。")

        # --- 2. 处理调研活动纪要 (重新启用) ---
        logger.info("\n---【任务2: 处理Q&A调研纪要】---")
        logger.info(f"读取文件: {TEST_QA_PDF_PATH}")
        try:
            qa_full_text = read_pdf_text(TEST_QA_PDF_PATH)
            if qa_full_text and qa_full_text.strip():
                logger.info("步骤 1/2: [Scraper] 使用AI提取Q&A...")

                # --- V4 关键修改: 使用工厂模式,不再硬编码模型 ---
                # 此脚本现在将根据 settings.py 中的 LLM_PROVIDER 设置自动选择提供商。
                # 我们不再传递 model_override，让每个提供商使用其最合适的默认模型。
                qa_list = extract_qa_with_ai(qa_full_text)

                if qa_list:
                    # 将Q&A列表格式化为JSON字符串进行存储
                    raw_content_qa = json.dumps(qa_list, ensure_ascii=False, indent=4)
                    output_path = os.path.join(TEST_OUTPUT_DIR, "3_qa_cleaned_content.json")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(raw_content_qa)
                    logger.info(f"✅ 已保存Q&A提取结果到: {output_path}")

                    logger.info("步骤 2/2: [Chunking] 模拟分块...")
                    # 对于Q&A，每个问答对就是一个独立的chunk
                    qa_chunks = [f"问题：{item.get('question', '')}\n回答：{item.get('answer', '')}" for item in qa_list]
                    chunk_output_path = os.path.join(TEST_OUTPUT_DIR, "4_qa_chunks.txt")
                    with open(chunk_output_path, "w", encoding="utf-8") as f:
                        f.write(f"--- Document: {os.path.basename(TEST_QA_PDF_PATH)} ---\n")
                        f.write(f"--- Total Chunks: {len(qa_chunks)} ---\n\n")
                        for i, chunk in enumerate(qa_chunks):
                            f.write(f"--- CHUNK {i+1}/{len(qa_chunks)} (Length: {len(chunk)}) ---\n")
                            f.write(chunk.strip())
                            f.write("\n\n")
                    logger.info(f"✅ 已保存Q&A分块结果到: {chunk_output_path}")
                else:
                    logger.warning("AI未能从Q&A文件中提取任何内容。")
            else:
                logger.warning("未能从Q&A PDF中提取任何文本。")
        except Exception as e:
            logger.error(f"处理Q&A文件时出错: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"测试流水线发生严重错误: {e}", exc_info=True)

    logger.info("\n" + "=" * 80)
    logger.info("✅ 数据处理与日志记录测试执行完毕。")
    logger.info(f"👉 请检查输出目录: {TEST_OUTPUT_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_data_processing_and_logging_test() 