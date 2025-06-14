import os
import glob
import re
import logging
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 尝试从项目中导入 settings，如果失败则使用默认值
try:
    from config.settings import settings
    SETTINGS_LOADED = True
except ImportError:
    settings = None
    SETTINGS_LOADED = False
    print("Warning: Could not import 'settings' from 'config.settings'. Using script defaults for parameters.")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 配置参数 (如果无法从 settings 加载，则使用这些默认值) ---
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TXT_FILES_DIRECTORY = SCRIPT_DIRECTORY # TXT文件与脚本在同一目录

# 页眉页脚移除参数
DEFAULT_HEADER_FOOTER_MIN_REPEATS = 3
DEFAULT_HEADER_FOOTER_MAX_LINE_LEN = 100

# 分块参数
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TEXT_SPLITTER_SEPARATORS = ["\n\n\n", "\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
DEFAULT_TEXT_SPLITTER_KEEP_SEPARATOR = True


# --- 从 settings 加载参数或使用默认值 ---
def get_setting(attr_name, default_value):
    if SETTINGS_LOADED and hasattr(settings, attr_name):
        return getattr(settings, attr_name)
    logger.warning(f"Setting '{attr_name}' not found in config/settings.py, using default: {default_value}")
    return default_value

HEADER_FOOTER_MIN_REPEATS = get_setting('HEADER_FOOTER_MIN_REPEATS', DEFAULT_HEADER_FOOTER_MIN_REPEATS)
HEADER_FOOTER_MAX_LINE_LEN = get_setting('HEADER_FOOTER_MAX_LINE_LEN', DEFAULT_HEADER_FOOTER_MAX_LINE_LEN)
CHUNK_SIZE = get_setting('CHUNK_SIZE', DEFAULT_CHUNK_SIZE)
CHUNK_OVERLAP = get_setting('CHUNK_OVERLAP', DEFAULT_CHUNK_OVERLAP)
TEXT_SPLITTER_SEPARATORS = get_setting('TEXT_SPLITTER_SEPARATORS', DEFAULT_TEXT_SPLITTER_SEPARATORS)
TEXT_SPLITTER_KEEP_SEPARATOR = get_setting('TEXT_SPLITTER_KEEP_SEPARATOR', DEFAULT_TEXT_SPLITTER_KEEP_SEPARATOR)


# --- 文本预处理函数 (与主处理脚本一致) ---
def remove_headers_footers_by_repetition(text: str, min_repeats: int, max_line_len: int) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    stripped_lines = [line.strip() for line in lines]
    line_counts = Counter(s_line for s_line in stripped_lines if s_line)
    lines_to_remove_content = set()
    for line_content, count in line_counts.items():
        if count >= min_repeats and len(line_content) <= max_line_len:
            lines_to_remove_content.add(line_content)
            logger.debug(f"Identified potential header/footer by repetition: '{line_content}' (count: {count})")
    if not lines_to_remove_content:
        return text
    cleaned_lines = [original_line for original_line in lines if original_line.strip() not in lines_to_remove_content]
    return "\n".join(cleaned_lines)

def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text

# --- 主测试逻辑 ---
def test_chunk_file(filepath: str):
    """读取单个文件，进行预处理和分块，并打印结果。"""
    logger.info(f"\nProcessing file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except Exception as e:
        logger.error(f"Could not read file {filepath}: {e}")
        return

    if not raw_text.strip():
        logger.info("File is empty or contains only whitespace.")
        return

    # 1. 预处理
    logger.debug("Preprocessing text...")
    text = remove_headers_footers_by_repetition(
        raw_text,
        min_repeats=HEADER_FOOTER_MIN_REPEATS,
        max_line_len=HEADER_FOOTER_MAX_LINE_LEN
    )
    text = normalize_whitespace(text)

    if not text.strip():
        logger.info("Text became empty after preprocessing.")
        return

    # logger.info("--- Text after Preprocessing ---")
    # print(text[:1000] + "..." if len(text) > 1000 else text) # 打印部分预处理后的文本
    # logger.info("---------------------------------")


    # 2. 递归分块
    logger.info(f"Recursively chunking text. Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=TEXT_SPLITTER_SEPARATORS,
        keep_separator=TEXT_SPLITTER_KEEP_SEPARATOR
    )
    chunks_text = text_splitter.split_text(text)

    if not chunks_text:
        logger.info("No text chunks generated.")
        return

    logger.info(f"Generated {len(chunks_text)} chunks for {filepath}.")
    print("\n" + "="*20 + f" Chunks for: {os.path.basename(filepath)} " + "="*20)
    for i, chunk in enumerate(chunks_text):
        print(f"\n--- Chunk {i+1} (Length: {len(chunk)}) ---")
        print(chunk)
        if i < len(chunks_text) -1 : # 如果不是最后一个块，打印一个分隔符
             print("--- End of Chunk / Overlap Starts ---")


if __name__ == "__main__":
    if not os.path.isdir(TXT_FILES_DIRECTORY):
        logger.error(f"Directory not found: {TXT_FILES_DIRECTORY}")
        logger.error("Please create this directory and place .txt files in it, or update TXT_FILES_DIRECTORY in the script.")
    else:
        # 查找目录下的所有 .txt 文件
        txt_files = glob.glob(os.path.join(TXT_FILES_DIRECTORY, "*.txt"))
        if not txt_files:
            logger.warning(f"No .txt files found in directory: {TXT_FILES_DIRECTORY}")
        else:
            logger.info(f"Found {len(txt_files)} .txt files to process in '{TXT_FILES_DIRECTORY}'.")
            for txt_file_path in txt_files:
                test_chunk_file(txt_file_path)
            logger.info("\nFinished processing all .txt files.")

    logger.info("\n--- Current Parameters Used ---")
    logger.info(f"HEADER_FOOTER_MIN_REPEATS: {HEADER_FOOTER_MIN_REPEATS}")
    logger.info(f"HEADER_FOOTER_MAX_LINE_LEN: {HEADER_FOOTER_MAX_LINE_LEN}")
    logger.info(f"CHUNK_SIZE: {CHUNK_SIZE}")
    logger.info(f"CHUNK_OVERLAP: {CHUNK_OVERLAP}")
    logger.info(f"TEXT_SPLITTER_SEPARATORS: {TEXT_SPLITTER_SEPARATORS}")
    logger.info(f"TEXT_SPLITTER_KEEP_SEPARATOR: {TEXT_SPLITTER_KEEP_SEPARATOR}")
    logger.info("--------------------------------")