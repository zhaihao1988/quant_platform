# main.py
import argparse
import logging
import time
import os
from sqlalchemy.orm import Session

# --- Project Modules ---
from db.database import get_db_session, create_database_tables # Use context manager for session
from db import crud # Import crud module
from data_processing import loader, scraper # Import loader and scraper
from core import vectorizer, prompting # Import vectorizer and prompting
from integrations import email_sender
from config import settings # Import settings for report path and email recipient

# --- Basic Logging Setup ---
# Configure logging level, format, and output (e.g., file and console)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
# Reduce noise from libraries if needed
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('selenium').setLevel(logging.INFO) # Show Selenium startup/shutdown
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('ollama').setLevel(logging.INFO) # Show Ollama calls

logger = logging.getLogger(__name__) # Logger for main script

# --- Helper Function for Data Processing ---
def run_data_processing(db: Session, symbol: str):
    """
    Runs the data processing pipeline for a single stock:
    1. Finds disclosures needing content.
    2. Scrapes text content for them.
    3. Generates embeddings.
    4. Updates the database.
    """
    logger.info(f"--- Starting Data Processing for: {symbol} ---")
    processed_count = 0
    failed_count = 0

    # 1. Find disclosures needing content
    disclosures_to_process = loader.get_disclosures_needing_content(db, symbol)

    if not disclosures_to_process:
        logger.info(f"No new disclosures requiring processing found for {symbol}.")
        return

    logger.info(f"Found {len(disclosures_to_process)} disclosures to process for {symbol}.")

    for disclosure in disclosures_to_process:
        logger.info(f"Processing Disclosure ID: {disclosure.id}, Title: {disclosure.title[:50]}...")
        text_content = None
        embedding_vector = None
        try:
            # 2. Scrape text content
            text_content = scraper.fetch_announcement_text(disclosure.url, disclosure.title)
            if not text_content:
                logger.warning(f"Failed to fetch/extract text for Disclosure ID: {disclosure.id}. Skipping.")
                failed_count += 1
                continue # Skip to next disclosure if scraping fails

            logger.info(f"Successfully scraped text (length: {len(text_content)}) for Disclosure ID: {disclosure.id}")

            # 3. Generate embedding
            embedding_vector = vectorizer.get_embedding(text_content, is_query=False)
            if not embedding_vector:
                logger.warning(f"Failed to generate embedding for Disclosure ID: {disclosure.id}. Vector will be NULL.")
                # Still save the text content even if embedding fails
            else:
                logger.info(f"Successfully generated embedding vector for Disclosure ID: {disclosure.id}")

            # 4. Update database (commit handled by context manager later)
            # We pass data to crud, crud updates the object in the session
            update_successful = crud.update_disclosure_content_vector(
                db=db, # Pass the session
                disclosure_id=disclosure.id,
                text_content=text_content,
                vector=embedding_vector
            )
            if update_successful:
                 processed_count += 1
            else:
                 failed_count += 1
                 # Error already logged in crud function

            # Optional: Add a small delay between processing each disclosure
            time.sleep(random.uniform(1, 3))

        except Exception as e:
            logger.error(f"Unhandled error processing Disclosure ID {disclosure.id}: {e}", exc_info=True)
            failed_count += 1
            # Continue to the next disclosure even if one fails unexpectedly

    logger.info(f"--- Finished Data Processing for {symbol}. Processed: {processed_count}, Failed: {failed_count} ---")


# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Stock Analysis Report Generator using Local LLM and Knowledge Base.")
    parser.add_argument("symbol", type=str, help="The stock symbol to analyze (e.g., 600519).")
    parser.add_argument("--skip-data", action="store_true", help="Skip the data processing (scraping, embedding) phase.")
    parser.add_argument("--recipient", type=str, default=settings.EMAIL_USER, help=f"Email address to send the report to (default: {settings.EMAIL_USER}).")
    parser.add_argument("--no-email", action="store_true", help="Generate and save the report but do not send email.")
    parser.add_argument("--force-report", action="store_true", help="Attempt to generate report even if data processing fails.")

    args = parser.parse_args()
    symbol = args.symbol.strip() # Clean up input symbol

    logger.info(f"--- Starting Stock Analysis for: {symbol} ---")
    start_time = time.time()

    # --- Database Initialization ---
    try:
        create_database_tables() # Ensure tables exist
    except Exception as e:
        logger.critical(f"Failed to initialize database tables: {e}. Exiting.")
        return # Cannot proceed without DB tables

    # --- Data Processing Phase ---
    if not args.skip_data:
        # Use the database session context manager
        try:
            with get_db_session() as db_session:
                if db_session: # Check if session was created successfully
                     run_data_processing(db_session, symbol)
                else:
                     logger.error("Failed to get database session for data processing.")
                     if not args.force_report: # Stop if DB fails and not forcing report
                          return
        except Exception as e:
            logger.error(f"Error during data processing phase for {symbol}: {e}", exc_info=True)
            if not args.force_report: # Stop if data processing fails and not forcing report
                 logger.critical("Exiting due to data processing error.")
                 return
            else:
                 logger.warning("Data processing failed, but continuing to report generation due to --force-report flag.")
    else:
        logger.info("Skipping data processing phase as requested.")

    # --- Report Generation Phase ---
    logger.info(f"--- Starting Report Generation for: {symbol} ---")
    report_content = None
    try:
        # Get a new session for report generation (or reuse if feasible, but separation is often cleaner)
        with get_db_session() as db_session:
             if db_session:
                  report_content = prompting.generate_stock_report(db_session, symbol)
             else:
                  logger.error("Failed to get database session for report generation.")

    except Exception as e:
        logger.error(f"Error during report generation phase for {symbol}: {e}", exc_info=True)


    # --- Save Report and Send Email ---
    if report_content:
        logger.info(f"Report generated successfully for {symbol}.")

        # 1. Save report locally
        report_filename = f"{symbol}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        save_path = os.path.join(settings.REPORT_SAVE_PATH, report_filename)
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Report saved locally to: {save_path}")
        except IOError as e:
            logger.error(f"Failed to save report to {save_path}: {e}")

        # 2. Send email (if not disabled)
        if not args.no_email:
            if args.recipient:
                logger.info(f"Attempting to send report via email to: {args.recipient}")
                email_subject = f"Stock Analysis Report: {symbol}"
                # Use a slightly different body for email if needed
                email_body = f"Stock analysis report for {symbol} generated on {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n" + report_content
                send_success = email_sender.send_email(email_subject, email_body, args.recipient)
                if send_success:
                    logger.info("Report email sent successfully.")
                else:
                    logger.error("Failed to send report email.")
            else:
                logger.warning("No recipient email address provided (--recipient). Skipping email.")
        else:
            logger.info("Email sending skipped due to --no-email flag.")

    else:
        logger.error(f"Report generation failed for {symbol}. No report to save or send.")

    # --- Finalization ---
    end_time = time.time()
    logger.info(f"--- Stock Analysis for {symbol} Finished. Total time: {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    # --- Setup before running main ---
    # Load .env file if it exists (pydantic-settings does this automatically if configured)
    from dotenv import load_dotenv
    load_dotenv()
    logger.info(".env file loaded (if exists).")

    # --- Run Main Application ---
    main()