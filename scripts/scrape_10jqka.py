from bs4 import BeautifulSoup
import logging
import requests
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_management_discussion(stock_code: str) -> Optional[Dict[str, str]]:
    """
    Fetches the "Management Discussion and Analysis" for a given stock code
    by directly accessing the iframe's source URL.

    Args:
        stock_code: The stock code (e.g., '000887').

    Returns:
        A dictionary where keys are report dates (e.g., '2023-12-31') and
        values are the corresponding discussion text. Returns None if fetching fails.
    """
    if not stock_code.isdigit():
        logging.error(f"Invalid stock code provided: {stock_code}. It should be a string of digits.")
        return None

    # This is the direct URL to the iframe content
    url = f"https://basic.10jqka.com.cn/{stock_code}/operate.html"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36',
        'Referer': f'https://stockpage.10jqka.com.cn/{stock_code}/', # It's good practice to include a Referer
    }

    try:
        logging.info(f"Requesting data from iframe URL: {url}")
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        # The request might return content with gb2312 encoding
        response.encoding = response.apparent_encoding
        
        soup = BeautifulSoup(response.text, 'html.parser')

        observe_div = soup.find('div', id='observe')
        if not observe_div:
            logging.warning("Could not find the main container div with id='observe'.")
            return None

        date_tabs = observe_div.select('.m_tab ul li a.operateTab')
        content_blocks = observe_div.select('.m_tab_content.m_tab_content2')

        if not date_tabs or not content_blocks:
            logging.warning("Could not find date tabs or content blocks within the 'observe' div.")
            return None
            
        discussion_data = {}
        
        for tab, content in zip(date_tabs, content_blocks):
            report_date = tab.get_text(strip=True)
            
            # The full text is within a <p> that has the 'clearfix' class, 
            # which is initially hidden with a 'none' class.
            full_text_p = content.find('p', class_='clearfix')
            
            if full_text_p:
                text = full_text_p.get_text(separator='\n', strip=True)
                if text.endswith('收起▲'):
                    text = text[:-3].strip()
                discussion_data[report_date] = text
                logging.info(f"Successfully extracted text for report date: {report_date}")
            else:
                # Fallback if the structure is slightly different
                fallback_p = content.find('p', class_='f14 pr')
                if fallback_p:
                    text = fallback_p.get_text(separator='\n', strip=True)
                    if text.endswith('查看全部▼'):
                        text = text[:-5].strip()
                    discussion_data[report_date] = text
                    logging.info(f"Successfully extracted text for report date (using fallback): {report_date}")
                else:
                    logging.warning(f"Could not find the text paragraph for report date: {report_date}")
        
        return discussion_data if discussion_data else None

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch data from URL: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred during parsing: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    stock_to_scrape = '000887'  # 中鼎股份
    print(f"--- Fetching Management Discussion for stock: {stock_to_scrape} ---")
    
    scraped_data = get_management_discussion(stock_to_scrape)
    
    if scraped_data:
        # Print summary for all found dates
        for date, text in scraped_data.items():
            print("\n" + "="*50)
            print(f"报告日期 (Report Date): {date}")
            print("="*50)
            print(text[:500] + "..." if len(text) > 500 else text)

        # Print full text for the first report to verify
        if scraped_data:
            first_report_date = next(iter(scraped_data))
            print(f"\n--- Full text for the first available report ({first_report_date}) ---")
            print(scraped_data[first_report_date])
    else:
        print("\n--- Failed to scrape any data. ---") 