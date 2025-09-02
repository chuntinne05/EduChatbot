import requests
from bs4 import BeautifulSoup
import os
import boto3
from botocore.exceptions import ClientError
from io import BytesIO
import time
import json
import re
from datetime import datetime
import pytesseract
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import cv2

load_dotenv()

textract_client = boto3.client(
    'textract',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'ap-southeast-1')
)

with open("sitemap_links.txt", "r") as f:
    urls = [line.strip() for line in f.readlines() if line.strip()]

output_dir = "structured_pages"
os.makedirs(output_dir, exist_ok=True)

def enhance_image_processing(image_bytes):
    """Enhance image for better OCR results"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)
        _, img_byte_arr = cv2.imencode('.jpg', denoised)
        return img_byte_arr.tobytes()
    except Exception as e:
        print(f"Image enhancement error: {str(e)}")
        return image_bytes

def extract_text_with_ocr(image_bytes: bytes) -> str:
    """Extract text from image using combined OCR approach"""
    try:
        response = textract_client.detect_document_text(Document={'Bytes': image_bytes})
        textract_text = ""
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                textract_text += item["Text"] + "\n"
        if not textract_text.strip():
            img = Image.open(BytesIO(image_bytes))
            tesseract_text = pytesseract.image_to_string(img, lang='vie')
            return tesseract_text.strip()
        
        return textract_text.strip()
    except ClientError:
        try:
            img = Image.open(BytesIO(image_bytes))
            return pytesseract.image_to_string(img, lang='vie').strip()
        except Exception as e:
            return f"OCR error: {str(e)}"
    except Exception as e:
        return f"Unexpected OCR error: {str(e)}"

def extract_tables(soup):
    """Extract and structure tables from HTML"""
    tables = []
    for table in soup.find_all('table'):
        table_data = []
        headers = []
        header_row = table.find('tr')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            if headers:
                table_data.append(headers)
        for row in table.find_all('tr'):
            if row == header_row:
                continue    
            cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
            if cells:
                table_data.append(cells)
        
        if table_data:
            tables.append(table_data)
    return tables

def extract_metadata(soup):
    """Extract metadata from the page"""
    metadata = {
        "title": "",
        "publish_date": "",
        "description": "",
        "keywords": []
    }
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.get_text(strip=True)
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{4}', 
        r'\d{4}-\d{2}-\d{2}',     
    ]
    date_selectors = [
        {'class': 'date'},
        {'class': 'post-date'},
        {'class': 'published'},
        {'property': 'article:published_time'},
        {'name': 'publish_date'}
    ]
    
    found_date = ""
    for pattern in date_patterns:
        date_match = re.search(pattern, soup.get_text())
        if date_match:
            found_date = date_match.group(0)
            break
    if not found_date:
        for selector in date_selectors:
            date_element = soup.find(attrs=selector)
            if date_element:
                found_date = date_element.get('content') or date_element.get_text(strip=True)
                if found_date:
                    break
    
    metadata['publish_date'] = found_date
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        metadata['description'] = meta_desc.get('content', '')
    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
    if meta_keywords:
        keywords = meta_keywords.get('content', '')
        metadata['keywords'] = [kw.strip() for kw in keywords.split(',') if kw.strip()]
    
    return metadata

def extract_content_sections(soup):
    """Segment content into structured sections"""
    sections = []
    content_containers = [
        soup.select_one("div.elementor-widget-theme-post-content"),
        soup.find("article"),
        soup.find("main"),
        soup.find("div", class_="content"),
        soup.find("div", class_="article-content")
    ]
    
    content_div = None
    for container in content_containers:
        if container:
            content_div = container
            break
    if not content_div:
        content_div = soup.body

    for element in content_div.find_all(recursive=False):
        el_type = None
        content = None

        if element.name in ['p', 'div']:
            text = element.get_text(strip=True)
            if text:
                el_type = "text"
                content = text
 
        elif element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            text = element.get_text(strip=True)
            if text:
                el_type = "heading"
                content = {
                    "level": int(element.name[1]),
                    "text": text
                }

        elif element.name in ['ul', 'ol']:
            list_items = []
            for li in element.find_all('li', recursive=False):
                list_items.append(li.get_text(strip=True))
            
            if list_items:
                el_type = "list"
                content = {
                    "type": "unordered" if element.name == 'ul' else "ordered",
                    "items": list_items
                }

        elif element.name == 'table':
            table_data = []
            for row in element.find_all('tr'):
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    row_data.append(cell.get_text(strip=True))
                if row_data:
                    table_data.append(row_data)
            
            if table_data:
                el_type = "table"
                content = table_data

        elif element.name == 'img':
            img_url = element.get('src') or element.get('data-src')
            if img_url and img_url.startswith('http'):
                el_type = "image"
                content = {
                    "url": img_url,
                    "ocr_text": ""
                }

        if el_type and content:
            sections.append({
                "type": el_type,
                "content": content
            })
    
    return sections

def process_page(url):
    """Process a single page and return structured data"""
    try:
        print(f"Processing: {url}")
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        metadata = extract_metadata(soup)
        sections = extract_content_sections(soup)
        for section in sections:
            if section['type'] == 'image':
                try:
                    img_url = section['content']['url']
                    print(f"  Downloading image: {img_url}")
                    img_response = requests.get(img_url, timeout=10)
                    img_response.raise_for_status()
                    enhanced_img = enhance_image_processing(img_response.content)
                    ocr_text = extract_text_with_ocr(enhanced_img)
                    section['content']['ocr_text'] = ocr_text
                except Exception as img_e:
                    print(f"    Image processing failed: {str(img_e)}")
                    section['content']['ocr_text'] = f"OCR failed: {str(img_e)}"
        tables = extract_tables(soup)
        for table_data in tables:
            sections.append({
                "type": "table",
                "content": table_data
            })

        word_count = sum(len(section['content'].split()) 
                      for section in sections 
                      if section['type'] == 'text' and isinstance(section['content'], str))
        
        return {
            "url": url,
            "metadata": metadata,
            "sections": sections,
            "word_count": word_count,
            "processing_time": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Failed to process {url}: {str(e)}")
        return {
            "url": url,
            "error": str(e)
        }

for idx, url in enumerate(urls):
    start_time = time.time()
    page_data = process_page(url)
    if 'error' not in page_data:
        filename = os.path.join(output_dir, f"doc_{idx+1:03}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)
        
        elapsed = time.time() - start_time
        print(f"  Saved structured data: {filename} ({elapsed:.2f}s)\n")

print("Processing completed.")