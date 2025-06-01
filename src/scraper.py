import requests
from bs4 import BeautifulSoup
import csv
import json
import time
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        logging.info(f"Successfully fetched the page: {url}")
        return response.text
    else:
        logging.error(f"Error fetching URL: {url}, status code: {response.status_code}")
        return None

def get_total_pages(page_source):
    soup = BeautifulSoup(page_source, 'html.parser')
    pagination = soup.find_all('a', class_='paging-button')
    if pagination:
        total_pages = max([int(btn.text) for btn in pagination if btn.text.isdigit()])
        logging.info(f"Found {total_pages} pages.")
        return total_pages
    return 1

def parse_apartments(page_source, district_id):
    soup = BeautifulSoup(page_source, 'html.parser')
    listings = soup.find_all('article', class_='realty-preview')
    data = []

    script_tag = soup.find('script', type='application/ld+json')
    if script_tag:
        try:
            json_data = json.loads(script_tag.string)
            items = json_data.get('itemListElement', [])
            
            for listing, item in zip(listings, items):
                try:
                    listing_id = listing.get('id') or 'N/A'
                    item_data = item['item']

                    geo_data = item_data.get('geo', {})
                    latitude = geo_data.get('latitude', 'N/A')
                    longitude = geo_data.get('longitude', 'N/A')
                    address = item_data.get('name', 'N/A')

                    price_tag = listing.find('div', class_='realty-preview-price--main')
                    price = price_tag.text.strip() if price_tag else 'N/A'
                    
                    price_sqm_tag = listing.find('div', class_='realty-preview-price--sqm')
                    price_sqm = price_sqm_tag.text.strip() if price_sqm_tag else 'N/A'
                    
                    property_items = listing.find_all('div', class_='realty-preview-properties-item')
                    rooms = property_items[0].text.strip() if len(property_items) > 0 else 'N/A'
                    area = property_items[1].text.strip() if len(property_items) > 1 else 'N/A'
                    floor = property_items[2].text.strip() if len(property_items) > 2 else 'N/A'

                    date_tag = listing.find('div', class_='realty-preview-dates')
                    date_info = date_tag.text.strip() if date_tag else 'N/A'

                    construction_type = 'N/A'
                    renovation_state = 'N/A'
                    info_spans = listing.find_all('span', class_='realty-preview-info')
                    for span in info_spans:
                        if 'панельні' in span.text:
                            construction_type = span.text.strip()
                        elif 'монолітно-каркасний' in span.text:
                            construction_type = span.text.strip()
                        elif 'цегляний будинок' in span.text:
                            construction_type = span.text.strip()
                        elif 'утеплена панель' in span.text:
                            construction_type = span.text.strip()
                          
                        elif 'з ремонтом' in span.text:
                            renovation_state = span.text.strip()
                        elif 'без ремонту' in span.text:
                            renovation_state = span.text.strip()

                    description_tag = listing.find('p', class_='realty-preview-description__text')
                    description = description_tag.text.strip() if description_tag else 'N/A'

                    # Extract construction year
                    construction_year = 'N/A'
                    info_spans = listing.find_all('span', class_='realty-preview-info')
                    for span in info_spans:
                        if span.text.isdigit() and len(span.text) == 4:  # Check if it's a year
                            construction_year = span.text.strip()
                            break  # Break after finding the year

                    data.append({
                        'ID': listing_id,
                        'Price': price,
                        'Price per sqm': price_sqm,
                        'Address': address,
                        'Rooms': rooms,
                        'Area': area,
                        'Floor': floor,
                        'Date Info': date_info,
                        'District ID': district_id,
                        'Latitude': latitude,
                        'Longitude': longitude,
                        'Construction Type': construction_type,
                        'Renovation State': renovation_state,
                        'Construction Year' : construction_year,
                        'Description': description
                    })

                except Exception as e:
                    logging.error(f"Error processing listing: {e}")
                    continue

            logging.info(f"Parsed {len(data)} listings from the page.")

        except json.JSONDecodeError:
            logging.error("JSON Decode Error: Invalid JSON format")
    else:
        logging.warning("No script tag found for JSON data")

    return data

def save_to_csv(data, filename='all_districts_data.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "ID", "Price", "Price per sqm", "Address", "Rooms", "Area", "Floor", 
            "Date Info", "District ID", "Latitude", "Longitude", "Construction Type", 
            "Renovation State", "Construction Year", "Description"
        ])
        writer.writeheader()
        writer.writerows(data)
    logging.info(f"Data saved to {filename}")

def collect_all_data(base_url, district_id):
    all_data = []
    first_page_source = fetch_page(base_url)
    if not first_page_source:
        return []

    total_pages = get_total_pages(first_page_source)

    for page in range(1, total_pages + 1):
        logging.info(f"Processing page {page} of {total_pages}...")
        page_url = f"{base_url}&page={page}"
        page_source = fetch_page(page_url)
        if page_source:
            parsed_data = parse_apartments(page_source, district_id)
            if parsed_data:
                all_data.extend(parsed_data)
        time.sleep(random.uniform(2, 5))

    return all_data

def main():
    base_url_template = "https://lun.ua/uk/search?currency=UAH&geo_id=10009580&has_eoselia=false&is_without_fee=false&price_sqm_currency=UAH&section_id=1&sort=relevance&sub_geo_id={}"
    
    with open('sub_geo_ids.txt', 'r') as f:
        ids = [line.strip() for line in f]

    
    all_districts_data = []

    for district_id in ids:
        base_url = base_url_template.format(district_id)
        logging.info(f"Scraping data for district ID: {district_id}")
        district_data = collect_all_data(base_url, district_id)
        all_districts_data.extend(district_data)
        logging.info(f"Finished scraping district ID: {district_id}")
        time.sleep(random.uniform(5, 10))

    if all_districts_data:
        save_to_csv(all_districts_data)
    else:
        logging.warning("No data collected for any district.")

if __name__ == "__main__":
    main()