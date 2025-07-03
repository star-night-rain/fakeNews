import time
import json
import urllib.parse
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By


def get_sohu_news(keyword, max_results=10):
    options = webdriver.EdgeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')

    encoded_keyword = urllib.parse.quote(keyword)
    sohu_search_url = f"https://search.sohu.com/?keyword={encoded_keyword}&type=10002&ie=utf8&queryType=default&spm=smpc.channel_258.search-box.174290507114188Y6uOI_1090"
    driver = webdriver.Edge(options=options)
    driver.get(sohu_search_url)
    time.sleep(2)

    result_area = driver.find_element(By.ID, "news-list")

    news_lis = result_area.find_elements(By.XPATH, ".//div[contains(@class, 'cards-small-plain') or contains(@class, 'cards-small-img')]")


    all_news = []

    for news_li in news_lis[:max_results]:
        try:
            news_link = news_li.find_element(By.TAG_NAME, 'a').get_attribute('href')

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(news_link, headers=headers)
            response.raise_for_status()
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')
            news_title_div = soup.find('div', class_="text-title")
            if not news_title_div:
                continue
            news_title = news_title_div.find('h1').text.strip().replace("\n", "")
            publish_time = news_title_div.find('span', id="news-time").text.strip().split(' ')[0]
            news_content_div = soup.find('article', class_="article")
            if not news_content_div:
                continue
            news_content = ""
            pic_url = []
            for p in news_content_div.find_all('p'):
                # 提取 p 标签的文本
                pic = p.find('img')
                if pic:
                    p_url = pic.get("src")
                    if p_url and not p_url.lower().startswith("data:image"):  # 排除 Base64 图片
                        if p_url.lower().endswith(('.jpg', '.jpeg', '.png')):  # 只允许 jpg 或 png
                            pic_url.append(p_url)
                if p.find('span'):
                    continue
                news_content += p.get_text(separator="", strip=True)
            # 去除两端空白
            news_content = news_content.strip()
            news_platform = "搜狐"
            news_data = {
                "news_title": news_title.replace('\u3000', ''),
                "platform": news_platform,
                "publish_time": publish_time,
                "news_link": news_link,
                "keyword": keyword,
                "news_content": news_content.replace('\u3000', ''),
                "pic_url": pic_url
            }
            all_news.append(news_data)

        except Exception as e:
            print(f"解析新闻时出错: {e}")

    driver.quit()
    return json.dumps(all_news, ensure_ascii=False, indent=4)

