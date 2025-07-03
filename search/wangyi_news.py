import time
import json
import urllib.parse
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By


def get_163_news(keyword, max_results=10):
    options = webdriver.EdgeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')

    encoded_keyword = urllib.parse.quote(keyword)
    search_url = f"https://www.163.com/search?keyword={encoded_keyword}"
    driver = webdriver.Edge(options=options)
    driver.get(search_url)
    time.sleep(2)

    result_area = driver.find_element(By.CLASS_NAME, "keyword_list ")

    news_lis = result_area.find_elements(By.CLASS_NAME, "keyword_new")


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
            news_title_h = soup.find('h1', class_="post_title")
            if not news_title_h:
                continue
            news_title = news_title_h.text.strip().replace("\n", "")
            publish_time = soup.find('div', class_="post_info").text.strip().split(' ')[0]
            news_content_div = soup.find('div', id="content").find('div', class_="post_body")
            news_content = ""
            pic_url = []
            for p in news_content_div.find_all('p'):
                pic = p.find('img')

                if pic:
                    # print(pic)
                    p_url = pic.get("src")
                    if p_url and not p_url.lower().startswith("data:image"):  # 排除 Base64 图片
                        if p_url.lower().endswith(('jpg', 'jpeg', 'png')):  # 只允许 jpg 或 png
                            pic_url.append(p_url)
                else:
                    news_content += p.text.strip()
            news_platform = "网易"
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