import time
import json
import urllib.parse
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By


def parse_publish_time(time_str):
    """
    解析发布时间，将其转换为 yyyy-MM-dd 格式
    """
    now = datetime.now()
    if "分钟前" in time_str:
        minutes = int(time_str.replace("分钟前", "").strip())
        publish_time = now - timedelta(minutes=minutes)
    elif "小时前" in time_str:
        hours = int(time_str.replace("小时前", "").strip())
        publish_time = now - timedelta(hours=hours)
    elif "天前" in time_str:
        days = int(time_str.replace("天前", "").strip())
        publish_time = now - timedelta(days=days)
    else:
        # 默认是 yyyy-MM-dd 格式
        return time_str
    return publish_time.strftime("%Y-%m-%d")


def get_sina_news(keyword, max_results=10):
    options = webdriver.EdgeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')

    encoded_keyword = urllib.parse.quote(keyword)
    sina_search_url = f"https://search.sina.com.cn/?q={encoded_keyword}&c=news&from=channel&ie=utf-8"

    driver = webdriver.Edge(options=options)
    driver.get(sina_search_url)
    time.sleep(2)

    result_area = driver.find_element(By.ID, "result")
    news_divs = result_area.find_elements(By.CLASS_NAME, "box-result")

    all_news = []

    for news_div in news_divs[:max_results]:
        try:
            news_title = news_div.find_element(By.TAG_NAME, 'h2').find_element(By.TAG_NAME, 'a').text
            news_time = news_div.find_element(By.CLASS_NAME, 'fgray_time').text.split(' ')[1]
            news_time = parse_publish_time(news_time)
            news_link = news_div.find_element(By.TAG_NAME, 'h2').find_element(By.TAG_NAME, 'a').get_attribute('href')

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(news_link, headers=headers)
            response.raise_for_status()
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')
            article_body = soup.find(class_="article")

            if not article_body:
                continue

            original_content = "".join([p.text for p in article_body.find_all('p')])
            original_img = [img.get('src') if img.get('src').startswith(('http', 'https')) else "http:" + img.get('src')
                            for img in article_body.find_all('img')]

            news_data = {
                "news_title": news_title.replace('\u3000', ''),
                "platform": "新浪",
                "publish_time": news_time,
                "news_link": news_link,
                "keyword": keyword,
                "news_content": original_content.strip().replace('\u3000', ''),
                "pic_url": original_img
            }

            all_news.append(news_data)
        except Exception as e:
            print(f"解析新闻时出错: {e}")

    driver.quit()

    return json.dumps(all_news, ensure_ascii=False, indent=4)


