import time
import json
import urllib.parse
from datetime import datetime
import re

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By


def parse_publish_time(time_str):
    """
    解析发布时间，将其转换为 yyyy-MM-dd 格式。
    如果 time_str 本身带年份，则直接提取；
    否则补上当前年份。
    """
    # 正则判断是否以年份开头，例如 "2024/03/25"
    if re.match(r"^\d{4}", time_str):
        # 直接提取年月日部分（忽略时间）
        date_part = time_str.split()[0]
        return date_part.replace("/", "-")
    else:
        # 补当前年份
        current_year = datetime.now().year
        month_day = time_str.split()[0]
        return f"{current_year}-{month_day.replace('/', '-')}"


def get_fenghuang_news(keyword, max_results=50):
    options = webdriver.EdgeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    encoded_keyword = urllib.parse.quote(keyword)
    fenghuang_search_url = f"https://so.ifeng.com/?q={encoded_keyword}&c=1"

    driver = webdriver.Edge(options=options)
    driver.get(fenghuang_search_url)
    time.sleep(2)

    result_area = driver.find_element(By.CLASS_NAME, "news-stream-basic-news-list")

    news_lis = result_area.find_elements(
        By.CLASS_NAME, "news-stream-newsStream-news-item-has-image"
    )

    all_news = []

    for news_li in news_lis[:max_results]:
        try:
            news_link = news_li.find_element(By.TAG_NAME, "a").get_attribute("href")

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            response = requests.get(news_link, headers=headers)
            response.raise_for_status()
            response.encoding = "utf-8"

            soup = BeautifulSoup(response.text, "html.parser")
            news_title_div = soup.find("div", id="articleTitle")
            if not news_title_div:
                continue
            news_title = news_title_div.text.strip()
            publish_time = parse_publish_time(
                soup.find("time", class_="index_time_22pEW").text
            )

            article_div = soup.find("div", id="articleBox")
            news_content = "".join([p.text for p in article_div.find_all("p")]).strip()
            news_platform = "凤凰"
            pic_url = []
            for p in article_div.find_all("p"):
                pic = p.find("img")
                if pic:
                    p_url = pic.get("data-lazyload")
                    if p_url and not p_url.lower().startswith(
                        "data:image"
                    ):  # 排除 Base64 图片
                        if p_url.lower().endswith(
                            (".jpg", ".jpeg", ".png")
                        ):  # 只允许 jpg 或 png
                            pic_url.append(p_url)
            news_data = {
                "news_title": news_title.replace("\u3000", ""),
                "platform": news_platform,
                "publish_time": publish_time,
                "news_link": news_link,
                "keyword": keyword,
                "news_content": news_content.replace("\u3000", ""),
                "pic_url": pic_url,
            }
            all_news.append(news_data)

        except Exception as e:
            print(f"解析新闻时出错: {e}")

    driver.quit()

    return json.dumps(all_news, ensure_ascii=False, indent=4)
