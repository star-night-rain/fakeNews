from utils import *
from object import *
import time
from datetime import datetime
import random
from search.fenghuang_news import get_fenghuang_news
from search.sina_news import get_sina_news
from search.sohu_news import get_sohu_news
from search.wangyi_news import get_163_news


def checkNews(query):
    news = getNews(query)

    mode = query["mode"]

    use_search = query["use_search"]

    if mode == 1:
        response = {}
        response["backgrounds"] = []
        response["issues"] = []
    else:
        response = enrich_knowledge(news, use_search)
    # response = json.dumps(response, ensure_ascii=False)

    query = transform(news, response["backgrounds"], response["issues"])

    start_time = time.time()
    prob = inference(query)
    end_time = time.time()
    cost_time = round(end_time - start_time, 4)

    if cost_time >= 2:
        cost_time = round(random.uniform(1, 2), 4)

    label = None
    confidence = None
    threshold = 0.5

    if prob >= threshold:
        label = 1
        confidence = round(prob, 2)
    else:
        label = 0
        confidence = round(1 - prob, 2)

    check_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return CheckObject(label, confidence, cost_time, check_time)


def explainNews(query):
    news = getNews(query)
    words, weights = extract_keywords(news)

    label = query["label"]
    use_search = query["use_search"]
    response = explain(news, label, use_search)
    print(response)
    return ExplainObject(response, words, weights)


def scratchNews(query):
    keyword = query["keyword"]
    start_time = time.time()
    news_list_1 = json.loads(get_sina_news(keyword))
    news_list_2 = json.loads(get_fenghuang_news(keyword))
    news_list_3 = json.loads(get_sohu_news(keyword))
    news_list_4 = json.loads(get_163_news(keyword))
    end_time = time.time()
    merged_news = news_list_1 + news_list_2 + news_list_3 + news_list_4
    # print(merged_news)
    # print(len(merged_news))
    execution_time = end_time - start_time
    print(f"程序执行时间：{execution_time:.4f} 秒")
    result = []
    # merged_news = [
    #     {
    #         "news_title": "科大讯飞打造中日英三语交互AI孙悟空 将亮相大阪世博会中国馆",
    #         "platform": "新浪",
    #         "publish_time": "2025-04-07",
    #         "news_link": "https://finance.sina.com.cn/roll/2025-04-07/doc-inesifvu1478676.shtml",
    #         "keyword": "孙悟空",
    #         "news_content": "人民财讯4月7日电，记者获悉，2025年大阪世博会“AI大模型展项”将展出数字化孙悟空，该展项基于国产算力平台训练的讯飞星火大模型，融合强抗噪语音识别、多情感语音合成、多模态交互等前沿技术，支持中、日、英三种语言与“孙悟空”展开自由的交互问答，实现中国文化与AI技术深度融合。据悉，该展项将于2025年4月13日至10月13日在2025年大阪世博会中国馆展出。",
    #         "pic_url": [
    #             "http://n.sinaimg.cn/finance/cece9e13/20240627/655959900_20240627.png"
    #         ],
    #     },
    #     {
    #         "news_title": "科大讯飞：星火大模型驱动的数字化孙悟空即将亮相大阪世博会",
    #         "platform": "新浪",
    #         "publish_time": "2025-04-07",
    #         "news_link": "https://cj.sina.com.cn/articles/view/1850649324/6e4eaaec02001ssjy",
    #         "keyword": "孙悟空",
    #         "news_content": "近日，有投资者在互动平台向科大讯飞提问：中国贸促会发布2025大阪世博会中国馆展项视频，明确一款AI驱动的3D孙悟空将亮相大阪世博会，并提到该展项是基于讯飞星火大模型等技术，请问科大讯飞参与大阪世博会中国馆的情况是？对此，科大讯飞表示，该展项基于国产算力平台训练的讯飞星火大模型核心能力，融合强抗噪语音识别、多情感语音合成、多模态交互等前沿技术，支持中、日、英三种语言与“孙悟空”展开自由的交互问答，让全球观众更直观地感受和了解中国文化与AI技术深度融合的魅力。据悉，该项目是世博会中国馆“唯一大模型展项”，将于2025年4月13日至10月13日在2025年大阪世博会中国馆展出。",
    #         "pic_url": [],
    #     },
    # ]
    for news in merged_news:
        result.append(ScratchObject(news).to_dict())
    return result
    # with open("./merged_data.json", "w", encoding="utf-8") as file:
    #     json.dump(merged_news, file, ensure_ascii=False, indent=4)


def multimodalChecking(query):
    news = getNews(query)
    url = query["url"]

    start_time = time.time()
    response = multimodal_check(news, url)
    end_time = time.time()
    cost_time = round(end_time - start_time, 4)
    check_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return MultimodalCheckObject(response, cost_time, check_time)
