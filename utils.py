import json
from openai import OpenAI
from transformers import AutoTokenizer
from model.train import inference
import jieba.analyse
import re

# prompt = (
#     "以下是一篇已经被核实为虚假的新闻报道。请基于该新闻内容撰写一份结构化的详细报告，报告应包括以下几个部分：\n\n"
#     "1. 虚假性质说明：明确指出该新闻为虚假信息，并简要说明其虚假性质。\n"
#     "2. 背景信息概述：提供与该新闻相关的真实背景信息，帮助读者了解事件的真实情况。\n"
#     "3. 问题内容分析：指出新闻中存在哪些具体问题（如捏造事实、误导性措辞、断章取义等），并针对每一项问题提供详细的理由或证据。\n"
#     "4. 用户建议与防范提醒：向读者提供一些实用建议，说明如何更好地识别和避免被虚假新闻误导，并推荐一些查证信息真伪的方法或工具。\n\n"
#     "请确保报告逻辑清晰、语言严谨，条理分明，适合普通用户阅读理解。"
# )


# prompt = (
#     "以下是一篇已经被核实为真实的新闻报道。请根据新闻内容撰写一份结构化的详细报告，报告应包括以下几个部分：\n\n"
#     "1. 新闻性质说明：明确指出该新闻内容属实，并简要说明其重要性或社会影响。\n"
#     "2. 背景信息介绍：补充与该新闻相关的背景资料，帮助读者更全面地理解事件的前因后果。\n"
#     "3. 用户启示与建议：结合新闻内容，向读者提供一些具有参考价值的建议，说明事件可能带来的影响，以及公众可采取的应对措施或行动建议。\n\n"
#     "请确保报告结构清晰、语言规范，内容具有可读性和实用性，便于普通用户理解和参考。"
# )


# def getNews(query):
#     news = ""
#     if query["publish_time"] != "":
#         news += query["publish_time"] + ","
#     if query["title"] != "":
#         news += query["title"] + "."
#     if query["content"] != "":
#         news += query["content"]
#     return news


def getNews(query):
    parts = []

    if query.get("publish_time"):
        parts.append(f"这篇新闻发布于 {query['publish_time']}。")

    if query.get("title"):
        parts.append(f"标题是《{query['title']}》。")

    if query.get("content"):
        parts.append(query["content"].strip())

    return " ".join(parts)


def enrich_knowledge(news, use_search):
    # client = OpenAI(
    #     base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
    #     api_key="8e704977-78ae-40e3-8f3a-7d3fa3f8edb4",
    # )

    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key="sk-3ee1d45f80ac4a25acbb367f638002cb",
    )

    not_search_content = "你是一位专业的事实核查助手，负责从新闻中提取关键信息，并识别其中可能存在的事实错误或逻辑问题。注意，不能使用markdown格式回答。"
    search_content = "你是一位专业的事实核查助手，负责从新闻中提取关键信息，并识别其中可能存在的事实错误或逻辑问题。你可以通过联网搜索获取当前时间及其他必要的事实信息，并应优先参考搜索结果进行回答。注意，回答内容不得使用 Markdown 格式。"

    response = client.chat.completions.create(
        # model="bot-20250409201752-rl95z",
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": search_content if use_search == 1 else not_search_content,
            },
            {
                "role": "user",
                "content": (
                    "请根据以下新闻提供背景信息(backgrounds)以及不符合常识的内容(issues)。"
                    '请返回标准 JSON 格式: {"backgrounds": ["b1", "b2", ...], "issues": ["c1", "c2", ...]},其中不包含任何markdown格式。\n'
                    f"{news}"
                ),
            },
        ],
    )
    try:
        response = json.loads(response.choices[0].message.content)
        return response
    except Exception as e:
        return None


def transform(news, backgrounds, issues):
    tokenizer = tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    max_length = 512

    news = tokenizer(
        news,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    s = ""
    for i in range(len(backgrounds)):
        if i > 0:
            s += ";"
        s += backgrounds[i]
    s += "."

    backgrounds = tokenizer(
        s,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    s = ""
    for i in range(len(issues)):
        if i > 0:
            s += ";"
        s += issues[i]
    s += "."

    issues = tokenizer(
        s,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": news["input_ids"],
        "attention_mask": news["attention_mask"],
        "backgrounds_input_ids": backgrounds["input_ids"],
        "backgrounds_attention_mask": backgrounds["attention_mask"],
        "issues_input_ids": issues["input_ids"],
        "issues_attention_mask": issues["attention_mask"],
    }


def explain(news, label, use_search):
    not_search_content = "你是一位专业的事实核查助手，负责从新闻中提取关键信息，并识别其中可能存在的事实错误或逻辑问题。注意，不能使用markdown格式回答。"
    search_content = "你是一位专业的事实核查助手，负责从新闻中提取关键信息，并识别其中可能存在的事实错误或逻辑问题。你可以通过联网搜索获取当前时间及其他必要的事实信息，并应优先参考搜索结果进行回答。注意，不能使用markdown格式回答。"

    true_content = (
        "以下是一篇已经被判定为虚假的新闻报道。请基于新闻内容撰写一份结构化的报告，报告应包括以下几个部分：\n\n"
        "1. 新闻性质说明：指出该新闻为虚假信息，并简要阐述其虚假性质，包括该新闻为何为虚假信息，以及可能存在的误导点。\n"
        "2. 问题内容分析：提供与该新闻相关的真实背景信息，帮助读者全面了解事件的真实情况与事实背景。\n"
        "3. 问题内容分析：分析新闻中存在的具体问题，如捏造事实、误导性措辞、断章取义等，并针对每个问题提供详细的分析、理由或证据。\n"
        "4. 用户建议与防范提醒：向读者提供实用建议，说明如何识别并避免被虚假新闻误导，推荐一些有效的查证真伪的方法或工具。\n\n"
        "请确保报告内容逻辑清晰、语言严谨、条理清晰，并确保普通读者能轻松理解。\n\n"
        "请返回标准的 JSON 格式：{\n"
        '"use_search"：数字，若使用了联网搜索则为1，否则为0，\n'
        '"description"：字符串,表示新闻的性质说明，\n'
        '"backgrounds"：数组,表示新闻的背景信息，\n'
        '"issue_title"：数组,表示存在的具体问题，\n'
        '"issue_content"：数组,表示"issue_title"每个具体问题对应的详细的分析、理由或证据，\n'
        '"suggestion_title"：数组,表示实用建议的概括，\n'
        '"suggestion_content"：数组,表示"suggestion_title"每个实用建议对应的详细解释。}\n'
        f"新闻内容: {news}"
    )
    false_content = (
        "以下是一篇已经被核实为真实的新闻报道。请根据新闻内容撰写一份结构化、详细的报告，报告应包括以下几个部分：\n\n"
        "1. 新闻性质说明：明确指出该新闻内容属实，并简要阐述其重要性及对社会的影响。\n"
        "2. 背景信息介绍：补充相关的背景资料，帮助读者全面了解事件的前因后果。\n"
        "3. 问题内容分析：分析新闻中可能存在的具体问题，如误导性措辞、片面描述等，针对每个问题提供详细的分析、理由或证据。\n"
        "4. 用户建议与防范提醒：结合新闻内容，向读者提供具有实用价值的建议，阐明事件可能带来的影响，并建议公众采取的应对措施或行动。\n\n"
        "请确保报告内容逻辑清晰、语言严谨、条理清晰，并确保普通读者能轻松理解。\n\n"
        "请返回标准的 JSON 格式：{\n"
        '"description"：字符串,表示新闻的性质说明，\n'
        '"backgrounds"：数组,表示新闻的背景信息，\n'
        '"issue_title"：数组,表示存在的具体问题，\n'
        '"issue_content"：数组,表示"issue_title"每个具体问题对应的详细的分析、理由或证据，\n'
        '"suggestion_title"：数组,表示实用建议的概括，\n'
        '"suggestion_content"：数组,表示"suggestion_title"每个实用建议对应的详细解释。}\n'
        f"新闻内容: {news}"
    )

    # client = OpenAI(
        # base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
        # api_key="8e704977-78ae-40e3-8f3a-7d3fa3f8edb4",
    # )

    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key="sk-3ee1d45f80ac4a25acbb367f638002cb",
    )

    response = client.chat.completions.create(
        # model="bot-20250409201752-rl95z",
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": search_content if use_search == 1 else not_search_content,
            },
            {
                "role": "user",
                "content": true_content if label == 0 else false_content,
            },
        ],
    )

    try:
        response = json.loads(response.choices[0].message.content)
        return response
    except Exception as e:
        # print(response.choices[0].message.content)
        # response = re.sub(r"```[\s\S]*?```", "", response.choices[0].message.content)
        print(f"error:{e}")
        # print(response)
        return None


def multimodal_check(news, url):
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
        api_key="8e704977-78ae-40e3-8f3a-7d3fa3f8edb4",
    )

    response = client.chat.completions.create(
        model="bot-20250409213632-zbw2t",
        messages=[
            {
                "role": "system",
                "content": "你是一位专业的事实核查助手，负责从新闻中提取关键信息，并识别其中可能存在的事实错误或逻辑问题。你可以通过联网搜索获取当前时间及其他必要的事实信息，并应优先参考搜索结果进行回答。注意，回答内容不得使用 Markdown 格式。",
            },
            {
                "role": "user",
                "content": (
                    "以下是新闻报道及其配套图片的链接。请根据你的专业知识判断新闻内容与图片是否一致，并评估新闻的真实性。"
                    "请按照标准的 JSON 格式返回报告：{\n"
                    '"consistency": 数字，若新闻内容与图片一致则为 1，否则为 0,\n'
                    '"label": 数字，若为虚假新闻或 "consistency" 等于 0 则为 1，否则为 0,\n'
                    '"confidence": 数字，表示对 "label" 判断的置信度,\n'
                    '"reason_title": 数组，若新闻内容与图片不一致，列出不一致的原因标题；若一致则为空数组,\n'
                    '"reason_content": 数组，对应每条 "reason_title"，提供具体的分析、理由或证据；若一致则为空数组,\n'
                    "}\n"
                    f"新闻内容: {news}\n"
                    f"图片链接： {url}"
                ),
            },
        ],
    )

    try:
        response = json.loads(response.choices[0].message.content)
        return response
    except Exception as e:
        print(e)
        return None


def multimodal_explanation(news, url):
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
        api_key="8e704977-78ae-40e3-8f3a-7d3fa3f8edb4",
    )

    response = client.chat.completions.create(
        model="bot-20250409213632-zbw2t",
        messages=[
            {
                "role": "system",
                "content": "你是一位专业的事实核查助手，负责从新闻中提取关键信息，并识别其中可能存在的事实错误或逻辑问题。你可以通过联网搜索获取当前时间及其他必要的事实信息，并应优先参考搜索结果进行回答。注意，回答内容不得使用 Markdown 格式。",
            },
            {
                "role": "user",
                "content": (
                    "以下是新闻报道及其配套图片的链接。请根据你的专业知识判断新闻内容与图片是否一致，并评估新闻的真实性。基于新闻内容，请撰写一份结构化的报告，报告应包括以下几个部分：\n\n"
                    "1. 新闻性质说明：如果该新闻为虚假信息，请明确指出并简要阐述其虚假性质，说明为何该新闻被认为是虚假信息，并指出其中可能存在的误导点；如果新闻属实，请明确指出，并简要分析新闻的重要性及其对社会的影响。\n"
                    "2. 背景信息分析：提供与该新闻相关的真实背景信息，帮助读者全面理解事件的真相及其背景。\n"
                    "3. 问题内容分析：详细分析新闻中存在的具体问题，如捏造事实、误导性措辞、断章取义等，并针对每个问题提供详细的分析、理由或证据。\n"
                    "4. 用户建议与防范提醒：向读者提供实用建议，帮助他们识别并避免虚假新闻的误导，并推荐一些有效的查证真伪的方法或工具。\n\n"
                    "请确保报告内容逻辑清晰、语言严谨、条理分明，并确保普通读者能够轻松理解。\n\n"
                    "请按照标准的 JSON 格式返回报告：{\n"
                    '"consistency": 数字，若新闻内容与图片一致则为 1，否则为 0,\n'
                    '"label": 数字，若为虚假新闻或 "consistency" 等于 0 则为 1，否则为 0,\n'
                    '"confidence": 数字，表示对 "label" 判断的置信度,\n'
                    '"reason_title": 数组，若新闻内容与图片不一致，列出不一致的原因标题；若一致则为空数组,\n'
                    '"reason_content": 数组，对应每条 "reason_title"，提供具体的分析、理由或证据；若一致则为空数组,\n'
                    '"description": 字符串，简要说明该新闻的性质,\n'
                    '"backgrounds": 数组，提供与该新闻相关的背景信息,\n'
                    '"issue_title": 数组，指出该新闻中存在的具体问题,\n'
                    '"issue_content": 数组，分别对应每条 "issue_title"，提供具体的分析、理由或证据,\n'
                    '"suggestion_title": 数组，提出与该新闻相关的实用建议概要,\n'
                    '"suggestion_content": 数组，分别对应每条 "suggestion_title"，提供详细解释\n'
                    "}\n"
                    f"新闻内容: {news}\n"
                    f"图片链接： {url}"
                ),
            },
        ],
    )

    try:
        response = json.loads(response.choices[0].message.content)
        return response
    except Exception as e:
        print(e)
        return None


def extract_keywords(news):
    keywords = jieba.analyse.textrank(news, topK=6, withWeight=True)
    words, weights = zip(*keywords)
    return list(words), list(weights)


# news = "2025年3月7日,在刚刚结束的全球气候变化峰会上，各国代表经过长达两周的紧张谈判，最终达成了一项历史性的协议。"
# label = 0
# use_search = 0
# response = explain(news, label, use_search)
# print(response)

# news = "雷军 张磊 俞敏洪：坚持长期主义，伟大需要时间的沉淀和积累"
# url = "https://cn.bing.com/images/search?view=detailV2&ccid=ZItTWANY&id=2A8171508BA0E36D0EC36A42669411FF8CE3579D&thid=OIP.ZItTWANYdG_qr0ql_2g4kAHaE8&mediaurl=https%3A%2F%2Fimage.c114.com.cn%2F20201231074626.jpg&exph=1280&expw=1920&q=%E9%9B%B7%E5%86%9B&simid=608005733423871393&FORM=IRPRST&ck=EE1FE1736A8FBBD195AA4E7905B44DBD&selectedIndex=12&itb=0&cw=1289&ch=698&ajaxhist=0&ajaxserp=0"
# response = multimodal_explanation(news, url)
# print(response)
