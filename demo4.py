import os
from openai import OpenAI

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
    api_key="8e704977-78ae-40e3-8f3a-7d3fa3f8edb4",
)

news = "2023/2/2,离婚窗口排长队"
completion = client.chat.completions.create(
    model="bot-20250409201752-rl95z",
    messages=[
        {
            "role": "system",
            "content": "你是一位专业的事实核查助手，负责从新闻中提取关键信息，并识别其中可能存在的事实错误或逻辑问题。注意，不能使用markdown格式回答。",
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
print(completion.choices[0].message.content)
# if hasattr(completion, "references"):
#     print(completion.references)
