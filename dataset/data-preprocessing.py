import pandas as pd
import json
from datetime import datetime
from utils import enrich_knowledge


def get_json():
    df = pd.read_csv("./training-data.csv", sep=",", encoding="utf-8")
    df = df[df["label"].isin(["谣言", "事实"])]
    with open("./data.json", "r", encoding="utf-8") as file:
        datas = json.load(file)
    cnt = 0
    for index, row in df.iterrows():
        if pd.isna(row["title"]) or pd.isna(row["content"]):
            cnt += 1
            if cnt <= 1500:
                continue
            data = {}
            data["id"] = len(datas) + 1

            publish_time = row["publish_time"]
            if pd.notna(publish_time):
                publish_time = datetime.strptime(publish_time, "%Y/%m/%d")
                data["publish_time"] = (
                    f"{publish_time.year}年{publish_time.month}月{publish_time.day}日"
                )
            else:
                data["publish_time"] = ""

            data["title"] = row["title"] if pd.notna(row["title"]) else ""
            data["content"] = row["content"] if pd.notna(row["content"]) else ""

            data["label"] = 1 if row["label"] == "谣言" else 0

            news = ""
            if pd.notna(row["publish_time"]):
                news = data["publish_time"] + ","

            if pd.notna(row["title"]):
                news += data["title"]

            if pd.notna(row["content"]):
                news += data["content"]

            response = enrich_knowledge(news)
            if response == None:
                continue

            data["backgrounds"] = response["backgrounds"]
            data["issues"] = response["issues"]

            datas.append(data)
            with open("data.json", "w", encoding="utf-8") as file:
                json.dump(datas, file, ensure_ascii=False, indent=4)

            if cnt >= 2400:
                break

    print(f"cnt:{cnt}")


if __name__ == "__main__":
    get_json()
