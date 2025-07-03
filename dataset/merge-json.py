# import json

# with open("./merged_data.json", "r", encoding="utf-8") as file:
#     datas = json.load(file)

# cnt = 1
# for row in datas:
#     row["id"] = cnt
#     cnt += 1
#     print(row)
# with open("./data.json", "w", encoding="utf-8") as file:
#     json.dump(datas, file, ensure_ascii=False, indent=4)

import json

# 读取数据
with open("./data.json", "r", encoding="utf-8") as file:
    datas = json.load(file)

with open("./data1.json", "r", encoding="utf-8") as file:
    datas1 = json.load(file)

# 合并两个字典列表
merged_list = []

# 将 datas 和 datas1 合并为一个列表
all_dicts = datas + datas1

# 使用字典的 (publish_time, title, content, label) 作为键，存储每个键对应的字典信息
dict_by_key = {}

for d in all_dicts:
    # 选取需要比较的字段
    key = (d["publish_time"], d["title"], d["content"], d["label"])

    # 如果 key 不存在，加入 dict_by_key 中
    if key not in dict_by_key:
        dict_by_key[key] = d
    else:
        # 如果 key 相同，则不添加该字典，避免重复
        pass

# 获取合并后的字典列表
merged_list = list(dict_by_key.values())

# 输出合并后的结果
print(merged_list)

# 如果需要将结果写回文件
with open("./merged_data.json", "w", encoding="utf-8") as file:
    json.dump(merged_list, file, ensure_ascii=False, indent=4)
