import serpapi


def search_information(query):
    params = {
        "engine": "bing",
        "q": query,
        "hl": "zh-CN",
        "gl": "cn",
        "api_key": "daafaf2bf38c2316ae3625999e642c9b15dec5e18ac492f8bc9ea00adaa98c1d",
    }
    search = serpapi.search(params)
    return search
