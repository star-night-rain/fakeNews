class Result(object):
    def __init__(self, code, message, data):
        self.code = code
        self.message = message
        self.data = data

    @classmethod
    def success(cls, data):
        return cls(200, "success", data)

    @classmethod
    def error(cls, data):
        return cls(500, "error", data)

    def to_dict(self):
        return {"code": self.code, "message": self.message, "data": self.data}


class CheckObject(object):
    def __init__(self, label, confidence, cost_time, check_time):
        self.label = label
        self.confidence = confidence
        self.cost_time = cost_time
        self.check_time = check_time


class ExplainObject(object):
    def __init__(
        self,
        response,
        words,
        weight,
    ):
        self.description = response["description"]
        self.backgrounds = response["backgrounds"]
        self.issue_title = response["issue_title"]
        self.issue_content = response["issue_content"]
        self.issue_count = len(self.issue_title)
        self.suggestion_title = response["suggestion_title"]
        self.suggestion_content = response["suggestion_content"]
        self.suggestion_count = len(self.suggestion_title)
        self.words = words
        self.weight = weight


class ScratchObject(object):
    def __init__(self, news):
        self.news_title = news["news_title"]
        self.platform = news["platform"]
        self.publish_time = news["publish_time"]
        self.news_link = news["news_link"]
        self.keyword = news["keyword"]
        self.news_content = news["news_content"]
        self.pic_url = news["pic_url"]

    def to_dict(self):
        return {
            "news_title": self.news_title,
            "platform": self.platform,
            "publish_time": self.publish_time,
            "news_link": self.news_link,
            "keyword": self.keyword,
            "news_content": self.news_content,
            "pic_url": self.pic_url,
        }

    # def __init__(
    #     self,
    #     news_title,
    #     platform,
    #     publish_time,
    #     news_link,
    #     keyword,
    #     news_content,
    #     pic_url,
    # ):
    #     self.news_title = news_title
    #     self.platform = platform
    #     self.publish_time = publish_time
    #     self.news_link = news_link
    #     self.keyword = keyword
    #     self.news_content = news_content
    #     self.pic_url = pic_url


class MultimodalCheckObject(object):
    def __init__(self, response, cost_time, check_time):
        self.label = response["label"]
        self.confidence = response["confidence"]
        self.consistency = response["consistency"]
        self.reason_title = response["reason_title"]
        self.reason_content = response["reason_content"]
        self.cost_time = cost_time
        self.check_time = check_time


class MultimodalExplainObject(object):
    def __init__(
        self,
        response,
        words,
        weight,
    ):
        self.description = response["description"]
        self.backgrounds = response["backgrounds"]
        self.issue_title = response["issue_title"]
        self.issue_content = response["issue_content"]
        self.issue_count = len(self.issue_title)
        self.suggestion_title = response["suggestion_title"]
        self.suggestion_content = response["suggestion_content"]
        self.suggestion_count = len(self.suggestion_title)
        self.words = words
        self.weight = weight
