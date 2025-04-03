import json
import requests

from collections import Counter
from dateutil.parser import parse


serialized = """{"title":  "Data Science Book",
                 "author": "Joel Grus",
                 "publicationYear": 2019,
                 "topics": ["data", "science", "data science"]}"""

# JSONからPython辞書を作る
deserialized = json.loads(serialized)
assert deserialized["publicationYear"] == 2019
assert "data science" in deserialized["topics"]

github_user = "baum-51"
endpoint = f"https://api.github.com/users/{github_user}/repos"

repos = json.loads(requests.get(endpoint).text)
# print(repos)

dates = [parse(repo["created_at"]) for repo in repos]
month_counts = Counter(date.month for date in dates)
weekday_counts = Counter(date.weekday() for date in dates)

last_5_repositories = sorted(repos,
                             key=lambda r: r["pushed_at"],
                             reverse=True)[:5]
last_5_languages = [repo["language"] for repo in last_5_repositories]
