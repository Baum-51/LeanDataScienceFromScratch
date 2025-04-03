import json
import requests

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