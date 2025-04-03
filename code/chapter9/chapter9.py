from collections import Counter
import csv
from bs4 import BeautifulSoup
import requests
import re

# 'r'は読み取り専用を意味する。省略した場合のデフォルト引数でもある
file_for_reading = open('./other_text/reading_file.txt', 'r')
file_for_reading2 = open('./other_text/reading_file.txt')

# 'w'はwrite(書き込み)を表す -- 既存のファイルを壊す可能性がある
file_for_writing = open('./other_text/writing_file.txt', 'w')

# 'a'はappend(追記)を表す -- ファイルの末尾に追記する
file_for_append = open('./other_text/appending_file.txt', 'a')

# ファイルを使い終わったら、クローズを忘れないように
file_for_reading.close()
file_for_reading2.close()
file_for_writing.close()
file_for_append.close()

def process(date: str, symbol: str, closing_price: float) -> None:
    # Imaginge that this function actually does something.
    assert closing_price > 0.0

def get_domain(email_address: str):
    """'@'で分割して、後ろの部分を返す"""
    return email_address.lower().split('@')[-1]

# テストをいくつか
assert get_domain('joelgrus@gmail.com') == 'gmail.com'
assert get_domain('joel@m.datasciencester.com') == 'm.datasciencester.com'

with open('./other_text/email_addresses.txt', 'r') as f:
    domain_counts = Counter(get_domain(line.strip()) for line in f if '@' in line)
print(domain_counts)

with open('./other_text/tab_delimited_stock_prices.txt') as f:
    tab_reader = csv.reader(f, delimiter='\t')
    for row in tab_reader:
        data = row[0]
        symbol = row[1]
        closing_price = float(row[2])
        process(data, symbol, closing_price)
        
with open('./other_text/colon_delimited_stock_prices.txt') as f:
    colon_reader = csv.DictReader(f, delimiter=':')
    for dict_row in colon_reader:
        date = dict_row["date"]
        symbol = dict_row["symbol"]
        closing_price = float(dict_row["closing_price"])
        process(date, symbol, closing_price)
        
todays_prices = {'AAPL': 90.91, 'MSFT': 41.68, 'FB': 64.5}

with open('other_text/comma_delimited_stock_prices.txt', 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    for stock, price in todays_prices.items():
        csv_writer.writerow([stock, price])
        
results = [["test1", "success", "Monday"],
           ["test2", "success, kind of", "Tuesday"],
           ["test3", "failure, kind of", "Wednesday"],
           ["test4", "failure, utter", "Thursday"]]

with open('./other_text/bad_csv.txt', 'w') as f:
    for row in results:
        f.write(",".join(map(str, row))) # おそらく必要以上のカンマ
                                        #  区切り文字が書き込まれる
        f.write("\n")

# 関連するHTMLファイルはGitHubにある
url = ("https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html")
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

first_paragraph_test  = soup.p.text
first_paragraph_words = soup.p.text.split()

print(first_paragraph_test)
print(first_paragraph_words)

first_paragraphs_id  = soup.p['id']     # 'id'がなければ、KeyErrorとなる
first_paragraphs_id2 = soup.p.get('id') # 'id'がなければNoneが返る

print(first_paragraphs_id)
print(first_paragraphs_id2)

all_paragraphs = soup.find_all('p')     # または、単にsoup('p')
paragraphs_with_ids = [p for p in soup('p') if p.get('id')]

print(all_paragraphs)
print(paragraphs_with_ids)

important_paragraphs  = soup('p', {'class': 'important'})
important_paragraphs2 = soup('p', 'important')
important_paragraphs3 = [p for p in soup('p') if 'important' in p.get('class', [])]

print(important_paragraphs)
print(important_paragraphs2)
print(important_paragraphs3)

# 注意：複数の<div>の中にある場合（div要素が他のdiv要素の入れ子になっている場合）
# 同じ<span>が複数回返る
# その場合は、さらに工夫が必要
spans_inside_divs = [span
                     for div in soup('div')   # ページ上の<div>要素ごとに繰り返し
                     for span in div('span')] # その中の<span>要素で繰り返し
print(spans_inside_divs)

url = "https://www.house.gov/representatives"
text = requests.get(url).text
soup = BeautifulSoup(text, "html5lib")

all_urls = [a['href']
            for a in soup('a')
            if a.has_attr('href')]
print(len(all_urls))

# http://またはhttps://で開始し
# .house.govまたは.house.gov/で終了する
regex = r"^https?://.*\.house\.gov/?$"

# テストをいくつか
assert re.match(regex, "http://joel.house.gov")
assert re.match(regex, "https://joel.house.gov")
assert re.match(regex, "http://joel.house.gov/")
assert re.match(regex, "https://joel.house.gov/")
assert not re.match(regex, "https://joel.house.com")
assert not re.match(regex, "https://joel.house.gov/biography")

# 適用する
good_urls = [url for url in all_urls if re.match(regex, url)]
print(len(good_urls))
good_urls = list(set(good_urls))
print(len(good_urls))

html = requests.get('https://jayapal.house.gov').text
soup = BeautifulSoup(html, 'html5lib')

# リンクが複数存在する可能性があるため、集合を使用する
links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}
print(links) # {'/media/press-releases'}

press_releases: dict[str, set[str]] = {}

for house_url in good_urls:
    html = requests.get(house_url).text
    soup = BeautifulSoup(html, 'html5lib')
    pr_links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}
    
    print(f"{house_url}: {pr_links}")
    press_releases[house_url] = pr_links
    
def paragraph_mentions(text: str, keyword: str) -> bool:
    """テキスト内の<p>が{keyword}に言及している場合にTrueを返す"""
    soup = BeautifulSoup(text, 'html5lib')
    paragraphs = [p.get_text() for p in soup('p')]
    
    return any(keyword.lower() in paragraph.lower() for paragraph in paragraphs)

text = """<body><h1>Facebook</h1><p>Twitter</p>"""
assert paragraph_mentions(text, "twitter")      # <p>の中で言及されている
assert not paragraph_mentions(text, "facebook") # <p>の中で言及されていない

for house_url, pr_links in press_releases.items():
    for pr_link in pr_links:
        url = f"{house_url}/{pr_link}"
        text = requests.get(url).text
        
        if paragraph_mentions(text, 'data'):
            print(f"{house_url}")
            break # 目的のhouse_urlを発見