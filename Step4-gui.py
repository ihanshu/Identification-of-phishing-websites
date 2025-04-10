import tkinter as tk
import urllib
from socket import socket
from tkinter import messagebox
import joblib
import pandas as pd
import urllib.parse
import requests
from bs4 import BeautifulSoup
import whois
from datetime import datetime

import socket
import tldextract
import whois
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import urllib.parse


def extract_features(url):
    features = {}

    # 提取地址栏特征
    features['url_length'] = len(url)

    try:
        domain = urllib.parse.urlparse(url).netloc
        socket.inet_aton(domain.split(':')[0])  # 检查是否是合法 IPv4
        features['uses_ip'] = 1
    except OSError:
        features['uses_ip'] = 0

    features['num_dots'] = url.count('.')
    features['protocol'] = 1 if urllib.parse.urlparse(url).scheme == 'https' else 0

    extracted = tldextract.extract(url)
    subdomains = extracted.subdomain.split('.')
    features['num_subdomains'] = len(subdomains) if extracted.subdomain else 0

    # 提取域名特征
    whois_info = None
    try:
        domain = f"{extracted.domain}.{extracted.suffix}"
        whois_info = whois.whois(domain)
        if whois_info.creation_date:
            if isinstance(whois_info.creation_date, list):
                creation_date = whois_info.creation_date[0]
            else:
                creation_date = whois_info.creation_date
            domain_age = (datetime.datetime.now() - creation_date).days
            features['domain_age_days'] = domain_age
        else:
            features['domain_age_days'] = -1
    except Exception as e:
        features['domain_age_days'] = -1

    try:
        socket.gethostbyname(domain)
        features['dns_valid'] = 1
    except OSError:
        features['dns_valid'] = 0

    features['whois_info_exists'] = 1 if whois_info else 0

    # 提取HTML和JavaScript特征
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')

        features['has_iframe'] = 1 if soup.find('iframe') else 0

        scripts = soup.find_all('script')
        obfuscated = 0
        for script in scripts:
            if script.string:
                if 'eval(' in script.string or '\\x' in script.string or 'unescape(' in script.string:
                    obfuscated = 1
                    break
        features['has_obfuscated_js'] = obfuscated

    except Exception as e:
        features['has_iframe'] = -1
        features['has_obfuscated_js'] = -1

    return features

# 加载模型
model = joblib.load("phishing_url_model.pkl")

def classify_url():
    url = url_entry.get()
    if not url:
        messagebox.showwarning("警告", "请输入网址！")
        return

    try:
        # 提取特征
        features = extract_features(url)
        df = pd.DataFrame([features])

        # 预测
        prediction = model.predict(df)
        result = "钓鱼网址" if prediction[0] == 1 else "合法网址"
        messagebox.showinfo("分类结果", f"该网址是：{result}")
    except Exception as e:
        messagebox.showerror("错误", f"分析网址时出错：{str(e)}")

# 创建主窗口
root = tk.Tk()
root.title("网址分类器")

# 创建输入框
url_label = tk.Label(root, text="请输入网址：")
url_label.pack(pady=10)

url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=10)

# 创建按钮
classify_button = tk.Button(root, text="分析网址", command=classify_url)
classify_button.pack(pady=20)

# 运行主循环
root.mainloop()