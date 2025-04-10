import csv
import re
import urllib.parse
import tldextract
import datetime
import whois
import requests
from bs4 import BeautifulSoup
import pandas as pd
import socket
import ssl
import time

def extract_address_bar_features(url):
    """提取基于地址栏的特征"""
    features = {}
    
    # URL 长度
    features['url_length'] = len(url)
    
    # 是否直接使用 IP 地址
    try:
        domain = urllib.parse.urlparse(url).netloc
        socket.inet_aton(domain.split(':')[0])  # 检查是否是合法 IPv4
        features['uses_ip'] = 1
    except (socket.error, ValueError):
        features['uses_ip'] = 0
    
    # 点号的数量
    features['num_dots'] = url.count('.')
    
    # 协议 (http/https)
    features['protocol'] = 1 if urllib.parse.urlparse(url).scheme == 'https' else 0
    
    # 子域名数量
    extracted = tldextract.extract(url)
    subdomains = extracted.subdomain.split('.')
    features['num_subdomains'] = len(subdomains) if extracted.subdomain else 0
    
    return features

def extract_domain_based_features(url):
    """提取基于域名的特征"""
    features = {}
    
    try:
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"
        
        # 域名年龄
        try:
            whois_info = whois.whois(domain)
            if whois_info.creation_date:
                if isinstance(whois_info.creation_date, list):
                    creation_date = whois_info.creation_date[0]
                else:
                    creation_date = whois_info.creation_date
                domain_age = (datetime.datetime.now() - creation_date).days
                features['domain_age_days'] = domain_age
            else:
                features['domain_age_days'] = -1  # 未知
        except Exception as e:
            features['domain_age_days'] = -1
        
        # DNS 记录有效性（简单检查是否可以解析）
        try:
            socket.gethostbyname(domain)
            features['dns_valid'] = 1
        except socket.gaierror:
            features['dns_valid'] = 0
        
        # WHOIS 信息（是否可获取）
        features['whois_info_exists'] = 1 if whois_info else 0
        
    except Exception as e:
        # 如果出现错误，填充默认值
        features['domain_age_days'] = -1
        features['dns_valid'] = 0
        features['whois_info_exists'] = 0
    
    return features

def extract_html_js_features(url):
    """提取基于HTML和JavaScript的特征"""
    features = {}
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 是否存在 iframe
        features['has_iframe'] = 1 if soup.find('iframe') else 0
        
        # 是否存在混淆的 JavaScript（检查 script 标签中的常见混淆模式）
        scripts = soup.find_all('script')
        obfuscated = 0
        for script in scripts:
            if script.string:
                # 简单的混淆检测：检查是否存在十六进制编码、长字符串、eval 等
                if 'eval(' in script.string or '\\x' in script.string or 'unescape(' in script.string:
                    obfuscated = 1
                    break
        features['has_obfuscated_js'] = obfuscated
        
    except Exception as e:
        # 如果请求失败，填充默认值
        features['has_iframe'] = -1
        features['has_obfuscated_js'] = -1
    
    return features

def process_csv(input_csv, output_csv, url_column='url', label_column=None):
    """处理CSV文件并提取特征"""
    df = pd.read_csv(input_csv)
    results = []
    
    for index, row in df.iterrows():
        url = row[url_column]
        print(f"Processing {index + 1}/{len(df)}: {url}")
        
        features = {}
        
        # 提取三类特征
        features.update(extract_address_bar_features(url))
        features.update(extract_domain_based_features(url))
        features.update(extract_html_js_features(url))
        
        # 保留原始数据（如标签）
        if label_column and label_column in row:
            features['label'] = row[label_column]
        
        results.append(features)
        time.sleep(1)  # 避免请求过于频繁
    
    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"特征提取完成，结果已保存到 {output_csv}")

if __name__ == "__main__":
    
    input_file = "dataset1.csv"  # 输入文件
    output_file = "url_features.csv"  # 输出文件
    
    # 根据你的CSV格式选择正确的列名
    process_csv(input_file, output_file, url_column='url', label_column='label')