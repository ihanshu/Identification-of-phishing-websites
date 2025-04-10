import pandas as pd
import requests
import logging
from urllib.parse import urlparse
from pathlib import Path
import time
import random

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_phishing_data(file_path: str) -> pd.DataFrame:
    """加载本地PhishTank CSV文件并清洗数据"""
    try:
        df = pd.read_csv(file_path)
        # 清洗数据：保留已验证的钓鱼URL并去重
        if "verified" in df.columns:
            df = df[df["verified"] == "yes"]
        df = df[["url"]].drop_duplicates()
        df["label"] = 1  # 标记为钓鱼
        logging.info(f"The phishing URL {len(df)} was successfully loaded")
        return df
    except FileNotFoundError:
        logging.error(f"文件 {file_path} 不存在，请检查路径")
        raise


def fetch_commoncrawl_urls(num_urls: int = 5000) -> pd.DataFrame:
    """从CommonCrawl索引中随机获取合法URL"""
    try:
        # 获取最新的CommonCrawl索引
        index_api = "https://index.commoncrawl.org/collinfo.json"
        response = requests.get(index_api)
        latest_crawl = response.json()[0]["id"]  # 选择最新的爬取批次

        # 查询随机域名（示例：获取.com域名）
        search_api = f"https://index.commoncrawl.org/{latest_crawl}-index?url=*.com&output=json"
        response = requests.get(search_api)
        records = response.text.strip().split("\n")

        # 解析URL并去重
        urls = [eval(record)["url"] for record in records if record]
        urls = list(set(urls))  # 去重
        random.shuffle(urls)  # 打乱顺序

        # 截取所需数量
        sampled_urls = urls[:num_urls]
        df = pd.DataFrame({"url": sampled_urls, "label": 0})
        logging.info(f"从CommonCrawl获取 {len(df)} 条合法URL")
        return df
    except Exception as e:
        logging.error(f"获取CommonCrawl数据失败: {str(e)}")
        raise


def filter_legit_urls(df: pd.DataFrame) -> pd.DataFrame:
    """过滤合法URL（基础规则）"""
    valid_domains = ["com", "org", "net", "edu", "gov"]
    df["domain"] = df["url"].apply(lambda x: urlparse(x).netloc.split(".")[-1])
    df = df[df["domain"].isin(valid_domains)]
    df = df.drop("domain", axis=1)
    return df


def balance_dataset(df_phish: pd.DataFrame, df_legit: pd.DataFrame) -> pd.DataFrame:
    """平衡数据集并打乱顺序"""
    min_samples = min(len(df_phish), len(df_legit))
    balanced_df = pd.concat([
        df_phish.sample(min_samples, random_state=42),
        df_legit.sample(min_samples, random_state=42)
    ])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    logging.info(f"have balanced: fishing {min_samples} 条, commoncrawl {min_samples} 条")
    return balanced_df


def main():
    # 配置路径和参数
    phish_path = r"D:\phish_url.csv"  # 本地PhishTank文件路径
    output_path = "./phishing_dataset1.csv"

    # 加载数据
    df_phish = load_phishing_data(phish_path)
    df_legit = fetch_commoncrawl_urls(num_urls=2 * len(df_phish))  # 多获取一些以应对过滤损失
    df_legit = filter_legit_urls(df_legit)

    # 平衡数据集
    balanced_df = balance_dataset(df_phish, df_legit)

    # 保存结果
    balanced_df.to_csv(output_path, index=False)
    logging.info(f"The dataset has been saved to {Path(output_path).resolve()}")


if __name__ == "__main__":
    main()