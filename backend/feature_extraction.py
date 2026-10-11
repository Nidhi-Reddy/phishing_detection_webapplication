import re
import pandas as pd
from urllib.parse import urlparse

def extract_features(url):

    domain = urlparse(url).netloc

    features = {
        "having_IP_Address": 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else -1,
        "URL_Length": 1 if len(url) >= 54 else -1,
        "Shortining_Service": 1 if re.search(r"bit\.ly|tinyurl|goo\.gl|t\.co", url) else -1,
        "having_At_Symbol": 1 if "@" in url else -1,
        "double_slash_redirecting": 1 if url.rfind("//") > 7 else -1,
        "Prefix_Suffix": 1 if "-" in domain else -1,
        "having_Sub_Domain": 1 if domain.count(".") > 2 else -1,
        "SSLfinal_State": 1 if url.startswith("https") else -1,
    }

    return pd.DataFrame([features])