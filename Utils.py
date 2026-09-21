import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import random
import json
import pandas as pd
from pathlib import Path
from Data_protocol import END_BLOCK, write_json

class DataSource:
    """
    Ethereum blockchain data retrieval class
    
    Provides methods to fetch transaction data from Etherscan API, including normal transactions,
    internal transactions, and token transfers. Uses multiple API keys in rotation to avoid
    exceeding API request limits.
    """

    def __init__(self):
        self.apikeys = [
            "**********************************",  # outlook  --reverse _reverse文件
            "**********************************"
        ]
        self.headers = {
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        }
        self.url_rpc = "http://**.**.**.**:****"
        # self.url_rpc = "https://rpc.ankr.com/eth"

    def _get_etherscan_data(self, module, action, address, startblock, endblock, page=1, offset=10000, sort="asc", **kwargs):
        """Generic method to fetch data from Etherscan API"""
        params = {
            "chainid": 1,
            "module": module,
            "action": action,
            "address": address,
            "startblock": startblock,
            "endblock": endblock,
            "page": page,
            "offset": offset,
            "sort": sort,
            "apikey": random.choice(self.apikeys)
        }
        
        # Add any additional parameters
        params.update(kwargs)
        
        # Build URL with parameters
        url_params = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"https://api.etherscan.io/v2/api?{url_params}"
        
        response = getDataFromUrl(url, self.headers)
        if response is None:
            raise RuntimeError('Etherscan request failed')
        payload = response.json()
        result = payload.get('result')
        if isinstance(result, list) and (payload.get('status') == '1' or payload.get('message') == 'No transactions found'):
            return result
        if payload.get('message') == 'No transactions found':
            return []
        raise RuntimeError('Etherscan did not return a successful transaction list')

    def getNormalTransactionsbyAddress(self, address, startblock, endblock, page, offset=10000, sort="asc"):
        """Get a list of 'Normal' Transactions By Address"""
        return self._get_etherscan_data("account", "txlist", address, startblock, endblock, page, offset, sort)

    def getInternalTransactionsbyAddress(self, address, startblock, endblock, page=1, offset=10000, sort="asc"):
        """Get a list of 'Internal' Transactions by Address (ETH)"""
        return self._get_etherscan_data("account", "txlistinternal", address, startblock, endblock, page, offset, sort)
    
    def getInternalTransactionsbyTransactionHash(self, txhash):
        """Get a list of 'Internal' Transactions by Transaction Hash"""
        params = {
            "chainid": 1,
            "module": "account",
            "action": "txlistinternal",
            "txhash": txhash,
            "apikey": random.choice(self.apikeys)
        }
        url_params = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"https://api.etherscan.io/v2/api?{url_params}"
        
        response = getDataFromUrl(url, self.headers)
        if response is None:
            raise RuntimeError('Etherscan request failed')
        payload = response.json()
        result = payload.get('result')
        if isinstance(result, list) and (payload.get('status') == '1' or payload.get('message') == 'No transactions found'):
            return result
        if payload.get('message') == 'No transactions found':
            return []
        raise RuntimeError('Etherscan did not return a successful transaction list')

    def getERCTokenTransferbyAddress(self, action, address, startblock, endblock, page, offset=10000, contractaddress="", sort="asc"):
        """
        Get a list of Token Transfer Events by Address
        
        action: one of [tokentx, tokennfttx, token1155tx]
        """
        kwargs = {}
        if contractaddress:
            kwargs["contractaddress"] = contractaddress
            
        return self._get_etherscan_data("account", action, address, startblock, endblock, page, offset, sort, **kwargs)

    # Core transaction columns used by Graph_construction.py.  ERC20 metadata
    # must be retained: token ``value`` is an integer in the token's smallest
    # unit and cannot be converted with the ETH 1e18 divisor.
    CORE_COLUMNS = [
        'blockNumber', 'from', 'to', 'value', 'gasUsed', 'gasPrice', 'timeStamp',
        'contractAddress', 'tokenDecimal', 'tokenSymbol', 'hash', 'traceId',
        'logIndex', 'transactionIndex', 'isError', 'txreceipt_status'
    ]
    
    def getTotalDatafromScan(self, address, ttype, saved_path, start_number=0, end_number=END_BLOCK):
        if start_number < 0 or end_number < start_number:
            raise ValueError('Invalid block interval')
        address = address.strip().lower()
        path = Path(saved_path) / f'{address}.csv'
        methods = {
            'Normal/': lambda start, end, page: self.getNormalTransactionsbyAddress(address, start, end, page),
            'Internal/': lambda start, end, page: self.getInternalTransactionsbyAddress(address, start, end, page),
            'ERC20/': lambda start, end, page: self.getERCTokenTransferbyAddress('tokentx', address, start, end, page),
        }
        if ttype not in methods:
            raise ValueError(f'Unknown transaction type: {ttype}')
        fetch = methods[ttype]
        pending = [(start_number, end_number)]
        rows = []
        while pending:
            lower, upper = pending.pop()
            response = fetch(lower, upper, 1)
            if not isinstance(response, list) or any(not isinstance(item, dict) for item in response):
                raise RuntimeError('Invalid transaction response; no cache was written')
            if len(response) >= 10000 and lower < upper:
                midpoint = (lower + upper) // 2
                pending.extend([(midpoint + 1, upper), (lower, midpoint)])
                continue
            rows.extend(response)
            page = 1
            previous_page = json.dumps(response, sort_keys=True)
            while len(response) >= 10000:
                page += 1
                response = fetch(lower, upper, page)
                if not isinstance(response, list) or any(not isinstance(item, dict) for item in response):
                    raise RuntimeError('Single-block pagination failed; no cache was written')
                current_page = json.dumps(response, sort_keys=True)
                if current_page == previous_page:
                    raise RuntimeError('Repeated transaction page; no cache was written')
                previous_page = current_page
                rows.extend(response)
        frame = pd.DataFrame([{key: row.get(key, '') for key in self.CORE_COLUMNS} for row in rows], columns=self.CORE_COLUMNS)
        if not frame.empty:
            blocks = pd.to_numeric(frame['blockNumber'], errors='coerce')
            if not blocks.between(start_number, end_number).all():
                raise ValueError('API returned transactions outside the requested block range')
            frame = frame.assign(_block=blocks).sort_values('_block', kind='stable').drop(columns='_block')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.csv.tmp')
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        write_json(path.with_suffix('.meta.json'), {'complete': True, 'format_version': 2,
                   'start_block': start_number, 'end_block': end_number, 'row_count': len(frame)})
        return True, len(frame)

    def getTransactionCountfromRPC(self, address):
        """Get transaction count for an address using RPC"""
        payload = {
            "method": "eth_getTransactionCount",
            "params": [address, "latest"],
            "id": 1,
            "jsonrpc": "2.0",
        }
        return int(getDatafromRPC(payload)["result"], 16)
        
    def getBalancefromRPC(self, address):
        """Get balance for an address using RPC"""
        payload = {
            "method": "eth_getBalance",
            "params": [address, "latest"],
            "id": 1,
            "jsonrpc": "2.0",
        }
        return int(getDatafromRPC(payload)["result"], 16) / 10**18  # Convert from wei to ETH

# 防止出错的多次请求
def getDataFromUrl(url, headers, data=None, sstype='get', timeout=20):
    # 设置重试策略
    retries = Retry(total=10, backoff_factor=0.9)
    
    with requests.Session() as session:
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        
        try:
            if sstype == 'get':
                response = session.get(url, headers=headers, data=data, timeout=timeout)
            else:  # post
                response = session.post(url, headers=headers, data=data, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            # 静默失败，不打印错误信息
            return None
            
# 访问Etherscan具体地址页面，查看是否存在标签
def getAddressLabelFromEthereumPage(address):
    headers = {
        'cookie': '__stripe_mid=901553e0-d3df-4424-8b9e-fcffccc0d698055c37; etherscan_offset_datetime=+8; etherscan_cookieconsent=True; etherscan_switch_token_amount_value=value; etherscan_switch_age_datetime=Age; _ga=GA1.1.882709905.1679023127; _ga_T1JC9RNQXV=GS1.1.1739777391.69.1.1739778303.60.0.0; __cflb=02DiuFnsSsHWYH8WqVXaqSXf986r8yFDsA1zoNWuQ6RPr; ASP.NET_SessionId=r5np40os0i1h3hdlekver0ce; cf_clearance=fukKVqTem3TBD..0bzxJsuuNQ_XmGLFkdQHmGi8e3zk-1744614664-1.2.1.1-IjP9nOVD0EyQ_SWh5Zq8nUHBJDPWTo2puZ_z3_iFGwCwEeNkxOz4wLrpPtLZ4hrGdRbUNgJnhl2lBWtevi4PszUA3veNvmy7k9VtarQYwTzqkYUd5DaxWmOAWKVsmyuyTBTXRbOQ.pG.dUx8AYDlclF0xOPeHmobO1DkVtikFNFTAavZ6tC29qFKgTYd5ZTi_M1L1SvBg687ZVf6bLYRzUwnMBmNC53laUGOZhhLxVz72LFKZ4VFFTBDkMP4ot6JE3qMfs_peH0rWb3iQAxi_AXXuDyRROaaHZgf.j97KJ0vyFXyLn6zGvWX3RGIzdgX4G0gbmX79eYqsNpaEhdf59ao_vfrTensmMltqiGi6N48iWKg.7A8d39cdj2zkdNr',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
        'origin': 'https://etherscan.io'
    }
    url_address = f'https://etherscan.io/address/{address}'
    
    response = getDataFromUrl(url=url_address, headers=headers)
    if not response:
        return "", "", ""
        
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 查找标签区域
    target_section = soup.select_one('body main section:nth-of-type(3) div:nth-of-type(1) div:nth-of-type(1)')
    
    label_text = ""
    span_text = ""
    
    if target_section:
        all_spans = target_section.find_all('span')
        all_span_contents = [span.get_text().strip() for span in all_spans if span.get_text().strip()]
        
        # 查找包含Fake_Phishing的标签
        for content in all_span_contents:
            if "Fake_Phishing" in content:
                span_text = content
                break
        
        # 如果没找到特定标签，合并所有span内容
        if not span_text and all_span_contents:
            unique_contents = list(dict.fromkeys(all_span_contents))
            span_text = ";".join(unique_contents)
        
        # 优先查找包含Phish的标签
        for content in all_span_contents:
            if "Phish" in content:
                label_text = content
                break
        
        # 如果没找到包含Phish的标签，使用第一个非空标签
        if not label_text and all_span_contents:
            label_text = all_span_contents[0]
    
    # 从页面标题获取标签
    title_text = ""
    if soup.title:
        title_parts = soup.title.string.split("|")
        if len(title_parts) >= 3:
            title_text = title_parts[0].strip().split("\n")[0]
    
    return span_text, title_text, label_text
            
def getDatafromRPC(payload):
    url_rpc = "http://**.**.**.**:****"
    headers = {
        "content-type": "application/json",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    }
    response = getDataFromUrl(url_rpc, headers=headers, data=json.dumps(payload), sstype='post')
    return response.json() if response else None

if __name__ == "__main__":
    data_source = DataSource()
    address = '0xdAC17F958D2ee523a2206206994597C13D831ec7'
    tx = data_source.getNormalTransactionsbyAddress(address, 22986299, 22986300, 1)
    print(tx)
