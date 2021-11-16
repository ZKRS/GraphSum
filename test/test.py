import datasets

# os.environ["http_proxy"] = "http://127.0.0.1:8118"
# os.environ["https_proxy"] = "http://127.0.0.1:8118"
# socks.set_default_proxy(socks.PROXY_TYPE_HTTP, "127.0.0.1", 8118)
# socket.socket = socks.socksocket
wikihow = datasets.load_dataset('wikihow', "all", data_dir="/tmp/Data/wikihow/url")
# gigaword = datasets.load_dataset('gigaword')
# datasets.load_dataset()
# import requests
#
#
# def main():
#   url = 'https://www.google.com'
#   html = requests.get(url).text
#   print(html)
#
#
# if __name__ == '__main__':
#   main()
# import requests
# requests.request()
