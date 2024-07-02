import bs4
from langchain_community.document_loaders import WebBaseLoader


def load_web_page(page_url: str, content_class: str):
    loader = WebBaseLoader(
        web_paths=(page_url),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=(content_class))),
        # mimic the behavior of a web browser to make the request
        header_template={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
        },
    )
    return loader
