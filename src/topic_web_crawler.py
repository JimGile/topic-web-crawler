from typing import List, Dict
import scrapy
from scrapy.crawler import CrawlerProcess
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from datetime import datetime


class CrawledResultsHandler:
    def __init__(self, crawled_results: List[Dict]):
        self.metadata_schema = {
            'url': str,
            'topic': str,
            'crawl_timestamp': str,
            'domain': str,
            'content_length': int
        }
        self.crawled_results = crawled_results

    def create_documents_with_metadata(self) -> List[Document]:
        documents = []

        for result in self.crawled_results:
            # Create metadata dictionary
            metadata = {
                'url': result['url'],
                'topic': result['topic'],
                'crawl_timestamp': datetime.now().isoformat(),
                'domain': result['url'].split('/')[2],
                'content_length': len(result['content']),
            }

            # Create Document object with content and metadata
            if self._validate_metadata(metadata):
                doc: Document = Document(
                    page_content=result['content'],
                    metadata=metadata
                )
                documents.append(doc)

        return documents

    def create_vectorstore(self, documents: List[Document], embedding_function=None, persist_directory: str = None) -> Chroma:
        # Create Chroma vectorstore with metadata
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function or OpenAIEmbeddings(),
            persist_directory=persist_directory,  # Optional: for persistence
        )

        return vectorstore

    def _validate_metadata(self, metadata: Dict) -> bool:
        try:
            for key, expected_type in self.metadata_schema.items():
                if key in metadata:
                    if not isinstance(metadata[key], expected_type):
                        return False
            return True
        except Exception:
            return False


class TopicWebCrawler:
    def __init__(self, domain: str, topics: List[str]):
        self.domain = domain
        self.topics = topics
        self.process = CrawlerProcess({
            'USER_AGENT': 'Mozilla/5.0',
            'ROBOTSTXT_OBEY': True,
            'CONCURRENT_REQUESTS': 32,
            'DOWNLOAD_DELAY': 1,
        })

    def crawl(self) -> CrawledResultsHandler:
        spider = TopicSpider(domain=self.domain, topics=self.topics)
        self.process.crawl(spider)
        self.process.start()
        results = CrawledResultsHandler(spider.results)
        return results


class TopicSpider(scrapy.Spider):
    name = 'topic_spider'

    def __init__(self, domain: str, topics: List[str], *args, **kwargs):
        super(TopicSpider, self).__init__(*args, **kwargs)
        self.allowed_domains = [domain]
        self.start_urls = [f'https://{domain}']
        self.topics = topics
        self.results = []

    def parse(self, response):
        # Extract text content from main content areas
        content = ' '.join(response.css(
            'p::text, h1::text, h2::text, h3::text').getall())

        # Check if any of our topics are mentioned
        for topic in self.topics:
            if topic.lower() in content.lower():
                self.results.append({
                    'url': response.url,
                    'content': content,
                    'topic': topic
                })

        # Follow links within the same domain
        for href in response.css('a::attr(href)').getall():
            yield response.follow(href, self.parse)
