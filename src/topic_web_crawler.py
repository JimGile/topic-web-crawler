import scrapy
import pandas as pd
from typing import List, Dict
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.signalmanager import dispatcher
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from datetime import datetime
from urllib.parse import urlparse, urljoin
from pathlib import Path


class CrawledResultsHandler:
    def __init__(self, crawled_results: List[Dict], crawled_urls: set = None):
        self.metadata_schema = {
            'crawl_timestamp': str,
            'domain': str,
            'url': str,
            'topics': str,
            'title': str,
            'preview': str,
            'content_length': int
        }
        self.crawled_results = crawled_results
        self.crawled_urls = crawled_urls

    def create_documents_with_metadata(self) -> List[Document]:
        documents = []

        for result in self.crawled_results:
            # Create metadata dictionary
            metadata = {
                'crawl_timestamp': datetime.now().isoformat(),
                'domain': result['url'].split('/')[2],
                'url': result['url'],
                'topics': result['topics'],
                'title': result['title'],
                'preview': result['preview'],
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

    def export_to_csv(self, documents: List[Document], output_dir: str = "crawler_output") -> str:
        """
        Export documents and metadata to CSV file.
        Returns path to created CSV file.
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepare data for documents CSV
            documents_data = []
            for doc in documents:
                doc_dict = {
                    # Unpack all metadata fields
                    **doc.metadata,
                    'content': doc.page_content
                }
                documents_data.append(doc_dict)

            # Define file paths
            documents_file = output_dir / f"documents_{timestamp}.csv"

            # Export documents to CSV
            if documents_data:
                df_documents = pd.DataFrame(documents_data)
                df_documents.to_csv(
                    documents_file, index=False, encoding='utf-8')
                print(f"Documents exported to: {documents_file}")

            return str(documents_file)

        except Exception as e:
            print(f"Error exporting to CSV: {str(e)}")
            raise

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
    def __init__(self, domain: str, start_url: str, topics: List[str]):
        self.domain = domain
        self.start_url = start_url
        self.topics = topics
        self.process = CrawlerProcess({
            'USER_AGENT': 'Mozilla/5.0',
            'ROBOTSTXT_OBEY': True,
            'CONCURRENT_REQUESTS': 16,
            'DOWNLOAD_DELAY': 1,
            'COOKIES_ENABLED': False,
            'RETRY_ENABLED': True,
            'RETRY_TIMES': 3,
            'DOWNLOAD_TIMEOUT': 30,
            'DEPTH_LIMIT': 10,  # Limit crawl depth
            'CLOSESPIDER_PAGECOUNT': 1000,  # Limit crawl to 100 pages
            'CLOSESPIDER_ITEMCOUNT': 1000,  # Limit crawl to 100 items
            'CLOSESPIDER_ERRORCOUNT': 10,  # Limit crawl to 10 errors
        })

    def spider_closed(self, spider):
        self.results = spider.results
        self.crawled_urls = spider.crawled_urls

    def crawl_and_process(self) -> CrawledResultsHandler:
        # Connect the spider_closed handler to the spider_closed signal
        dispatcher.connect(self.spider_closed, signal=signals.spider_closed)

        # Crawl - Correct way to pass spider class and settings
        self.process.crawl(
            TopicSpider,
            domain=self.domain,
            start_url=self.start_url,
            topics=self.topics
        )

        # This will block until crawling is finished
        self.process.start()

        # Process results
        return CrawledResultsHandler(self.results, self.crawled_urls)


class TopicSpider(scrapy.Spider):
    name = 'topic_spider'

    def __init__(self, domain: str, start_url: str, topics: List[str], *args, **kwargs):
        super(TopicSpider, self).__init__(*args, **kwargs)
        self.allowed_domains = [domain]
        self.start_urls = [start_url]
        self.topics = topics
        self.crawled_urls = set()
        self.results = []

    def is_valid_url(self, url: str) -> bool:
        """Check if the URL is valid and should be followed."""
        if not url \
                or url.startswith('mailto:') \
                or url.startswith('tel:') \
                or url.startswith('javascript:') \
                or any(url.lower().endswith(ext) for ext in [
                    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar',
                    '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3'
                ]):
            # print(f"Ignoring invalid URL: {url}")
            return False
        elif url in self.crawled_urls:
            # print(f"Ignoring already crawled URL: {url}")
            return False
        else:
            # Check if URL is in allowed domain
            try:
                parsed_url = urlparse(url)
                return parsed_url.netloc in self.allowed_domains
            except Exception:
                return False

    def extract_title(self, response) -> str:
        """Extract page title with priority given to oc-page-title class."""
        title = None

        # Method 1: Try the oc-page-title class first
        title = response.css('h1.oc-page-title::text').get()

        # Method 2: Fallback to title tag if oc-page-title not found
        if not title:
            title = response.css('title::text').get()

        # Method 3: Try og:title meta tag
        if not title:
            title = response.css(
                'meta[property="og:title"]::attr(content)').get()

        # Clean up the title
        if title:
            # Remove extra whitespace and newlines
            title = ' '.join(title.strip().split())

        return title or "No title found"

    # def extract_main_content(self, response) -> LiteralString | Literal['']:
    #     """Extract content specifically from main-content div."""
    #     try:
    #         # First, select the main-content div
    #         main_content = response.css('#main-content')

    #         if not main_content:
    #             self.logger.warning(
    #                 f"No #main-content div found on {response.url}")
    #             return ""

    #         # Extract text from main content areas
    #         content_selectors = main_content.xpath(
    #             './/p|.//h1|.//h2|.//h3|.//ul)]'
    #         )

    #         # Extract and clean content
    #         content_parts = []
    #         for text in content_selectors.css('::text').getall():
    #             text = text.strip()
    #             if text and not text.isspace():
    #                 content_parts.append(text)

    #         # Debug logging
    #         content = ' '.join(content_parts)
    #         self.logger.debug(f"main-content length: {len(content)}")

    #         return content

    #     except Exception as e:
    #         self.logger.error(f"Error in extract_clean_content: {str(e)}")
    #         return ""

    def extract_clean_content(self, response):
        """Extract content while excluding the PublicEmergencyAnnouncementList div."""
        try:
            # First, get a copy of the response selector without the emergency div
            content_selectors = response.xpath(
                '//p[not(ancestor::div[@id="PublicEmergencyAnnouncementList"])]|' +
                '//h1[not(ancestor::div[@id="PublicEmergencyAnnouncementList"])]|' +
                '//h2[not(ancestor::div[@id="PublicEmergencyAnnouncementList"])]|' +
                '//h3[not(ancestor::div[@id="PublicEmergencyAnnouncementList"])]'
            )

            # Extract and clean content
            content_parts = []
            for text in content_selectors.css('::text').getall():
                text = text.strip()
                if text and not text.isspace():
                    content_parts.append(text)

            return ' '.join(content_parts)

        except Exception as e:
            self.logger.error(f"Error in extract_clean_content: {str(e)}")
            return ""

    def parse(self, response):
        url: str = str(response.url).split('?lang_update=', 1)[0]
        if url not in self.crawled_urls:
            # Add current URL to crawled_urls
            self.crawled_urls.add(response.url)
            try:
                # Extract title and text content from main content areas
                title = self.extract_title(response)
                content = self.extract_clean_content(response)
                content_lower = content.lower()

                # Check if any of our topics are mentioned
                mentioned_topics = [topic for topic in self.topics if topic.lower() in content_lower]
                if mentioned_topics:
                    self.results.append({
                        'url': url,
                        'topics': mentioned_topics,
                        'title': title,
                        'preview': content.replace(title, '', 1)[:200] + "...",
                        'content': content,
                    })

                # Follow valid links within the same domain
                for href in response.css('a::attr(href)').getall():
                    try:
                        # Convert relative URLs to absolute URLs
                        absolute_url = urljoin(url, href)
                        # print(f"Processing URL: {absolute_url}")
                        if self.is_valid_url(absolute_url):
                            # print(f"valid url: {absolute_url}")
                            yield response.follow(absolute_url, self.parse)
                    except Exception as e:
                        self.logger.error(
                            f"Error processing URL {href}: {str(e)}")

            except Exception as e:
                self.logger.error(f"Error parsing {response.url}: {str(e)}")


# Usage example
def main():
    domain = "www.denvergov.org"
    start_url = (
        "https://www.denvergov.org/Government/Agencies-Departments-Offices/"
        "Agencies-Departments-Offices-Directory/Denver-City-Council/"
        "Council-Members-Websites-Info/District-4"
    )
    topics = ["trash", "recycling", "waste", "compost",
              "community", "environment", "safety"]

    # Initialize the crawler
    crawler = TopicWebCrawler(domain, start_url, topics)

    # Crawl and process
    results: CrawledResultsHandler = crawler.crawl_and_process()

    # Get documents and Export to CSV
    documents: List[Document] = results.create_documents_with_metadata()
    results.export_to_csv(documents)

    print(f"Number of crawled URLs: {len(results.crawled_urls)}")
    # print(f"Crawled URLs: {results.crawled_urls}")
    print('---')

    print(f"Number of documents: {len(documents)}")
    print("\nSearch results:")
    for i, doc in enumerate(documents, 1):
        print(f"\nResult {i}:")
        print(f"URL: {doc.metadata['url']}")
        print(f"Topics: {doc.metadata['topics']}")
        print(f"Title: {doc.metadata['title']}")
        print(f"Preview: {doc.metadata['preview']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
