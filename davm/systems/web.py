"""
DAVM Web System - Web scraping and research capabilities.

This system gives the mech the ability to fetch and analyze web content,
enabling research and information gathering from the internet.
"""

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag


@dataclass
class WebLink:
    """A link extracted from a web page."""
    
    url: str
    text: str
    title: str | None = None


@dataclass
class WebImage:
    """An image extracted from a web page."""
    
    url: str
    alt: str | None = None
    title: str | None = None


@dataclass
class WebPage:
    """Parsed content from a web page."""
    
    url: str
    title: str | None
    text: str  # Main text content
    html: str  # Raw HTML
    links: list[WebLink] = field(default_factory=list)
    images: list[WebImage] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)  # Meta tags
    headers: dict[str, str] = field(default_factory=dict)  # HTTP headers
    status_code: int = 200
    
    @property
    def domain(self) -> str:
        """Get the domain of the URL."""
        parsed = urlparse(self.url)
        return parsed.netloc
    
    @property
    def text_preview(self) -> str:
        """Get a preview of the text content."""
        if len(self.text) <= 500:
            return self.text
        return self.text[:500] + "..."


@dataclass
class FetchResult:
    """Result of a web fetch operation."""
    
    success: bool
    message: str
    page: WebPage | None = None
    error: str | None = None


class WebSystem:
    """
    Web scraping and research system for DAVM.
    
    This system allows the mech to fetch web pages, extract content,
    and gather information from the internet.
    
    Features:
    - Fetch web pages with proper headers
    - Parse HTML and extract text content
    - Extract links, images, and metadata
    - Handle redirects and errors gracefully
    
    Example:
        web = WebSystem()
        
        # Fetch a page
        result = await web.fetch("https://example.com")
        if result.success:
            print(result.page.title)
            print(result.page.text[:500])
        
        # Extract links
        for link in result.page.links[:10]:
            print(f"{link.text}: {link.url}")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_redirects: int = 5,
        user_agent: str | None = None,
    ):
        """
        Initialize the web system.
        
        Args:
            timeout: Request timeout in seconds.
            max_redirects: Maximum number of redirects to follow.
            user_agent: Custom user agent string.
        """
        self._timeout = timeout
        self._max_redirects = max_redirects
        self._user_agent = user_agent or (
            "DAVM/0.1 (Digital Assistant Virtual Mech; "
            "+https://github.com/davm) httpx/0.27"
        )
        
        # Common headers for requests
        self._headers = {
            "User-Agent": self._user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    async def fetch(
        self,
        url: str,
        extract_content: bool = True,
        follow_redirects: bool = True,
    ) -> FetchResult:
        """
        Fetch a web page.
        
        Args:
            url: URL to fetch.
            extract_content: If True, parse HTML and extract content.
            follow_redirects: If True, follow redirects.
            
        Returns:
            FetchResult with the page content or error.
        """
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme:
            url = "https://" + url
            parsed = urlparse(url)
        
        if parsed.scheme not in ("http", "https"):
            return FetchResult(
                success=False,
                message=f"Invalid URL scheme: {parsed.scheme}",
                error="Only HTTP and HTTPS URLs are supported",
            )

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=follow_redirects,
                max_redirects=self._max_redirects,
            ) as client:
                response = await client.get(url, headers=self._headers)
                
                # Get response headers
                headers = dict(response.headers)
                
                # Check content type
                content_type = headers.get("content-type", "")
                
                if "text/html" not in content_type and "application/xhtml" not in content_type:
                    # Non-HTML content
                    return FetchResult(
                        success=True,
                        message=f"Fetched non-HTML content: {content_type}",
                        page=WebPage(
                            url=str(response.url),
                            title=None,
                            text=f"[Non-HTML content: {content_type}]",
                            html="",
                            headers=headers,
                            status_code=response.status_code,
                        ),
                    )
                
                html = response.text
                
                if extract_content:
                    page = self._parse_html(str(response.url), html, headers, response.status_code)
                else:
                    page = WebPage(
                        url=str(response.url),
                        title=None,
                        text="",
                        html=html,
                        headers=headers,
                        status_code=response.status_code,
                    )
                
                return FetchResult(
                    success=True,
                    message=f"Fetched {len(html)} bytes from {page.domain}",
                    page=page,
                )

        except httpx.TimeoutException:
            return FetchResult(
                success=False,
                message=f"Request timed out after {self._timeout} seconds",
                error="timeout",
            )
        except httpx.TooManyRedirects:
            return FetchResult(
                success=False,
                message=f"Too many redirects (max: {self._max_redirects})",
                error="too_many_redirects",
            )
        except httpx.RequestError as e:
            return FetchResult(
                success=False,
                message=f"Request error: {e}",
                error=str(e),
            )
        except Exception as e:
            return FetchResult(
                success=False,
                message=f"Unexpected error: {e}",
                error=str(e),
            )

    def _parse_html(
        self,
        url: str,
        html: str,
        headers: dict[str, str],
        status_code: int,
    ) -> WebPage:
        """Parse HTML and extract content."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract title
        title = None
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Extract meta tags
        meta = {}
        for tag in soup.find_all("meta"):
            name = tag.get("name") or tag.get("property")
            content = tag.get("content")
            if name and content:
                meta[name] = content
        
        # Extract main text content
        text = self._extract_text(soup)
        
        # Extract links
        links = self._extract_links(soup, url)
        
        # Extract images
        images = self._extract_images(soup, url)
        
        return WebPage(
            url=url,
            title=title,
            text=text,
            html=html,
            links=links,
            images=images,
            meta=meta,
            headers=headers,
            status_code=status_code,
        )

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract readable text from HTML."""
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        
        # Try to find main content areas
        main_content = None
        for selector in ["main", "article", '[role="main"]', ".content", "#content"]:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            # Fall back to body
            body = soup.find("body")
            if body:
                text = body.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)
        
        # Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[WebLink]:
        """Extract links from HTML."""
        links = []
        seen_urls = set()
        
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            
            # Skip empty, anchor-only, and javascript links
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            
            # Make absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Skip duplicates
            if absolute_url in seen_urls:
                continue
            seen_urls.add(absolute_url)
            
            # Get link text
            text = a.get_text(strip=True)
            if not text:
                # Try to get text from child elements
                img = a.find("img")
                if img:
                    text = img.get("alt", "") or "[image]"
                else:
                    text = "[link]"
            
            links.append(WebLink(
                url=absolute_url,
                text=text[:200],  # Truncate long text
                title=a.get("title"),
            ))
        
        return links

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> list[WebImage]:
        """Extract images from HTML."""
        images = []
        seen_urls = set()
        
        for img in soup.find_all("img", src=True):
            src = img.get("src", "")
            
            if not src or src.startswith("data:"):
                continue
            
            # Make absolute URL
            absolute_url = urljoin(base_url, src)
            
            # Skip duplicates
            if absolute_url in seen_urls:
                continue
            seen_urls.add(absolute_url)
            
            images.append(WebImage(
                url=absolute_url,
                alt=img.get("alt"),
                title=img.get("title"),
            ))
        
        return images

    async def fetch_text(self, url: str) -> str:
        """
        Fetch a URL and return just the text content.
        
        Convenience method for simple text extraction.
        
        Args:
            url: URL to fetch.
            
        Returns:
            Extracted text content, or error message.
        """
        result = await self.fetch(url)
        
        if not result.success:
            return f"Error fetching {url}: {result.message}"
        
        if result.page:
            return result.page.text
        
        return ""

    async def fetch_links(self, url: str) -> list[WebLink]:
        """
        Fetch a URL and return just the links.
        
        Args:
            url: URL to fetch.
            
        Returns:
            List of links from the page.
        """
        result = await self.fetch(url)
        
        if result.success and result.page:
            return result.page.links
        
        return []

    async def search_page(
        self,
        url: str,
        query: str,
        case_sensitive: bool = False,
    ) -> list[str]:
        """
        Search for text within a web page.
        
        Args:
            url: URL to fetch.
            query: Text to search for.
            case_sensitive: Whether search is case-sensitive.
            
        Returns:
            List of matching lines/paragraphs.
        """
        result = await self.fetch(url)
        
        if not result.success or not result.page:
            return []
        
        text = result.page.text
        search_query = query if case_sensitive else query.lower()
        
        matches = []
        for paragraph in text.split("\n\n"):
            compare_text = paragraph if case_sensitive else paragraph.lower()
            if search_query in compare_text:
                matches.append(paragraph.strip())
        
        return matches

    async def get_page_summary(self, url: str) -> dict[str, Any]:
        """
        Get a summary of a web page.
        
        Args:
            url: URL to fetch.
            
        Returns:
            Dictionary with page summary information.
        """
        result = await self.fetch(url)
        
        if not result.success:
            return {
                "success": False,
                "error": result.message,
                "url": url,
            }
        
        page = result.page
        if not page:
            return {
                "success": False,
                "error": "No page content",
                "url": url,
            }
        
        return {
            "success": True,
            "url": page.url,
            "domain": page.domain,
            "title": page.title,
            "description": page.meta.get("description", ""),
            "text_length": len(page.text),
            "text_preview": page.text_preview,
            "link_count": len(page.links),
            "image_count": len(page.images),
            "status_code": page.status_code,
        }

    async def download(
        self,
        url: str,
        save_path: str | None = None,
    ) -> tuple[bool, bytes | str]:
        """
        Download content from a URL.
        
        Args:
            url: URL to download.
            save_path: Optional path to save the content.
            
        Returns:
            Tuple of (success, content_or_error_message)
        """
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(url, headers=self._headers)
                response.raise_for_status()
                
                content = response.content
                
                if save_path:
                    with open(save_path, "wb") as f:
                        f.write(content)
                    return True, f"Saved {len(content)} bytes to {save_path}"
                
                return True, content
                
        except Exception as e:
            return False, str(e)
