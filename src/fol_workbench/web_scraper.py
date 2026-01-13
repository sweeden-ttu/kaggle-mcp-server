"""
Web Scraper (Soup) and Pourage

BeautifulSoup-based web scraping and content extraction.
Pourage: Pour/aggregate content from multiple sources.
"""

from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import requests
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    SOUP_AVAILABLE = True
except ImportError:
    SOUP_AVAILABLE = False
    BeautifulSoup = None


class WebScraper:
    """Web scraper using BeautifulSoup."""
    
    def __init__(self):
        if not SOUP_AVAILABLE:
            raise ImportError("BeautifulSoup4 is required. Install with: pip install beautifulsoup4")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a URL."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_text(self, url: str) -> Optional[str]:
        """Extract text content from URL."""
        soup = self.fetch(url)
        if soup:
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator='\n', strip=True)
        return None
    
    def extract_links(self, url: str) -> List[str]:
        """Extract all links from URL."""
        soup = self.fetch(url)
        if not soup:
            return []
        
        links = []
        base_url = url
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            links.append(absolute_url)
        return links
    
    def extract_formulas(self, url: str) -> List[str]:
        """Extract formulas/logic expressions from page."""
        text = self.extract_text(url)
        if not text:
            return []
        
        # Simple pattern matching for formulas
        import re
        # Look for common FOL patterns
        patterns = [
            r'\b(And|Or|Not|Implies|ForAll|Exists)\s*\([^)]+\)',
            r'[A-Za-z]\s*[∧∨¬→↔]\s*[A-Za-z]',
            r'∀[a-z]\s*[A-Z]\([a-z]\)',
            r'∃[a-z]\s*[A-Z]\([a-z]\)'
        ]
        
        formulas = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            formulas.extend(matches)
        
        return formulas
    
    def extract_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from page."""
        soup = self.fetch(url)
        if not soup:
            return {}
        
        metadata = {
            'title': soup.title.string if soup.title else '',
            'description': '',
            'keywords': []
        }
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        # Meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            keywords = meta_keywords.get('content', '')
            metadata['keywords'] = [k.strip() for k in keywords.split(',')]
        
        return metadata


class Pourage:
    """
    Pourage: Pour/aggregate content from multiple sources.
    
    Collects and aggregates content from various sources:
    - Web pages
    - Files
    - APIs
    - Databases
    """
    
    def __init__(self):
        self.scraper = WebScraper() if SOUP_AVAILABLE else None
        self.sources: List[Dict[str, Any]] = []
    
    def add_web_source(self, url: str, extract_formulas: bool = True) -> Dict[str, Any]:
        """Add a web source to aggregate."""
        source = {
            'type': 'web',
            'url': url,
            'content': None,
            'formulas': [],
            'metadata': {}
        }
        
        if self.scraper:
            source['content'] = self.scraper.extract_text(url)
            source['metadata'] = self.scraper.extract_metadata(url)
            if extract_formulas:
                source['formulas'] = self.scraper.extract_formulas(url)
        
        self.sources.append(source)
        return source
    
    def add_file_source(self, filepath: str) -> Dict[str, Any]:
        """Add a file source to aggregate."""
        source = {
            'type': 'file',
            'path': filepath,
            'content': None
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source['content'] = f.read()
        except Exception:
            pass
        
        self.sources.append(source)
        return source
    
    def aggregate(self) -> Dict[str, Any]:
        """Aggregate all sources into a single collection."""
        aggregated = {
            'total_sources': len(self.sources),
            'total_content_length': 0,
            'all_formulas': [],
            'all_metadata': [],
            'sources': []
        }
        
        for source in self.sources:
            if source.get('content'):
                aggregated['total_content_length'] += len(source['content'])
            
            if source.get('formulas'):
                aggregated['all_formulas'].extend(source['formulas'])
            
            if source.get('metadata'):
                aggregated['all_metadata'].append(source['metadata'])
            
            aggregated['sources'].append({
                'type': source['type'],
                'url': source.get('url'),
                'path': source.get('path'),
                'formula_count': len(source.get('formulas', [])),
                'content_length': len(source.get('content', ''))
            })
        
        return aggregated
    
    def extract_unique_formulas(self) -> List[str]:
        """Extract unique formulas from all sources."""
        all_formulas = []
        for source in self.sources:
            all_formulas.extend(source.get('formulas', []))
        return list(set(all_formulas))
    
    def save_aggregated(self, filepath: str):
        """Save aggregated content to file."""
        aggregated = self.aggregate()
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
