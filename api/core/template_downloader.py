"""
Online Contract Template Downloader

This module downloads and processes real contract templates from legal websites
to create ideal contract benchmarks for the Guardian Score system.
"""

import os
import requests
import tempfile
from typing import List, Dict, Optional
import time
from urllib.parse import urlparse
import logging

# Setup logging
logger = logging.getLogger(__name__)


class ContractTemplateDownloader:
    """Downloads and processes contract templates from legal websites."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def download_template(self, url: str, filename: str) -> Optional[str]:
        """
        Download a contract template from a URL.
        
        Args:
            url: URL to download the template from
            filename: Local filename to save as
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            print(f"📥 Downloading template from: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Create downloads directory if it doesn't exist
            downloads_dir = os.path.join(os.getcwd(), "ideal_contracts_downloads")
            os.makedirs(downloads_dir, exist_ok=True)
            
            # Save the file
            file_path = os.path.join(downloads_dir, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded: {filename}")
            return file_path
            
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            return None
    
    def get_recommended_templates(self) -> List[Dict]:
        """
        Get a curated list of recommended contract templates from reliable Indian legal sources.
        
        Returns:
            List of template metadata with download URLs
        """
        return [
            # Rental/Leave & License Agreements (India)
            {
                "category": "rental",
                "title": "Leave and License Agreement - Mumbai",
                "description": "Standard leave and license agreement for Mumbai under Maharashtra Rent Control Act",
                "url": "https://www.vakilsearch.com/advice/sample-leave-and-license-agreement/",
                "filename": "leave_license_mumbai.pdf",
                "source": "Vakil Search",
                "jurisdiction": "Maharashtra, India"
            },
            {
                "category": "rental",
                "title": "Rental Agreement - Delhi Format",
                "description": "Delhi rent agreement format as per Delhi Rent Control Act",
                "url": "https://www.legaldesk.com/rental-agreement/",
                "filename": "rental_agreement_delhi.pdf",
                "source": "Legal Desk India", 
                "jurisdiction": "Delhi, India"
            },
            
            # Employment Contracts (India)
            {
                "category": "employment",
                "title": "Employment Agreement - Indian Format",
                "description": "Comprehensive employment contract under Indian Labour Laws",
                "url": "https://www.vakilsearch.com/advice/employment-agreement/",
                "filename": "employment_agreement_india.pdf",
                "source": "Vakil Search",
                "jurisdiction": "India (All States)"
            },
            {
                "category": "employment", 
                "title": "Service Agreement - Employee India",
                "description": "Employee service agreement as per Industrial Disputes Act",
                "url": "https://www.indiafilings.com/learn/employment-agreement/",
                "filename": "service_agreement_employee_india.pdf",
                "source": "India Filings",
                "jurisdiction": "India (All States)"
            },
            
            # Service Agreements (India)
            {
                "category": "service_agreement",
                "title": "Professional Service Agreement India",
                "description": "Professional service agreement under Indian Contract Act 1872",
                "url": "https://www.vakilsearch.com/advice/service-agreement/",
                "filename": "service_agreement_india.pdf",
                "source": "Vakil Search",
                "jurisdiction": "India (All States)"
            },
            
            # Non-Disclosure Agreements (India)
            {
                "category": "nda",
                "title": "Non-Disclosure Agreement India",
                "description": "Mutual NDA template under Indian Contract Act 1872",
                "url": "https://www.legaldesk.com/nda/",
                "filename": "nda_india.pdf",
                "source": "Legal Desk India",
                "jurisdiction": "India (All States)"
            },
            
            # Partnership Agreements (India)
            {
                "category": "partnership",
                "title": "Partnership Deed India Format",
                "description": "Partnership deed under Indian Partnership Act 1932",
                "url": "https://www.vakilsearch.com/advice/partnership-deed/",
                "filename": "partnership_deed_india.pdf",
                "source": "Vakil Search",
                "jurisdiction": "India (All States)"
            },
            
            # Purchase/Sale Agreements (India)
            {
                "category": "purchase",
                "title": "Sale Agreement India",
                "description": "Standard sale agreement under Sale of Goods Act 1930",
                "url": "https://www.legaldesk.com/sale-agreement/",
                "filename": "sale_agreement_india.pdf",
                "source": "Legal Desk India",
                "jurisdiction": "India (All States)"
            }
        ]
    
    def download_all_recommended(self) -> List[Dict]:
        """
        Download all recommended contract templates.
        
        Returns:
            List of successfully downloaded templates with file paths
        """
        templates = self.get_recommended_templates()
        downloaded_templates = []
        
        print(f"📥 Starting download of {len(templates)} contract templates...")
        
        for template in templates:
            print(f"\n--- Downloading {template['title']} ---")
            
            # Add delay between downloads to be respectful
            time.sleep(2)
            
            file_path = self.download_template(template['url'], template['filename'])
            
            if file_path:
                template['local_file_path'] = file_path
                downloaded_templates.append(template)
                print(f"✅ Successfully downloaded: {template['title']}")
            else:
                print(f"❌ Failed to download: {template['title']}")
        
        print(f"\n🎉 Downloaded {len(downloaded_templates)} out of {len(templates)} templates")
        return downloaded_templates


# Indian Contract Template URLs - Direct PDF downloads from reliable sources
FREE_CONTRACT_TEMPLATE_URLS = {
    "rental": [
        {
            "title": "Leave and License Agreement Mumbai Format",
            "url": "https://www.mhada.gov.in/sites/default/files/2020-07/Leave%20and%20License%20Agreement%20Format.pdf",
            "source": "MHADA (Maharashtra Housing Authority)",
            "description": "Official leave and license agreement format from Maharashtra government"
        },
        {
            "title": "Model Tenancy Act Format",
            "url": "https://www.mhupa.gov.in/sites/default/files/pdf/MODEL%20TENANCY%20ACT%202021.pdf",
            "source": "Ministry of Housing and Urban Affairs",
            "description": "Model Tenancy Act 2021 format from Government of India"
        },
        {
            "title": "Rental Agreement Sample Delhi",
            "url": "https://dda.org.in/tendernotices_docs/jan18/Annexure-F.pdf",
            "source": "Delhi Development Authority",
            "description": "Sample rental agreement format from Delhi Development Authority"
        }
    ],
    "employment": [
        {
            "title": "Employment Contract Sample Format",
            "url": "https://labour.gov.in/sites/default/files/employment_contract_sample.pdf",
            "source": "Ministry of Labour and Employment",
            "description": "Sample employment contract from Government of India"
        },
        {
            "title": "Service Agreement Format",
            "url": "https://www.epfindia.gov.in/site_docs/PDFs/Downloads_PDFs/service_agreement_format.pdf",
            "source": "EPFO India",
            "description": "Service agreement format for employees under EPF"
        }
    ],
    "nda": [
        {
            "title": "Non-Disclosure Agreement Template",
            "url": "https://www.startupindia.gov.in/content/dam/invest-india/Templates/public/NDA_Template.pdf",
            "source": "Startup India",
            "description": "NDA template from Government of India's Startup India initiative"
        }
    ],
    "service_agreement": [
        {
            "title": "Professional Service Agreement",
            "url": "https://www.cag.gov.in/sites/default/files/tender_document/Service_Agreement_Format.pdf",
            "source": "Comptroller and Auditor General of India",
            "description": "Professional service agreement format from CAG India"
        }
    ],
    "partnership": [
        {
            "title": "Partnership Deed Format",
            "url": "https://www.mca.gov.in/MinistryV2/partnership_deed_format.pdf",
            "source": "Ministry of Corporate Affairs",
            "description": "Partnership deed format from MCA India"
        }
    ],
    "purchase": [
        {
            "title": "Sale Purchase Agreement",
            "url": "https://www.igr.up.gov.in/sites/default/files/all/pdf/Sale_Purchase_Agreement_Format.pdf",
            "source": "Inspector General of Registration, UP",
            "description": "Sale purchase agreement format from Uttar Pradesh government"
        }
    ],
    "lease": [
        {
            "title": "Commercial Lease Agreement",
            "url": "https://www.dda.org.in/ddanew/pdf/Commercial_Lease_Format.pdf",
            "source": "Delhi Development Authority",
            "description": "Commercial lease agreement format from DDA"
        }
    ]
}


def get_direct_template_urls(category: str) -> List[Dict]:
    """
    Get direct URLs to free contract templates.
    
    Args:
        category: Contract category (rental, employment, etc.)
        
    Returns:
        List of template URLs for the category
    """
    return FREE_CONTRACT_TEMPLATE_URLS.get(category, [])


def download_free_templates(category: str = None) -> List[str]:
    """
    Download free contract templates from direct URLs.
    
    Args:
        category: Optional category filter (rental, employment, etc.)
        
    Returns:
        List of downloaded file paths
    """
    downloader = ContractTemplateDownloader()
    downloaded_files = []
    
    categories_to_download = [category] if category else FREE_CONTRACT_TEMPLATE_URLS.keys()
    
    for cat in categories_to_download:
        templates = get_direct_template_urls(cat)
        
        for template in templates:
            filename = f"{cat}_{template['title'].replace(' ', '_').lower()}.pdf"
            file_path = downloader.download_template(template['url'], filename)
            
            if file_path:
                downloaded_files.append(file_path)
    
    return downloaded_files


if __name__ == "__main__":
    # Test the downloader
    print("Testing Contract Template Downloader...")
    
    # Download free templates
    downloaded = download_free_templates("rental")
    print(f"Downloaded {len(downloaded)} rental templates")
    
    for file_path in downloaded:
        print(f"📄 {file_path}")