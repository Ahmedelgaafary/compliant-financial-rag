"""Company configuration and mapping."""

from typing import Dict, List, Optional


class CompanyConfig:
    """Configuration for supported companies."""
    
    COMPANIES = {
        "apple": {
            "name": "Apple Inc.",
            "variations": ["Apple", "Apple Inc.", "AAPL", "Apple Inc", "Apple Computer"],
            "filename_patterns": ["apple", "aapl"],
            "keywords": ["iPhone", "iPad", "Mac", "iOS", "macOS", "App Store", "Apple Watch"],
            "document_ids": ["Apple 10K 2025", "Apple 10K 2024", "Apple 10k 2023"],
        },
        "microsoft": {
            "name": "Microsoft Corp.",
            "variations": ["Microsoft", "Microsoft Corp.", "MSFT", "Microsoft Corporation"],
            "filename_patterns": ["microsoft", "msft"],
            "keywords": ["Windows", "Azure", "Office", "Xbox", "LinkedIn", "GitHub", "Teams"],
            "document_ids": ["MSFT_10K_2025"],
        },
        "google": {
            "name": "Google LLC",
            "variations": ["Google", "Alphabet", "Google LLC", "GOOGL", "GOOG"],
            "filename_patterns": ["google", "alphabet", "googl"],
            "keywords": ["Search", "YouTube", "Android", "Cloud", "Gmail", "Chrome"],
            "document_ids": [],
        },
        "amazon": {
            "name": "Amazon.com Inc.",
            "variations": ["Amazon", "Amazon.com", "AMZN", "Amazon Inc."],
            "filename_patterns": ["amazon", "amzn"],
            "keywords": ["Prime", "AWS", "Kindle", "Echo", "Alexa", "Marketplace"],
            "document_ids": [],
        },
        "real_brokerage": {
            "name": "The Real Brokerage Inc.",
            "variations": ["Real Brokerage", "The Real Brokerage", "Real Brokerage Inc.", "REAL"],
            "filename_patterns": ["real_brokerage", "real", "brokerage"],
            "keywords": ["real estate", "brokerage", "agent", "commission"],
            "document_ids": ["Brokerage 10k 2025"],
        },
        "tesla": {
            "name": "Tesla Inc.",
            "variations": ["Tesla", "Tesla Inc.", "TSLA"],
            "filename_patterns": ["tesla", "tsla"],
            "keywords": ["electric vehicle", "EV", "Autopilot", "Gigafactory", "Solar"],
            "document_ids": [],
        },
    }
    
    @classmethod
    def get_company(cls, company_name: str) -> Optional[Dict]:
        """Get company configuration by name or variation."""
        company_lower = company_name.lower()
        
        if company_lower in cls.COMPANIES:
            return cls.COMPANIES[company_lower]
        
        for _key, config in cls.COMPANIES.items():
            for variation in config.get("variations", []):
                if company_lower == variation.lower():
                    return config
            
        return None
    
    @classmethod
    def detect_company(cls, text: str) -> Optional[str]:
        """Detect company from text - returns the first match."""
        text_lower = text.lower()
        
        for key, config in cls.COMPANIES.items():
            for pattern in config.get("filename_patterns", []):
                if pattern.lower() in text_lower:
                    return key
        
        for key, config in cls.COMPANIES.items():
            if config["name"].lower() in text_lower:
                return key
            for variation in config.get("variations", []):
                if variation.lower() in text_lower:
                    return key
        
        return None
    
    @classmethod
    def detect_all_companies(cls, text: str) -> List[str]:
        """Detect ALL companies mentioned in text (deduplicated)."""
        text_lower = text.lower()
        found = []
        seen = set()
        
        for key, config in cls.COMPANIES.items():
            # Check name
            if config["name"].lower() in text_lower:
                if key not in seen:
                    found.append(key)
                    seen.add(key)
                continue
            
            # Check variations
            for variation in config.get("variations", []):
                if variation.lower() in text_lower:
                    if key not in seen:
                        found.append(key)
                        seen.add(key)
                    break
        
        return found
    
    @classmethod
    def get_document_id_for_company(cls, company_key: str) -> Optional[str]:
        """Get the document ID pattern for a company."""
        config = cls.COMPANIES.get(company_key)
        if config:
            doc_ids = config.get("document_ids", [])
            return doc_ids[0] if doc_ids else None
        return None
    
    @classmethod
    def get_all_companies(cls) -> List[str]:
        return list(cls.COMPANIES.keys())
    
    @classmethod
    def get_company_names(cls) -> List[str]:
        return [config["name"] for config in cls.COMPANIES.values()]