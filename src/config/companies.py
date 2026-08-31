"""Company configuration and mapping."""

from typing import Dict, List, Optional


class CompanyConfig:
    """Configuration for supported companies."""
    
    # Define known companies with their variations
    COMPANIES = {
        "apple": {
            "name": "Apple Inc.",
            "variations": ["Apple", "Apple Inc.", "AAPL", "Apple Inc"],
            "filename_patterns": ["apple", "aapl", "10k"],
            "keywords": ["iPhone", "iPad", "Mac", "iOS", "macOS", "App Store", "Apple Watch", "AirPods"],
        },
        "microsoft": {
            "name": "Microsoft Corp.",
            "variations": ["Microsoft", "Microsoft Corp.", "MSFT", "Microsoft Corporation"],
            "filename_patterns": ["microsoft", "msft", "10k"],
            "keywords": ["Windows", "Azure", "Office", "Xbox", "LinkedIn", "GitHub", "Teams"],
        },
        "google": {
            "name": "Google LLC",
            "variations": ["Google", "Alphabet", "Google LLC", "GOOGL", "GOOG"],
            "filename_patterns": ["google", "alphabet", "googl", "10k"],
            "keywords": ["Search", "YouTube", "Android", "Cloud", "Gmail", "Chrome", "Maps"],
        },
        "amazon": {
            "name": "Amazon.com Inc.",
            "variations": ["Amazon", "Amazon.com", "AMZN", "Amazon Inc."],
            "filename_patterns": ["amazon", "amzn", "10k"],
            "keywords": ["Prime", "AWS", "Kindle", "Echo", "Alexa", "Marketplace", "Fulfillment"],
        },
        "real_brokerage": {
            "name": "The Real Brokerage Inc.",
            "variations": ["Real Brokerage", "The Real Brokerage", "Real Brokerage Inc.", "REAL"],
            "filename_patterns": ["real_brokerage", "real", "brokerage"],
            "keywords": ["real estate", "brokerage", "agent", "commission", "title"],
        },
        "tesla": {
            "name": "Tesla Inc.",
            "variations": ["Tesla", "Tesla Inc.", "TSLA"],
            "filename_patterns": ["tesla", "tsla", "10k"],
            "keywords": ["electric vehicle", "EV", "Autopilot", "Gigafactory", "Solar", "Battery"],
        },
    }
    
    @classmethod
    def get_company(cls, company_name: str) -> Optional[Dict]:
        """Get company configuration by name or variation."""
        company_lower = company_name.lower()
        
        # Direct match
        if company_lower in cls.COMPANIES:
            return cls.COMPANIES[company_lower]
        
        # Check variations - use _company_key to indicate intentionally unused
        for _company_key, config in cls.COMPANIES.items():
            for variation in config.get("variations", []):
                if company_lower == variation.lower():
                    return config
        
        return None
    
    @classmethod
    def detect_company(cls, text: str) -> Optional[str]:
        """Detect company from text."""
        text_lower = text.lower()
        
        for company_key, config in cls.COMPANIES.items():
            # Check for company name in text
            if config["name"].lower() in text_lower:
                return company_key
            
            # Check for variations
            for variation in config.get("variations", []):
                if variation.lower() in text_lower:
                    return company_key
            
            # Check for keywords
            for keyword in config.get("keywords", []):
                if keyword.lower() in text_lower:
                    return company_key
        
        return None
    
    @classmethod
    def get_all_companies(cls) -> List[str]:
        """Get list of all supported company names."""
        return list(cls.COMPANIES.keys())
    
    @classmethod
    def get_company_names(cls) -> List[str]:
        """Get display names of all supported companies."""
        return [config["name"] for config in cls.COMPANIES.values()]