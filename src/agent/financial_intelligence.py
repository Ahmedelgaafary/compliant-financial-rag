"""Financial intelligence module for calculations and analysis."""

import re
from typing import Dict, List, Optional


class FinancialIntelligence:
    """Perform financial calculations and analysis on extracted claims."""
    
    @staticmethod
    def calculate_growth_rate(
        current_value: float,
        previous_value: float,
        period: str = "year-over-year"
    ) -> Dict[str, any]:
        """
        Calculate growth rate between two values.
        
        Args:
            current_value: Current period value
            previous_value: Previous period value
            period: Time period (e.g., "year-over-year", "quarter-over-quarter")
            
        Returns:
            Dictionary with growth rate, absolute change, and percentage change
        """
        if previous_value == 0:
            return {
                "growth_rate": None,
                "absolute_change": current_value - previous_value,
                "percentage_change": None,
                "period": period,
                "error": "Previous value is zero"
            }
        
        absolute_change = current_value - previous_value
        percentage_change = (absolute_change / previous_value) * 100
        
        return {
            "growth_rate": percentage_change,
            "absolute_change": absolute_change,
            "percentage_change": f"{percentage_change:.2f}%",
            "period": period,
            "direction": "increase" if percentage_change > 0 else "decrease" if percentage_change < 0 else "no_change"
        }
    
    @staticmethod
    def calculate_margin(
        numerator: float,
        denominator: float,
        margin_type: str = "gross"
    ) -> Dict[str, any]:
        """
        Calculate financial margin.
        
        Args:
            numerator: Profit value (e.g., gross profit, operating profit, net income)
            denominator: Revenue value
            margin_type: "gross", "operating", "net"
            
        Returns:
            Dictionary with margin percentage and value
        """
        if denominator == 0:
            return {
                "margin": None,
                "margin_percentage": None,
                "margin_type": margin_type,
                "error": "Denominator is zero"
            }
        
        margin_value = numerator / denominator
        margin_percentage = margin_value * 100
        
        return {
            "margin": margin_value,
            "margin_percentage": f"{margin_percentage:.2f}%",
            "margin_type": margin_type,
            "formula": f"{numerator} / {denominator}",
            "numerator": numerator,
            "denominator": denominator
        }
    
    @staticmethod
    def calculate_ratio(
        value_a: float,
        value_b: float,
        ratio_type: str = "debt_to_equity"
    ) -> Dict[str, any]:
        """
        Calculate financial ratios.
        
        Args:
            value_a: First value (e.g., total debt)
            value_b: Second value (e.g., total equity)
            ratio_type: Type of ratio
            
        Returns:
            Dictionary with ratio value
        """
        if value_b == 0:
            return {
                "ratio": None,
                "ratio_type": ratio_type,
                "error": "Denominator is zero"
            }
        
        ratio_value = value_a / value_b
        
        return {
            "ratio": ratio_value,
            "ratio_display": f"{ratio_value:.2f}x",
            "ratio_type": ratio_type,
            "formula": f"{value_a} / {value_b}"
        }
    
    @staticmethod
    def calculate_variance(
        actual: float,
        budgeted: float,
        variance_type: str = "revenue"
    ) -> Dict[str, any]:
        """
        Calculate variance between actual and budgeted values.
        
        Args:
            actual: Actual value
            budgeted: Budgeted/expected value
            variance_type: Type of variance
            
        Returns:
            Dictionary with variance details
        """
        variance = actual - budgeted
        variance_percentage = (variance / budgeted) * 100 if budgeted != 0 else None
        
        return {
            "variance": variance,
            "variance_percentage": f"{variance_percentage:.2f}%" if variance_percentage else None,
            "variance_type": variance_type,
            "favorable": variance > 0,
            "actual": actual,
            "budgeted": budgeted
        }
    
    @staticmethod
    def calculate_ebitda_from_income(
        net_income: float,
        interest: float,
        taxes: float,
        depreciation: float,
        amortization: float
    ) -> Dict[str, any]:
        """
        Calculate EBITDA from income statement components.
        
        Args:
            net_income: Net income
            interest: Interest expense
            taxes: Tax expense
            depreciation: Depreciation expense
            amortization: Amortization expense
            
        Returns:
            Dictionary with EBITDA value
        """
        ebitda = net_income + interest + taxes + depreciation + amortization
        
        return {
            "ebitda": ebitda,
            "formula": "Net Income + Interest + Taxes + Depreciation + Amortization",
            "components": {
                "net_income": net_income,
                "interest": interest,
                "taxes": taxes,
                "depreciation": depreciation,
                "amortization": amortization
            }
        }
    
    @staticmethod
    def analyze_financial_statement(
        claims: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Analyze multiple claims and calculate derived metrics.
        
        Args:
            claims: List of claim dictionaries with keys like 'subject', 'value', 'period'
            
        Returns:
            Dictionary with analysis results
        """
        result = {}
        
        # Extract values by subject
        values = {}
        for claim in claims:
            subject = claim.get("subject", "").lower()
            value_str = claim.get("value", "")
            
            # Extract numeric value
            numeric_value = FinancialIntelligence._extract_numeric_value(value_str)
            if numeric_value:
                values[subject] = numeric_value
        
        # Calculate margins if revenue and profit are available
        if "revenue" in values:
            if "gross profit" in values:
                result["gross_margin"] = FinancialIntelligence.calculate_margin(
                    values["gross profit"],
                    values["revenue"],
                    "gross"
                )
            
            if "operating income" in values or "operating profit" in values:
                operating_profit = values.get("operating income") or values.get("operating profit")
                if operating_profit:
                    result["operating_margin"] = FinancialIntelligence.calculate_margin(
                        operating_profit,
                        values["revenue"],
                        "operating"
                    )
            
            if "net income" in values or "net profit" in values:
                net_income = values.get("net income") or values.get("net profit")
                if net_income:
                    result["net_margin"] = FinancialIntelligence.calculate_margin(
                        net_income,
                        values["revenue"],
                        "net"
                    )
        
        # Calculate growth if multiple years available
        revenue_values = []
        for claim in claims:
            if "revenue" in claim.get("subject", "").lower():
                period = claim.get("period")
                value = FinancialIntelligence._extract_numeric_value(claim.get("value", ""))
                if value and period:
                    revenue_values.append({
                        "period": period,
                        "value": value
                    })
        
        if len(revenue_values) >= 2:
            # Sort by period
            revenue_values.sort(key=lambda x: x["period"])
            current = revenue_values[-1]
            previous = revenue_values[-2]
            
            result["revenue_growth"] = FinancialIntelligence.calculate_growth_rate(
                current["value"],
                previous["value"],
                f"{previous['period']} to {current['period']}"
            )
        
        return result
    
    @staticmethod
    def _extract_numeric_value(value_str: str) -> Optional[float]:
        """Extract numeric value from string like '$416.16B', '$8.07M', etc."""
        if not value_str:
            return None
        
        # Remove $, commas, parentheses
        cleaned = value_str.replace('$', '').replace(',', '').replace('(', '').replace(')', '')
        
        # Extract number and unit
        match = re.match(r'([\d.]+)\s*([BMK]?)$', cleaned, re.IGNORECASE)
        if not match:
            return None
        
        num = float(match.group(1))
        unit = match.group(2).upper() if match.group(2) else ""
        
        if unit == 'B':
            return num * 1_000_000_000
        elif unit == 'M':
            return num * 1_000_000
        elif unit == 'K':
            return num * 1_000
        else:
            return num