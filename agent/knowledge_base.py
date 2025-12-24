"""
Supply Chain Knowledge Base - Agent_SHIP
"""

SUPPLY_CHAIN_KNOWLEDGE = {
    "delay_factors": {
        "scheduled_shipping_days": {
            "description": "Planned delivery timeframe set at order creation",
            "impact": "Tighter schedules (1-2 days) have 40-60% higher delay risk vs standard (3-5 days)",
            "mitigation": "Set realistic expectations, buffer time for high-value orders, expedite processing",
            "best_practices": [
                "Use data-driven delivery estimates",
                "Account for carrier performance history",
                "Add buffer for international shipments",
                "Communicate clearly with customers"
            ]
        },
        "shipping_mode": {
            "description": "Delivery service level selected",
            "impact": "Same Day: 15% delay rate, First Class: 8%, Second Class: 12%, Standard: 18%",
            "mitigation": "Carrier diversification, SLA monitoring, backup options for premium services",
            "best_practices": [
                "Match service level to customer expectations",
                "Monitor carrier performance by mode",
                "Maintain backup carriers for each tier",
                "Price services to reflect reliability"
            ]
        },
        "payment_type": {
            "description": "Payment method used for transaction",
            "impact": "Transfer: 22% delay (fraud checks), Cash: 10%, Debit: 12%, Payment: 15%",
            "mitigation": "Streamline payment verification, pre-authorize high-value transfers, fast-track cash payments",
            "best_practices": [
                "Fast-track verified payment methods",
                "Implement real-time fraud detection for transfers",
                "Expedite cash on delivery processing",
                "Automate payment confirmation",
                "Reduce processing time for trusted customers"
            ]
        },
        "ship_hour": {
            "description": "Hour of day when shipment is processed",
            "impact": "Peak hours (7-9am, 5-7pm) have 25% higher delays due to volume",
            "mitigation": "Stagger processing, hire for peak coverage, automate routing",
            "best_practices": [
                "Schedule bulk shipments for off-peak",
                "Staff appropriately for peak hours",
                "Use automation during high volume",
                "Pre-stage orders for morning rush"
            ]
        },
        "is_weekend": {
            "description": "Whether shipment processed on weekend",
            "impact": "Weekend shipments 30-40% more likely to delay (limited carrier availability)",
            "mitigation": "Partner with 7-day carriers, set customer expectations, premium pricing",
            "best_practices": [
                "Clearly communicate weekend limitations",
                "Offer premium weekend service",
                "Partner with weekend-capable carriers",
                "Consider Monday processing for Sat/Sun orders"
            ]
        },
        "is_peak_hour": {
            "description": "Shipment during morning (7-9am) or evening (5-7pm) rush",
            "impact": "Peak hour processing adds average 0.5 days to delivery time",
            "mitigation": "Extend processing windows, automate where possible, optimize staffing",
            "best_practices": [
                "Incentivize off-peak processing",
                "Use batch processing overnight",
                "Automate label generation",
                "Pre-pick orders for peak times"
            ]
        },
        "is_night": {
            "description": "Overnight processing (10pm-5am)",
            "impact": "Night shipments may wait until morning pickup, adding 0.3-0.5 day delays",
            "mitigation": "Negotiate overnight pickups, batch for early morning, manage expectations",
            "best_practices": [
                "Batch night orders for first morning pickup",
                "Communicate processing cutoff times",
                "Use overnight fulfillment centers",
                "Partner with 24-hour carriers"
            ]
        },
        "order_item_quantity": {
            "description": "Number of units in the order",
            "impact": "Bulk orders (5+ items) have 15% higher delay risk due to picking complexity",
            "mitigation": "Optimize warehouse layout, use batch picking, automate inventory",
            "best_practices": [
                "Zone picking for multi-item orders",
                "Pre-kitting common combinations",
                "Wave picking during peak times",
                "Automated storage and retrieval"
            ]
        },
        "order_item_discount_rate": {
            "description": "Percentage discount applied to order",
            "impact": "High discounts (>20%) correlate with promotional periods, 18% more delays",
            "mitigation": "Plan staffing for promotions, pre-position inventory, set expectations",
            "best_practices": [
                "Forecast promotional demand",
                "Pre-hire temporary staff",
                "Increase safety stock before sales",
                "Extend delivery windows during promotions"
            ]
        },
        "sales": {
            "description": "Total revenue from the order",
            "impact": "High-value orders (>$500) get priority, 12% faster processing",
            "mitigation": "Implement value-based prioritization, VIP handling for premium orders",
            "best_practices": [
                "Tiered processing queues",
                "White-glove service for high-value",
                "Expedited QA for premium orders",
                "Dedicated account management"
            ]
        },
        "order_profit_per_order": {
            "description": "Profit margin on the order",
            "impact": "Low-margin orders may receive lower priority, affecting speed",
            "mitigation": "Balance efficiency with service level, avoid margin-based discrimination",
            "best_practices": [
                "Optimize fulfillment costs",
                "Bundle low-margin items",
                "Maintain service standards across margins",
                "Use automation to reduce costs"
            ]
        },
        "market": {
            "description": "Geographic market region",
            "impact": "LATAM: 28% delay rate, Africa: 32%, Pacific Asia: 15%, Europe: 10%, USCA: 12%",
            "mitigation": "Region-specific carriers, local fulfillment centers, customs expertise",
            "best_practices": [
                "Establish regional fulfillment centers",
                "Partner with local carriers",
                "Customs brokerage relationships",
                "Market-specific delivery windows"
            ]
        },
        "latitude_longitude": {
            "description": "Geographic coordinates of delivery location",
            "impact": "Remote locations (>50km from distribution) have 25% higher delays",
            "mitigation": "Zone-based shipping rates, partner with last-mile carriers, clear expectations",
            "best_practices": [
                "Distance-based pricing",
                "Last-mile carrier network",
                "Consolidated rural delivery",
                "Parcel locker partnerships"
            ]
        },
        "product_price": {
            "description": "Individual product value",
            "impact": "High-value items (>$300) require enhanced security, adding 0.2-0.4 days",
            "mitigation": "Streamline verification, signature confirmation, insurance automation",
            "best_practices": [
                "Automated insurance for high-value",
                "Signature required threshold",
                "Enhanced package tracking",
                "Direct signature for >$500"
            ]
        }
    },

    "shipping_modes": {
        "same_day": {
            "description": "Delivery within 24 hours",
            "typical_sla": "Same business day",
            "delay_rate": "15%",
            "key_factors": "Processing speed, local availability, traffic",
            "recommendations": "Limited to metro areas, premium pricing, real-time tracking"
        },
        "first_class": {
            "description": "Priority 1-2 day delivery",
            "typical_sla": "1-2 business days",
            "delay_rate": "8%",
            "key_factors": "Airport proximity, customs for international",
            "recommendations": "Best for time-sensitive, worth premium cost"
        },
        "second_class": {
            "description": "Standard 2-3 day delivery",
            "typical_sla": "2-3 business days",
            "delay_rate": "12%",
            "key_factors": "Ground transportation, weather, volume",
            "recommendations": "Good balance of cost and speed"
        },
        "standard_class": {
            "description": "Economy 3-5 day delivery",
            "typical_sla": "3-5 business days",
            "delay_rate": "18%",
            "key_factors": "Route optimization, consolidation, lower priority",
            "recommendations": "Cost-effective for non-urgent"
        }
    },

    "market_characteristics": {
        "africa": {
            "avg_delay_rate": "32%",
            "key_challenges": "Infrastructure, customs, remote locations",
            "recommendations": "Local partnerships, extended windows, tracking investments",
            "top_carriers": "DHL, Aramex, local postal services"
        },
        "europe": {
            "avg_delay_rate": "10%",
            "key_challenges": "Multi-country routing, VAT complexity",
            "recommendations": "EU fulfillment centers, IOSS compliance",
            "top_carriers": "DPD, Hermes, national posts"
        },
        "latam": {
            "avg_delay_rate": "28%",
            "key_challenges": "Customs delays, infrastructure, security",
            "recommendations": "Regional hubs (Brazil, Mexico), customs brokers",
            "top_carriers": "Correios, Mercado Envios, local carriers"
        },
        "pacific_asia": {
            "avg_delay_rate": "15%",
            "key_challenges": "Distance, island nations, customs",
            "recommendations": "Ocean consolidation, air for premium",
            "top_carriers": "SF Express, ZTO, Australia Post"
        },
        "usca": {
            "avg_delay_rate": "12%",
            "key_challenges": "Last-mile rural, weather events",
            "recommendations": "Multi-carrier strategy, regional warehouses",
            "top_carriers": "USPS, FedEx, UPS, regional carriers"
        }
    },

    "risk_mitigation": {
        "high_delay_risk": {
            "threshold": ">60% probability",
            "actions": [
                "Immediately alert customer with realistic ETA",
                "Upgrade to faster shipping mode if cost-effective",
                "Activate backup carrier if available",
                "Expedite warehouse processing",
                "Assign dedicated account manager",
                "Document delay factors for analysis"
            ]
        },
        "moderate_delay_risk": {
            "threshold": "40-60% probability",
            "actions": [
                "Proactive customer communication",
                "Monitor shipment closely with alerts",
                "Prepare contingency options",
                "Check for routing optimization",
                "Brief customer service team"
            ]
        },
        "low_delay_risk": {
            "threshold": "<40% probability",
            "actions": [
                "Standard processing and monitoring",
                "Maintain service level commitments",
                "Continue performance tracking",
                "Identify efficiency improvements"
            ]
        }
    },

    "business_metrics": {
        "order_value_tiers": {
            "premium": ">$500 (white-glove service)",
            "standard": "$100-$500 (normal processing)",
            "economy": "<$100 (cost-optimized)"
        },
        "discount_impact": {
            "promotional": ">15% discount (plan for volume)",
            "standard": "5-15% discount (normal flow)",
            "full_price": "0-5% discount (premium customers)"
        },
        "profitability_focus": {
            "high_margin": ">30% profit (maintain service)",
            "standard_margin": "15-30% profit (optimize costs)",
            "low_margin": "<15% profit (efficiency critical)"
        }
    }
}


def get_knowledge(topic: str, subtopic: str = None) -> dict:
    """Retrieve knowledge base information"""
    if topic in SUPPLY_CHAIN_KNOWLEDGE:
        if subtopic and subtopic in SUPPLY_CHAIN_KNOWLEDGE[topic]:
            return SUPPLY_CHAIN_KNOWLEDGE[topic][subtopic]
        return SUPPLY_CHAIN_KNOWLEDGE[topic]
    return {}


def get_recommendations(prediction_label: str, drivers: list) -> list:
    """Get actionable recommendations based on prediction"""
    recommendations = []

    probability = 0.5

    if prediction_label == "Delayed":
        recommendations.extend(SUPPLY_CHAIN_KNOWLEDGE["risk_mitigation"]["high_delay_risk"]["actions"][:5])

        for driver in drivers[:3]:
            feature = driver["feature"]
            if feature in SUPPLY_CHAIN_KNOWLEDGE["delay_factors"]:
                factor_info = SUPPLY_CHAIN_KNOWLEDGE["delay_factors"][feature]
                recommendations.append(
                    f"For {feature.replace('_', ' ')}: {factor_info.get('mitigation', 'Monitor closely')}")
    else:
        recommendations.extend(SUPPLY_CHAIN_KNOWLEDGE["risk_mitigation"]["low_delay_risk"]["actions"][:3])

    return recommendations[:8]


def format_knowledge_for_prompt(topic: str) -> str:
    """Format knowledge for inclusion in AI prompt"""
    knowledge = get_knowledge(topic)
    if not knowledge:
        return ""

    formatted = f"\n {topic.replace('_', ' ').title()} \n"

    def format_dict(d, indent=0):
        text = ""
        for key, value in d.items():
            spaces = "  " * indent
            if isinstance(value, dict):
                text += f"{spaces}{key}:\n"
                text += format_dict(value, indent + 1)
            elif isinstance(value, list):
                text += f"{spaces}{key}:\n"
                for item in value:
                    text += f"{spaces}  - {item}\n"
            else:
                text += f"{spaces}{key}: {value}\n"
        return text

    formatted += format_dict(knowledge)
    return formatted