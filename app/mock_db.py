from datetime import datetime, timezone, timedelta

NOW = datetime.now(timezone.utc)

MOCK_CUSTOMERS = {
    "CUST-101": {
        "name": "Sarah Jenkins",
        "email": "sarah.j@example.com",
        "loyalty_tier": "Gold",
        "total_orders": 54,
        "refund_risk_score": 0.12
    },
    "CUST-909": {
        "name": "Alex Miller",
        "email": "alex.m99@example.com",
        "loyalty_tier": "Bronze",
        "total_orders": 5,
        "refund_risk_score": 0.85
    }
}

MOCK_ORDERS = {
    "ORD-8821": {
        "customer_id": "CUST-101",
        "status": "delivered",
        "delivered_at": (NOW - timedelta(minutes=25)).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "rider_name": "Dave (Bicycle Courier)",
        "items": [
            {"item_id": "ITM-01", "name": "Organic Whole Milk 1 Gallon", "qty": 1, "price": 4.99, "is_perishable": True},
            {"item_id": "ITM-02", "name": "Hass Avocados (4-pack)", "qty": 1, "price": 5.49, "is_perishable": True},
            {"item_id": "ITM-03", "name": "Artisan Sourdough Loaf", "qty": 1, "price": 6.50, "is_perishable": False}
        ],
        "total_amount": 16.98
    },
    "ORD-9904": {
        "customer_id": "CUST-909",
        "status": "delivered",
        "delivered_at": (NOW - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "rider_name": "Maria (Scooter Courier)",
        "items": [
            {"item_id": "ITM-90", "name": "A5 Japanese Wagyu Ribeye 16oz", "qty": 2, "price": 65.00, "is_perishable": True},
            {"item_id": "ITM-91", "name": "Fresh Black Truffle 50g", "qty": 1, "price": 45.00, "is_perishable": True}
        ],
        "total_amount": 175.00
    }
}

MOCK_POLICIES = {
    "damaged_perishables": "Produce, dairy, and meat received damaged or spoiled must be reported within 2 hours of delivery for an instant full refund or store credit.",
    "missing_items": "If an item on the receipt was not delivered, we issue an immediate refund for that item plus a $3 courtesy credit voucher.",
    "delivery_delay": "We guarantee 20-minute delivery. Delays over 35 minutes qualify for an automatic $5 store credit upon request."
}