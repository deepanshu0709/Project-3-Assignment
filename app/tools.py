from langchain_core.tools import tool
from app.mock_db import MOCK_ORDERS, MOCK_CUSTOMERS, MOCK_POLICIES


@tool
def get_order_details(order_id: str) -> str:
    """
    Look up customer order details, items, delivery timestamps, and courier status.
    Args:
        order_id: The order identifier, e.g. ORD-8821.
    """
    clean_id = order_id.strip().upper()
    order = MOCK_ORDERS.get(clean_id)
    if not order:
        return f"Order '{order_id}' was not found in the GroceryOnTheGo database."
    
    items_summary = ", ".join([f"{item['name']} (x{item['qty']})" for item in order['items']])
    return (
        f"Order: {clean_id}\n"
        f"Status: {order['status']} (Delivered at: {order['delivered_at']})\n"
        f"Courier: {order['rider_name']}\n"
        f"Items: {items_summary}\n"
        f"Total Paid: ${order['total_amount']:.2f}"
    )


@tool
def search_grocery_policy(query: str) -> str:
    """
    Search GroceryOnTheGo store policies on damaged items, returns, delivery delays, and refund eligibility.
    Args:
        query: Keyword or phrase to search (e.g. 'damaged milk', 'delay policy').
    """
    q = query.lower()
    matches = []
    for policy_key, policy_text in MOCK_POLICIES.items():
        if any(word in policy_text.lower() for word in q.split()):
            matches.append(f"[{policy_key.upper()}]: {policy_text}")
    
    if not matches:
        return f"Standard Policy: All issues must be reported within 2 hours. {MOCK_POLICIES['damaged_perishables']}"
    
    return "\n\n".join(matches)


# List of all tools available in Session 2
SESSION_2_TOOLS = [get_order_details, search_grocery_policy]