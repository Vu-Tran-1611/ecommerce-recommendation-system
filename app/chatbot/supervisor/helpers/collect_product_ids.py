from langchain_core.messages import ToolMessage

from app.chatbot.supervisor.tools.delegate_to_product_search_agent import delegate_to_product_search_agent


# Collect product IDs from the product search delegate's artifacts in a supervisor run
def collect_product_ids(messages) -> list[int]:
    product_ids = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == delegate_to_product_search_agent.name:
            for product_id in message.artifact or []:
                if product_id not in product_ids:
                    product_ids.append(product_id)
    return product_ids
