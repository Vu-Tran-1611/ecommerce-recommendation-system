from langchain_core.messages import ToolMessage

from app.chatbot.supervisor.tools.delegate_to_product_search_agent import delegate_to_product_search_agent


# Collect one artifact field from the product search delegate's results, without duplicates
def collect_artifact_values(messages, field: str) -> list:
    values = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == delegate_to_product_search_agent.name:
            for value in (message.artifact or {}).get(field, []):
                if value not in values:
                    values.append(value)
    return values


# Collect product IDs from a supervisor run
def collect_product_ids(messages) -> list[int]:
    return collect_artifact_values(messages, "product_ids")


# Collect categories the user should narrow down with subcategory buttons
def collect_narrow_categories(messages) -> list[str]:
    return collect_artifact_values(messages, "narrow_categories")
