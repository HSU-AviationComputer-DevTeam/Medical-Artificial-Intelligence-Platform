import requests
from urllib.parse import urlparse

from common.types import AgentCard


def get_agent_card(remote_agent_address: str) -> AgentCard:
    """Get the agent card."""
    # Handle URL parsing properly
    if not remote_agent_address.startswith(('http://', 'https://')):
        # If no protocol specified, assume http for localhost, https for others
        if remote_agent_address.startswith('localhost') or remote_agent_address.startswith('127.0.0.1'):
            url = f'http://{remote_agent_address}/.well-known/agent.json'
        else:
            url = f'https://{remote_agent_address}/.well-known/agent.json'
    else:
        # Protocol already specified
        url = f'{remote_agent_address}/.well-known/agent.json'
    
    agent_card = requests.get(url)
    agent_card.raise_for_status()  # Raise exception for HTTP errors
    return AgentCard(**agent_card.json())
