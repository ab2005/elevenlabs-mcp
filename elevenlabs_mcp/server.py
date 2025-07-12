"""
ElevenLabs MCP Server

⚠️ IMPORTANT: This server provides access to ElevenLabs API endpoints which may incur costs.
Each tool that makes an API call is marked with a cost warning. Please follow these guidelines:

1. Only use tools when explicitly requested by the user
2. For tools that generate audio, consider the length of the text as it affects costs
3. Some operations like voice cloning or text-to-voice may have higher costs

Tools without cost warnings in their description are free to use as they only read existing data.
"""

import httpx
import os
from datetime import datetime
from io import BytesIO
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from elevenlabs.client import ElevenLabs
from elevenlabs_mcp.utils import (
    make_error,
    handle_input_file,
)
from elevenlabs_mcp.curl_tools import (
    create_weather_tool,
    create_webhook_tool,
    create_rest_api_tool,
    create_crm_webhook_tool,
    create_notification_tool,
    create_database_api_tool,
    create_search_tool,
    generate_curl_command,
    validate_tool_config,
    TOOL_TEMPLATES
)
from elevenlabs_mcp.convai import create_conversation_config, create_platform_settings
from elevenlabs.types.knowledge_base_locator import KnowledgeBaseLocator

from elevenlabs_mcp import __version__

load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")
base_path = os.getenv("ELEVENLABS_MCP_BASE_PATH")
DEFAULT_VOICE_ID = "cgSgspJ2msm6clMCkdW9"

if not api_key:
    raise ValueError("ELEVENLABS_API_KEY environment variable is required")

# Add custom client to ElevenLabs to set User-Agent header
custom_client = httpx.Client(
    headers={
        "User-Agent": f"ElevenLabs-MCP/{__version__}",
    }
)

client = ElevenLabs(api_key=api_key, httpx_client=custom_client)
mcp = FastMCP("ElevenLabs")


@mcp.tool(
    description="""Create a conversational AI agent with custom configuration.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        name: Name of the agent
        first_message: First message the agent will say i.e. "Hi, how can I help you today?"
        system_prompt: System prompt for the agent
        voice_id: ID of the voice to use for the agent
        language: ISO 639-1 language code for the agent
        llm: LLM to use for the agent
        temperature: Temperature for the agent. The lower the temperature, the more deterministic the agent's responses will be. Range is 0 to 1.
        max_tokens: Maximum number of tokens to generate.
        asr_quality: Quality of the ASR. `high` or `low`.
        model_id: ID of the ElevenLabs model to use for the agent.
        optimize_streaming_latency: Optimize streaming latency. Range is 0 to 4.
        stability: Stability for the agent. Range is 0 to 1.
        similarity_boost: Similarity boost for the agent. Range is 0 to 1.
        turn_timeout: Timeout for the agent to respond in seconds. Defaults to 7 seconds.
        max_duration_seconds: Maximum duration of a conversation in seconds. Defaults to 600 seconds (10 minutes).
        record_voice: Whether to record the agent's voice.
        retention_days: Number of days to retain the agent's data.
    """
)
def create_agent(
    name: str,
    first_message: str,
    system_prompt: str,
    voice_id: str | None = DEFAULT_VOICE_ID,
    language: str = "en",
    llm: str = "gemini-2.0-flash-001",
    temperature: float = 0.5,
    max_tokens: int | None = None,
    asr_quality: str = "high",
    model_id: str = "eleven_turbo_v2",
    optimize_streaming_latency: int = 3,
    stability: float = 0.5,
    similarity_boost: float = 0.8,
    turn_timeout: int = 7,
    max_duration_seconds: int = 300,
    record_voice: bool = True,
    retention_days: int = 730,
) -> TextContent:
    conversation_config = create_conversation_config(
        language=language,
        system_prompt=system_prompt,
        llm=llm,
        first_message=first_message,
        temperature=temperature,
        max_tokens=max_tokens,
        asr_quality=asr_quality,
        voice_id=voice_id,
        model_id=model_id,
        optimize_streaming_latency=optimize_streaming_latency,
        stability=stability,
        similarity_boost=similarity_boost,
        turn_timeout=turn_timeout,
        max_duration_seconds=max_duration_seconds,
    )

    platform_settings = create_platform_settings(
        record_voice=record_voice,
        retention_days=retention_days,
    )

    response = client.conversational_ai.agents.create(
        name=name,
        conversation_config=conversation_config,
        platform_settings=platform_settings,
    )

    return TextContent(
        type="text",
        text=f"""Agent created successfully: Name: {name}, Agent ID: {response.agent_id}, System Prompt: {system_prompt}, Voice ID: {voice_id or "Default"}, Language: {language}, LLM: {llm}, You can use this agent ID for future interactions with the agent.""",
    )


@mcp.tool(
    description="""Add a knowledge base to ElevenLabs workspace. Allowed types are epub, pdf, docx, txt, html.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        agent_id: ID of the agent to add the knowledge base to.
        knowledge_base_name: Name of the knowledge base.
        url: URL of the knowledge base.
        input_file_path: Path to the file to add to the knowledge base.
        text: Text to add to the knowledge base.
    """
)
def add_knowledge_base_to_agent(
    agent_id: str,
    knowledge_base_name: str,
    url: str | None = None,
    input_file_path: str | None = None,
    text: str | None = None,
) -> TextContent:
    provided_params = [
        param for param in [url, input_file_path, text] if param is not None
    ]
    if len(provided_params) == 0:
        make_error("Must provide either a URL, a file, or text")
    if len(provided_params) > 1:
        make_error("Must provide exactly one of: URL, file, or text")

    if url is not None:
        response = client.conversational_ai.knowledge_base.documents.create_from_url(
            name=knowledge_base_name,
            url=url,
        )
    else:
        if text is not None:
            text_bytes = text.encode("utf-8")
            text_io = BytesIO(text_bytes)
            text_io.name = "text.txt"
            text_io.content_type = "text/plain"
            file = text_io
        elif input_file_path is not None:
            path = handle_input_file(
                file_path=input_file_path, audio_content_check=False
            )
            file = open(path, "rb")

        response = client.conversational_ai.knowledge_base.documents.create_from_file(
            name=knowledge_base_name,
            file=file,
        )

    agent = client.conversational_ai.agents.get(agent_id=agent_id)
    agent.conversation_config.agent.prompt.knowledge_base.append(
        KnowledgeBaseLocator(
            type="file" if file else "url",
            name=knowledge_base_name,
            id=response.id,
        )
    )
    client.conversational_ai.agents.update(
        agent_id=agent_id, conversation_config=agent.conversation_config
    )
    return TextContent(
        type="text",
        text=f"""Knowledge base created with ID: {response.id} and added to agent {agent_id} successfully.""",
    )


@mcp.tool(
    description="""List all documents in an agent's knowledge base.

    Args:
        agent_id: The ID of the agent whose knowledge base to list
    """
)
def list_knowledge_base_documents(agent_id: str) -> TextContent:
    response = client.conversational_ai.knowledge_base.list(agent_id=agent_id)

    if not response.documents:
        return TextContent(type="text", text="No knowledge base documents found.")

    docs_list = []
    for doc in response.documents:
        docs_list.append(
            f"Name: {doc.name}, ID: {doc.document_id}, Type: {doc.type}, Status: {doc.status}"
        )

    formatted_list = "\n".join(docs_list)
    return TextContent(type="text", text=f"Knowledge Base Documents:\n{formatted_list}")


@mcp.tool(
    description="""Get details of a specific knowledge base document.

    Args:
        document_id: The ID of the document to retrieve
    """
)
def get_knowledge_base_document(document_id: str) -> TextContent:
    response = client.conversational_ai.knowledge_base.documents.get(
        document_id=document_id
    )

    metadata_info = f"Metadata: {response.metadata}" if response.metadata else "No metadata"
    
    return TextContent(
        type="text",
        text=f"Document Details: Name: {response.name}, ID: {response.document_id}, Type: {response.type}, Status: {response.status}, Created: {response.created_at}, {metadata_info}",
    )


@mcp.tool(
    description="""Remove a document from a knowledge base.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        document_id: The ID of the document to delete
    """
)
def delete_knowledge_base_document(document_id: str) -> TextContent:
    client.conversational_ai.knowledge_base.documents.delete(
        document_id=document_id
    )

    return TextContent(
        type="text",
        text=f"Knowledge base document deleted successfully: Document ID: {document_id}",
    )


@mcp.tool(
    description="""Trigger recomputation of the RAG (Retrieval-Augmented Generation) index for a document.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        document_id: The ID of the document to recompute RAG index for
    """
)
def compute_rag_index(document_id: str) -> TextContent:
    response = client.conversational_ai.knowledge_base.document.compute_rag_index(document_id=document_id)

    estimated_time = getattr(response, 'estimated_time_seconds', 'Unknown')
    
    return TextContent(
        type="text",
        text=f"RAG index computation started for document {document_id}. Estimated time: {estimated_time} seconds",
    )


@mcp.tool(
    description="""Get the full content of a knowledge base document.

    Args:
        document_id: The ID of the document to retrieve content for
    """
)
def get_document_content(document_id: str) -> TextContent:
    response = client.conversational_ai.knowledge_base.documents.get_content(
        document_id=document_id
    )

    # Truncate very long content for readability
    content = response.content if hasattr(response, 'content') else str(response)
    if len(content) > 2000:
        content = content[:2000] + "... [content truncated]"

    return TextContent(
        type="text",
        text=f"Document Content for {document_id}:\n\n{content}",
    )


@mcp.tool(description="List all available conversational AI agents")
def list_agents() -> TextContent:
    """List all available conversational AI agents.

    Returns:
        TextContent with a formatted list of available agents
    """
    response = client.conversational_ai.agents.list()

    if not response.agents:
        return TextContent(type="text", text="No agents found.")

    agent_list = ",".join(
        f"{agent.name} (ID: {agent.agent_id})" for agent in response.agents
    )

    return TextContent(type="text", text=f"Available agents: {agent_list}")


@mcp.tool(description="Get details about a specific conversational AI agent")
def get_agent(agent_id: str) -> TextContent:
    """Get details about a specific conversational AI agent.

    Args:
        agent_id: The ID of the agent to retrieve

    Returns:
        TextContent with detailed information about the agent
    """
    response = client.conversational_ai.agents.get(agent_id=agent_id)

    voice_info = "None"
    if response.conversation_config.tts:
        voice_info = f"Voice ID: {response.conversation_config.tts.voice_id}"

    return TextContent(
        type="text",
        text=f"Agent Details: Name: {response.name}, Agent ID: {response.agent_id}, Voice Configuration: {voice_info}, Created At: {datetime.fromtimestamp(response.metadata.created_at_unix_secs).strftime('%Y-%m-%d %H:%M:%S')}",
    )


@mcp.tool(
    description="""Update an existing agent's configuration.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        agent_id: The ID of the agent to update
        name: New name for the agent (optional)
        conversation_config: Updated conversation configuration (optional)
    """
)
def update_agent(
    agent_id: str,
    name: str | None = None,
    conversation_config: dict | None = None,
) -> TextContent:
    update_data = {}
    if name is not None:
        update_data["name"] = name
    if conversation_config is not None:
        update_data["conversation_config"] = conversation_config

    if not update_data:
        make_error("At least one field (name or conversation_config) must be provided for update")

    response = client.conversational_ai.agents.update(
        agent_id=agent_id,
        **update_data
    )

    return TextContent(
        type="text",
        text=f"Agent updated successfully: Name: {response.name}, Agent ID: {response.agent_id}",
    )


@mcp.tool(
    description="""Delete an agent permanently.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        agent_id: The ID of the agent to delete
    """
)
def delete_agent(agent_id: str) -> TextContent:
    client.conversational_ai.agents.delete(agent_id=agent_id)

    return TextContent(
        type="text",
        text=f"Agent deleted successfully: Agent ID: {agent_id}",
    )


@mcp.tool(
    description="""Get widget embed information for an agent (equivalent to signed URL for web integration).

    Args:
        agent_id: The ID of the agent to get widget information for
    """
)
def get_signed_url(agent_id: str) -> TextContent:
    response = client.conversational_ai.agents.widget.get(agent_id=agent_id)

    # Generate embed information
    embed_url = f"https://elevenlabs.io/convai-widget/{agent_id}"
    
    return TextContent(
        type="text",
        text=f"Widget embed URL: {embed_url} (Agent ID: {response.agent_id}, Widget configured: {'Yes' if response.widget_config else 'No'})",
    )







# =============================================================================
# TOOLS API - UPDATED 2025 IMPLEMENTATION
# =============================================================================
# 
# NOTE: The Tools API is available via HTTP endpoints but not directly through 
# the conversational_ai client attributes. Using direct HTTP requests to the API.
# Endpoints: /v1/convai/tools/* as documented in 2025 API updates.

@mcp.tool(
    description="""Create a new tool for agents to use.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        tool_type: Type of tool ("server" or "client")
        name: Name of the tool
        description: Description of what the tool does
        server_tool: Server tool configuration (required if tool_type is "server")
        client_tool: Client tool configuration (required if tool_type is "client")
    """
)
def create_tool(
    tool_type: str,
    name: str,
    description: str,
    server_tool: dict | None = None,
    client_tool: dict | None = None,
) -> TextContent:
    if tool_type not in ["server", "client"]:
        return make_error("tool_type must be either 'server' or 'client'")

    if tool_type == "server" and server_tool is None:
        return make_error("server_tool configuration is required for server tools")
    
    if tool_type == "client" and client_tool is None:
        return make_error("client_tool configuration is required for client tools")

    # For server tools, extract webhook config from tool_config wrapper if present
    if server_tool and "tool_config" in server_tool:
        server_tool = server_tool["tool_config"]

    # Map tool_type to the correct API type
    api_tool_type = "webhook" if tool_type == "server" else "client"
    
    # Build the tool data according to the API structure
    tool_config = {
        "type": api_tool_type,
        "name": name,
        "description": description,
    }

    if tool_type == "server" and server_tool:
        # Server tools need a webhook configuration
        if "webhook" in server_tool:
            tool_config["webhook"] = server_tool["webhook"]
        else:
            # If no webhook key, assume the entire server_tool is the webhook config
            tool_config["webhook"] = server_tool
    elif tool_type == "client" and client_tool:
        tool_config["client_tool"] = client_tool
    
    tool_data = {"tool_config": tool_config}

    # Use HTTP request to tools API endpoint
    try:
        response = custom_client.post(
            "https://api.elevenlabs.io/v1/convai/tools",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json=tool_data
        )
        response.raise_for_status()
        result = response.json()
        
        return TextContent(
            type="text",
            text=f"Tool created successfully: Name: {name}, Tool ID: {result.get('id')}, Type: {tool_type}",
        )
    except Exception as e:
        return make_error(f"Failed to create tool: {str(e)}")


@mcp.tool(
    description="""Get details of a specific tool.

    Args:
        tool_id: The ID of the tool to retrieve
    """
)
def get_tool(tool_id: str) -> TextContent:
    try:
        response = custom_client.get(
            f"https://api.elevenlabs.io/v1/convai/tools/{tool_id}",
            headers={"xi-api-key": api_key}
        )
        response.raise_for_status()
        result = response.json()

        tool_config = result.get('tool_config', {})
        name = tool_config.get('name', 'Unknown')
        tool_type = tool_config.get('type', 'Unknown')
        tool_id = result.get('id', 'Unknown')
        description = tool_config.get('description', 'No description')
        
        return TextContent(
            type="text",
            text=f"Tool Details: Name: {name}, Tool ID: {tool_id}, Type: {tool_type}, Description: {description}",
        )
    except Exception as e:
        return make_error(f"Failed to get tool: {str(e)}")

@mcp.tool(
    description="""Update an existing tool.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        tool_id: The ID of the tool to update
        name: New name for the tool (optional)
        description: New description for the tool (optional)
        server_tool: Updated server tool configuration (optional)
        client_tool: Updated client tool configuration (optional)
    """
)
def update_tool(
    tool_id: str,
    name: str | None = None,
    description: str | None = None,
    server_tool: dict | None = None,
    client_tool: dict | None = None,
) -> TextContent:
    tool_config_updates = {}
    if name is not None:
        tool_config_updates["name"] = name
    if description is not None:
        tool_config_updates["description"] = description
    if server_tool is not None:
        tool_config_updates["server_tool"] = server_tool
    if client_tool is not None:
        tool_config_updates["client_tool"] = client_tool

    if not tool_config_updates:
        return make_error("At least one field must be provided for update")

    update_data = {"tool_config": tool_config_updates}

    try:
        response = custom_client.patch(
            f"https://api.elevenlabs.io/v1/convai/tools/{tool_id}",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json=update_data
        )
        response.raise_for_status()
        result = response.json()

        tool_config = result.get('tool_config', {})
        name = tool_config.get('name', 'Unknown')
        tool_id = result.get('id', 'Unknown')
        
        return TextContent(
            type="text",
            text=f"Tool updated successfully: Name: {name}, Tool ID: {tool_id}",
        )
    except Exception as e:
        return make_error(f"Failed to update tool: {str(e)}")

@mcp.tool(
    description="""Delete a tool permanently.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        tool_id: The ID of the tool to delete
    """
)
def delete_tool(tool_id: str) -> TextContent:
    try:
        response = custom_client.delete(
            f"https://api.elevenlabs.io/v1/convai/tools/{tool_id}",
            headers={"xi-api-key": api_key}
        )
        response.raise_for_status()

        return TextContent(
            type="text",
            text=f"Tool deleted successfully: Tool ID: {tool_id}",
        )
    except Exception as e:
        return make_error(f"Failed to delete tool: {str(e)}")
@mcp.tool(
    description="""List all tools in your workspace.

    Args:
        tool_type: Filter by tool type ("server" or "client") (optional)
        page: Page number for pagination (optional)
        page_size: Number of tools per page (optional)
    """
)
def list_tools(
    tool_type: str | None = None,
    page: int = 0,
    page_size: int = 20,
) -> TextContent:
    params = {"page": page, "page_size": page_size}
    if tool_type is not None:
        if tool_type not in ["server", "client"]:
            return make_error("tool_type must be either 'server' or 'client'")
        params["type"] = tool_type

    try:
        response = custom_client.get(
            "https://api.elevenlabs.io/v1/convai/tools",
            headers={"xi-api-key": api_key},
            params=params
        )
        response.raise_for_status()
        result = response.json()

        tools = result.get("tools", [])
        if not tools:
            return TextContent(type="text", text="No tools found.")

        tools_list = []
        for tool in tools:
            tool_config = tool.get('tool_config', {})
            name = tool_config.get('name', 'Unknown')
            tool_type = tool_config.get('type', 'Unknown')
            tool_id = tool.get('id', 'Unknown')
            description = tool_config.get('description', '')[:100] + ('...' if len(tool_config.get('description', '')) > 100 else '')
            
            tools_list.append(f"Name: {name}, ID: {tool_id}, Type: {tool_type}, Description: {description}")

        formatted_list = "\n".join(tools_list)
        return TextContent(type="text", text=f"Tools:\n{formatted_list}")
    except Exception as e:
        return make_error(f"Failed to list tools: {str(e)}")


@mcp.tool(
    description="""Get all agents that depend on a specific tool.

    Args:
        tool_id: The ID of the tool to check for dependencies
    """
)
def get_tool_dependent_agents(tool_id: str) -> TextContent:
    try:
        response = custom_client.get(
            f"https://api.elevenlabs.io/v1/convai/tools/{tool_id}/dependent-agents",
            headers={"xi-api-key": api_key}
        )
        response.raise_for_status()
        result = response.json()

        agents = result.get("agents", [])
        if not agents:
            return TextContent(type="text", text=f"No agents depend on tool {tool_id}.")

        agents_list = []
        for agent in agents:
            agents_list.append(f"Agent ID: {agent.get('agent_id')}, Name: {agent.get('name', 'Unknown')}")

        formatted_list = "\n".join(agents_list)
        return TextContent(type="text", text=f"Agents depending on tool {tool_id}:\n{formatted_list}")
    except Exception as e:
        return make_error(f"Failed to get dependent agents: {str(e)}")


@mcp.tool(
    description="""Get information about ElevenLabs Tools API deprecation and migration guidance.
    
    This tool provides critical information about the upcoming Tools API changes in July 2025
    and guidance on how to migrate from the legacy prompt.tools format to the new format.
    """
)
def get_tools_deprecation_info() -> TextContent:
    deprecation_info = """
🚨 CRITICAL: ElevenLabs Tools API Deprecation Timeline

BREAKING CHANGES COMING:
• July 14, 2025: Full backwards compatibility ends
• July 15, 2025: GET endpoints stop returning 'tools' field
• July 23, 2025: Legacy 'prompt.tools' field permanently removed

MIGRATION REQUIRED:
OLD FORMAT (being removed):
{
  "conversation_config": {
    "agent": {
      "prompt": {
        "tools": [
          {"type": "system", "name": "end_call"},
          {"type": "client", "name": "custom_tool"}
        ]
      }
    }
  }
}

NEW FORMAT (required after July 15):
{
  "conversation_config": {
    "agent": {
      "prompt": {
        "tool_ids": ["tool_123456789abcdef0"],
        "built_in_tools": {
          "end_call": {"name": "end_call", "type": "system"},
          "language_detection": null,
          "transfer_to_agent": null,
          "transfer_to_number": null,
          "skip_turn": null
        }
      }
    }
  }
}

MIGRATION STEPS:
1. Use create_tool() to create workspace tools
2. Get tool IDs from responses
3. Update agent configs to use tool_ids instead of tools array
4. Configure built_in_tools for system tools
5. Test before July 14, 2025

BENEFITS:
• Tool reuse across multiple agents
• Simplified tool management and audits
• Cleaner agent configurations

WARNING: Cannot mix prompt.tools and prompt.tool_ids in same request!
"""
    
    return TextContent(type="text", text=deprecation_info)




@mcp.tool(description="List all phone numbers associated with the ElevenLabs account")
def list_phone_numbers() -> TextContent:
    """List all phone numbers associated with the ElevenLabs account.

    Returns:
        TextContent containing formatted information about the phone numbers
    """
    response = client.conversational_ai.phone_numbers.list()

    if not response:
        return TextContent(type="text", text="No phone numbers found.")

    phone_info = []
    for phone in response:
        assigned_agent = "None"
        if phone.assigned_agent:
            assigned_agent = f"{phone.assigned_agent.agent_name} (ID: {phone.assigned_agent.agent_id})"

        phone_info.append(
            f"Phone Number: {phone.phone_number}\n"
            f"ID: {phone.phone_number_id}\n"
            f"Provider: {phone.provider}\n"
            f"Label: {phone.label}\n"
            f"Assigned Agent: {assigned_agent}"
        )

    formatted_info = "\n\n".join(phone_info)
    return TextContent(type="text", text=f"Phone Numbers:\n\n{formatted_info}")


@mcp.tool(
    description="""Create a new phone number and assign it to an agent.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        agent_id: The ID of the agent to assign the phone number to
        area_code: Preferred area code for the phone number
        country: Country code (e.g., "US", "CA", "GB")
        capabilities: Phone capabilities configuration (optional)
        webhook_url: Webhook URL for phone events (optional)
    """
)
def create_phone_number(
    agent_id: str,
    area_code: str,
    country: str = "US",
    capabilities: dict | None = None,
    webhook_url: str | None = None,
) -> TextContent:
    phone_data = {
        "agent_id": agent_id,
        "area_code": area_code,
        "country": country,
    }
    
    if capabilities is not None:
        phone_data["capabilities"] = capabilities
    if webhook_url is not None:
        phone_data["webhook_url"] = webhook_url

    response = client.conversational_ai.phone_numbers.create(**phone_data)

    return TextContent(
        type="text",
        text=f"Phone number created successfully: {response.phone_number}, ID: {response.phone_number_id}, Agent: {response.agent_id}",
    )


@mcp.tool(
    description="""Get details of a specific phone number.

    Args:
        phone_number_id: The ID of the phone number to retrieve
    """
)
def get_phone_number(phone_number_id: str) -> TextContent:
    response = client.conversational_ai.phone_numbers.get(phone_number_id=phone_number_id)

    assigned_agent = "None"
    if response.assigned_agent:
        assigned_agent = f"{response.assigned_agent.agent_name} (ID: {response.assigned_agent.agent_id})"

    return TextContent(
        type="text",
        text=f"Phone Number Details: Number: {response.phone_number}, ID: {response.phone_number_id}, Provider: {response.provider}, Label: {response.label}, Assigned Agent: {assigned_agent}",
    )


@mcp.tool(
    description="""Update phone number configuration.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        phone_number_id: The ID of the phone number to update
        agent_id: New agent ID to assign (optional)
        webhook_url: New webhook URL (optional)
        greeting_message: Custom greeting message (optional)
    """
)
def update_phone_number(
    phone_number_id: str,
    agent_id: str | None = None,
    webhook_url: str | None = None,
    greeting_message: str | None = None,
) -> TextContent:
    update_data = {}
    if agent_id is not None:
        update_data["agent_id"] = agent_id
    if webhook_url is not None:
        update_data["webhook_url"] = webhook_url
    if greeting_message is not None:
        update_data["greeting_message"] = greeting_message

    if not update_data:
        make_error("At least one field must be provided for update")

    response = client.conversational_ai.phone_numbers.update(
        phone_number_id=phone_number_id, **update_data
    )

    return TextContent(
        type="text",
        text=f"Phone number updated successfully: {response.phone_number}, ID: {response.phone_number_id}",
    )


@mcp.tool(
    description="""Release/delete a phone number.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        phone_number_id: The ID of the phone number to delete
    """
)
def delete_phone_number(phone_number_id: str) -> TextContent:
    client.conversational_ai.phone_numbers.delete(phone_number_id=phone_number_id)

    return TextContent(
        type="text",
        text=f"Phone number deleted/released successfully: Phone Number ID: {phone_number_id}",
    )


# =============================================================================
# WIDGET API
# =============================================================================

@mcp.tool(
    description="""Get widget configuration for embedding an agent on websites.

    Args:
        agent_id: The ID of the agent to get widget configuration for
    """
)
def get_widget(agent_id: str) -> TextContent:
    response = client.conversational_ai.widgets.get(agent_id=agent_id)

    config_info = []
    if response.config:
        config = response.config
        config_info.extend([
            f"Position: {getattr(config, 'position', 'N/A')}",
            f"Theme: {getattr(config, 'theme', 'N/A')}",
            f"Launcher Text: {getattr(config, 'launcher_text', 'N/A')}",
        ])
        if hasattr(config, 'avatar_url') and config.avatar_url:
            config_info.append(f"Avatar URL: {config.avatar_url}")

    config_text = ", ".join(config_info) if config_info else "Default configuration"

    return TextContent(
        type="text",
        text=f"Widget Configuration: Widget ID: {response.widget_id}, Agent ID: {response.agent_id}, Config: {config_text}, Embed Code: {response.embed_code}",
    )


@mcp.tool(
    description="""Upload a custom avatar for an agent's widget.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        agent_id: The ID of the agent to upload avatar for
        file_path: Path to the image file (PNG, JPG, GIF)
    """
)
def create_widget_avatar(agent_id: str, file_path: str) -> TextContent:
    file_path_obj = handle_input_file(file_path, audio_content_check=False)
    
    with file_path_obj.open("rb") as f:
        response = client.conversational_ai.widgets.upload_avatar(
            agent_id=agent_id,
            file=f
        )

    return TextContent(
        type="text",
        text=f"Widget avatar uploaded successfully for agent {agent_id}. Avatar URL: {getattr(response, 'avatar_url', 'uploaded')}",
    )


# =============================================================================
# WORKSPACE API
# =============================================================================

@mcp.tool(
    description="""Get workspace settings and configuration.

    Returns:
        TextContent containing workspace settings information
    """
)
def get_workspace_settings() -> TextContent:
    response = client.conversational_ai.workspace.settings.get()

    settings_info = []
    if response.settings:
        settings = response.settings
        settings_info.extend([
            f"Data Retention Days: {getattr(settings, 'data_retention_days', 'N/A')}",
            f"GDPR Compliant: {getattr(settings, 'gdpr_compliant', 'N/A')}",
            f"EU Data Residency: {getattr(settings, 'eu_data_residency', 'N/A')}",
        ])
        
        if hasattr(settings, 'allowed_llm_providers') and settings.allowed_llm_providers:
            providers = ", ".join(settings.allowed_llm_providers)
            settings_info.append(f"Allowed LLM Providers: {providers}")

    settings_text = "\n".join(settings_info) if settings_info else "No settings available"

    return TextContent(
        type="text",
        text=f"Workspace Settings:\nWorkspace ID: {response.workspace_id}\nName: {response.name}\n\nSettings:\n{settings_text}",
    )


@mcp.tool(
    description="""Update workspace settings.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        settings: Dictionary of settings to update
    """
)
def update_workspace_settings(settings: dict) -> TextContent:
    if not settings:
        make_error("Settings dictionary is required")

    response = client.conversational_ai.workspace.settings.update(settings=settings)

    return TextContent(
        type="text",
        text=f"Workspace settings updated successfully for workspace: {response.workspace_id}",
    )


@mcp.tool(
    description="""List all stored secrets in the workspace (returns secret names only, not values).

    Returns:
        TextContent containing list of secret names and metadata
    """
)
def list_secrets() -> TextContent:
    response = client.conversational_ai.workspace.secrets.list()

    if not response.secrets:
        return TextContent(type="text", text="No secrets found in workspace.")

    secrets_list = []
    for secret in response.secrets:
        last_used = getattr(secret, 'last_used', 'Never')
        secrets_list.append(
            f"Name: {secret.name}, ID: {secret.secret_id}, Created: {secret.created_at}, Last Used: {last_used}"
        )

    formatted_list = "\n".join(secrets_list)
    return TextContent(type="text", text=f"Workspace Secrets:\n{formatted_list}")


@mcp.tool(
    description="""Create a new secret in the workspace.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        name: Name of the secret (e.g., "OPENAI_API_KEY")
        value: The secret value to store
        description: Description of the secret (optional)
    """
)
def create_secret(
    name: str,
    value: str,
    description: str | None = None,
) -> TextContent:
    secret_data = {
        "name": name,
        "value": value,
    }
    if description is not None:
        secret_data["description"] = description

    response = client.conversational_ai.workspace.secrets.create(**secret_data)

    return TextContent(
        type="text",
        text=f"Secret created successfully: Name: {name}, Secret ID: {response.secret_id}",
    )


@mcp.tool(
    description="""Delete a stored secret from the workspace.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        secret_id: The ID of the secret to delete
    """
)
def delete_secret(secret_id: str) -> TextContent:
    client.conversational_ai.workspace.secrets.delete(secret_id=secret_id)

    return TextContent(
        type="text",
        text=f"Secret deleted successfully: Secret ID: {secret_id}",
    )


# =============================================================================
# CURL-BASED TOOL CREATION UTILITIES
# =============================================================================

@mcp.tool(
    description="""Create a weather integration tool using popular weather APIs.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        api_provider: Weather API provider ("open-meteo" - free, "openweathermap" - requires API key)
        weather_api_key: API key for services that require authentication (required for openweathermap)
    """
)
def create_weather_integration_tool(
    api_provider: str = "open-meteo",
    weather_api_key: str | None = None,
) -> TextContent:
    try:
        # Create the weather tool configuration
        tool_config = create_weather_tool(api_provider, weather_api_key)
        
        # Update tool_config to have correct type
        tool_config["tool_config"]["type"] = "webhook"
        
        # Create the tool using ElevenLabs API
        response = custom_client.post(
            "https://api.elevenlabs.io/v1/convai/tools",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json=tool_config
        )
        response.raise_for_status()
        result = response.json()
        
        return TextContent(
            type="text",
            text=f"Weather tool created successfully: Tool ID: {result.get('id')}, Provider: {api_provider}, Name: get_weather",
        )
    except Exception as e:
        return make_error(f"Failed to create weather tool: {str(e)}")


@mcp.tool(
    description="""Create a custom webhook tool for integrating with external services.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        name: Tool name (snake_case recommended)
        description: Clear description of what the tool does
        webhook_url: Target webhook URL
        method: HTTP method (GET, POST, PUT, DELETE)
        auth_header: Authorization header template (e.g., "Bearer {api_key}")
        custom_headers: Additional headers as JSON string (optional)
        parameters_schema: Parameter schema as JSON string (optional)
    """
)
def create_custom_webhook_tool(
    name: str,
    description: str,
    webhook_url: str,
    method: str = "POST",
    auth_header: str | None = None,
    custom_headers: str | None = None,
    parameters_schema: str | None = None,
) -> TextContent:
    try:
        import json
        
        # Parse optional JSON parameters
        headers_dict = None
        if custom_headers:
            headers_dict = json.loads(custom_headers)
        
        params_dict = None
        if parameters_schema:
            params_dict = json.loads(parameters_schema)
        
        # Create the webhook tool configuration
        tool_config = create_webhook_tool(
            name=name,
            description=description,
            url=webhook_url,
            method=method,
            auth_header=auth_header,
            custom_headers=headers_dict,
            parameters=params_dict
        )
        
        # Create the tool using ElevenLabs API
        response = custom_client.post(
            "https://api.elevenlabs.io/v1/convai/tools",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json=tool_config
        )
        response.raise_for_status()
        result = response.json()
        
        return TextContent(
            type="text",
            text=f"Webhook tool created successfully: Tool ID: {result.get('id')}, Name: {name}, URL: {webhook_url}",
        )
    except Exception as e:
        return make_error(f"Failed to create webhook tool: {str(e)}")


@mcp.tool(
    description="""Create a REST API integration tool for external services.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        name: Tool name
        description: Tool description
        base_url: Base API URL
        endpoint: API endpoint path (optional)
        method: HTTP method (GET, POST, PUT, DELETE)
        auth_type: Authentication type ("bearer", "api_key", "basic")
        auth_token: Authentication token/key
        path_params: Path parameters as JSON string (optional)
        query_params: Query parameters as JSON string (optional)
        body_params: Body parameters as JSON string (optional)
    """
)
def create_api_integration_tool(
    name: str,
    description: str,
    base_url: str,
    endpoint: str = "",
    method: str = "GET",
    auth_type: str = "bearer",
    auth_token: str | None = None,
    path_params: str | None = None,
    query_params: str | None = None,
    body_params: str | None = None,
) -> TextContent:
    try:
        import json
        
        # Parse optional JSON parameters
        path_params_dict = json.loads(path_params) if path_params else None
        query_params_dict = json.loads(query_params) if query_params else None
        body_params_dict = json.loads(body_params) if body_params else None
        
        # Create the REST API tool configuration
        tool_config = create_rest_api_tool(
            name=name,
            description=description,
            base_url=base_url,
            endpoint=endpoint,
            method=method,
            auth_type=auth_type,
            auth_token=auth_token,
            path_params=path_params_dict,
            query_params=query_params_dict,
            body_params=body_params_dict
        )
        
        # Create the tool using ElevenLabs API
        response = custom_client.post(
            "https://api.elevenlabs.io/v1/convai/tools",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json=tool_config
        )
        response.raise_for_status()
        result = response.json()
        
        return TextContent(
            type="text",
            text=f"API integration tool created successfully: Tool ID: {result.get('id')}, Name: {name}, URL: {base_url}{endpoint}",
        )
    except Exception as e:
        return make_error(f"Failed to create API integration tool: {str(e)}")


@mcp.tool(
    description="""Create a CRM integration tool for popular platforms.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        crm_platform: CRM platform ("salesforce", "hubspot")
        webhook_url: CRM webhook endpoint URL
        auth_token: Authentication token for the CRM
    """
)
def create_crm_integration_tool(
    crm_platform: str,
    webhook_url: str,
    auth_token: str | None = None,
) -> TextContent:
    try:
        # Create the CRM tool configuration
        tool_config = create_crm_webhook_tool(crm_platform, webhook_url, auth_token)
        
        # Create the tool using ElevenLabs API
        response = custom_client.post(
            "https://api.elevenlabs.io/v1/convai/tools",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json=tool_config
        )
        response.raise_for_status()
        result = response.json()
        
        tool_name = tool_config["tool_config"]["name"]
        
        return TextContent(
            type="text",
            text=f"CRM integration tool created successfully: Tool ID: {result.get('id')}, Platform: {crm_platform}, Name: {tool_name}",
        )
    except Exception as e:
        return make_error(f"Failed to create CRM integration tool: {str(e)}")


@mcp.tool(
    description="""Create a notification tool for messaging platforms.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        service: Notification service ("slack", "discord")
        webhook_url: Service webhook URL
        auth_token: Authentication token if required
    """
)
def create_notification_tool_integration(
    service: str,
    webhook_url: str,
    auth_token: str | None = None,
) -> TextContent:
    try:
        # Create the notification tool configuration
        tool_config = create_notification_tool(service, webhook_url, auth_token)
        
        # Extract the webhook config from tool_config wrapper
        webhook_config = tool_config["tool_config"]["webhook"]
        name = tool_config["tool_config"]["name"]
        description = tool_config["tool_config"]["description"]
        
        # Create the tool using ElevenLabs API with proper structure
        tool_data = {
            "type": "server",
            "name": name,
            "description": description,
            "webhook": webhook_config
        }
        
        response = custom_client.post(
            "https://api.elevenlabs.io/v1/convai/tools",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json=tool_data
        )
        response.raise_for_status()
        result = response.json()
        
        tool_name = tool_config["tool_config"]["name"]
        
        return TextContent(
            type="text",
            text=f"Notification tool created successfully: Tool ID: {result.get('id')}, Service: {service}, Name: {tool_name}",
        )
    except Exception as e:
        return make_error(f"Failed to create notification tool: {str(e)}")


@mcp.tool(
    description="""Generate a curl command equivalent for testing a tool configuration.

    Args:
        tool_id: The ID of the tool to generate curl command for
        parameter_values: Parameter values as JSON string
    """
)
def generate_curl_test_command(
    tool_id: str,
    parameter_values: str,
) -> TextContent:
    try:
        import json
        
        # Get tool configuration
        response = custom_client.get(
            f"https://api.elevenlabs.io/v1/convai/tools/{tool_id}",
            headers={"xi-api-key": api_key}
        )
        response.raise_for_status()
        tool_data = response.json()
        
        # Parse parameter values
        params = json.loads(parameter_values)
        
        # Generate curl command
        curl_command = generate_curl_command({"tool_config": tool_data.get("tool_config", {})}, params)
        
        return TextContent(
            type="text",
            text=f"Curl command for tool {tool_id}:\n\n{curl_command}",
        )
    except Exception as e:
        return make_error(f"Failed to generate curl command: {str(e)}")


@mcp.tool(
    description="""Validate a tool configuration for errors and best practices.

    Args:
        tool_id: The ID of the tool to validate
    """
)
def validate_tool_configuration(tool_id: str) -> TextContent:
    try:
        # Get tool configuration
        response = custom_client.get(
            f"https://api.elevenlabs.io/v1/convai/tools/{tool_id}",
            headers={"xi-api-key": api_key}
        )
        response.raise_for_status()
        tool_data = response.json()
        
        # Validate configuration
        errors = validate_tool_config({"tool_config": tool_data.get("tool_config", {})})
        
        if not errors:
            return TextContent(
                type="text",
                text=f"Tool {tool_id} configuration is valid and follows best practices.",
            )
        else:
            error_list = "\n".join(f"- {error}" for error in errors)
            return TextContent(
                type="text",
                text=f"Tool {tool_id} configuration has the following issues:\n{error_list}",
            )
    except Exception as e:
        return make_error(f"Failed to validate tool configuration: {str(e)}")


@mcp.tool(
    description="""List available tool templates for common integrations.

    Returns information about predefined tool templates that can be created.
    """
)
def list_tool_templates() -> TextContent:
    template_info = []
    for category, info in TOOL_TEMPLATES.items():
        variants = ", ".join(info["variants"].keys())
        template_info.append(f"{category}: {info['description']} (variants: {variants})")
    
    formatted_templates = "\n".join(template_info)
    return TextContent(
        type="text",
        text=f"Available tool templates:\n{formatted_templates}",
    )


def main():
    print("Starting MCP server")
    """Run the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
