"""
ElevenLabs MCP Server - curl-based Tool Utilities

This module provides utilities for creating server tools that use HTTP requests,
equivalent to curl commands. These tools enable conversational AI agents to
interact with external APIs and webhooks.

⚠️ IMPORTANT: Server tools created with these utilities will make HTTP requests
to external services, which may incur costs or expose data. Only use with
trusted endpoints and proper authentication.
"""

from typing import Dict, List, Optional, Any, Union
import json
import urllib.parse


def generate_tool_config(
    name: str,
    description: str,
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    path_parameters: Optional[Dict[str, Dict[str, str]]] = None,
    query_parameters: Optional[Dict[str, Dict[str, str]]] = None,
    body_parameters: Optional[Dict[str, Any]] = None,
    auth_header: Optional[str] = None,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """
    Generate a complete tool configuration for a server tool that makes HTTP requests.
    
    Args:
        name: Tool name (snake_case recommended)
        description: Clear description of what the tool does and when to use it
        url: Target URL, can include path parameters in {brackets}
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: Static headers to include in requests
        path_parameters: Path parameter definitions for URL templating
        query_parameters: Query parameter definitions
        body_parameters: Request body parameter schema
        auth_header: Authorization header template (e.g., "Bearer {api_key}")
        timeout_seconds: Request timeout in seconds
        
    Returns:
        Dict containing complete tool_config for ElevenLabs API
        
    Example:
        >>> config = generate_tool_config(
        ...     name="get_weather",
        ...     description="Get current weather for a location",
        ...     url="https://api.openweathermap.org/data/2.5/weather",
        ...     query_parameters={
        ...         "q": {"type": "string", "description": "City name"},
        ...         "appid": {"type": "string", "description": "API key"}
        ...     }
        ... )
    """
    # Build parameters schema first
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    # Add path parameters
    if path_parameters:
        for param_name, param_def in path_parameters.items():
            # Remove 'required' from property definition - it belongs in the required array
            param_copy = param_def.copy()
            is_required = param_copy.pop("required", False)
            parameters_schema["properties"][param_name] = param_copy
            if is_required:
                parameters_schema["required"].append(param_name)
    
    # Add query parameters
    if query_parameters:
        for param_name, param_def in query_parameters.items():
            # Remove 'required' from property definition - it belongs in the required array
            param_copy = param_def.copy()
            is_required = param_copy.pop("required", False)
            parameters_schema["properties"][param_name] = param_copy
            if is_required:
                parameters_schema["required"].append(param_name)
    
    # Add body parameters
    if body_parameters:
        if method.upper() in ["POST", "PUT", "PATCH"]:
            # For methods that support request body
            for param_name, param_def in body_parameters.get("properties", {}).items():
                parameters_schema["properties"][param_name] = param_def
            if "required" in body_parameters:
                parameters_schema["required"].extend(body_parameters["required"])
        else:
            raise ValueError(f"Body parameters not supported for {method} method")

    # Create webhook configuration
    webhook_config = {
        "url": url,
        "method": method.upper(),
        "api_schema": {
            "url": url,
            "method": method.upper(),
            "parameters": parameters_schema
        }
    }
    
    # Add headers if provided
    if headers or auth_header:
        request_headers = {}
        if headers:
            request_headers.update(headers)
        if auth_header:
            request_headers["Authorization"] = auth_header
        webhook_config["headers"] = request_headers

    # Return the webhook configuration directly for server tools
    webhook_config["response_timeout_secs"] = timeout_seconds
    webhook_config["parameters"] = parameters_schema
    
    return {
        "tool_config": {
            "name": name,
            "description": description,
            "webhook": webhook_config
        }
    }


def create_weather_tool(
    api_provider: str = "open-meteo",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a weather tool configuration for common weather APIs.
    
    Args:
        api_provider: Weather API provider ("open-meteo", "openweathermap")
        api_key: API key for services that require authentication
        
    Returns:
        Tool configuration dict
    """
    if api_provider == "open-meteo":
        # Open-Meteo is free and doesn't require API key
        return generate_tool_config(
            name="get_weather",
            description="Get current weather and forecast for a location. Use when users ask about weather conditions, temperature, or forecasts.",
            url="https://api.open-meteo.com/v1/forecast",
            method="GET",
            query_parameters={
                "latitude": {
                    "type": "number",
                    "description": "Latitude coordinate of the location",
                    "required": True
                },
                "longitude": {
                    "type": "number", 
                    "description": "Longitude coordinate of the location",
                    "required": True
                },
                "current": {
                    "type": "string",
                    "description": "Current weather parameters (comma-separated): temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                    "default": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
                },
                "daily": {
                    "type": "string",
                    "description": "Daily forecast parameters: temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                    "default": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone for the location (e.g., 'America/New_York')",
                    "default": "auto"
                },
                "forecast_days": {
                    "type": "number",
                    "description": "Number of forecast days (1-16)",
                    "default": 7
                }
            }
        )
    
    elif api_provider == "openweathermap":
        if not api_key:
            raise ValueError("API key required for OpenWeatherMap")
        
        return generate_tool_config(
            name="get_weather",
            description="Get current weather and forecast for a location using OpenWeatherMap API.",
            url="https://api.openweathermap.org/data/2.5/weather",
            method="GET",
            query_parameters={
                "q": {
                    "type": "string",
                    "description": "City name (e.g., 'London,UK' or 'New York,US')",
                    "required": True
                },
                "appid": {
                    "type": "string",
                    "description": "OpenWeatherMap API key",
                    "default": api_key
                },
                "units": {
                    "type": "string",
                    "description": "Temperature units (metric, imperial, kelvin)",
                    "default": "metric"
                }
            }
        )
    
    else:
        raise ValueError(f"Unsupported weather API provider: {api_provider}")


def create_webhook_tool(
    name: str,
    description: str,
    url: str,
    method: str = "POST",
    auth_header: Optional[str] = None,
    custom_headers: Optional[Dict[str, str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 30
) -> Dict[str, Any]:
    """
    Create a generic webhook tool configuration.
    
    Args:
        name: Tool name
        description: Tool description
        url: Webhook URL
        method: HTTP method
        auth_header: Authorization header template
        custom_headers: Additional headers
        parameters: Parameter schema
        timeout_seconds: Request timeout
        
    Returns:
        Tool configuration dict
    """
    headers = {}
    if custom_headers:
        headers.update(custom_headers)
    if method.upper() in ["POST", "PUT", "PATCH"]:
        headers["Content-Type"] = "application/json"
    
    # Default parameter schema for webhooks
    if parameters is None:
        parameters = {
            "properties": {
                "data": {
                    "type": "object",
                    "description": "Data to send in the webhook payload",
                    "additionalProperties": True
                }
            },
            "required": ["data"]
        }
    
    return generate_tool_config(
        name=name,
        description=description,
        url=url,
        method=method,
        headers=headers,
        auth_header=auth_header,
        body_parameters=parameters,
        timeout_seconds=timeout_seconds
    )


def create_rest_api_tool(
    name: str,
    description: str,
    base_url: str,
    endpoint: str = "",
    method: str = "GET",
    auth_type: str = "bearer",
    auth_token: Optional[str] = None,
    path_params: Optional[Dict[str, Dict[str, str]]] = None,
    query_params: Optional[Dict[str, Dict[str, str]]] = None,
    body_params: Optional[Dict[str, Any]] = None,
    custom_headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Create a REST API tool configuration.
    
    Args:
        name: Tool name
        description: Tool description  
        base_url: Base API URL
        endpoint: API endpoint path
        method: HTTP method
        auth_type: Authentication type ("bearer", "api_key", "basic")
        auth_token: Authentication token/key
        path_params: Path parameter definitions
        query_params: Query parameter definitions
        body_params: Body parameter schema
        custom_headers: Additional headers
        
    Returns:
        Tool configuration dict
    """
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}" if endpoint else base_url
    
    headers = {}
    if custom_headers:
        headers.update(custom_headers)
    
    auth_header = None
    if auth_token:
        if auth_type == "bearer":
            auth_header = f"Bearer {auth_token}"
        elif auth_type == "api_key":
            headers["X-API-Key"] = auth_token
        elif auth_type == "basic":
            auth_header = f"Basic {auth_token}"
    
    if method.upper() in ["POST", "PUT", "PATCH"] and "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"
    
    return generate_tool_config(
        name=name,
        description=description,
        url=url,
        method=method,
        headers=headers if headers else None,
        auth_header=auth_header,
        path_parameters=path_params,
        query_parameters=query_params,
        body_parameters=body_params
    )


def generate_curl_command(
    tool_config: Dict[str, Any],
    parameter_values: Dict[str, Any]
) -> str:
    """
    Generate a curl command equivalent to the tool configuration with given parameters.
    
    Args:
        tool_config: Tool configuration from generate_tool_config()
        parameter_values: Actual parameter values to use
        
    Returns:
        Formatted curl command string
        
    Example:
        >>> config = create_weather_tool()
        >>> params = {"latitude": 40.7128, "longitude": -74.0060}
        >>> curl_cmd = generate_curl_command(config, params)
        >>> print(curl_cmd)
        curl -X GET "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
    """
    tool = tool_config["tool_config"]
    webhook = tool.get("webhook", tool)  # Fallback to tool if no webhook nested
    url = webhook["url"]
    method = webhook["method"] 
    headers = webhook.get("headers", {})
    
    # Replace path parameters in URL
    for param_name, param_value in parameter_values.items():
        url = url.replace(f"{{{param_name}}}", str(param_value))
    
    # Build query parameters
    query_params = {}
    body_params = {}
    
    # Determine which parameters go where based on method
    if method in ["GET", "DELETE"]:
        # All parameters become query parameters
        query_params = parameter_values.copy()
    else:
        # For POST/PUT/PATCH, need to separate query vs body parameters
        # This is a simplified approach - in reality, you'd need to check the parameter definitions
        query_params = {k: v for k, v in parameter_values.items() if k not in ["data", "payload", "body"]}
        body_params = {k: v for k, v in parameter_values.items() if k in ["data", "payload", "body"]}
        
        # If no explicit body params but method supports body, put non-query params in body
        if not body_params and method in ["POST", "PUT", "PATCH"]:
            body_params = parameter_values.copy()
            query_params = {}
    
    # Add query parameters to URL
    if query_params:
        query_string = urllib.parse.urlencode(query_params)
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{query_string}"
    
    # Build curl command
    curl_parts = [f'curl -X {method}']
    
    # Add URL
    curl_parts.append(f'"{url}"')
    
    # Add headers
    for header_name, header_value in headers.items():
        curl_parts.append(f'-H "{header_name}: {header_value}"')
    
    # Add body data for methods that support it
    if body_params and method in ["POST", "PUT", "PATCH"]:
        if isinstance(body_params, dict):
            json_data = json.dumps(body_params)
            curl_parts.append(f"-d '{json_data}'")
        else:
            curl_parts.append(f"-d '{body_params}'")
    
    return " \\\n  ".join(curl_parts)


def validate_tool_config(tool_config: Dict[str, Any]) -> List[str]:
    """
    Validate a tool configuration and return any errors found.
    
    Args:
        tool_config: Tool configuration to validate
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    if "tool_config" not in tool_config:
        errors.append("Missing 'tool_config' key")
        return errors
    
    config = tool_config["tool_config"]
    
    # Check required fields
    required_fields = ["type", "name", "description"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Check tool type
    if config.get("type") != "webhook":
        errors.append("Tool type must be 'webhook' for curl-based tools")
    
    # Check webhook required fields
    webhook_required = ["url", "method"]
    for field in webhook_required:
        if field not in config:
            errors.append(f"Missing webhook field: {field}")
    
    # Validate HTTP method
    valid_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
    method = config.get("method", "").upper()
    if method not in valid_methods:
        errors.append(f"Invalid HTTP method: {method}. Must be one of {valid_methods}")
    
    # Validate URL
    url = config.get("url", "")
    if not url.startswith(("http://", "https://")):
        errors.append("URL must start with http:// or https://")
    
    # Validate parameters schema if present
    if "parameters" in config:
        params = config["parameters"]
        if not isinstance(params, dict):
            errors.append("Parameters must be a dictionary")
        elif "type" in params and params["type"] != "object":
            errors.append("Parameters type must be 'object'")
    
    return errors


def create_database_api_tool(
    name: str,
    description: str,
    api_base_url: str,
    table_name: str,
    api_key: Optional[str] = None,
    operation: str = "query"
) -> Dict[str, Any]:
    """
    Create a database API tool for common database-as-a-service platforms.
    
    Args:
        name: Tool name
        description: Tool description
        api_base_url: Base URL of the database API
        table_name: Table/collection name to query
        api_key: API authentication key
        operation: Database operation ("query", "insert", "update", "delete")
        
    Returns:
        Tool configuration dict
    """
    if operation == "query":
        return generate_tool_config(
            name=name,
            description=description,
            url=f"{api_base_url}/query",
            method="POST",
            auth_header=f"Bearer {api_key}" if api_key else None,
            body_parameters={
                "properties": {
                    "table": {
                        "type": "string",
                        "description": f"Table name to query",
                        "default": table_name
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query or filter conditions"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of records to return",
                        "default": 100
                    }
                },
                "required": ["query"]
            }
        )
    elif operation == "insert":
        return generate_tool_config(
            name=name,
            description=description,
            url=f"{api_base_url}/insert",
            method="POST",
            auth_header=f"Bearer {api_key}" if api_key else None,
            body_parameters={
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name to insert into",
                        "default": table_name
                    },
                    "data": {
                        "type": "object",
                        "description": "Data to insert as key-value pairs",
                        "additionalProperties": True
                    }
                },
                "required": ["data"]
            }
        )
    else:
        raise ValueError(f"Unsupported database operation: {operation}")


def create_crm_webhook_tool(
    crm_platform: str,
    webhook_url: str,
    auth_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a CRM integration webhook tool for common platforms.
    
    Args:
        crm_platform: CRM platform ("salesforce", "hubspot", "pipedrive")
        webhook_url: Webhook endpoint URL
        auth_token: Authentication token
        
    Returns:
        Tool configuration dict
    """
    if crm_platform.lower() == "salesforce":
        return create_webhook_tool(
            name="update_salesforce_record",
            description="Update or create records in Salesforce CRM. Use when users want to save contact information, leads, or opportunities.",
            url=webhook_url,
            method="POST",
            auth_header=f"Bearer {auth_token}" if auth_token else None,
            parameters={
                "properties": {
                    "object_type": {
                        "type": "string",
                        "description": "Salesforce object type (Lead, Contact, Opportunity, Account)",
                        "enum": ["Lead", "Contact", "Opportunity", "Account"]
                    },
                    "record_data": {
                        "type": "object",
                        "description": "Record data with field names and values",
                        "additionalProperties": True
                    },
                    "record_id": {
                        "type": "string",
                        "description": "Existing record ID for updates (optional)"
                    }
                },
                "required": ["object_type", "record_data"]
            }
        )
    
    elif crm_platform.lower() == "hubspot":
        return create_webhook_tool(
            name="update_hubspot_contact",
            description="Create or update contacts in HubSpot CRM. Use when users provide contact information or want to track interactions.",
            url=webhook_url,
            method="POST",
            auth_header=f"Bearer {auth_token}" if auth_token else None,
            parameters={
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "Contact email address",
                        "format": "email"
                    },
                    "properties": {
                        "type": "object",
                        "description": "Contact properties (firstname, lastname, company, phone, etc.)",
                        "additionalProperties": True
                    },
                    "associations": {
                        "type": "array",
                        "description": "Associated records (companies, deals, etc.)",
                        "items": {"type": "object"}
                    }
                },
                "required": ["email", "properties"]
            }
        )
    
    else:
        raise ValueError(f"Unsupported CRM platform: {crm_platform}")


def create_notification_tool(
    service: str,
    webhook_url: str,
    auth_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create notification/messaging tools for common services.
    
    Args:
        service: Notification service ("slack", "discord", "teams", "email")
        webhook_url: Webhook URL for the service
        auth_token: Authentication token if required
        
    Returns:
        Tool configuration dict
    """
    if service.lower() == "slack":
        return create_webhook_tool(
            name="send_slack_message",
            description="Send messages to Slack channels. Use when users want to notify team members or share information.",
            url=webhook_url,
            method="POST",
            parameters={
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Message text to send"
                    },
                    "channel": {
                        "type": "string",
                        "description": "Slack channel (optional, uses webhook default)"
                    },
                    "username": {
                        "type": "string",
                        "description": "Display name for the bot (optional)"
                    },
                    "attachments": {
                        "type": "array",
                        "description": "Rich message attachments (optional)",
                        "items": {"type": "object"}
                    }
                },
                "required": ["text"]
            }
        )
    
    elif service.lower() == "discord":
        return create_webhook_tool(
            name="send_discord_message",
            description="Send messages to Discord channels. Use for team notifications or community updates.",
            url=webhook_url,
            method="POST",
            parameters={
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Message content to send"
                    },
                    "username": {
                        "type": "string",
                        "description": "Override username for the webhook"
                    },
                    "embeds": {
                        "type": "array",
                        "description": "Rich embed objects (optional)",
                        "items": {"type": "object"}
                    }
                },
                "required": ["content"]
            }
        )
    
    else:
        raise ValueError(f"Unsupported notification service: {service}")


def create_file_upload_tool(
    name: str,
    description: str,
    upload_url: str,
    auth_header: Optional[str] = None,
    max_file_size_mb: int = 10
) -> Dict[str, Any]:
    """
    Create a file upload tool configuration.
    
    Args:
        name: Tool name
        description: Tool description
        upload_url: File upload endpoint URL
        auth_header: Authorization header
        max_file_size_mb: Maximum file size in MB
        
    Returns:
        Tool configuration dict
    """
    return generate_tool_config(
        name=name,
        description=description,
        url=upload_url,
        method="POST",
        headers={"Content-Type": "multipart/form-data"},
        auth_header=auth_header,
        body_parameters={
            "properties": {
                "file": {
                    "type": "string",
                    "description": f"File to upload (base64 encoded, max {max_file_size_mb}MB)",
                    "format": "byte"
                },
                "filename": {
                    "type": "string",
                    "description": "Original filename"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional file metadata",
                    "additionalProperties": True
                }
            },
            "required": ["file", "filename"]
        }
    )


def create_search_tool(
    search_engine: str,
    api_key: str,
    custom_search_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a web search tool for common search APIs.
    
    Args:
        search_engine: Search engine ("google", "bing", "duckduckgo")
        api_key: API key for the search service
        custom_search_id: Custom search engine ID (for Google)
        
    Returns:
        Tool configuration dict
    """
    if search_engine.lower() == "google":
        if not custom_search_id:
            raise ValueError("custom_search_id required for Google Custom Search")
        
        return generate_tool_config(
            name="web_search",
            description="Search the web for current information. Use when users ask questions that require up-to-date information not in your training data.",
            url="https://www.googleapis.com/customsearch/v1",
            method="GET",
            query_parameters={
                "key": {
                    "type": "string",
                    "description": "Google API key",
                    "default": api_key
                },
                "cx": {
                    "type": "string",
                    "description": "Custom search engine ID",
                    "default": custom_search_id
                },
                "q": {
                    "type": "string",
                    "description": "Search query",
                    "required": True
                },
                "num": {
                    "type": "number",
                    "description": "Number of results to return (1-10)",
                    "default": 5
                },
                "safe": {
                    "type": "string",
                    "description": "Safe search setting",
                    "default": "active"
                }
            }
        )
    
    elif search_engine.lower() == "bing":
        return generate_tool_config(
            name="web_search",
            description="Search the web using Bing Search API for current information.",
            url="https://api.bing.microsoft.com/v7.0/search",
            method="GET",
            headers={"Ocp-Apim-Subscription-Key": api_key},
            query_parameters={
                "q": {
                    "type": "string",
                    "description": "Search query",
                    "required": True
                },
                "count": {
                    "type": "number",
                    "description": "Number of results to return",
                    "default": 5
                },
                "safeSearch": {
                    "type": "string",
                    "description": "Safe search setting (Off, Moderate, Strict)",
                    "default": "Moderate"
                }
            }
        )
    
    else:
        raise ValueError(f"Unsupported search engine: {search_engine}")


# Predefined tool templates for common use cases
TOOL_TEMPLATES = {
    "weather": {
        "description": "Weather information tools",
        "variants": {
            "open-meteo": lambda: create_weather_tool("open-meteo"),
            "openweathermap": lambda api_key: create_weather_tool("openweathermap", api_key)
        }
    },
    "crm": {
        "description": "CRM integration tools",
        "variants": {
            "salesforce": lambda url, token: create_crm_webhook_tool("salesforce", url, token),
            "hubspot": lambda url, token: create_crm_webhook_tool("hubspot", url, token)
        }
    },
    "notifications": {
        "description": "Notification and messaging tools",
        "variants": {
            "slack": lambda url: create_notification_tool("slack", url),
            "discord": lambda url: create_notification_tool("discord", url)
        }
    },
    "search": {
        "description": "Web search tools",
        "variants": {
            "google": lambda api_key, cx: create_search_tool("google", api_key, cx),
            "bing": lambda api_key: create_search_tool("bing", api_key)
        }
    }
}