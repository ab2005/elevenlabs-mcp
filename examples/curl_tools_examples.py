"""
ElevenLabs MCP Server - curl-based Tool Examples

This file demonstrates how to use the curl-based tool creation utilities
to build server tools for ElevenLabs conversational AI agents.

These examples show various integration patterns for common APIs and services.
"""

import json
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


def example_weather_tool():
    """Example: Create a weather tool using Open-Meteo (free service)"""
    print("=== Weather Tool Example ===")
    
    # Create weather tool configuration
    weather_config = create_weather_tool("open-meteo")
    
    print("Weather tool configuration:")
    print(json.dumps(weather_config, indent=2))
    
    # Generate example curl command
    params = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "current": "temperature_2m,relative_humidity_2m,weather_code",
        "timezone": "America/New_York"
    }
    
    curl_cmd = generate_curl_command(weather_config, params)
    print(f"\nExample curl command:\n{curl_cmd}")
    
    # Validate configuration
    errors = validate_tool_config(weather_config)
    print(f"\nValidation result: {'Valid' if not errors else f'Errors: {errors}'}")
    print()


def example_slack_webhook():
    """Example: Create a Slack notification webhook tool"""
    print("=== Slack Webhook Example ===")
    
    # Create Slack webhook tool
    slack_config = create_notification_tool(
        service="slack",
        webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    )
    
    print("Slack webhook tool configuration:")
    print(json.dumps(slack_config, indent=2))
    
    # Generate example curl command
    params = {
        "text": "Hello from ElevenLabs agent!",
        "channel": "#general",
        "username": "AI Assistant"
    }
    
    curl_cmd = generate_curl_command(slack_config, params)
    print(f"\nExample curl command:\n{curl_cmd}")
    print()


def example_custom_api():
    """Example: Create a custom REST API tool"""
    print("=== Custom REST API Example ===")
    
    # Create a tool for a hypothetical user management API
    api_config = create_rest_api_tool(
        name="get_user_profile",
        description="Get user profile information from the user management API",
        base_url="https://api.example.com",
        endpoint="/users/{user_id}",
        method="GET",
        auth_type="bearer",
        auth_token="your_api_token_here",
        path_params={
            "user_id": {
                "type": "string",
                "description": "User ID to retrieve",
                "required": True
            }
        },
        query_params={
            "include_avatar": {
                "type": "boolean",
                "description": "Include avatar URL in response",
                "default": True
            }
        }
    )
    
    print("Custom API tool configuration:")
    print(json.dumps(api_config, indent=2))
    
    # Generate example curl command
    params = {
        "user_id": "12345",
        "include_avatar": True
    }
    
    curl_cmd = generate_curl_command(api_config, params)
    print(f"\nExample curl command:\n{curl_cmd}")
    print()


def example_crm_integration():
    """Example: Create a CRM integration tool"""
    print("=== CRM Integration Example (HubSpot) ===")
    
    # Create HubSpot CRM tool
    crm_config = create_crm_webhook_tool(
        crm_platform="hubspot",
        webhook_url="https://api.hubapi.com/contacts/v1/contact/",
        auth_token="your_hubspot_token"
    )
    
    print("HubSpot CRM tool configuration:")
    print(json.dumps(crm_config, indent=2))
    
    # Generate example curl command
    params = {
        "email": "john.doe@example.com",
        "properties": {
            "firstname": "John",
            "lastname": "Doe",
            "company": "Example Corp",
            "phone": "+1-555-123-4567"
        }
    }
    
    curl_cmd = generate_curl_command(crm_config, params)
    print(f"\nExample curl command:\n{curl_cmd}")
    print()


def example_database_api():
    """Example: Create a database API tool"""
    print("=== Database API Example ===")
    
    # Create database query tool
    db_config = create_database_api_tool(
        name="query_customer_orders",
        description="Query customer orders from the database",
        api_base_url="https://db-api.example.com",
        table_name="orders",
        api_key="your_db_api_key",
        operation="query"
    )
    
    print("Database API tool configuration:")
    print(json.dumps(db_config, indent=2))
    
    # Generate example curl command
    params = {
        "query": "SELECT * FROM orders WHERE customer_id = '12345' ORDER BY created_at DESC",
        "limit": 10
    }
    
    curl_cmd = generate_curl_command(db_config, params)
    print(f"\nExample curl command:\n{curl_cmd}")
    print()


def example_search_tool():
    """Example: Create a web search tool"""
    print("=== Web Search Tool Example (Google) ===")
    
    # Create Google Custom Search tool
    search_config = create_search_tool(
        search_engine="google",
        api_key="your_google_api_key",
        custom_search_id="your_custom_search_engine_id"
    )
    
    print("Google Search tool configuration:")
    print(json.dumps(search_config, indent=2))
    
    # Generate example curl command
    params = {
        "q": "ElevenLabs conversational AI",
        "num": 5,
        "safe": "active"
    }
    
    curl_cmd = generate_curl_command(search_config, params)
    print(f"\nExample curl command:\n{curl_cmd}")
    print()


def example_custom_webhook():
    """Example: Create a completely custom webhook tool"""
    print("=== Custom Webhook Example ===")
    
    # Create a custom webhook for order processing
    webhook_config = create_webhook_tool(
        name="process_order",
        description="Process a new order through the order management webhook",
        url="https://orders.example.com/webhook/process",
        method="POST",
        auth_header="Bearer your_order_api_token",
        custom_headers={
            "X-Source": "ElevenLabs-Agent",
            "Content-Type": "application/json"
        },
        parameters={
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Customer ID placing the order"
                },
                "items": {
                    "type": "array",
                    "description": "Array of order items",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string"},
                            "quantity": {"type": "number"},
                            "price": {"type": "number"}
                        }
                    }
                },
                "shipping_address": {
                    "type": "object",
                    "description": "Shipping address information",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "zip": {"type": "string"}
                    }
                },
                "priority": {
                    "type": "string",
                    "description": "Order priority level",
                    "enum": ["standard", "expedited", "rush"],
                    "default": "standard"
                }
            },
            "required": ["customer_id", "items", "shipping_address"]
        }
    )
    
    print("Custom webhook tool configuration:")
    print(json.dumps(webhook_config, indent=2))
    
    # Generate example curl command
    params = {
        "data": {
            "customer_id": "CUST-12345",
            "items": [
                {
                    "product_id": "PROD-001",
                    "quantity": 2,
                    "price": 29.99
                },
                {
                    "product_id": "PROD-002", 
                    "quantity": 1,
                    "price": 49.99
                }
            ],
            "shipping_address": {
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "zip": "10001"
            },
            "priority": "expedited"
        }
    }
    
    curl_cmd = generate_curl_command(webhook_config, params)
    print(f"\nExample curl command:\n{curl_cmd}")
    print()


def show_available_templates():
    """Show all available tool templates"""
    print("=== Available Tool Templates ===")
    
    for category, info in TOOL_TEMPLATES.items():
        print(f"\n{category.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Variants: {', '.join(info['variants'].keys())}")
    
    print()


def example_validation():
    """Example: Validate tool configurations"""
    print("=== Tool Validation Examples ===")
    
    # Valid configuration
    valid_config = create_weather_tool("open-meteo")
    errors = validate_tool_config(valid_config)
    print(f"Valid weather tool: {len(errors)} errors")
    
    # Invalid configuration (missing required fields)
    invalid_config = {
        "tool_config": {
            "type": "server",
            "name": "",  # Empty name - invalid
            "server_tool": {
                "url": "not-a-url",  # Invalid URL
                "method": "INVALID"  # Invalid HTTP method
            }
        }
    }
    errors = validate_tool_config(invalid_config)
    print(f"Invalid configuration: {len(errors)} errors")
    for error in errors:
        print(f"  - {error}")
    
    print()


if __name__ == "__main__":
    """Run all examples"""
    print("ElevenLabs MCP Server - curl-based Tool Examples")
    print("=" * 50)
    print()
    
    # Show available templates first
    show_available_templates()
    
    # Run all examples
    example_weather_tool()
    example_slack_webhook()
    example_custom_api()
    example_crm_integration()
    example_database_api()
    example_search_tool()
    example_custom_webhook()
    example_validation()
    
    print("All examples completed!")
    print("\nTo use these tools with ElevenLabs agents:")
    print("1. Create the tool using the MCP server functions")
    print("2. Get the tool ID from the response")
    print("3. Add the tool ID to your agent's configuration")
    print("4. Test the tool using the generated curl commands")