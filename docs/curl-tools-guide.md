# curl-based Tools Guide for ElevenLabs Conversational AI

This guide explains how to create and use server tools that make HTTP requests (equivalent to curl commands) within the ElevenLabs Conversational AI framework.

## Overview

Server tools enable your conversational AI agents to connect to external APIs, databases, and webhooks. The curl-based tool utilities in this MCP server make it easy to create, test, and deploy these integrations.

## Key Features

- **One-command tool creation** for common integrations (weather, CRM, notifications)
- **curl command generation** for testing and debugging
- **Validation utilities** to check tool configurations
- **Predefined templates** for popular services
- **Comprehensive parameter handling** for complex APIs

## Getting Started

### 1. Basic Tool Creation

Use the MCP server functions to create tools that will be available to your conversational AI agents:

```python
# Create a weather tool
create_weather_integration_tool(api_provider="open-meteo")

# Create a custom webhook
create_custom_webhook_tool(
    name="send_notification",
    description="Send notifications to our system",
    webhook_url="https://api.yourservice.com/notify",
    method="POST"
)
```

### 2. Advanced Configuration

For more complex integrations, use the detailed configuration options:

```python
# Create a REST API tool with authentication
create_api_integration_tool(
    name="get_user_data",
    description="Retrieve user information from our API",
    base_url="https://api.yourservice.com",
    endpoint="/users/{user_id}",
    method="GET",
    auth_type="bearer",
    auth_token="your_api_token",
    path_params='{"user_id": {"type": "string", "description": "User ID", "required": true}}',
    query_params='{"include_details": {"type": "boolean", "description": "Include detailed info"}}'
)
```

## Available Tool Types

### Weather Tools

Create weather integration tools using popular APIs:

```python
# Free service (no API key required)
create_weather_integration_tool(api_provider="open-meteo")

# OpenWeatherMap (requires API key)
create_weather_integration_tool(
    api_provider="openweathermap", 
    api_key="your_openweathermap_key"
)
```

**Generated curl equivalent:**
```bash
curl -X GET "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code&timezone=auto&forecast_days=7"
```

### CRM Integration Tools

Connect to popular CRM platforms:

```python
# Salesforce integration
create_crm_integration_tool(
    crm_platform="salesforce",
    webhook_url="https://your-salesforce-webhook.com/api",
    auth_token="your_salesforce_token"
)

# HubSpot integration
create_crm_integration_tool(
    crm_platform="hubspot",
    webhook_url="https://api.hubapi.com/contacts/v1/contact/",
    auth_token="your_hubspot_token"
)
```

### Notification Tools

Send messages to team communication platforms:

```python
# Slack notifications
create_notification_tool_integration(
    service="slack",
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
)

# Discord notifications  
create_notification_tool_integration(
    service="discord",
    webhook_url="https://discord.com/api/webhooks/YOUR/WEBHOOK/URL"
)
```

### Custom Webhook Tools

Create completely custom webhook integrations:

```python
create_custom_webhook_tool(
    name="process_order",
    description="Process customer orders",
    webhook_url="https://orders.yourcompany.com/webhook",
    method="POST",
    auth_header="Bearer your_api_token",
    custom_headers='{"X-Source": "ElevenLabs-Agent"}',
    parameters_schema='''{
        "properties": {
            "customer_id": {"type": "string", "description": "Customer ID"},
            "items": {
                "type": "array",
                "description": "Order items",
                "items": {"type": "object"}
            },
            "total": {"type": "number", "description": "Order total"}
        },
        "required": ["customer_id", "items", "total"]
    }'''
)
```

## Tool Configuration Structure

All server tools follow this configuration pattern:

```json
{
  "tool_config": {
    "type": "server",
    "name": "tool_name",
    "description": "What the tool does and when to use it",
    "server_tool": {
      "url": "https://api.example.com/endpoint/{param}",
      "method": "POST",
      "headers": {
        "Authorization": "Bearer {api_token}",
        "Content-Type": "application/json"
      },
      "parameters": {
        "type": "object",
        "properties": {
          "param": {
            "type": "string",
            "description": "Parameter description",
            "required": true
          }
        },
        "required": ["param"]
      },
      "response_timeout_secs": 30
    }
  }
}
```

## Parameter Types and Handling

### Path Parameters
Used in URL templating with `{parameter_name}` syntax:
```json
{
  "url": "https://api.example.com/users/{user_id}/posts/{post_id}",
  "path_parameters": {
    "user_id": {"type": "string", "description": "User ID"},
    "post_id": {"type": "string", "description": "Post ID"}
  }
}
```

### Query Parameters
Added to URL as query string:
```json
{
  "query_parameters": {
    "limit": {"type": "number", "description": "Max results", "default": 10},
    "filter": {"type": "string", "description": "Filter criteria"}
  }
}
```

### Body Parameters
Sent in request body for POST/PUT/PATCH requests:
```json
{
  "body_parameters": {
    "properties": {
      "data": {
        "type": "object",
        "description": "Request payload",
        "additionalProperties": true
      }
    },
    "required": ["data"]
  }
}
```

## Authentication Patterns

### Bearer Token Authentication
```python
auth_header="Bearer your_api_token"
```

### API Key Authentication
```python
auth_type="api_key"
auth_token="your_api_key"
# Results in header: X-API-Key: your_api_key
```

### Basic Authentication
```python
auth_type="basic"
auth_token="base64_encoded_credentials"
```

### Custom Headers
```python
custom_headers={"Authorization": "Custom your_token", "X-API-Version": "v2"}
```

## Testing and Debugging

### Generate curl Commands

Test your tools by generating equivalent curl commands:

```python
# Get a tool configuration and generate test command
generate_curl_test_command(
    tool_id="your_tool_id",
    parameter_values='{"user_id": "12345", "include_details": true}'
)
```

This generates output like:
```bash
curl -X GET "https://api.example.com/users/12345?include_details=true" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json"
```

### Validate Tool Configurations

Check for common issues in your tool setup:

```python
validate_tool_configuration(tool_id="your_tool_id")
```

Returns validation results:
- ✅ Valid configurations
- ❌ Error details for invalid configurations

## Best Practices

### 1. Tool Naming
- Use descriptive, action-oriented names: `get_weather`, `create_contact`, `send_notification`
- Use snake_case for consistency
- Keep names under 50 characters

### 2. Descriptions
Write clear descriptions that help the AI understand:
- **What** the tool does
- **When** to use it
- **What parameters** are required

Good example:
```
"Get current weather and 7-day forecast for any location worldwide. 
Use when users ask about weather conditions, temperature, or forecasts. 
Requires latitude and longitude coordinates."
```

### 3. Parameter Definitions
- Provide clear descriptions for all parameters
- Use appropriate data types (`string`, `number`, `boolean`, `array`, `object`)
- Set sensible defaults where possible
- Mark required parameters clearly

### 4. Error Handling
- Set appropriate timeout values (default: 30 seconds)
- Handle authentication failures gracefully
- Validate inputs before making requests

### 5. Security
- Never hardcode sensitive tokens in tool configurations
- Use environment variables or secure credential storage
- Validate webhook signatures when possible
- Implement proper CORS and rate limiting on your endpoints

## Agent Integration

Once tools are created, add them to your agent configuration:

### Legacy Format (being deprecated July 2025)
```json
{
  "conversation_config": {
    "agent": {
      "prompt": {
        "tools": [
          {"type": "server", "name": "get_weather"},
          {"type": "server", "name": "send_notification"}
        ]
      }
    }
  }
}
```

### New Format (required after July 15, 2025)
```json
{
  "conversation_config": {
    "agent": {
      "prompt": {
        "tool_ids": ["tool_123456789abcdef0", "tool_987654321fedcba0"],
        "built_in_tools": {
          "end_call": {"name": "end_call", "type": "system"}
        }
      }
    }
  }
}
```

## Common Integration Examples

### E-commerce Order Processing
```python
create_custom_webhook_tool(
    name="process_order",
    description="Process customer orders with payment and inventory checks",
    webhook_url="https://orders.yourstore.com/api/process",
    method="POST",
    parameters_schema='''{
        "properties": {
            "customer_email": {"type": "string", "format": "email"},
            "items": {"type": "array", "items": {"type": "object"}},
            "payment_method": {"type": "string", "enum": ["card", "paypal"]},
            "shipping_address": {"type": "object"}
        },
        "required": ["customer_email", "items", "payment_method"]
    }'''
)
```

### Customer Support Ticketing
```python
create_api_integration_tool(
    name="create_support_ticket",
    description="Create support tickets for customer issues",
    base_url="https://support.yourcompany.com/api/v1",
    endpoint="/tickets",
    method="POST",
    auth_type="bearer",
    body_params='''{
        "properties": {
            "subject": {"type": "string", "description": "Ticket subject"},
            "description": {"type": "string", "description": "Issue description"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "customer_email": {"type": "string", "format": "email"}
        },
        "required": ["subject", "description", "customer_email"]
    }'''
)
```

### Database Queries
```python
# Using the database API helper
from elevenlabs_mcp.curl_tools import create_database_api_tool

create_database_api_tool(
    name="query_customer_orders",
    description="Look up customer order history",
    api_base_url="https://db-api.yourcompany.com",
    table_name="orders",
    api_key="your_db_api_key",
    operation="query"
)
```

## Troubleshooting

### Common Issues

1. **405 Method Not Allowed**
   - Check HTTP method is correct for the endpoint
   - Verify the API supports the specified method

2. **401 Unauthorized**
   - Verify authentication token is correct
   - Check token hasn't expired
   - Ensure proper authentication header format

3. **422 Validation Error**
   - Check parameter types match schema
   - Verify required parameters are provided
   - Validate JSON formatting for complex parameters

4. **Timeout Errors**
   - Increase `response_timeout_secs` value
   - Check if external API is responsive
   - Consider implementing retry logic

### Debugging Steps

1. **Generate and test curl command manually**
   ```python
   generate_curl_test_command(tool_id="your_tool_id", parameter_values="{}")
   ```

2. **Validate tool configuration**
   ```python
   validate_tool_configuration(tool_id="your_tool_id")
   ```

3. **Check ElevenLabs logs** for tool execution details

4. **Test external API independently** using Postman or curl

## Migration from Legacy Tools

If you have existing tools created before the 2025 migration:

1. **Create new tools** using the curl utilities
2. **Get the new tool IDs** from the creation response
3. **Update agent configurations** to use `tool_ids` instead of `tools` array
4. **Test thoroughly** before the July 2025 deadline
5. **Clean up unused tools** using the delete functions

## Support and Resources

- **Examples**: See `examples/curl_tools_examples.py` for complete working examples
- **API Reference**: ElevenLabs Conversational AI API documentation
- **Tool Templates**: Use `list_tool_templates()` to see available presets
- **Validation**: Always run `validate_tool_configuration()` before deployment

## Conclusion

The curl-based tool utilities provide a powerful and flexible way to integrate your conversational AI agents with external services. By following the patterns and best practices in this guide, you can create robust, reliable integrations that enhance your agents' capabilities.

Remember to:
- Start with simple integrations and build complexity gradually
- Test thoroughly using the curl generation features
- Follow security best practices for authentication
- Keep tool descriptions clear and actionable
- Monitor tool performance and usage patterns

Happy building! 🚀