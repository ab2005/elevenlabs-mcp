![export](https://github.com/user-attachments/assets/ee379feb-348d-48e7-899c-134f7f7cd74f)

<div class="title-block" style="text-align: center;" align="center">

  [![Discord Community](https://img.shields.io/badge/discord-@elevenlabs-000000.svg?style=for-the-badge&logo=discord&labelColor=000)](https://discord.gg/elevenlabs)
  [![Twitter](https://img.shields.io/badge/Twitter-@elevenlabsio-000000.svg?style=for-the-badge&logo=twitter&labelColor=000)](https://x.com/ElevenLabsDevs)
  [![PyPI](https://img.shields.io/badge/PyPI-elevenlabs--agents--mcp-000000.svg?style=for-the-badge&logo=pypi&labelColor=000)](https://pypi.org/project/elevenlabs-agents-mcp)
  [![Tests](https://img.shields.io/badge/tests-passing-000000.svg?style=for-the-badge&logo=github&labelColor=000)](https://github.com/elevenlabs/elevenlabs-mcp-server/actions/workflows/test.yml)

</div>


<p align="center">
  Official ElevenLabs <a href="https://github.com/modelcontextprotocol">Model Context Protocol (MCP)</a> server focused on Conversational AI. This specialized server enables MCP clients like <a href="https://www.anthropic.com/claude">Claude Desktop</a>, <a href="https://www.cursor.so">Cursor</a>, <a href="https://codeium.com/windsurf">Windsurf</a>, <a href="https://github.com/openai/openai-agents-python">OpenAI Agents</a> and others to create and manage AI agents, knowledge bases, phone integrations, and more.
</p>

## Quickstart with Claude Desktop

1. Get your API key from [ElevenLabs](https://elevenlabs.io/app/settings/api-keys). There is a free tier with 10k credits per month.
2. Install `uv` (Python package manager), install with `curl -LsSf https://astral.sh/uv/install.sh | sh` or see the `uv` [repo](https://github.com/astral-sh/uv) for additional install methods.
3. Go to Claude > Settings > Developer > Edit Config > claude_desktop_config.json to include the following:

```
{
  "mcpServers": {
    "ElevenLabs": {
      "command": "uvx",
      "args": ["elevenlabs-agents-mcp"],
      "env": {
        "ELEVENLABS_API_KEY": "<insert-your-api-key-here>"
      }
    }
  }
}

```

If you're using Windows, you will have to enable "Developer Mode" in Claude Desktop to use the MCP server. Click "Help" in the hamburger menu at the top left and select "Enable Developer Mode".

## Other MCP clients

For other clients like Cursor and Windsurf, run:
1. `pip install elevenlabs-agents-mcp`
2. `python -m elevenlabs_mcp --api-key={{PUT_YOUR_API_KEY_HERE}} --print` to get the configuration. Paste it into appropriate configuration directory specified by your MCP client.

That's it. Your MCP client can now interact with ElevenLabs Conversational AI through these tools:

## Available Tools

This MCP server provides 29 specialized tools for ElevenLabs Conversational AI:

### Agent Management
- `create_agent` - Create conversational AI agents with custom configurations
- `update_agent` - Update agent settings and configurations
- `delete_agent` - Delete an agent
- `get_agent` - Get agent details
- `list_agents` - List all available conversational AI agents
- `get_signed_url` - Get a signed URL for agent interactions

### Knowledge Base Management
- `add_knowledge_base_to_agent` - Add documents from files, URLs, or text
- `list_knowledge_base_documents` - List documents in an agent's knowledge base
- `get_knowledge_base_document` - Get document details
- `delete_knowledge_base_document` - Remove documents from knowledge base
- `get_knowledge_base_document_content` - Retrieve document content
- `compute_knowledge_base_rag_index` - Optimize knowledge base for retrieval

### Tools API
- `create_server_tool` - Create server-side tools for agents
- `create_client_tool` - Create client-side tools for agents
- `get_tool` - Get tool details
- `update_tool` - Update tool configurations
- `delete_tool` - Remove tools
- `list_tools` - List available tools

### Phone Integration
- `create_phone_number` - Set up phone numbers for agents
- `get_phone_number` - Get phone number details
- `update_phone_number` - Update phone configurations
- `delete_phone_number` - Remove phone numbers
- `list_phone_numbers` - List all phone numbers

### Widget & Workspace
- `get_widget_config` - Get widget configuration for website embedding
- `upload_avatar` - Upload custom avatar for agents
- `get_workspace` - Get workspace settings
- `update_workspace` - Update workspace configuration
- `list_secrets` - List workspace secrets
- `create_secret` - Create new secrets (API keys, tokens)

## Example usage

⚠️ Warning: ElevenLabs API usage may incur costs. Only tools marked with cost warnings make API calls.

Try asking Claude:

- "Create an AI agent that can help customers with technical support inquiries"
- "Set up a knowledge base for my agent with product documentation and FAQs"
- "Configure a phone number for my agent to handle inbound calls"
- "Create a custom tool that lets my agent check order status in our database"
- "Get the widget code to embed my agent on our company website"

## Optional features

You can add the `ELEVENLABS_MCP_BASE_PATH` environment variable to the `claude_desktop_config.json` to specify the base path MCP server should look for and output files specified with relative paths.

## Contributing

If you want to contribute or run from source:

1. Clone the repository:

```bash
git clone https://github.com/elevenlabs/elevenlabs-agents-mcp
cd elevenlabs-agents-mcp
```

2. Create a virtual environment and install dependencies [using uv](https://github.com/astral-sh/uv):

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

3. Copy `.env.example` to `.env` and add your ElevenLabs API key:

```bash
cp .env.example .env
# Edit .env and add your API key
```

4. Run the tests to make sure everything is working:

```bash
./scripts/test.sh
# Or with options
./scripts/test.sh --verbose --fail-fast
```

5. Install the server in Claude Desktop: `mcp install elevenlabs_mcp/server.py`

6. Debug and test locally with MCP Inspector: `mcp dev elevenlabs_mcp/server.py`

## Troubleshooting

Logs when running with Claude Desktop can be found at:

- **Windows**: `%APPDATA%\Claude\logs\mcp-server-elevenlabs.log`
- **macOS**: `~/Library/Logs/Claude/mcp-server-elevenlabs.log`

### Timeouts when using certain tools

Certain ElevenLabs API operations, like voice design and audio isolation, can take a long time to resolve. When using the MCP inspector in dev mode, you might get timeout errors despite the tool completing its intended task.

This shouldn't occur when using a client like Claude.

### MCP ElevenLabs: spawn uvx ENOENT

If you encounter the error "MCP ElevenLabs: spawn uvx ENOENT", confirm its absolute path by running this command in your terminal:

```bash
which uvx
```

Once you obtain the absolute path (e.g., `/usr/local/bin/uvx`), update your configuration to use that path (e.g., `"command": "/usr/local/bin/uvx"`). This ensures that the correct executable is referenced.



