# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
- `./scripts/test.sh` - Run pytest with coverage
- `./scripts/test.sh --verbose` - Run with verbose output
- `./scripts/test.sh --fail-fast` - Stop on first failure
- `./scripts/test.sh --no-coverage` - Skip coverage analysis
- `python -m pytest tests/` - Direct pytest execution

### Development & Debugging
- `./scripts/dev.sh` - Start MCP server in development mode with fastmcp inspector
- `uv run elevenlabs_mcp/server.py` - Run server directly
- `mcp dev elevenlabs_mcp/server.py` - Alternative dev mode (requires mcp install)

### Building & Distribution
- `./scripts/build.sh` - Clean and build package using uv
- `uv build` - Build package directly
- `uv pip install -e ".[dev]"` - Install in development mode

### Environment Setup
- `uv venv` - Create virtual environment
- `source .venv/bin/activate` - Activate virtual environment
- Copy `.env.example` to `.env` and add `ELEVENLABS_API_KEY`

## Architecture Overview

### Core Components

**elevenlabs_mcp/server.py**: Main MCP server implementation using FastMCP framework. Contains 47 tools for complete ElevenLabs API integration including:

**Core Audio Tools:**
- Text-to-speech with voice customization
- Speech-to-text with diarization
- Voice cloning and management
- Audio processing (isolation, effects, speech-to-speech)
- Sound effects generation

**Agent Management:**
- Create, update, delete agents
- Get agent details and signed URLs
- List agents with pagination

**Conversation Management:**
- Get conversations with transcripts
- List conversations with filtering
- Send conversation feedback
- Outbound calling via Twilio/SIP

**Tools API:**
- Create server/client tools for agents
- Get, update, delete tools
- List tools with filtering

**Knowledge Base Management:**
- Add documents from files, URLs, or text
- List, get, delete knowledge base documents
- Compute RAG index for improved retrieval
- Get document content

**Phone Numbers:**
- Create, get, update, delete phone numbers
- Assign phone numbers to agents
- Configure webhooks and capabilities

**Widget Integration:**
- Get widget configuration for website embedding
- Upload custom avatars

**Workspace Administration:**
- Get/update workspace settings
- Manage secrets (API keys, tokens)
- Configure data retention and compliance

**elevenlabs_mcp/utils.py**: Utility functions for file handling, path management, error handling, and large text processing.

**elevenlabs_mcp/model.py**: Pydantic models for structured data including:
- McpVoice, McpModel, McpLanguage (core voice/model data)
- McpTool (tool definitions and metadata)
- McpKnowledgeBaseDocument (knowledge base document info)
- McpPhoneNumber (phone number configurations)
- McpSecret (workspace secret metadata)

**elevenlabs_mcp/convai.py**: Configuration builders for conversational AI agents and platform settings.

### Key Patterns

**Tool Registration**: All tools use `@mcp.tool()` decorator with comprehensive descriptions including cost warnings for API calls.

**File Handling**: Uses `handle_input_file()` and `make_output_path()`/`make_output_file()` utilities for consistent file operations across tools.

**Error Handling**: Custom `make_error()` function for consistent error responses across the MCP interface.

**Environment Configuration**: 
- `ELEVENLABS_API_KEY` (required)
- `ELEVENLABS_MCP_BASE_PATH` (optional, for file path resolution)

**API Client**: Custom httpx client with User-Agent header for ElevenLabs API calls.

### API Coverage
The implementation provides complete coverage of ElevenLabs Conversational AI API including:
- Agents API (create, read, update, delete, signed URLs)
- Conversations API (list, get, feedback)
- Tools API (server/client tools management)
- Knowledge Base API (documents, RAG indexing)
- Phone Numbers API (Twilio/SIP integration)
- Widget API (website embedding)
- Workspace API (settings, secrets management)

### Cost Awareness
All tools that make API calls include "⚠️ COST WARNING" in descriptions. Tools without warnings only read existing data and are free to use.

### File Structure
- Main package: `elevenlabs_mcp/` (~1600 lines with 47 tools)
- Scripts: `scripts/` (test, dev, build, setup)
- Tests: `tests/` with pytest configuration
- Docker support via `Dockerfile`
- Distribution via PyPI as `elevenlabs-mcp`

## Project Goals
- The goal of this project is to extend mcp server with full set of tools to use elevenlabs agents api