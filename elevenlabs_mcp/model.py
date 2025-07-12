from pydantic import BaseModel
from typing import Dict, Optional


class McpVoice(BaseModel):
    id: str
    name: str
    category: str
    fine_tuning_status: Optional[Dict] = None


class ConvAiAgentListItem(BaseModel):
    name: str
    agent_id: str


class ConvaiAgent(BaseModel):
    name: str
    agent_id: str
    system_prompt: str
    voice_id: str | None
    language: str
    llm: str


class McpLanguage(BaseModel):
    language_id: str
    name: str


class McpModel(BaseModel):
    id: str
    name: str
    languages: list[McpLanguage]


class McpTool(BaseModel):
    tool_id: str
    type: str  # "server" or "client"
    name: str
    description: str
    created_at: Optional[str] = None


class McpKnowledgeBaseDocument(BaseModel):
    document_id: str
    name: str
    type: str  # "file", "url", "text"
    status: str  # "processing", "ready", "failed"
    created_at: str
    metadata: Optional[Dict] = None


class McpPhoneNumber(BaseModel):
    phone_number_id: str
    phone_number: str
    agent_id: Optional[str] = None
    status: str  # "active", "inactive", "released"
    provider: Optional[str] = None
    label: Optional[str] = None


class McpSecret(BaseModel):
    secret_id: str
    name: str
    created_at: str
    last_used: Optional[str] = None
