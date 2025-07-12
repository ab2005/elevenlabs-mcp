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
import base64
from datetime import datetime
from io import BytesIO
from typing import Literal
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from elevenlabs.client import ElevenLabs
from elevenlabs_mcp.model import McpVoice, McpModel, McpLanguage, McpTool, McpKnowledgeBaseDocument, McpPhoneNumber, McpSecret
from elevenlabs_mcp.utils import (
    make_error,
    make_output_path,
    make_output_file,
    handle_input_file,
    parse_conversation_transcript,
    handle_large_text,
)
from elevenlabs_mcp.convai import create_conversation_config, create_platform_settings
from elevenlabs.types.knowledge_base_locator import KnowledgeBaseLocator

from elevenlabs import play
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
    description="""Convert text to speech with a given voice and save the output audio file to a given directory.
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop.
    Only one of voice_id or voice_name can be provided. If none are provided, the default voice will be used.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

     Args:
        text (str): The text to convert to speech.
        voice_name (str, optional): The name of the voice to use.
        stability (float, optional): Stability of the generated audio. Determines how stable the voice is and the randomness between each generation. Lower values introduce broader emotional range for the voice. Higher values can result in a monotonous voice with limited emotion. Range is 0 to 1.
        similarity_boost (float, optional): Similarity boost of the generated audio. Determines how closely the AI should adhere to the original voice when attempting to replicate it. Range is 0 to 1.
        style (float, optional): Style of the generated audio. Determines the style exaggeration of the voice. This setting attempts to amplify the style of the original speaker. It does consume additional computational resources and might increase latency if set to anything other than 0. Range is 0 to 1.
        use_speaker_boost (bool, optional): Use speaker boost of the generated audio. This setting boosts the similarity to the original speaker. Using this setting requires a slightly higher computational load, which in turn increases latency.
        speed (float, optional): Speed of the generated audio. Controls the speed of the generated speech. Values range from 0.7 to 1.2, with 1.0 being the default speed. Lower values create slower, more deliberate speech while higher values produce faster-paced speech. Extreme values can impact the quality of the generated speech. Range is 0.7 to 1.2.
        output_directory (str, optional): Directory where files should be saved.
            Defaults to $HOME/Desktop if not provided.
        language: ISO 639-1 language code for the voice.
        output_format (str, optional): Output format of the generated audio. Formatted as codec_sample_rate_bitrate. So an mp3 with 22.05kHz sample rate at 32kbs is represented as mp3_22050_32. MP3 with 192kbps bitrate requires you to be subscribed to Creator tier or above. PCM with 44.1kHz sample rate requires you to be subscribed to Pro tier or above. Note that the μ-law format (sometimes written mu-law, often approximated as u-law) is commonly used for Twilio audio inputs.
            Defaults to "mp3_44100_128". Must be one of:
            mp3_22050_32
            mp3_44100_32
            mp3_44100_64
            mp3_44100_96
            mp3_44100_128
            mp3_44100_192
            pcm_8000
            pcm_16000
            pcm_22050
            pcm_24000
            pcm_44100
            ulaw_8000
            alaw_8000
            opus_48000_32
            opus_48000_64
            opus_48000_96
            opus_48000_128
            opus_48000_192

    Returns:
        Text content with the path to the output file and name of the voice used.
    """
)
def text_to_speech(
    text: str,
    voice_name: str | None = None,
    output_directory: str | None = None,
    voice_id: str | None = None,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0,
    use_speaker_boost: bool = True,
    speed: float = 1.0,
    language: str = "en",
    output_format: str = "mp3_44100_128",
):
    if text == "":
        make_error("Text is required.")

    if voice_id is not None and voice_name is not None:
        make_error("voice_id and voice_name cannot both be provided.")

    voice = None
    if voice_id is not None:
        voice = client.voices.get(voice_id=voice_id)
    elif voice_name is not None:
        voices = client.voices.search(search=voice_name)
        if len(voices.voices) == 0:
            make_error("No voices found with that name.")
        voice = next((v for v in voices.voices if v.name == voice_name), None)
        if voice is None:
            make_error(f"Voice with name: {voice_name} does not exist.")

    voice_id = voice.voice_id if voice else DEFAULT_VOICE_ID

    output_path = make_output_path(output_directory, base_path)
    output_file_name = make_output_file("tts", text, output_path, "mp3")

    model_id = (
        "eleven_flash_v2_5"
        if language in ["hu", "no", "vi"]
        else "eleven_multilingual_v2"
    )

    audio_data = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
        voice_settings={
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost,
            "speed": speed,
        },
    )
    audio_bytes = b"".join(audio_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path / output_file_name, "wb") as f:
        f.write(audio_bytes)

    return TextContent(
        type="text",
        text=f"Success. File saved as: {output_path / output_file_name}. Voice used: {voice.name if voice else DEFAULT_VOICE_ID}",
    )


@mcp.tool(
    description="""Transcribe speech from an audio file and either save the output text file to a given directory or return the text to the client directly.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        file_path: Path to the audio file to transcribe
        language_code: ISO 639-3 language code for transcription (default: "eng" for English)
        diarize: Whether to diarize the audio file. If True, which speaker is currently speaking will be annotated in the transcription.
        save_transcript_to_file: Whether to save the transcript to a file.
        return_transcript_to_client_directly: Whether to return the transcript to the client directly.
        output_directory: Directory where files should be saved.
            Defaults to $HOME/Desktop if not provided.

    Returns:
        TextContent containing the transcription. If save_transcript_to_file is True, the transcription will be saved to a file in the output directory.
    """
)
def speech_to_text(
    input_file_path: str,
    language_code: str = "eng",
    diarize: bool = False,
    save_transcript_to_file: bool = True,
    return_transcript_to_client_directly: bool = False,
    output_directory: str | None = None,
) -> TextContent:
    if not save_transcript_to_file and not return_transcript_to_client_directly:
        make_error("Must save transcript to file or return it to the client directly.")
    file_path = handle_input_file(input_file_path)
    if save_transcript_to_file:
        output_path = make_output_path(output_directory, base_path)
        output_file_name = make_output_file("stt", file_path.name, output_path, "txt")
    with file_path.open("rb") as f:
        audio_bytes = f.read()
    transcription = client.speech_to_text.convert(
        model_id="scribe_v1",
        file=audio_bytes,
        language_code=language_code,
        enable_logging=True,
        diarize=diarize,
        tag_audio_events=True,
    )

    if save_transcript_to_file:
        with open(output_path / output_file_name, "w") as f:
            f.write(transcription.text)

    if return_transcript_to_client_directly:
        return TextContent(type="text", text=transcription.text)
    else:
        return TextContent(
            type="text", text=f"Transcription saved to {output_path / output_file_name}"
        )


@mcp.tool(
    description="""Convert text description of a sound effect to sound effect with a given duration and save the output audio file to a given directory.
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop.
    Duration must be between 0.5 and 5 seconds.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        text: Text description of the sound effect
        duration_seconds: Duration of the sound effect in seconds
        output_directory: Directory where files should be saved.
            Defaults to $HOME/Desktop if not provided.
        output_format (str, optional): Output format of the generated audio. Formatted as codec_sample_rate_bitrate. So an mp3 with 22.05kHz sample rate at 32kbs is represented as mp3_22050_32. MP3 with 192kbps bitrate requires you to be subscribed to Creator tier or above. PCM with 44.1kHz sample rate requires you to be subscribed to Pro tier or above. Note that the μ-law format (sometimes written mu-law, often approximated as u-law) is commonly used for Twilio audio inputs.
            Defaults to "mp3_44100_128". Must be one of:
            mp3_22050_32
            mp3_44100_32
            mp3_44100_64
            mp3_44100_96
            mp3_44100_128
            mp3_44100_192
            pcm_8000
            pcm_16000
            pcm_22050
            pcm_24000
            pcm_44100
            ulaw_8000
            alaw_8000
            opus_48000_32
            opus_48000_64
            opus_48000_96
            opus_48000_128
            opus_48000_192
    """
)
def text_to_sound_effects(
    text: str,
    duration_seconds: float = 2.0,
    output_directory: str | None = None,
    output_format: str = "mp3_44100_128",
) -> list[TextContent]:
    if duration_seconds < 0.5 or duration_seconds > 5:
        make_error("Duration must be between 0.5 and 5 seconds")
    output_path = make_output_path(output_directory, base_path)
    output_file_name = make_output_file("sfx", text, output_path, "mp3")

    audio_data = client.text_to_sound_effects.convert(
        text=text,
        output_format=output_format,
        duration_seconds=duration_seconds,
    )
    audio_bytes = b"".join(audio_data)

    with open(output_path / output_file_name, "wb") as f:
        f.write(audio_bytes)

    return TextContent(
        type="text",
        text=f"Success. File saved as: {output_path / output_file_name}",
    )


@mcp.tool(
    description="""
    Search for existing voices, a voice that has already been added to the user's ElevenLabs voice library.
    Searches in name, description, labels and category.

    Args:
        search: Search term to filter voices by. Searches in name, description, labels and category.
        sort: Which field to sort by. `created_at_unix` might not be available for older voices.
        sort_direction: Sort order, either ascending or descending.

    Returns:
        List of voices that match the search criteria.
    """
)
def search_voices(
    search: str | None = None,
    sort: Literal["created_at_unix", "name"] = "name",
    sort_direction: Literal["asc", "desc"] = "desc",
) -> list[McpVoice]:
    response = client.voices.search(
        search=search, sort=sort, sort_direction=sort_direction
    )
    return [
        McpVoice(id=voice.voice_id, name=voice.name, category=voice.category)
        for voice in response.voices
    ]


@mcp.tool(description="List all available models")
def list_models() -> list[McpModel]:
    response = client.models.list()
    return [
        McpModel(
            id=model.model_id,
            name=model.name,
            languages=[
                McpLanguage(language_id=lang.language_id, name=lang.name)
                for lang in model.languages
            ],
        )
        for model in response
    ]


@mcp.tool(description="Get details of a specific voice")
def get_voice(voice_id: str) -> McpVoice:
    """Get details of a specific voice."""
    response = client.voices.get(voice_id=voice_id)
    return McpVoice(
        id=response.voice_id,
        name=response.name,
        category=response.category,
        fine_tuning_status=response.fine_tuning.state,
    )


@mcp.tool(
    description="""Create an instant voice clone of a voice using provided audio files.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """
)
def voice_clone(
    name: str, files: list[str], description: str | None = None
) -> TextContent:
    input_files = [str(handle_input_file(file).absolute()) for file in files]
    voice = client.voices.ivc.create(
        name=name, description=description, files=input_files
    )

    return TextContent(
        type="text",
        text=f"""Voice cloned successfully: Name: {voice.name}
        ID: {voice.voice_id}
        Category: {voice.category}
        Description: {voice.description or "N/A"}""",
    )


@mcp.tool(
    description="""Isolate audio from a file and save the output audio file to a given directory.
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """
)
def isolate_audio(
    input_file_path: str, output_directory: str | None = None
) -> list[TextContent]:
    file_path = handle_input_file(input_file_path)
    output_path = make_output_path(output_directory, base_path)
    output_file_name = make_output_file("iso", file_path.name, output_path, "mp3")
    with file_path.open("rb") as f:
        audio_bytes = f.read()
    audio_data = client.audio_isolation.convert(
        audio=audio_bytes,
    )
    audio_bytes = b"".join(audio_data)

    with open(output_path / output_file_name, "wb") as f:
        f.write(audio_bytes)

    return TextContent(
        type="text",
        text=f"Success. File saved as: {output_path / output_file_name}",
    )


@mcp.tool(
    description="Check the current subscription status. Could be used to measure the usage of the API."
)
def check_subscription() -> TextContent:
    subscription = client.user.subscription.get()
    return TextContent(type="text", text=f"{subscription.model_dump_json(indent=2)}")


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
    response = client.conversational_ai.knowledge_base.documents.list(agent_id=agent_id)

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
        agent_id: The ID of the agent
        document_id: The ID of the document to retrieve
    """
)
def get_knowledge_base_document(agent_id: str, document_id: str) -> TextContent:
    response = client.conversational_ai.knowledge_base.documents.get(
        agent_id=agent_id, document_id=document_id
    )

    metadata_info = f"Metadata: {response.metadata}" if response.metadata else "No metadata"
    
    return TextContent(
        type="text",
        text=f"Document Details: Name: {response.name}, ID: {response.document_id}, Type: {response.type}, Status: {response.status}, Created: {response.created_at}, {metadata_info}",
    )


@mcp.tool(
    description="""Remove a document from an agent's knowledge base.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        agent_id: The ID of the agent
        document_id: The ID of the document to delete
    """
)
def delete_knowledge_base_document(agent_id: str, document_id: str) -> TextContent:
    client.conversational_ai.knowledge_base.documents.delete(
        agent_id=agent_id, document_id=document_id
    )

    return TextContent(
        type="text",
        text=f"Knowledge base document deleted successfully: Document ID: {document_id}",
    )


@mcp.tool(
    description="""Trigger recomputation of the RAG (Retrieval-Augmented Generation) index for an agent's knowledge base.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        agent_id: The ID of the agent whose knowledge base index to recompute
    """
)
def compute_rag_index(agent_id: str) -> TextContent:
    response = client.conversational_ai.knowledge_base.compute_index(agent_id=agent_id)

    estimated_time = getattr(response, 'estimated_time_seconds', 'Unknown')
    
    return TextContent(
        type="text",
        text=f"RAG index computation started for agent {agent_id}. Estimated time: {estimated_time} seconds",
    )


@mcp.tool(
    description="""Get the full content of a knowledge base document.

    Args:
        agent_id: The ID of the agent
        document_id: The ID of the document to retrieve content for
    """
)
def get_document_content(agent_id: str, document_id: str) -> TextContent:
    response = client.conversational_ai.knowledge_base.documents.get_content(
        agent_id=agent_id, document_id=document_id
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
    description="""Generate a signed URL for secure client connections to an agent.

    Args:
        agent_id: The ID of the agent to generate signed URL for
    """
)
def get_signed_url(agent_id: str) -> TextContent:
    response = client.conversational_ai.agents.get_signed_url(agent_id=agent_id)

    return TextContent(
        type="text",
        text=f"Signed URL: {response.signed_url} (expires at: {response.expires_at})",
    )


@mcp.tool(
    description="""Gets conversation with transcript. Returns: conversation details and full transcript. Use when: analyzing completed agent conversations.
    
    Args:
        conversation_id: The unique identifier of the conversation to retrieve, you can get the ids from the list_conversations tool.
    """
)
def get_conversation(
    conversation_id: str,
) -> TextContent:
    """Get conversation details with transcript"""
    try:
        response = client.conversational_ai.conversations.get(conversation_id)

        # Parse transcript using utility function
        transcript, _ = parse_conversation_transcript(response.transcript)

        response_text = f"""Conversation Details:
ID: {response.conversation_id}
Status: {response.status}
Agent ID: {response.agent_id}
Message Count: {len(response.transcript)}

Transcript:
{transcript}"""

        if response.metadata:
            metadata = response.metadata
            duration = getattr(
                metadata,
                "call_duration_secs",
                getattr(metadata, "duration_seconds", "N/A"),
            )
            started_at = getattr(
                metadata, "start_time_unix_secs", getattr(metadata, "started_at", "N/A")
            )
            response_text += (
                f"\n\nMetadata:\nDuration: {duration} seconds\nStarted: {started_at}"
            )

        if response.analysis:
            analysis_summary = getattr(
                response.analysis, "summary", "Analysis available but no summary"
            )
            response_text += f"\n\nAnalysis:\n{analysis_summary}"

        return TextContent(type="text", text=response_text)

    except Exception as e:
        make_error(f"Failed to fetch conversation: {str(e)}")


@mcp.tool(
    description="""Lists agent conversations. Returns: conversation list with metadata. Use when: asked about conversation history.
    
    Args:
        agent_id (str, optional): Filter conversations by specific agent ID
        cursor (str, optional): Pagination cursor for retrieving next page of results
        call_start_before_unix (int, optional): Filter conversations that started before this Unix timestamp
        call_start_after_unix (int, optional): Filter conversations that started after this Unix timestamp
        page_size (int, optional): Number of conversations to return per page (1-100, defaults to 30)
    """
)
def list_conversations(
    agent_id: str | None = None,
    cursor: str | None = None,
    call_start_before_unix: int | None = None,
    call_start_after_unix: int | None = None,
    page_size: int = 30,
    max_length: int = 10000,
) -> TextContent:
    """List conversations with filtering options."""
    page_size = min(page_size, 100)

    try:
        response = client.conversational_ai.conversations.list(
            cursor=cursor,
            agent_id=agent_id,
            call_start_before_unix=call_start_before_unix,
            call_start_after_unix=call_start_after_unix,
            page_size=page_size,
        )

        if not response.conversations:
            return TextContent(type="text", text="No conversations found.")

        conv_list = []
        for conv in response.conversations:
            start_time = datetime.fromtimestamp(conv.start_time_unix_secs).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            conv_info = f"""Conversation ID: {conv.conversation_id}
Status: {conv.status}
Agent: {conv.agent_name or 'N/A'} (ID: {conv.agent_id})
Started: {start_time}
Duration: {conv.call_duration_secs} seconds
Messages: {conv.message_count}
Call Successful: {conv.call_successful}"""

            conv_list.append(conv_info)

        formatted_list = "\n\n".join(conv_list)

        pagination_info = f"Showing {len(response.conversations)} conversations"
        if response.has_more:
            pagination_info += f" (more available, next cursor: {response.next_cursor})"

        full_text = f"{pagination_info}\n\n{formatted_list}"

        # Use utility to handle large text content
        result_text = handle_large_text(full_text, max_length, "conversation list")

        # If content was saved to file, prepend pagination info
        if result_text != full_text:
            result_text = f"{pagination_info}\n\n{result_text}"

        return TextContent(type="text", text=result_text)

    except Exception as e:
        make_error(f"Failed to list conversations: {str(e)}")
        # This line is unreachable but satisfies type checker
        return TextContent(type="text", text="")


@mcp.tool(
    description="""Send feedback for a completed conversation.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        conversation_id: The ID of the conversation to provide feedback for
        rating: Rating score (1-5)
        feedback_text: Detailed feedback text (optional)
        categories: List of feedback categories (optional)
    """
)
def send_conversation_feedback(
    conversation_id: str,
    rating: int,
    feedback_text: str | None = None,
    categories: list[str] | None = None,
) -> TextContent:
    if rating < 1 or rating > 5:
        make_error("Rating must be between 1 and 5")

    feedback_data = {"rating": rating}
    if feedback_text is not None:
        feedback_data["feedback_text"] = feedback_text
    if categories is not None:
        feedback_data["categories"] = categories

    client.conversational_ai.conversations.feedback(
        conversation_id=conversation_id,
        **feedback_data
    )

    return TextContent(
        type="text",
        text=f"Feedback submitted successfully for conversation {conversation_id} with rating {rating}",
    )


# =============================================================================
# TOOLS API
# =============================================================================

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
        make_error("tool_type must be either 'server' or 'client'")

    if tool_type == "server" and server_tool is None:
        make_error("server_tool configuration is required for server tools")
    
    if tool_type == "client" and client_tool is None:
        make_error("client_tool configuration is required for client tools")

    tool_data = {
        "type": tool_type,
        "name": name,
        "description": description,
    }

    if server_tool is not None:
        tool_data["server_tool"] = server_tool
    if client_tool is not None:
        tool_data["client_tool"] = client_tool

    response = client.conversational_ai.tools.create(**tool_data)

    return TextContent(
        type="text",
        text=f"Tool created successfully: Name: {name}, Tool ID: {response.tool_id}, Type: {tool_type}",
    )


@mcp.tool(
    description="""Get details of a specific tool.

    Args:
        tool_id: The ID of the tool to retrieve
    """
)
def get_tool(tool_id: str) -> TextContent:
    response = client.conversational_ai.tools.get(tool_id=tool_id)

    return TextContent(
        type="text",
        text=f"Tool Details: Name: {response.name}, Tool ID: {response.tool_id}, Type: {response.type}, Description: {response.description}",
    )


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
    update_data = {}
    if name is not None:
        update_data["name"] = name
    if description is not None:
        update_data["description"] = description
    if server_tool is not None:
        update_data["server_tool"] = server_tool
    if client_tool is not None:
        update_data["client_tool"] = client_tool

    if not update_data:
        make_error("At least one field must be provided for update")

    response = client.conversational_ai.tools.update(tool_id=tool_id, **update_data)

    return TextContent(
        type="text",
        text=f"Tool updated successfully: Name: {response.name}, Tool ID: {response.tool_id}",
    )


@mcp.tool(
    description="""Delete a tool permanently.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        tool_id: The ID of the tool to delete
    """
)
def delete_tool(tool_id: str) -> TextContent:
    client.conversational_ai.tools.delete(tool_id=tool_id)

    return TextContent(
        type="text",
        text=f"Tool deleted successfully: Tool ID: {tool_id}",
    )


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
            make_error("tool_type must be either 'server' or 'client'")
        params["type"] = tool_type

    response = client.conversational_ai.tools.list(**params)

    if not response.tools:
        return TextContent(type="text", text="No tools found.")

    tools_list = []
    for tool in response.tools:
        tools_list.append(f"Name: {tool.name}, ID: {tool.tool_id}, Type: {tool.type}")

    formatted_list = "\n".join(tools_list)
    return TextContent(type="text", text=f"Tools:\n{formatted_list}")


@mcp.tool(
    description="""Transform audio from one voice to another using provided audio files.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """
)
def speech_to_speech(
    input_file_path: str,
    voice_name: str = "Adam",
    output_directory: str | None = None,
) -> TextContent:
    voices = client.voices.search(search=voice_name)

    if len(voices.voices) == 0:
        make_error("No voice found with that name.")

    voice = next((v for v in voices.voices if v.name == voice_name), None)

    if voice is None:
        make_error(f"Voice with name: {voice_name} does not exist.")

    assert voice is not None  # Type assertion for type checker
    file_path = handle_input_file(input_file_path)
    output_path = make_output_path(output_directory, base_path)
    output_file_name = make_output_file("sts", file_path.name, output_path, "mp3")

    with file_path.open("rb") as f:
        audio_bytes = f.read()

    audio_data = client.speech_to_speech.convert(
        model_id="eleven_multilingual_sts_v2",
        voice_id=voice.voice_id,
        audio=audio_bytes,
    )

    audio_bytes = b"".join(audio_data)

    with open(output_path / output_file_name, "wb") as f:
        f.write(audio_bytes)

    return TextContent(
        type="text", text=f"Success. File saved as: {output_path / output_file_name}"
    )


@mcp.tool(
    description="""Create voice previews from a text prompt. Creates three previews with slight variations. Saves the previews to a given directory. If no text is provided, the tool will auto-generate text.

    Voice preview files are saved as: voice_design_(generated_voice_id)_(timestamp).mp3

    Example file name: voice_design_Ya2J5uIa5Pq14DNPsbC1_20250403_164949.mp3

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """
)
def text_to_voice(
    voice_description: str,
    text: str | None = None,
    output_directory: str | None = None,
) -> TextContent:
    if voice_description == "":
        make_error("Voice description is required.")

    previews = client.text_to_voice.create_previews(
        voice_description=voice_description,
        text=text,
        auto_generate_text=True if text is None else False,
    )

    output_path = make_output_path(output_directory, base_path)

    generated_voice_ids = []
    output_file_paths = []

    for preview in previews.previews:
        output_file_name = make_output_file(
            "voice_design", preview.generated_voice_id, output_path, "mp3", full_id=True
        )
        output_file_paths.append(str(output_file_name))
        generated_voice_ids.append(preview.generated_voice_id)
        audio_bytes = base64.b64decode(preview.audio_base_64)

        with open(output_path / output_file_name, "wb") as f:
            f.write(audio_bytes)

    return TextContent(
        type="text",
        text=f"Success. Files saved at: {', '.join(output_file_paths)}. Generated voice IDs are: {', '.join(generated_voice_ids)}",
    )


@mcp.tool(
    description="""Add a generated voice to the voice library. Uses the voice ID from the `text_to_voice` tool.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """
)
def create_voice_from_preview(
    generated_voice_id: str,
    voice_name: str,
    voice_description: str,
) -> TextContent:
    voice = client.text_to_voice.create_voice_from_preview(
        voice_name=voice_name,
        voice_description=voice_description,
        generated_voice_id=generated_voice_id,
    )

    return TextContent(
        type="text",
        text=f"Success. Voice created: {voice.name} with ID:{voice.voice_id}",
    )


def _get_phone_number_by_id(phone_number_id: str):
    """Helper function to get phone number details by ID."""
    phone_numbers = client.conversational_ai.phone_numbers.list()
    for phone in phone_numbers:
        if phone.phone_number_id == phone_number_id:
            return phone
    make_error(f"Phone number with ID {phone_number_id} not found.")


@mcp.tool(
    description="""Make an outbound call using an ElevenLabs agent. Automatically detects provider type (Twilio or SIP trunk) and uses the appropriate API.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        agent_id: The ID of the agent that will handle the call
        agent_phone_number_id: The ID of the phone number to use for the call
        to_number: The phone number to call (E.164 format: +1xxxxxxxxxx)

    Returns:
        TextContent containing information about the call
    """
)
def make_outbound_call(
    agent_id: str,
    agent_phone_number_id: str,
    to_number: str,
) -> TextContent:
    # Get phone number details to determine provider type
    phone_number = _get_phone_number_by_id(agent_phone_number_id)
    
    if phone_number.provider.lower() == "twilio":
        response = client.conversational_ai.twilio.outbound_call(
            agent_id=agent_id,
            agent_phone_number_id=agent_phone_number_id,
            to_number=to_number,
        )
        provider_info = "Twilio"
    elif phone_number.provider.lower() == "sip_trunk":
        response = client.conversational_ai.sip_trunk.outbound_call(
            agent_id=agent_id,
            agent_phone_number_id=agent_phone_number_id,
            to_number=to_number,
        )
        provider_info = "SIP trunk"
    else:
        make_error(f"Unsupported provider type: {phone_number.provider}")

    return TextContent(type="text", text=f"Outbound call initiated via {provider_info}: {response}.")


@mcp.tool(
    description="""Search for a voice across the entire ElevenLabs voice library.

    Args:
        page: Page number to return (0-indexed)
        page_size: Number of voices to return per page (1-100)
        search: Search term to filter voices by

    Returns:
        TextContent containing information about the shared voices
    """
)
def search_voice_library(
    page: int = 0,
    page_size: int = 10,
    search: str | None = None,
) -> TextContent:
    response = client.voices.get_shared(
        page=page,
        page_size=page_size,
        search=search,
    )

    if not response.voices:
        return TextContent(
            type="text", text="No shared voices found with the specified criteria."
        )

    voice_list = []
    for voice in response.voices:
        language_info = "N/A"
        if hasattr(voice, "verified_languages") and voice.verified_languages:
            languages = []
            for lang in voice.verified_languages:
                accent_info = (
                    f" ({lang.accent})"
                    if hasattr(lang, "accent") and lang.accent
                    else ""
                )
                languages.append(f"{lang.language}{accent_info}")
            language_info = ", ".join(languages)

        details = [
            f"Name: {voice.name}",
            f"ID: {voice.voice_id}",
            f"Category: {getattr(voice, 'category', 'N/A')}",
        ]
        # TODO: Make cleaner
        if hasattr(voice, "gender") and voice.gender:
            details.append(f"Gender: {voice.gender}")
        if hasattr(voice, "age") and voice.age:
            details.append(f"Age: {voice.age}")
        if hasattr(voice, "accent") and voice.accent:
            details.append(f"Accent: {voice.accent}")
        if hasattr(voice, "description") and voice.description:
            details.append(f"Description: {voice.description}")
        if hasattr(voice, "use_case") and voice.use_case:
            details.append(f"Use Case: {voice.use_case}")

        details.append(f"Languages: {language_info}")

        if hasattr(voice, "preview_url") and voice.preview_url:
            details.append(f"Preview URL: {voice.preview_url}")

        voice_info = "\n".join(details)
        voice_list.append(voice_info)

    formatted_info = "\n\n".join(voice_list)
    return TextContent(type="text", text=f"Shared Voices:\n\n{formatted_info}")


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
        text=f"Phone Number Details: Number: {response.phone_number}, ID: {response.phone_number_id}, Provider: {response.provider}, Status: {response.status}, Assigned Agent: {assigned_agent}",
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


@mcp.tool(description="Play an audio file. Supports WAV and MP3 formats.")
def play_audio(input_file_path: str) -> TextContent:
    file_path = handle_input_file(input_file_path)
    play(open(file_path, "rb").read(), use_ffmpeg=False)
    return TextContent(type="text", text=f"Successfully played audio file: {file_path}")


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


def main():
    print("Starting MCP server")
    """Run the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
