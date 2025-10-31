#!/usr/bin/env python3
"""
Simple test to verify Claude Agent SDK is working.
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("Testing Claude Agent SDK")
print("=" * 60)

# Check if API key is set
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ANTHROPIC_API_KEY not set")
    exit(1)
else:
    print(f"✅ ANTHROPIC_API_KEY is set (length: {len(api_key)})")

# Test 1: Import the SDK
print("\n1. Testing SDK import...")
try:
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
    print("✅ SDK imported successfully")
except ImportError as e:
    print(f"❌ Failed to import SDK: {e}")
    exit(1)

# Test 2: Simple query using the query() function
print("\n2. Testing simple query() function...")
try:
    from claude_agent_sdk import query
    
    async def test_simple_query():
        print("   Sending query: 'What is 2 + 2?'")
        response_text = ""
        async for message in query(prompt="What is 2 + 2?"):
            print(f"   Received message type: {type(message).__name__}")
            if hasattr(message, 'content'):
                for block in message.content:
                    if hasattr(block, 'text'):
                        response_text += block.text
        print(f"   Response: {response_text[:200]}")
        return response_text
    
    result = asyncio.run(test_simple_query())
    if result:
        print("✅ Simple query worked!")
    else:
        print("⚠️  Query returned empty response")
        
except Exception as e:
    print(f"❌ Simple query failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: ClaudeSDKClient with options
print("\n3. Testing ClaudeSDKClient with options...")
try:
    async def test_client_with_options():
        options = ClaudeAgentOptions(
            cwd=str(Path.cwd()),
            allowed_tools=["Read", "Grep", "Glob"],
            disallowed_tools=["Write", "Edit", "Bash", "Task"],
            permission_mode="plan",
            max_turns=5
        )
        
        print(f"   Options: cwd={options.cwd}, allowed_tools={options.allowed_tools}")
        
        async with ClaudeSDKClient(options=options) as client:
            print("   Client created successfully")
            
            # Send a query
            await client.query("List the Python files in the current directory using the Glob tool.")
            print("   Query sent, receiving response...")
            
            response_text = ""
            async for message in client.receive_response():
                print(f"   Received: {type(message).__name__}")
                if hasattr(message, 'content'):
                    for block in message.content:
                        if hasattr(block, 'text'):
                            response_text += block.text
            
            print(f"   Response length: {len(response_text)} chars")
            return response_text
    
    result = asyncio.run(test_client_with_options())
    if result:
        print("✅ ClaudeSDKClient worked!")
    else:
        print("⚠️  Client returned empty response")
        
except Exception as e:
    print(f"❌ ClaudeSDKClient test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)

