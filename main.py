import anyio
import click
import openai
import os
from dotenv import load_dotenv
import mcp.types as types
from mcp.server.lowlevel import Server
import logging
from openai import AsyncOpenAI
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

async def analyze_code_with_context(code: str, context: str) -> dict:
    """
    Analyze code using OpenAI's API with the given context.
    """
    try:
        system_prompt = """You are a code review expert. Analyze the provided code and context.
        Determine if the code meets the stated goals and requirements.
        Provide specific feedback and suggestions for improvement."""
        
        user_prompt = f"""Context: {context}
        
        Code:
        {code}
        
        Please analyze:
        1. Does the code fulfill the stated intent?
        2. Are there any efficiency concerns?
        3. What specific improvements would you suggest?"""

        response = await client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        analysis = response.choices[0].message.content
        is_approved = "looks good" in analysis.lower() and "suggest" not in analysis.lower()
        suggestions = [line.strip() for line in analysis.split('\n') if line.strip().startswith('-')]
        
        if is_approved:
            prompt = "Code aligns well with the intent. Proceed to the next task."
        else:
            prompt = f"Please revise the code based on the following feedback: {analysis}"

        return {
            "prompt": prompt,
            "suggestions": suggestions if suggestions else ["No specific suggestions provided"],
            "is_approved": is_approved
        }
    except Exception as e:
        logger.error(f"Error in analyze_code_with_context: {str(e)}")
        raise ValueError(f"Error analyzing code: {str(e)}")

def create_app():
    app = Server("code-review-server")

    @app.call_tool()
    async def review_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent]:
        try:
            if name != "review_code":
                raise ValueError(f"Unknown tool: {name}")
            if "code" not in arguments or "context" not in arguments:
                raise ValueError("Missing required arguments 'code' or 'context'")
            
            result = await analyze_code_with_context(arguments["code"], arguments["context"])
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            logger.error(f"Error in review_tool: {str(e)}")
            return [types.TextContent(type="text", text=str({"error": str(e)}))]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="review_code",
                description="Reviews code against provided context and suggests improvements",
                inputSchema={
                    "type": "object",
                    "required": ["code", "context"],
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to review"
                        },
                        "context": {
                            "type": "string",
                            "description": "The context or intent of what the code should do"
                        }
                    }
                }
            )
        ]

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        try:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
        except Exception as e:
            logger.error(f"Error in handle_sse: {str(e)}")
            raise

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    routes = [
        Route("/sse", endpoint=handle_sse),
        Mount("/messages", app=sse.handle_post_message),
    ]

    return Starlette(
        debug=True,
        routes=routes,
        middleware=middleware,
    )

starlette_app = create_app()

@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    if transport == "sse":
        import uvicorn
        uvicorn.run(
            "main:starlette_app",
            host="127.0.0.1",
            port=port,
            log_level="info",
            reload=True
        )
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            app = create_app()
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        anyio.run(arun)

    return 0

if __name__ == "__main__":
    main()