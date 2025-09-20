"""
Usage examples for the Telemetry SDK
Demonstrates all integration patterns: context managers, decorators, auto-instrumentation, and logging
"""

import asyncio
import openai

from telemetry_sdk.client.telemetry_client import TelemetryClient
from telemetry_sdk.instrumentation.instrument import AutoInstrumentation
from telemetry_sdk.utils.logger import setup_telemetry_logging



# ==============================================================================
# BASIC SETUP
# ==============================================================================

# Initialize the telemetry client
telemetry = TelemetryClient(
    api_key="your-api-key",
    endpoint="https://your-telemetry-server.com",
    project_id="my-ai-project",
    tenant_id="acme-corp",
    user_id="john-doe",
    application_id="chatbot-v1"
)


# ==============================================================================
# 1. CONTEXT MANAGER APPROACH (Recommended)
# ==============================================================================

async def example_context_managers():
    """Examples using context managers for explicit control"""
    
    # Model call tracing
    async with telemetry.trace_model_call(provider="openai", model="gpt-4") as span:
        span.set_input("What is the weather like?")
        
        # Your LLM call here
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is the weather like?"}]
        )
        
        span.set_output(response.choices[0].message.content)
        span.set_tokens(response.usage.total_tokens)
        span.set_cost(0.03)  # Your cost calculation

    # Tool execution tracing
    async with telemetry.trace_tool_execution(tool_name="weather_api") as span:
        span.set_metadata("endpoint", "https://api.weather.com/v1/current")
        span.set_metadata("location", "San Francisco")
        
        # Your tool call here
        weather_data = await call_weather_api("San Francisco")
        span.set_output(str(weather_data))

    # Agent action tracing
    async with telemetry.trace_agent_action(action_type="planning") as span:
        span.set_input("User wants to know about weather")
        
        plan = {
            "steps": [
                {"action": "get_location", "tool": "location_service"},
                {"action": "get_weather", "tool": "weather_api"},
                {"action": "format_response", "tool": "formatter"}
            ]
        }
        
        span.set_metadata("plan_steps", len(plan["steps"]))
        span.set_output("Created 3-step plan for weather query")

    # Nested tracing for complex workflows
    async with telemetry.trace_agent_action(action_type="conversation") as conversation:
        conversation.set_input("User started new conversation")
        
        async with telemetry.trace_model_call(provider="openai") as model_call:
            # LLM call
            pass
        
        async with telemetry.trace_tool_execution(tool_name="database") as db_call:
            # Database lookup
            pass


# ==============================================================================
# 2. DECORATOR APPROACH (Clean for existing functions)
# ==============================================================================

@telemetry.trace_model_call_decorator(provider="openai", model="gpt-4")
async def generate_response(prompt: str, temperature: float = 0.7):
    """Function automatically traced with decorator"""
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

@telemetry.trace_tool_execution_decorator(tool_name="web_search")
async def search_web(query: str):
    """Tool execution automatically traced"""
    # Your web search implementation
    results = await web_search_api.search(query)
    return results

@telemetry.trace_agent_action_decorator(action_type="reasoning")
async def agent_think(context: str):
    """Agent reasoning automatically traced"""
    # Your reasoning logic
    thought = await reasoning_engine.process(context)
    return thought

# Usage of decorated functions
async def example_decorators():
    response = await generate_response("Explain quantum computing")
    search_results = await search_web("latest AI news")
    reasoning = await agent_think("How to solve this problem?")


# ==============================================================================
# 3. AUTO-INSTRUMENTATION (Zero code changes)
# ==============================================================================

def example_auto_instrumentation():
    """Auto-instrument popular libraries"""
    
    # Set up auto-instrumentation
    auto_instr = AutoInstrumentation(telemetry)
    auto_instr.instrument_all()  # Instruments OpenAI, Anthropic, LangChain, etc.
    
    # Now all calls to instrumented libraries are automatically traced
    # No code changes needed!

async def example_auto_instrumented_code():
    """This code gets automatically traced after instrumentation"""
    
    # OpenAI calls automatically traced
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # LangChain operations automatically traced
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    
    llm = OpenAI()
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a short poem about {topic}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # This call will be automatically traced
    result = chain.run(topic="artificial intelligence")


# ==============================================================================
# 4. LOGGING INTEGRATION (Existing logging users)
# ==============================================================================

def example_logging_integration():
    """Use standard Python logging to send telemetry"""
    
    # Setup telemetry logging
    logger = setup_telemetry_logging(
        api_key="your-api-key",
        endpoint="https://your-telemetry-server.com",
        project_id="my-project",
        logger_name="my_app"
    )
    
    # Model call via logging
    logger.model_call(
        "GPT-4 call completed",
        provider="openai",
        model="gpt-4",
        input_text="What is AI?",
        output_text="AI is artificial intelligence...",
        token_count=150,
        latency_ms=250,
        cost=0.003
    )
    
    # Tool execution via logging
    logger.tool_execution(
        "Web search completed",
        tool_name="bing_search",
        action="search",
        endpoint="https://api.bing.com/search",
        http_method="GET",
        http_status_code=200,
        latency_ms=500
    )
    
    # Agent action via logging
    logger.agent_action(
        "Agent completed reasoning",
        action_type="planning",
        agent_name="task_planner",
        thought_process="Analyzed user request and created execution plan",
        selected_tool="web_search"
    )
    
    # Regular logging also works
    logger.info("Application started", meta_version="1.0.0")
    logger.error("Something went wrong", meta_error_code="E001")


# ==============================================================================
# 5. MANUAL EVENT CREATION (Full control)
# ==============================================================================

async def example_manual_events():
    """Manual event creation for maximum control"""
    
    # Create single event
    event = telemetry.create_event(
        event_type=telemetry.EventType.MODEL_CALL,
        source_component="custom_llm"
    )
    
    event.set_input("Custom input text")
    event.set_metadata("custom_field", "custom_value")
    event.start_timing()
    
    # Your custom logic here
    await asyncio.sleep(0.1)  # Simulate work
    
    event.end_timing()
    event.set_output("Custom output")
    event.set_tokens(100)
    
    # Send the event
    event_id = await event.send()
    print(f"Event sent with ID: {event_id}")
    
    # Create batch of events
    batch = telemetry.create_batch()
    
    for i in range(5):
        event = telemetry.create_event(
            event_type=telemetry.EventType.TOOL_EXECUTION,
            source_component=f"tool_{i}"
        ).set_metadata("batch_index", i).build()
        
        batch.add_event(event, {"tool_name": f"tool_{i}"})
    
    # Send batch
    result = await batch.send()
    print(f"Sent batch: {result}")


# ==============================================================================
# 6. FRAMEWORK INTEGRATIONS
# ==============================================================================

def example_fastapi_integration():
    """FastAPI application with telemetry"""
    from fastapi import FastAPI
    from telemetry_sdk import FrameworkIntegrations
    
    app = FastAPI()
    
    # Add telemetry middleware
    integrations = FrameworkIntegrations(telemetry)
    integrations.wrap_fastapi_app(app)
    
    @app.post("/chat")
    async def chat_endpoint(message: str):
        # This endpoint will be automatically traced
        async with telemetry.trace_model_call(provider="openai") as span:
            response = await generate_ai_response(message)
            span.set_input(message)
            span.set_output(response)
            return {"response": response}


def example_langchain_integration():
    """LangChain with telemetry"""
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from telemetry_sdk import FrameworkIntegrations
    
    # Create chain
    llm = OpenAI()
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer this question: {question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Wrap with telemetry
    integrations = FrameworkIntegrations(telemetry)
    traced_chain = integrations.wrap_langchain_chain(chain)
    
    # Use wrapped chain (automatically traced)
    result = traced_chain.run(question="What is machine learning?")


def example_llamaindex_integration():
    """LlamaIndex with telemetry"""
    from llama_index import GPTSimpleVectorIndex, Document
    from telemetry_sdk import FrameworkIntegrations
    
    # Create index
    documents = [Document("Sample document content")]
    index = GPTSimpleVectorIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    # Wrap with telemetry
    integrations = FrameworkIntegrations(telemetry)
    traced_engine = integrations.wrap_llamaindex_query_engine(query_engine)
    
    # Use wrapped engine (automatically traced)
    response = traced_engine.query("What is in the document?")


# ==============================================================================
# 7. REAL-WORLD EXAMPLES
# ==============================================================================

class AIAgent:
    """Example AI agent with comprehensive telemetry"""
    
    def __init__(self, telemetry_client: TelemetryClient):
        self.telemetry = telemetry_client
        self.openai_client = openai.AsyncOpenAI()
    
    async def process_user_request(self, user_input: str) -> str:
        """Main agent workflow with nested telemetry"""
        
        async with self.telemetry.trace_agent_action(
            action_type="conversation",
            source_component="ai_agent"
        ) as conversation:
            conversation.set_input(user_input)
            
            # Step 1: Analyze user intent
            intent = await self._analyze_intent(user_input)
            conversation.set_metadata("detected_intent", intent)
            
            # Step 2: Plan actions
            plan = await self._create_plan(intent, user_input)
            conversation.set_metadata("plan_steps", len(plan.steps))
            
            # Step 3: Execute plan
            results = []
            for step in plan.steps:
                result = await self._execute_step(step)
                results.append(result)
            
            # Step 4: Generate final response
            response = await self._generate_response(user_input, results)
            conversation.set_output(response)
            
            return response
    
    @telemetry.trace_model_call_decorator(provider="openai", model="gpt-4")
    async def _analyze_intent(self, user_input: str) -> str:
        """Analyze user intent with automatic tracing"""
        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Analyze the user's intent"},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    
    async def _create_plan(self, intent: str, user_input: str):
        """Create execution plan"""
        async with self.telemetry.trace_agent_action(
            action_type="planning",
            source_component="planner"
        ) as span:
            span.set_input(f"Intent: {intent}, Input: {user_input}")
            
            # Your planning logic here
            plan = ExecutionPlan([
                PlanStep("search", "web_search", {"query": user_input}),
                PlanStep("summarize", "summarizer", {})
            ])
            
            span.set_output(f"Created plan with {len(plan.steps)} steps")
            return plan
    
    async def _execute_step(self, step):
        """Execute a single plan step"""
        async with self.telemetry.trace_tool_execution(
            tool_name=step.tool,
            action=step.action
        ) as span:
            span.set_metadata("step_params", step.params)
            
            # Your step execution logic
            if step.tool == "web_search":
                result = await self._web_search(step.params["query"])
            elif step.tool == "summarizer":
                result = await self._summarize(step.params.get("text", ""))
            else:
                result = "Unknown tool"
            
            span.set_output(str(result)[:500])  # Truncate for logging
            return result
    
    @telemetry.trace_tool_execution_decorator(tool_name="web_search")
    async def _web_search(self, query: str):
        """Web search with automatic tracing"""
        # Your web search implementation
        await asyncio.sleep(0.5)  # Simulate API call
        return f"Search results for: {query}"
    
    @telemetry.trace_model_call_decorator(provider="openai", model="gpt-3.5-turbo")
    async def _summarize(self, text: str):
        """Summarization with automatic tracing"""
        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize the following text"},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    
    @telemetry.trace_model_call_decorator(provider="openai", model="gpt-4")
    async def _generate_response(self, user_input: str, results: list):
        """Generate final response with automatic tracing"""
        context = "\n".join(str(r) for r in results)
        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Generate a helpful response based on the context"},
                {"role": "user", "content": f"User: {user_input}\nContext: {context}"}
            ]
        )
        return response.choices[0].message.content


class ChatBot:
    """Example chatbot with logging-based telemetry"""
    
    def __init__(self):
        # Setup telemetry logging
        self.logger = setup_telemetry_logging(
            api_key="your-api-key",
            endpoint="https://your-telemetry-server.com",
            project_id="chatbot-project",
            logger_name="chatbot"
        )
        self.openai_client = openai.AsyncOpenAI()
    
    async def chat(self, user_message: str) -> str:
        """Chat method using logging for telemetry"""
        
        # Log conversation start
        self.logger.agent_action(
            "Conversation started",
            action_type="chat_session",
            agent_name="chatbot",
            input_text=user_message
        )
        
        try:
            # Make LLM call
            start_time = asyncio.get_event_loop().time()
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_message}]
            )
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = int((end_time - start_time) * 1000)
            
            response_text = response.choices[0].message.content
            
            # Log model call
            self.logger.model_call(
                "GPT response generated",
                provider="openai",
                model="gpt-3.5-turbo",
                input_text=user_message,
                output_text=response_text,
                token_count=response.usage.total_tokens,
                latency_ms=latency_ms,
                cost=self._calculate_cost(response.usage.total_tokens)
            )
            
            return response_text
            
        except Exception as e:
            # Log error
            self.logger.error(
                f"Chat error: {str(e)}",
                input_text=user_message,
                error_type=type(e).__name__
            )
            raise
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate cost based on token usage"""
        return tokens * 0.000002  # Example rate


# ==============================================================================
# 8. RUNNING THE EXAMPLES
# ==============================================================================

async def main():
    """Run all examples"""
    
    print("🚀 Telemetry SDK Examples")
    print("=" * 50)
    
    # 1. Context managers
    print("\n1. Context Manager Examples:")
    await example_context_managers()
    
    # 2. Decorators
    print("\n2. Decorator Examples:")
    await example_decorators()
    
    # 3. Auto-instrumentation
    print("\n3. Auto-instrumentation:")
    example_auto_instrumentation()
    await example_auto_instrumented_code()
    
    # 4. Logging integration
    print("\n4. Logging Integration:")
    example_logging_integration()
    
    # 5. Manual events
    print("\n5. Manual Events:")
    await example_manual_events()
    
    # 6. Real-world agent example
    print("\n6. AI Agent Example:")
    agent = AIAgent(telemetry)
    response = await agent.process_user_request("What's the weather like today?")
    print(f"Agent response: {response}")
    
    # 7. Chatbot example
    print("\n7. Chatbot Example:")
    chatbot = ChatBot()
    chat_response = await chatbot.chat("Hello, how are you?")
    print(f"Chatbot response: {chat_response}")
    
    # Cleanup
    await telemetry.flush()
    await telemetry.close()
    
    print("\n✅ All examples completed!")


# Helper classes for examples
class ExecutionPlan:
    def __init__(self, steps):
        self.steps = steps

class PlanStep:
    def __init__(self, action, tool, params):
        self.action = action
        self.tool = tool
        self.params = params


# Dummy implementations for examples
async def call_weather_api(location: str):
    await asyncio.sleep(0.1)
    return {"temperature": "72°F", "condition": "sunny"}

async def generate_ai_response(message: str):
    await asyncio.sleep(0.2)
    return f"AI response to: {message}"

class WebSearchAPI:
    async def search(self, query: str):
        await asyncio.sleep(0.3)
        return [{"title": "Result 1", "url": "http://example.com"}]

web_search_api = WebSearchAPI()

class ReasoningEngine:
    async def process(self, context: str):
        await asyncio.sleep(0.1)
        return f"Reasoning result for: {context}"

reasoning_engine = ReasoningEngine()


if __name__ == "__main__":
    # Set up OpenAI API key (you'll need to set this)
    # openai.api_key = "your-openai-api-key"
    
    asyncio.run(main())