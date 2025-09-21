#!/usr/bin/env python3
"""
Advanced Agent Telemetry Example
Demonstrates comprehensive telemetry for a multi-tool AI agent with:
- Planning and reasoning loops
- Tool selection and execution
- Error handling and recovery
- Performance optimization
- Memory management
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


from telemetry_sdk import quick_setup, TelemetryClient


class TaskType(Enum):
    """Types of tasks the agent can handle"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    CALCULATION = "calculation"
    CREATIVE = "creative"


class AgentState(Enum):
    """Agent execution states"""
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Task:
    """Task structure for the agent"""
    id: str
    type: TaskType
    description: str
    requirements: List[str]
    priority: int = 1
    max_retries: int = 3
    timeout_seconds: int = 300


@dataclass
class Tool:
    """Tool available to the agent"""
    name: str
    description: str
    capabilities: List[str]
    cost_per_use: float
    avg_latency_ms: int
    reliability_score: float  # 0.0 to 1.0


@dataclass
class ExecutionPlan:
    """Agent's execution plan"""
    steps: List[str]
    estimated_cost: float
    estimated_time_ms: int
    confidence: float
    fallback_plan: Optional['ExecutionPlan'] = None


class AdvancedAIAgent:
    """
    Advanced AI Agent with comprehensive telemetry integration
    Demonstrates real-world agent patterns with full observability
    """
    
    def __init__(self, agent_id: str, telemetry_client: TelemetryClient):
        self.agent_id = agent_id
        self.client = telemetry_client
        self.state = AgentState.PLANNING
        self.memory: Dict[str, Any] = {}
        self.execution_history: List[Dict] = []
        
        # Available tools
        self.tools = {
            "web_search": Tool(
                name="web_search",
                description="Search the web for information",
                capabilities=["research", "fact_checking", "current_events"],
                cost_per_use=0.01,
                avg_latency_ms=800,
                reliability_score=0.95
            ),
            "calculator": Tool(
                name="calculator",
                description="Perform mathematical calculations", 
                capabilities=["math", "statistics", "computation"],
                cost_per_use=0.001,
                avg_latency_ms=50,
                reliability_score=0.99
            ),
            "code_executor": Tool(
                name="code_executor",
                description="Execute code and return results",
                capabilities=["programming", "data_analysis", "automation"],
                cost_per_use=0.05,
                avg_latency_ms=2000,
                reliability_score=0.90
            ),
            "llm_writer": Tool(
                name="llm_writer",
                description="Generate high-quality text content",
                capabilities=["writing", "summarization", "creative"],
                cost_per_use=0.02,
                avg_latency_ms=1500,
                reliability_score=0.92
            ),
            "data_analyzer": Tool(
                name="data_analyzer", 
                description="Analyze datasets and extract insights",
                capabilities=["analysis", "visualization", "insights"],
                cost_per_use=0.03,
                avg_latency_ms=3000,
                reliability_score=0.88
            )
        }

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task with full telemetry tracking
        This is the main entry point that orchestrates the entire agent workflow
        """
        
        async with self.client.trace_agent_action(
            action_type="task_execution",
            source_component=f"agent_{self.agent_id}",
            agent_name=self.agent_id
        ) as task_span:
            
            task_span.set_input(f"Task: {task.description}")
            task_span.set_metadata("task_id", task.id)
            task_span.set_metadata("task_type", task.type.value)
            task_span.set_metadata("priority", task.priority)
            task_span.set_metadata("max_retries", task.max_retries)
            
            try:
                result = await self._execute_task_with_retries(task, task_span)
                task_span.set_output(json.dumps(result, indent=2)[:1000])
                task_span.set_metadata("success", True)
                task_span.set_metadata("final_cost", result.get("total_cost", 0))
                return result
                
            except Exception as e:
                task_span.set_error(e)
                task_span.set_metadata("success", False)
                raise

    async def _execute_task_with_retries(self, task: Task, parent_span) -> Dict[str, Any]:
        """Execute task with retry logic and comprehensive telemetry"""
        
        for attempt in range(task.max_retries):
            try:
                result = await self._single_task_execution(task, attempt + 1)
                return result
                
            except Exception as e:
                parent_span.set_metadata(f"attempt_{attempt + 1}_error", str(e))
                
                if attempt == task.max_retries - 1:
                    # Final attempt failed
                    await self._handle_final_failure(task, e)
                    raise
                else:
                    # Retry with backoff
                    backoff_time = 2 ** attempt
                    #await asyncio.sleep(backoff_time)
                    await self._log_retry_attempt(task, attempt + 1, e)

    async def _single_task_execution(self, task: Task, attempt: int) -> Dict[str, Any]:
        """Single task execution attempt with full workflow tracking"""
        
        execution_id = f"{task.id}_attempt_{attempt}"
        start_time = time.time()
        
        # Phase 1: Planning
        plan = await self._plan_execution(task, execution_id)
        
        # Phase 2: Tool Selection and Preparation
        selected_tools = await self._select_tools(task, plan, execution_id)
        
        # Phase 3: Execution
        execution_results = await self._execute_plan(task, plan, selected_tools, execution_id)
        
        # Phase 4: Evaluation and Synthesis
        final_result = await self._evaluate_and_synthesize(task, execution_results, execution_id)
        
        # Phase 5: Learning and Memory Update
        await self._update_memory(task, plan, execution_results, final_result)
        
        total_time = int((time.time() - start_time) * 1000)
        
        return {
            "task_id": task.id,
            "execution_id": execution_id,
            "attempt": attempt,
            "result": final_result,
            "total_time_ms": total_time,
            "total_cost": execution_results.get("total_cost", 0),
            "tools_used": list(selected_tools.keys()),
            "plan_confidence": plan.confidence,
            "success": True
        }

    async def _plan_execution(self, task: Task, execution_id: str) -> ExecutionPlan:
        """Create execution plan with reasoning telemetry"""
        
        async with self.client.trace_agent_action(
            action_type="planning",
            source_component=f"agent_{self.agent_id}",
            agent_name=self.agent_id
        ) as plan_span:
            
            plan_span.set_input(f"Planning for: {task.description}")
            plan_span.set_metadata("task_type", task.type.value)
            plan_span.set_metadata("execution_id", execution_id)
            
            # Simulate planning with model call
            async with self.client.trace_model_call(
                provider="openai",
                model="gpt-4",
                source_component="planning_llm"
            ) as llm_span:
                
                llm_span.set_input(f"Create execution plan for {task.type.value} task: {task.description}")
                
                # Simulate LLM planning call
                #await asyncio.sleep(0.2)
                
                # Create plan based on task type
                if task.type == TaskType.RESEARCH:
                    steps = ["search_information", "verify_sources", "synthesize_findings"]
                    estimated_cost = 0.05
                    confidence = 0.85
                elif task.type == TaskType.ANALYSIS:
                    steps = ["gather_data", "analyze_patterns", "generate_insights"]
                    estimated_cost = 0.08
                    confidence = 0.78
                elif task.type == TaskType.WRITING:
                    steps = ["research_topic", "create_outline", "write_content", "review_quality"]
                    estimated_cost = 0.06
                    confidence = 0.82
                elif task.type == TaskType.CALCULATION:
                    steps = ["parse_requirements", "calculate_result", "verify_accuracy"]
                    estimated_cost = 0.02
                    confidence = 0.95
                else:  # CREATIVE
                    steps = ["brainstorm_ideas", "develop_concept", "create_content"]
                    estimated_cost = 0.07
                    confidence = 0.70
                
                plan_text = f"Plan: {' -> '.join(steps)}"
                llm_span.set_output(plan_text)
                llm_span.set_tokens(len(plan_text.split()) * 4)  # Rough estimate
                llm_span.set_cost(0.003)
            
            # Create execution plan
            plan = ExecutionPlan(
                steps=steps,
                estimated_cost=estimated_cost,
                estimated_time_ms=len(steps) * 2000,
                confidence=confidence
            )
            
            # Create fallback plan for low confidence
            if confidence < 0.8:
                fallback_steps = ["use_simple_approach", "manual_verification"]
                plan.fallback_plan = ExecutionPlan(
                    steps=fallback_steps,
                    estimated_cost=estimated_cost * 0.5,
                    estimated_time_ms=len(fallback_steps) * 1000,
                    confidence=0.9
                )
                plan_span.set_metadata("fallback_plan_created", True)
            
            plan_span.set_output(f"Created {len(steps)}-step plan")
            plan_span.set_metadata("step_count", len(steps))
            plan_span.set_metadata("estimated_cost", estimated_cost)
            plan_span.set_metadata("confidence", confidence)
            plan_span.set_metadata("has_fallback", plan.fallback_plan is not None)
            
            return plan

    async def _select_tools(self, task: Task, plan: ExecutionPlan, execution_id: str) -> Dict[str, Tool]:
        """Select optimal tools for execution with decision telemetry"""
        
        async with self.client.trace_agent_action(
            action_type="tool_selection", 
            source_component=f"agent_{self.agent_id}",
            agent_name=self.agent_id
        ) as selection_span:
            
            selection_span.set_input(f"Selecting tools for {len(plan.steps)} steps")
            selection_span.set_metadata("execution_id", execution_id)
            selection_span.set_metadata("available_tools", len(self.tools))
            
            selected_tools = {}
            total_estimated_cost = 0
            
            # Tool selection logic based on capabilities and requirements
            for step in plan.steps:
                best_tool = None
                best_score = 0
                
                for tool_name, tool in self.tools.items():
                    # Calculate selection score
                    capability_match = any(cap in step for cap in tool.capabilities)
                    score = (
                        (1.0 if capability_match else 0.3) * 0.4 +  # Capability match
                        tool.reliability_score * 0.3 +                # Reliability
                        (1.0 - tool.cost_per_use / 0.1) * 0.2 +      # Cost efficiency  
                        (1.0 - tool.avg_latency_ms / 5000) * 0.1     # Speed
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_tool = tool
                
                if best_tool:
                    selected_tools[step] = best_tool
                    total_estimated_cost += best_tool.cost_per_use
            
            selection_span.set_output(f"Selected {len(selected_tools)} tools")
            selection_span.set_metadata("selected_tools", list(t.name for t in selected_tools.values()))
            selection_span.set_metadata("total_estimated_cost", total_estimated_cost)
            selection_span.set_metadata("avg_tool_reliability", 
                                      sum(t.reliability_score for t in selected_tools.values()) / len(selected_tools))
            
            return selected_tools

    async def _execute_plan(self, task: Task, plan: ExecutionPlan, tools: Dict[str, Tool], execution_id: str) -> Dict[str, Any]:
        """Execute the plan step by step with detailed telemetry"""
        
        async with self.client.trace_agent_action(
            action_type="plan_execution",
            source_component=f"agent_{self.agent_id}",
            agent_name=self.agent_id
        ) as execution_span:
            
            execution_span.set_input(f"Executing {len(plan.steps)}-step plan")
            execution_span.set_metadata("execution_id", execution_id)
            execution_span.set_metadata("plan_confidence", plan.confidence)
            
            step_results = {}
            total_cost = 0
            execution_start = time.time()
            
            for i, step in enumerate(plan.steps):
                step_result = await self._execute_step(step, tools.get(step), task, i + 1, len(plan.steps))
                step_results[step] = step_result
                total_cost += step_result.get("cost", 0)
                
                # Check if we should continue or abort
                if step_result.get("should_abort", False):
                    execution_span.set_metadata("execution_aborted", True)
                    execution_span.set_metadata("abort_reason", step_result.get("abort_reason"))
                    break
            
            execution_time = int((time.time() - execution_start) * 1000)
            
            execution_span.set_output(f"Completed {len(step_results)} steps")
            execution_span.set_metadata("total_cost", total_cost)
            execution_span.set_metadata("execution_time_ms", execution_time)
            execution_span.set_metadata("steps_completed", len(step_results))
            execution_span.set_metadata("steps_planned", len(plan.steps))
            
            return {
                "step_results": step_results,
                "total_cost": total_cost,
                "execution_time_ms": execution_time,
                "completed_steps": len(step_results),
                "planned_steps": len(plan.steps)
            }

    async def _execute_step(self, step: str, tool: Tool, task: Task, step_num: int, total_steps: int) -> Dict[str, Any]:
        """Execute individual step with tool telemetry"""
        
        tool_name = tool.name if tool else "no_tool"
        
        async with self.client.trace_tool_execution(
            tool_name=tool_name,
            source_component=f"agent_{self.agent_id}"
        ) as tool_span:
            
            tool_span.set_input(f"Step {step_num}/{total_steps}: {step}")
            tool_span.set_metadata("step_name", step)
            tool_span.set_metadata("step_number", step_num)
            tool_span.set_metadata("total_steps", total_steps)
            
            if tool:
                tool_span.set_metadata("tool_cost", tool.cost_per_use)
                tool_span.set_metadata("tool_reliability", tool.reliability_score)
                tool_span.set_metadata("expected_latency_ms", tool.avg_latency_ms)
            
            # Simulate tool execution
            if tool:
                # Add some variance to simulate real tool behavior
                actual_latency = int(tool.avg_latency_ms * (0.8 + random.random() * 0.4))
                #await asyncio.sleep(actual_latency / 1000)
                
                # Simulate occasional tool failures
                if random.random() > tool.reliability_score:
                    error_msg = f"Tool {tool.name} failed during step: {step}"
                    tool_span.set_metadata("tool_failed", True)
                    tool_span.set_metadata("failure_reason", "simulated_failure")
                    raise Exception(error_msg)
                
                # Simulate successful execution
                result_data = {
                    "step": step,
                    "tool_used": tool.name,
                    "success": True,
                    "cost": tool.cost_per_use,
                    "latency_ms": actual_latency,
                    "output": f"Successfully completed {step} using {tool.name}"
                }
                
                tool_span.set_output(result_data["output"])
                tool_span.set_cost(tool.cost_per_use)
                
            else:
                # No tool execution - manual processing
                #await asyncio.sleep(0.1)
                result_data = {
                    "step": step,
                    "tool_used": None,
                    "success": True,
                    "cost": 0,
                    "latency_ms": 100,
                    "output": f"Manually processed step: {step}"
                }
                tool_span.set_output(result_data["output"])
            
            tool_span.set_metadata("result_success", result_data["success"])
            
            return result_data

    async def _evaluate_and_synthesize(self, task: Task, execution_results: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Evaluate results and synthesize final output"""
        
        async with self.client.trace_agent_action(
            action_type="evaluation_synthesis",
            source_component=f"agent_{self.agent_id}",
            agent_name=self.agent_id
        ) as eval_span:
            
            eval_span.set_input("Evaluating execution results and synthesizing final output")
            eval_span.set_metadata("execution_id", execution_id)
            eval_span.set_metadata("steps_executed", execution_results["completed_steps"])
            
            # Use LLM to synthesize results
            async with self.client.trace_model_call(
                provider="openai",
                model="gpt-4",
                source_component="synthesis_llm"
            ) as synth_span:
                
                step_summaries = []
                for step, result in execution_results["step_results"].items():
                    step_summaries.append(f"{step}: {result['output']}")
                
                synthesis_input = f"Synthesize results for {task.type.value} task: {'; '.join(step_summaries)}"
                synth_span.set_input(synthesis_input[:1000])
                
                # Simulate synthesis
                #await asyncio.sleep(0.3)
                
                if task.type == TaskType.RESEARCH:
                    final_output = f"Research findings on '{task.description}': Comprehensive analysis completed with verified sources and synthesized insights."
                elif task.type == TaskType.CALCULATION:
                    final_output = f"Calculation result for '{task.description}': Mathematical computation completed with verified accuracy."
                elif task.type == TaskType.WRITING:
                    final_output = f"Written content for '{task.description}': High-quality text generated with proper structure and tone."
                else:
                    final_output = f"Task completed for '{task.description}': All requirements addressed successfully."
                
                synth_span.set_output(final_output[:1000])
                synth_span.set_tokens(len(final_output.split()) * 4)
                synth_span.set_cost(0.004)
            
            # Quality assessment
            quality_score = min(1.0, execution_results["completed_steps"] / execution_results["planned_steps"])
            confidence = quality_score * 0.9  # Slightly lower than completion rate
            
            eval_span.set_output(final_output[:500])
            eval_span.set_metadata("quality_score", quality_score)
            eval_span.set_metadata("confidence", confidence)
            eval_span.set_metadata("completion_rate", quality_score)
            
            return {
                "output": final_output,
                "quality_score": quality_score,
                "confidence": confidence,
                "metadata": {
                    "execution_id": execution_id,
                    "steps_completed": execution_results["completed_steps"],
                    "total_cost": execution_results["total_cost"]
                }
            }

    async def _update_memory(self, task: Task, plan: ExecutionPlan, execution_results: Dict[str, Any], final_result: Dict[str, Any]):
        """Update agent memory with learning from execution"""
        
        async with self.client.trace_agent_action(
            action_type="memory_update",
            source_component=f"agent_{self.agent_id}",
            agent_name=self.agent_id
        ) as memory_span:
            
            memory_span.set_input("Updating agent memory with execution learnings")
            
            # Store execution history
            execution_record = {
                "task_id": task.id,
                "task_type": task.type.value,
                "plan_confidence": plan.confidence,
                "actual_quality": final_result["quality_score"],
                "total_cost": execution_results["total_cost"],
                "execution_time": execution_results["execution_time_ms"],
                "tools_used": list(execution_results["step_results"].keys()),
                "timestamp": time.time()
            }
            
            self.execution_history.append(execution_record)
            
            # Update memory with patterns
            task_type_key = f"performance_{task.type.value}"
            if task_type_key not in self.memory:
                self.memory[task_type_key] = {
                    "total_executions": 0,
                    "avg_quality": 0,
                    "avg_cost": 0,
                    "preferred_tools": {}
                }
            
            stats = self.memory[task_type_key]
            stats["total_executions"] += 1
            stats["avg_quality"] = (stats["avg_quality"] * (stats["total_executions"] - 1) + final_result["quality_score"]) / stats["total_executions"]
            stats["avg_cost"] = (stats["avg_cost"] * (stats["total_executions"] - 1) + execution_results["total_cost"]) / stats["total_executions"]
            
            # Update tool preferences
            for step, result in execution_results["step_results"].items():
                if result.get("tool_used"):
                    tool_name = result["tool_used"]
                    if tool_name not in stats["preferred_tools"]:
                        stats["preferred_tools"][tool_name] = {"uses": 0, "avg_success": 0}
                    
                    tool_stats = stats["preferred_tools"][tool_name]
                    tool_stats["uses"] += 1
                    success_rate = (tool_stats["avg_success"] * (tool_stats["uses"] - 1) + (1 if result["success"] else 0)) / tool_stats["uses"]
                    tool_stats["avg_success"] = success_rate
            
            memory_span.set_output(f"Updated memory for {task.type.value} tasks")
            memory_span.set_metadata("total_history_records", len(self.execution_history))
            memory_span.set_metadata("task_type_executions", stats["total_executions"])
            memory_span.set_metadata("learned_avg_quality", stats["avg_quality"])
            memory_span.set_metadata("learned_avg_cost", stats["avg_cost"])

    async def _handle_final_failure(self, task: Task, error: Exception):
        """Handle final task failure with telemetry"""
        
        async with self.client.trace_agent_action(
            action_type="failure_handling",
            source_component=f"agent_{self.agent_id}",
            agent_name=self.agent_id
        ) as failure_span:
            
            failure_span.set_input(f"Handling final failure for task: {task.description}")
            failure_span.set_error(error)
            failure_span.set_metadata("task_id", task.id)
            failure_span.set_metadata("task_type", task.type.value)
            failure_span.set_metadata("max_retries_reached", True)
            
            # Log failure pattern for learning
            failure_record = {
                "task_id": task.id,
                "task_type": task.type.value,
                "error": str(error),
                "timestamp": time.time()
            }
            
            if "failures" not in self.memory:
                self.memory["failures"] = []
            self.memory["failures"].append(failure_record)
            
            failure_span.set_metadata("total_failures_recorded", len(self.memory["failures"]))

    async def _log_retry_attempt(self, task: Task, attempt: int, error: Exception):
        """Log retry attempt with context"""
        
        async with self.client.trace_agent_action(
            action_type="retry_attempt",
            source_component=f"agent_{self.agent_id}",
            agent_name=self.agent_id
        ) as retry_span:
            
            retry_span.set_input(f"Retrying task attempt {attempt}")
            retry_span.set_metadata("task_id", task.id)
            retry_span.set_metadata("attempt_number", attempt)
            retry_span.set_metadata("max_retries", task.max_retries)
            retry_span.set_metadata("previous_error", str(error))
            retry_span.set_output(f"Preparing retry {attempt}/{task.max_retries}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get agent performance summary for monitoring"""
        
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        total_executions = len(self.execution_history)
        avg_quality = sum(record["actual_quality"] for record in self.execution_history) / total_executions
        avg_cost = sum(record["total_cost"] for record in self.execution_history) / total_executions
        avg_time = sum(record["execution_time"] for record in self.execution_history) / total_executions
        
        # Task type breakdown
        task_type_stats = {}
        for record in self.execution_history:
            task_type = record["task_type"]
            if task_type not in task_type_stats:
                task_type_stats[task_type] = {"count": 0, "avg_quality": 0, "avg_cost": 0}
            
            stats = task_type_stats[task_type]
            stats["count"] += 1
            stats["avg_quality"] = (stats["avg_quality"] * (stats["count"] - 1) + record["actual_quality"]) / stats["count"]
            stats["avg_cost"] = (stats["avg_cost"] * (stats["count"] - 1) + record["total_cost"]) / stats["count"]
        
        return {
            "agent_id": self.agent_id,
            "total_executions": total_executions,
            "overall_performance": {
                "avg_quality_score": round(avg_quality, 3),
                "avg_cost": round(avg_cost, 4),
                "avg_execution_time_ms": round(avg_time, 1)
            },
            "task_type_breakdown": task_type_stats,
            "failure_count": len(self.memory.get("failures", [])),
            "available_tools": len(self.tools),
            "memory_size": len(self.memory)
        }


# Demo scenarios
async def demo_single_task_execution():
    """Demonstrate single task execution with full telemetry"""
    print("🤖 Single Task Execution Demo")
    print("=" * 40)
    
    # Setup telemetry
    client = quick_setup(
        api_key="change-me",
        endpoint="https://localhost:8443",
        project_id="agent-demo"
    )
    
    # Create agent
    agent = AdvancedAIAgent("research_agent_001", client)
    
    # Create task
    task = Task(
        id="task_001",
        type=TaskType.RESEARCH,
        description="Research the latest developments in AI agent architectures",
        requirements=["current_sources", "technical_depth", "practical_applications"],
        priority=1
    )
    
    print(f"📋 Executing task: {task.description}")
    print(f"🔧 Task type: {task.type.value}")
    
    try:
        result = await agent.execute_task(task)
        
        print("✅ Task completed successfully!")
        print(f"📊 Execution time: {result['total_time_ms']}ms")
        print(f"💰 Total cost: ${result['total_cost']:.4f}")
        print(f"🔧 Tools used: {', '.join(result['tools_used'])}")
        print(f"🎯 Plan confidence: {result['plan_confidence']:.2f}")
        print(f"📝 Result: {result['result']['output'][:200]}...")
        
    except Exception as e:
        print(f"❌ Task failed: {e}")
    
    await client.close()


async def demo_multiple_task_types():
    """Demonstrate agent handling different task types"""
    print("\n🎯 Multiple Task Types Demo")
    print("=" * 40)
    
    # Setup telemetry
    client = quick_setup(
        api_key="change-me",
        endpoint="https://localhost:8443", 
        project_id="multi-task-agent"
    )
    
    # Create agent
    agent = AdvancedAIAgent("multi_agent_002", client)
    
    # Create various tasks
    tasks = [
        Task("calc_001", TaskType.CALCULATION, "Calculate the ROI of AI implementation", ["accuracy", "business_context"]),
        Task("write_001", TaskType.WRITING, "Write a technical blog post about agent architectures", ["technical_accuracy", "readability"]),
        Task("analysis_001", TaskType.ANALYSIS, "Analyze user behavior patterns from app data", ["data_insights", "visualization"]),
        Task("research_001", TaskType.RESEARCH, "Research competitive landscape for AI tools", ["market_analysis", "feature_comparison"]),
        Task("creative_001", TaskType.CREATIVE, "Generate creative marketing campaign concepts", ["originality", "brand_alignment"])
    ]
    
    print(f"🔄 Processing {len(tasks)} different task types...")
    
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"\n📋 Task {i}/{len(tasks)}: {task.description}")
        print(f"   Type: {task.type.value}")
        
        try:
            result = await agent.execute_task(task)
            results.append(result)
            
            print(f"   ✅ Completed in {result['total_time_ms']}ms")
            print(f"   💰 Cost: ${result['total_cost']:.4f}")
            print(f"   🎯 Quality: {result['result']['quality_score']:.2f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({"error": str(e), "task_id": task.id})
    
    # Show performance summary
    print(f"\n📊 Agent Performance Summary:")
    summary = agent.get_performance_summary()
    print(f"   Total executions: {summary['total_executions']}")
    print(f"   Average quality: {summary['overall_performance']['avg_quality_score']:.2f}")
    print(f"   Average cost: ${summary['overall_performance']['avg_cost']:.4f}")
    print(f"   Average time: {summary['overall_performance']['avg_execution_time_ms']:.1f}ms")
    
    await client.close()


async def demo_agent_learning_and_adaptation():
    """Demonstrate agent learning from previous executions"""
    print("\n🧠 Agent Learning and Adaptation Demo")
    print("=" * 40)
    
    # Setup telemetry
    client = quick_setup(
        api_key="change-me",
        endpoint="https://localhost:8443",
        project_id="learning-agent"
    )
    
    # Create agent
    agent = AdvancedAIAgent("learning_agent_003", client)
    
    # Execute similar tasks multiple times to show learning
    base_task = Task(
        id="learning_task",
        type=TaskType.ANALYSIS,
        description="Analyze data patterns and extract insights",
        requirements=["statistical_analysis", "visualization", "actionable_insights"]
    )
    
    print("🔄 Executing similar tasks to demonstrate learning...")
    
    for iteration in range(3):
        task = Task(
            id=f"learning_task_{iteration + 1}",
            type=base_task.type,
            description=f"Iteration {iteration + 1}: {base_task.description}",
            requirements=base_task.requirements
        )
        
        print(f"\n📊 Learning Iteration {iteration + 1}")
        
        try:
            result = await agent.execute_task(task)
            
            # Show how the agent adapts over time
            memory_stats = agent.memory.get(f"performance_{task.type.value}", {})
            
            print(f"   ✅ Completed successfully")
            print(f"   📈 Quality score: {result['result']['quality_score']:.2f}")
            print(f"   💰 Cost: ${result['total_cost']:.4f}")
            print(f"   🧠 Agent has executed {memory_stats.get('total_executions', 0)} {task.type.value} tasks")
            print(f"   📊 Learned average quality: {memory_stats.get('avg_quality', 0):.2f}")
            print(f"   💡 Preferred tools: {list(memory_stats.get('preferred_tools', {}).keys())}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    await client.close()


async def demo_error_handling_and_recovery():
    """Demonstrate agent error handling and recovery mechanisms"""
    print("\n🛡️ Error Handling and Recovery Demo")
    print("=" * 40)
    
    # Setup telemetry
    client = quick_setup(
        api_key="change-me",
        endpoint="https://localhost:8443",
        project_id="resilient-agent"
    )
    
    # Create agent with modified tools to simulate failures
    agent = AdvancedAIAgent("resilient_agent_004", client)
    
    # Reduce tool reliability to simulate failures
    for tool in agent.tools.values():
        tool.reliability_score *= 0.7  # Make tools less reliable
    
    # Create challenging task
    challenging_task = Task(
        id="challenge_001",
        type=TaskType.RESEARCH,
        description="Complex research task that may encounter tool failures",
        requirements=["robustness", "error_recovery"],
        max_retries=3
    )
    
    print("🔧 Testing error handling with unreliable tools...")
    print(f"📋 Task: {challenging_task.description}")
    
    try:
        result = await agent.execute_task(challenging_task)
        
        print("✅ Task completed despite tool failures!")
        print(f"📊 Final result quality: {result['result']['quality_score']:.2f}")
        print(f"🔄 Attempt number: {result['attempt']}")
        print(f"💰 Total cost: ${result['total_cost']:.4f}")
        
        # Show failure patterns learned
        failures = agent.memory.get("failures", [])
        if failures:
            print(f"🧠 Agent recorded {len(failures)} failure patterns for learning")
        
    except Exception as e:
        print(f"❌ Task failed after all retries: {e}")
        failures = agent.memory.get("failures", [])
        print(f"📝 Recorded failure patterns: {len(failures)}")
    
    await client.close()


async def demo_agent_collaboration():
    """Demonstrate multiple agents working together"""
    print("\n🤝 Agent Collaboration Demo")
    print("=" * 40)
    
    # Setup telemetry
    client = quick_setup(
        api_key="change-me",
        endpoint="https://localhost:8443",
        project_id="collaborative-agents"
    )
    
    # Create specialized agents
    research_agent = AdvancedAIAgent("researcher_005", client)
    analysis_agent = AdvancedAIAgent("analyst_006", client)
    writer_agent = AdvancedAIAgent("writer_007", client)
    
    # Collaborative task: Create comprehensive report
    collaborative_task = Task(
        id="collab_001",
        type=TaskType.RESEARCH,
        description="Create comprehensive market analysis report",
        requirements=["research", "analysis", "professional_writing"]
    )
    
    print("👥 Simulating agent collaboration on complex task...")
    print(f"📋 Task: {collaborative_task.description}")
    
    async with client.trace_agent_action(
        action_type="agent_collaboration",
        source_component="collaboration_coordinator",
        agent_name="coordinator"
    ) as collab_span:
        
        collab_span.set_input(f"Coordinating {collaborative_task.description}")
        collab_span.set_metadata("participating_agents", 3)
        collab_span.set_metadata("task_complexity", "high")
        
        try:
            # Phase 1: Research
            research_task = Task(
                id="collab_research",
                type=TaskType.RESEARCH,
                description="Research market data and trends",
                requirements=["comprehensive_sources", "recent_data"]
            )
            
            print("🔍 Phase 1: Research Agent working...")
            research_result = await research_agent.execute_task(research_task)
            print(f"   ✅ Research completed (Quality: {research_result['result']['quality_score']:.2f})")
            
            # Phase 2: Analysis
            analysis_task = Task(
                id="collab_analysis",
                type=TaskType.ANALYSIS,
                description="Analyze research findings and extract insights",
                requirements=["statistical_analysis", "trend_identification"]
            )
            
            print("📊 Phase 2: Analysis Agent working...")
            analysis_result = await analysis_agent.execute_task(analysis_task)
            print(f"   ✅ Analysis completed (Quality: {analysis_result['result']['quality_score']:.2f})")
            
            # Phase 3: Writing
            writing_task = Task(
                id="collab_writing",
                type=TaskType.WRITING,
                description="Create professional report from research and analysis",
                requirements=["professional_tone", "clear_structure", "executive_summary"]
            )
            
            print("✍️ Phase 3: Writer Agent working...")
            writing_result = await writer_agent.execute_task(writing_task)
            print(f"   ✅ Writing completed (Quality: {writing_result['result']['quality_score']:.2f})")
            
            # Calculate collaborative metrics
            total_cost = (research_result['total_cost'] + 
                         analysis_result['total_cost'] + 
                         writing_result['total_cost'])
            
            total_time = (research_result['total_time_ms'] + 
                         analysis_result['total_time_ms'] + 
                         writing_result['total_time_ms'])
            
            avg_quality = (research_result['result']['quality_score'] + 
                          analysis_result['result']['quality_score'] + 
                          writing_result['result']['quality_score']) / 3
            
            collab_span.set_output("Collaborative task completed successfully")
            collab_span.set_metadata("total_cost", total_cost)
            collab_span.set_metadata("total_time_ms", total_time)
            collab_span.set_metadata("average_quality", avg_quality)
            collab_span.set_metadata("phases_completed", 3)
            
            print(f"\n🎉 Collaborative task completed!")
            print(f"📊 Combined Results:")
            print(f"   Total cost: ${total_cost:.4f}")
            print(f"   Total time: {total_time}ms")
            print(f"   Average quality: {avg_quality:.2f}")
            print(f"   Agents involved: 3")
            
        except Exception as e:
            collab_span.set_error(e)
            print(f"❌ Collaboration failed: {e}")
    
    await client.close()


async def demo_real_time_monitoring():
    """Demonstrate real-time agent monitoring and performance tracking"""
    print("\n📡 Real-time Monitoring Demo")
    print("=" * 40)
    
    # Setup telemetry
    client = quick_setup(
        api_key="change-me",
        endpoint="https://localhost:8443",
        project_id="monitored-agent"
    )
    
    # Create monitored agent
    agent = AdvancedAIAgent("monitored_agent_008", client)
    
    # Create monitoring task that runs continuously
    monitoring_tasks = [
        Task(f"monitor_{i}", TaskType.ANALYSIS, f"Monitoring task {i}", ["real_time", "metrics"])
        for i in range(5)
    ]
    
    print("📊 Running continuous monitoring simulation...")
    
    for i, task in enumerate(monitoring_tasks, 1):
        print(f"\n⏱️ Monitoring Cycle {i}/5")
        
        try:
            # Execute task
            result = await agent.execute_task(task)
            
            # Get current performance metrics
            summary = agent.get_performance_summary()
            
            print(f"   ✅ Task {i} completed")
            print(f"   📈 Current avg quality: {summary['overall_performance']['avg_quality_score']:.2f}")
            print(f"   💰 Current avg cost: ${summary['overall_performance']['avg_cost']:.4f}")
            print(f"   ⚡ Current avg time: {summary['overall_performance']['avg_execution_time_ms']:.1f}ms")
            print(f"   🧠 Memory size: {summary['memory_size']} patterns")
            
            # Simulate alerting on performance issues
            if result['result']['quality_score'] < 0.7:
                print("   🚨 ALERT: Quality score below threshold!")
                
                async with client.trace_agent_action(
                    action_type="performance_alert",
                    source_component="monitoring_system"
                ) as alert_span:
                    alert_span.set_input("Quality threshold violation detected")
                    alert_span.set_metadata("quality_score", result['result']['quality_score'])
                    alert_span.set_metadata("threshold", 0.7)
                    alert_span.set_metadata("agent_id", agent.agent_id)
                    alert_span.set_metadata("task_id", task.id)
            
            if result['total_cost'] > 0.1:
                print("   💸 ALERT: Cost exceeds budget!")
                
                async with client.trace_agent_action(
                    action_type="cost_alert",
                    source_component="monitoring_system"
                ) as cost_alert_span:
                    cost_alert_span.set_input("Cost threshold violation detected")
                    cost_alert_span.set_metadata("actual_cost", result['total_cost'])
                    cost_alert_span.set_metadata("budget_threshold", 0.1)
                    cost_alert_span.set_metadata("agent_id", agent.agent_id)
            
        except Exception as e:
            print(f"   ❌ Monitoring cycle {i} failed: {e}")
            
            # Log monitoring failure
            async with client.trace_agent_action(
                action_type="monitoring_failure",
                source_component="monitoring_system"
            ) as failure_span:
                failure_span.set_error(e)
                failure_span.set_metadata("cycle_number", i)
                failure_span.set_metadata("agent_id", agent.agent_id)
        
        # Brief pause between monitoring cycles
        #await asyncio.sleep(0.1)
    
    print(f"\n📋 Final Performance Report:")
    final_summary = agent.get_performance_summary()
    print(json.dumps(final_summary, indent=2))
    
    await client.close()


async def demo_advanced_debugging():
    """Demonstrate advanced debugging capabilities with detailed telemetry"""
    print("\n🔍 Advanced Debugging Demo")
    print("=" * 40)
    
    # Setup telemetry with debug-level tracing
    client = quick_setup(
        api_key="change-me",
        endpoint="https://localhost:8443",
        project_id="debug-agent"
    )
    
    # Create agent for debugging
    agent = AdvancedAIAgent("debug_agent_009", client)
    
    # Create a task that will have issues to debug
    debug_task = Task(
        id="debug_001",
        type=TaskType.CALCULATION,
        description="Complex calculation with potential edge cases",
        requirements=["precision", "error_handling"],
        max_retries=2
    )
    
    print("🐛 Executing task with detailed debugging telemetry...")
    
    async with client.trace_agent_action(
        action_type="debug_session",
        source_component="debugging_framework"
    ) as debug_session:
        
        debug_session.set_input("Starting debug session for complex calculation task")
        debug_session.set_metadata("debug_level", "verbose")
        debug_session.set_metadata("task_id", debug_task.id)
        
        try:
            # Execute with debug tracing
            result = await agent.execute_task(debug_task)
            
            debug_session.set_output("Debug session completed successfully")
            debug_session.set_metadata("final_quality", result['result']['quality_score'])
            debug_session.set_metadata("debug_successful", True)
            
            print("✅ Debug session completed successfully")
            print(f"📊 Task executed with quality score: {result['result']['quality_score']:.2f}")
            print(f"🔧 Tools used in execution: {', '.join(result['tools_used'])}")
            print(f"💰 Debug execution cost: ${result['total_cost']:.4f}")
            
            # Show debug insights
            print(f"\n🔍 Debug Insights:")
            print(f"   Plan confidence was: {result['plan_confidence']:.2f}")
            print(f"   Execution took {result['attempt']} attempt(s)")
            print(f"   Memory patterns learned: {len(agent.memory)}")
            
        except Exception as e:
            debug_session.set_error(e)
            debug_session.set_metadata("debug_successful", False)
            
            print(f"❌ Debug session revealed critical issue: {e}")
            print("🔧 Debug telemetry captured for analysis")
    
    await client.close()


async def main():
    """Run all advanced agent examples"""
    print("🤖 Advanced AI Agent Telemetry Examples")
    print("=" * 50)
    
    try:
        await demo_single_task_execution()
        await demo_multiple_task_types()
        await demo_agent_learning_and_adaptation()
        await demo_error_handling_and_recovery()
        await demo_agent_collaboration()
        await demo_real_time_monitoring()
        await demo_advanced_debugging()
        
        print("\n🎉 All Advanced Agent Examples Completed!")
        print("\n🎯 Key Agent Telemetry Benefits Demonstrated:")
        print("   🔍 Complete execution visibility")
        print("   📊 Performance and cost tracking") 
        print("   🧠 Learning and adaptation monitoring")
        print("   🛡️ Error handling and recovery tracking")
        print("   🤝 Multi-agent collaboration insights")
        print("   📡 Real-time monitoring and alerting")
        print("   🐛 Advanced debugging capabilities")
        print("   🎯 Decision-making transparency")
        print("   💡 Tool usage optimization")
        print("   📈 Continuous improvement tracking")
        
        print("\n🚀 Production Use Cases:")
        print("   • Monitor agent performance in production")
        print("   • Debug complex multi-step agent failures")
        print("   • Optimize tool selection and usage")
        print("   • Track learning and adaptation over time")
        print("   • Coordinate multiple specialized agents")
        print("   • Implement cost and performance budgets")
        print("   • Provide transparency for agent decisions")
        print("   • Enable continuous improvement through data")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        print("💭 Ensure your telemetry server is running and accessible")


if __name__ == "__main__":
    asyncio.run(main())