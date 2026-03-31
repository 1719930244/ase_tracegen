"""
Base classes for synthesis agent tools.
合成 Agent 工具的基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from loguru import logger

class BaseTool(ABC):
    """
    Abstract base class for all tools used by the SynthesisAgent.
    SynthesisAgent 使用的所有工具的抽象基类
    """
    
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    
    def __init__(self, context: Dict[str, Any]):
        """
        Initialize tool with shared context.
        使用共享上下文初始化工具
        
        Args:
            context: Shared context dictionary (graph, repo_path, etc.)
        """
        self.context = context
        self.graph = context.get("graph")
        self.repo_path = context.get("repo_path")
        self.seed_id = context.get("seed_id")

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        Execute the tool with given arguments.
        使用给定参数执行工具
        
        Returns:
            String representation of the result/observation.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool definition to dictionary for LLM prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class ToolRegistry:
    """
    Registry for managing and invoking tools.
    用于管理和调用工具的注册表
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.call_count = 0
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def register_all(self, tools: List[BaseTool]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register(tool)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def call(self, name: str, params: Dict[str, Any]) -> str:
        """
        Call a tool by name with parameters.
        根据名称和参数调用工具
        """
        tool = self.get_tool(name)
        if not tool:
            return f"Error: Tool '{name}' not found."
        
        try:
            self.call_count += 1
            logger.info(f"Executing tool '{name}' with params: {params}")
            
            # Filter parameters to only pass those expected by the tool's execute method
            import inspect
            sig = inspect.signature(tool.execute)
            valid_params = {
                k: v for k, v in params.items() 
                if k in sig.parameters or sig.parameters.get("kwargs")
            }
            
            result = tool.execute(**valid_params)
            return result
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {str(e)}")
            return f"Error executing tool '{name}': {str(e)}"
    
    def get_descriptions(self) -> str:
        """Get formatted descriptions of all registered tools."""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
            descriptions.append(f"  Parameters: {tool.parameters}")
        return "\n".join(descriptions)
