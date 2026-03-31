"""
LLM 客户端封装 - 支持多种 LLM 提供商
"""
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import openai
import anthropic
import time
import json
from pathlib import Path


class LLMClient(ABC):
    """
    LLM 客户端抽象基类
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None):
        """
        初始化 LLM 客户端
        
        Args:
            config: LLM 配置字典
            output_dir: 输出目录路径（用于保存资源记录）
        """
        self.config = config
        self.model = config.get("model", "gpt-4-turbo-preview")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 15000)
        self.timeout = config.get("timeout", 60)

        # 资源记录配置 (JSONL 格式，追加写入，线程安全)
        if output_dir:
            self.resource_log_file = Path(output_dir) / "llm_resource_usage.jsonl"
            self.resource_log_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.resource_log_file = Path(".cache/llm_resource_usage.jsonl")
            self.resource_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        发送完成请求
        
        Args:
            prompt: 用户提示
            system_message: 系统消息
            **kwargs: 额外参数
            
        Returns:
            LLM 响应文本
        """
        pass
    
    @abstractmethod
    def complete_with_json(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        发送请求并要求返回 JSON 格式
        
        Args:
            prompt: 用户提示
            system_message: 系统消息
            **kwargs: 额外参数
            
        Returns:
            解析后的 JSON 对象
        """
        pass
    
    def _save_resource_usage(self, usage: Dict[str, Any]) -> None:
        """
        保存 LLM 调用资源记录 (JSONL 追加写入，线程安全)
        """
        try:
            line = json.dumps(usage, ensure_ascii=False)
            with open(self.resource_log_file, 'a') as f:
                f.write(line + "\n")
        except Exception as e:
            logger.warning(f"保存资源记录失败: {e}")


class OpenAIClient(LLMClient):
    """
    OpenAI API 客户端
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None):
        super().__init__(config, output_dir=output_dir)
        
        # 初始化 OpenAI 客户端
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,
        )
        logger.info(f"OpenAI 客户端初始化: model={self.model}")
    
    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        发送 Chat Completion 请求
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )
            content = response.choices[0].message.content
            duration = time.time() - start_time
            
            # 记录资源使用
            usage = {
                "model": self.model,
                "duration_seconds": round(duration, 3),
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "call_count": 1,
                "system_message": system_message if system_message else None,
                "prompt": prompt,
                "response": content
            }
            self._save_resource_usage(usage)
            
            logger.debug(f"OpenAI 响应: {len(content)} 字符, tokens: {response.usage.prompt_tokens}/{response.usage.completion_tokens}")
            return content
        except Exception as e:
            logger.error(f"OpenAI 请求失败: {e}")
            raise
    
    def complete_with_json(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        请求 JSON 格式响应
        """
        # 在 prompt 中明确要求 JSON 格式
        json_prompt = f"{prompt}\n\n请以 JSON 格式返回结果。"
        
        response_text = self.complete(
            json_prompt,
            system_message=system_message,
            response_format={"type": "json_object"},
            **kwargs,
        )
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}\n响应: {response_text}")
            raise


class AnthropicClient(LLMClient):
    """
    Anthropic (Claude) API 客户端
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None):
        super().__init__(config, output_dir=output_dir)
        
        api_key = config.get("api_key")
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Anthropic 客户端初始化: model={self.model}")
    
    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        发送 Messages 请求
        """
        start_time = time.time()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message or "",
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            content = response.content[0].text
            duration = time.time() - start_time
            
            # 记录资源使用
            usage = {
                "model": self.model,
                "duration_seconds": round(duration, 3),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "call_count": 1,
                "system_message": system_message if system_message else None,
                "prompt": prompt,
                "response": content
            }
            self._save_resource_usage(usage)
            
            logger.debug(f"Anthropic 响应: {len(content)} 字符, tokens: {response.usage.input_tokens}/{response.usage.output_tokens}")
            return content
        except Exception as e:
            logger.error(f"Anthropic 请求失败: {e}")
            raise
    
    def complete_with_json(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        请求 JSON 格式响应
        """
        json_prompt = f"{prompt}\n\n请以有效的 JSON 格式返回结果,不要包含其他文本。"
        response_text = self.complete(json_prompt, system_message=system_message, **kwargs)
        
        # 尝试提取 JSON (Claude 有时会添加额外文本)
        try:
            # 尝试直接解析
            return json.loads(response_text)
        except json.JSONDecodeError:
            # 尝试提取 ```json ... ``` 代码块
            import re
            match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            else:
                logger.error(f"无法提取 JSON: {response_text}")
                raise


def create_llm_client(config: Dict[str, Any], output_dir: Optional[Path] = None) -> LLMClient:
    """
    工厂函数:根据配置创建 LLM 客户端
    
    Args:
        config: LLM 配置
        output_dir: 输出目录（用于保存资源记录）
        
    Returns:
        LLMClient 实例
        
    Raises:
        ValueError: 不支持的提供商
    """
    provider = config.get("provider", "openai").lower()
    
    if provider == "openai":
        return OpenAIClient(config, output_dir=output_dir)
    elif provider == "anthropic":
        return AnthropicClient(config, output_dir=output_dir)
    elif provider == "azure":
        # TODO: 实现 Azure OpenAI 客户端
        raise NotImplementedError("Azure OpenAI 尚未实现")
    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")
