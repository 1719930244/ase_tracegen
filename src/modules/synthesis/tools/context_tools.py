"""
Context tools for synthesis agent.
合成 Agent 的上下文工具
"""

from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger
import json

from .base import BaseTool


class ReadCodeTool(BaseTool):
    """
    Tool to read source code from the repository.
    支持按节点（方法/类）或按文件范围读取代码。
    """
    name = "read_code"
    description = "Read source code. Recommended: use node_id to read a specific function or class."
    parameters = {
        "node_id": "string (optional, e.g., 'path/to/file.py:MyClass.my_method')",
        "file_path": "string (required if node_id is not provided)",
        "start_line": "integer (optional)",
        "end_line": "integer (optional)",
        "include_line_numbers": "boolean (optional, default: false)"
    }

    def execute(
        self, 
        node_id: Optional[str] = None, 
        file_path: Optional[str] = None, 
        start_line: Optional[int] = None, 
        end_line: Optional[int] = None,
        include_line_numbers: bool = False
    ) -> str:
        if not self.repo_path:
            return "Error: Repository path not set."
        
        graph = self.context.get("graph")
        
        # 1. 优先处理 node_id (语义化读取)
        if node_id:
            if graph and node_id in graph:
                node_data = graph.nodes[node_id]
                # 兼容性处理：如果 file_path 为空，尝试从 node_id 解析
                file_path = node_data.get("file_path") or (node_id.split(":")[0] if ":" in node_id else "")
                line_range = node_data.get("line_range")
                if line_range:
                    start_line, end_line = line_range
            else:
                # 关键改进：如果给了 node_id 但图中找不到（可能是缓存问题或 node_id 格式略有差异）
                # 不要报错，尝试自愈提取 file_path
                if ":" in node_id:
                    file_path = node_id.split(":")[0]
                    logger.info(f"node_id {node_id} 不在图中，降级为读取文件 {file_path}")
                else:
                    file_path = node_id

        if not file_path:
            return "Error: Either node_id or file_path must be provided."

        full_path = Path(self.repo_path) / file_path
        if not full_path.exists():
            return f"Error: File not found: {file_path}"
        
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
            
            total_lines = len(all_lines)
            
            # 2. 智能识别：如果提供了行号但没有 node_id，或者虽然有 node_id 但没拿到行范围
            if start_line and (not end_line or end_line - start_line < 5) and graph:
                # 寻找包含该行的最具体实体（函数优先于类）
                best_node = None
                best_range = None
                
                for nid, data in graph.nodes(data=True):
                    # 只有当节点的 file_path 匹配或者是 node_id 的一部分时才考虑
                    n_file = data.get("file_path") or (nid.split(":")[0] if ":" in nid else "")
                    if n_file == file_path:
                        l_range = data.get("line_range")
                        if l_range and l_range[0] <= start_line <= l_range[1]:
                            # 优先选择范围更小的（通常是方法而不是类）
                            if best_range is None or (l_range[1] - l_range[0] < best_range[1] - best_range[0]):
                                best_node = nid
                                best_range = l_range
                
                if best_node:
                    node_id = best_node
                    start_line, end_line = best_range
                    logger.info(f"智能识别行 {start_line} 属于实体: {node_id}")

            # 确定读取范围
            s_idx = max(0, (start_line - 1) if start_line else 0)
            e_idx = min(total_lines, end_line if end_line else total_lines)
            
            # 如果依然没确定范围且文件太长，则截断
            if not start_line and not end_line and total_lines > 300:
                e_idx = 300
                warning = f"\n(Note: Showing first 300 lines. Use node_id or start_line/end_line for specific parts.)"
            else:
                warning = ""
            
            selected_lines = all_lines[s_idx:e_idx]
            
            # 格式化输出：带行号参考
            output_lines = []
            for i, line in enumerate(selected_lines):
                if include_line_numbers:
                    curr_line_num = s_idx + i + 1
                    output_lines.append(f"{curr_line_num:4} | {line}")
                else:
                    output_lines.append(line)
            
            content = "".join(output_lines)

            # 构建输出头部
            header = f"--- Source: {file_path}"
            if node_id:
                header += f" (Entity: {node_id})"
            header += f" (Lines {s_idx+1}-{e_idx} of {total_lines}) ---\n"

            # 添加清晰的代码区域标记，帮助 Agent 正确复制代码
            code_start_marker = "=== CODE START (Copy from here) ===\n"
            code_end_marker = "\n=== CODE END (Copy until here) ==="

            return header + code_start_marker + content + code_end_marker + warning
        except Exception as e:
            return f"Error reading code: {str(e)}"
