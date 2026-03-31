"""
通用工具函数
"""
import json
import os
import shutil
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from git import Repo
from loguru import logger

from .exceptions import LLMResponseError


def robust_json_load(content: str) -> Any:
    """
    Robustly parse JSON string from LLM output, handling Markdown code blocks 
    and common escape issues.
    
    Args:
        content: The string containing JSON, potentially wrapped in markdown 
                 or containing invalid escapes.
                 
    Returns:
        Parsed JSON object (dict or list)
        
    Raises:
        LLMResponseError: If JSON cannot be parsed
    """
    if not content:
        raise LLMResponseError("Empty content provided for JSON parsing")
        
    cleaned_content = content.strip()
    
    # 1. Remove Markdown code blocks if present
    # Matches ```json ... ``` or just ``` ... ```
    code_block_pattern = r'^```(?:json)?\s+(.*?)\s+```$'
    match = re.search(code_block_pattern, cleaned_content, re.DOTALL)
    if match:
        cleaned_content = match.group(1)
    
    # 2. Extract JSON object/array if embedded in text
    # Finds the first outermost {} or []
    json_match = re.search(r'(\{.*\}|\[.*\])', cleaned_content, re.DOTALL)
    if json_match:
        cleaned_content = json_match.group(1).strip()
        
    # 3. Handle common escape issues (backslashes in file paths/regex)
    # Replace single backslashes that aren't part of valid JSON escapes with double backslashes
    # Valid JSON escapes: \, \", \/, \b, \f, \n, \r, \t, \uXXXX
    # We use negative lookbehind/lookahead to avoid breaking valid escapes
    processed_json = re.sub(r'(?<!\\)\\(?![nrtbf"\\/u])', r'\\\\', cleaned_content)
    
    try:
        return json.loads(processed_json)
    except json.JSONDecodeError:
        # If processing failed, try the original cleaned content
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse failed. Content preview: {cleaned_content[:200]}...")
            raise LLMResponseError(f"Failed to parse JSON response: {str(e)}") from e


def load_json(file_path: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
    """
    加载 JSON 文件
    
    Args:
        file_path: JSON 文件路径
        
    Returns:
        解析后的数据
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 格式错误
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    原子保存数据为 JSON 文件 (先写临时文件再 os.replace)

    Args:
        data: 待保存的数据
        file_path: 保存路径
        indent: 缩进空格数
    """
    import tempfile
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=file_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.debug(f"已保存 JSON 文件: {file_path}")


def clone_repository(repo_url: str, target_dir: Union[str, Path], commit: Optional[str] = None) -> Repo:
    """
    克隆 Git 仓库到本地 (增强版)
    
    Args:
        repo_url: 仓库 URL
        target_dir: 目标目录
        commit: 可选的提交 SHA,如果提供则 checkout 到该提交
        
    Returns:
        GitPython Repo 对象
    """
    target_dir = Path(target_dir)
    
    def _clean_and_clone():
        if target_dir.exists():
            logger.warning(f"清理无效仓库目录: {target_dir}")
            shutil.rmtree(target_dir)
        logger.info(f"克隆仓库: {repo_url} -> {target_dir}")
        return Repo.clone_from(repo_url, target_dir)

    repo = None
    if target_dir.exists():
        try:
            repo = Repo(target_dir)
            # 简单验证是否为有效 git 仓库
            _ = repo.git_dir
            logger.info(f"使用已有仓库: {target_dir}")

            def _has_commit(commit_sha: str) -> bool:
                """Return True if the repo already contains the commit object."""
                try:
                    _ = repo.commit(commit_sha)
                    return True
                except Exception:
                    return False

            # 只有在本地不存在目标 commit 时才尝试 fetch，避免网络不稳定导致的长时间卡顿
            should_fetch = False
            if commit:
                if not _has_commit(commit):
                    should_fetch = True
            else:
                should_fetch = True

            if should_fetch:
                try:
                    logger.info("Fetching updates...")
                    with repo.git.custom_environment(GIT_TERMINAL_PROMPT="0"):
                        repo.remotes.origin.fetch(kill_after_timeout=30)
                except Exception as fetch_err:
                    logger.warning(f"Fetch 跳过或失败: {fetch_err}")
        except Exception as e:
            logger.warning(f"已有目录不是有效仓库或更新失败: {e}")
            repo = _clean_and_clone()
    else:
        repo = _clean_and_clone()
    
    if commit:
        logger.info(f"切换到提交: {commit}")
        try:
            # 强制切换，丢弃之前的任何补丁修改
            repo.git.checkout(commit, force=True)
        except Exception as e:
            logger.error(f"切换提交失败: {e}")
            # 如果切换失败，可能是 fetch 不够或者仓库损坏，尝试重新 clone
            logger.warning("尝试重新克隆仓库...")
            repo = _clean_and_clone()
            repo.git.checkout(commit, force=True)
    
    return repo


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    确保目录存在,如果不存在则创建
    
    Args:
        dir_path: 目录路径
        
    Returns:
        Path 对象
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def parse_diff_hunks(patch: str) -> List[Dict[str, Any]]:
    """
    解析 unified diff 格式的补丁
    
    Args:
        patch: Diff 补丁字符串
        
    Returns:
        Hunk 列表,每个 hunk 包含文件路径、行号等信息
    """
    hunks = []
    current_file = None
    
    for line in patch.split("\n"):
        if line.startswith("--- a/"):
            current_file = line[6:]
        elif line.startswith("+++ b/"):
            pass
        elif line.startswith("@@"):
            # 解析行号范围 @@ -old_start,old_count +new_start,new_count @@
            parts = line.split("@@")[1].strip().split()
            old_range = parts[0][1:].split(",")
            new_range = parts[1][1:].split(",")
            
            hunks.append({
                "file": current_file,
                "old_start": int(old_range[0]),
                "old_count": int(old_range[1]) if len(old_range) > 1 else 1,
                "new_start": int(new_range[0]),
                "new_count": int(new_range[1]) if len(new_range) > 1 else 1,
            })
    
    return hunks


def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """
    原子保存数据为 pkl 文件 (先写临时文件再 os.replace，防止并发读到半截文件)

    Args:
        data: 待保存的数据（如 NetworkX 图对象）
        file_path: 保存路径
    """
    import tempfile
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入同目录临时文件，然后原子替换
    fd, tmp_path = tempfile.mkstemp(dir=file_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    except BaseException:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.debug(f"已保存 pkl 文件: {file_path}")


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    加载 pkl 文件
    
    Args:
        file_path: pkl 文件路径
        
    Returns:
        反序列化的对象
        
    Raises:
        FileNotFoundError: 文件不存在
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"pkl 文件不存在: {file_path}")
    
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    logger.debug(f"已加载 pkl 文件: {file_path}")
    return data
