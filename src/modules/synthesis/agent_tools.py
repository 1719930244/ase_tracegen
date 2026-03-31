"""
ToolsMixin - Tool/action handling, patch building, and test-related methods for SynthesisAgent.

Extracted from agent.py to reduce file size. Contains:
- LLM response parsing
- Patch generation (git-based and difflib-based)
- Code reading and fuzzy matching
- Chain parsing
- Test discovery, collection, and generation
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re
import json
import difflib

from loguru import logger

from .tools.context_tools import ReadCodeTool


class ToolsMixin:
    """
    Mixin providing tool/action handling, patch building, and test-related
    methods for SynthesisAgent.
    """

    def _parse_response(self, response: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parse LLM response to extract thought, action, and action input.
        支持更加灵活的格式（如 Markdown 标题, XML 标签）
        """
        thought = ""
        action = ""
        action_input = {}

        # 1. 尝试提取 Thought
        # 优先寻找 XML 标签
        thought_tag_match = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL | re.IGNORECASE)
        if thought_tag_match:
            thought = thought_tag_match.group(1).strip()
        else:
            # 回退到标准格式
            thought_match = re.search(r"(?:Thought|### Thought|\*\*Thought:\*\*):\s*(.+?)(?=(?:Action|### Action|\*\*Action:\*\*|<action>)|$)", response, re.DOTALL | re.IGNORECASE)
            if thought_match:
                thought = thought_match.group(1).strip()
            else:
                # 最后的保底：尝试提取 Action 之前的所有内容
                thought_fallback = re.split(r"(?:Action|### Action|\*\*Action:\*\*|<action>)", response, flags=re.IGNORECASE)[0].strip()
                if thought_fallback:
                    thought = thought_fallback

        # 2. 尝试提取 Action
        # 优先寻找 XML 标签
        action_tag_match = re.search(r"<action>(.*?)</action>", response, re.DOTALL | re.IGNORECASE)
        if action_tag_match:
            action = action_tag_match.group(1).strip()
        else:
            # 回退到标准格式
            action_match = re.search(r"(?:Action|### Action|\*\*Action:\*\*):\s*(\w+)", response, re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip()
            else:
                # 处理只有 <action> 开始标签的情况
                action_loose_match = re.search(r"<action>:?\s*(\w+)", response, re.IGNORECASE)
                if action_loose_match:
                    action = action_loose_match.group(1).strip()

        # 3. 尝试提取 Action Input
        all_json_strs = []

        # A. 优先寻找 XML 标签
        action_input_tag_match = re.search(r"<action_input>(.*?)</action_input>", response, re.DOTALL | re.IGNORECASE)
        if action_input_tag_match:
            all_json_strs.append(action_input_tag_match.group(1).strip())

        # B. 寻找 Markdown 代码块中的 JSON
        markdown_json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
        all_json_strs.extend(re.findall(markdown_json_pattern, response))

        # C. 寻找标准标签后的 JSON
        direct_input_pattern = r"(?:Action Input|### Action Input|\*\*Action Input:\*\*):\s*(\{[\s\S]*)"
        direct_matches = re.findall(direct_input_pattern, response, re.IGNORECASE)
        for dm in direct_matches:
            # 贪婪匹配最后一个 }
            last_brace = dm.rfind('}')
            if last_brace != -1:
                all_json_strs.append(dm[:last_brace+1])

        # D. 处理只有 <action_input> 开始标签的情况
        loose_input_pattern = r"<action_input>:?\s*(\{[\s\S]*)"
        loose_matches = re.findall(loose_input_pattern, response, re.IGNORECASE)
        for lm in loose_matches:
            last_brace = lm.rfind('}')
            if last_brace != -1:
                all_json_strs.append(lm[:last_brace+1])

        # 验证并解析找到的 JSON
        for json_str in reversed(all_json_strs):
            try:
                clean_json = json_str.strip()
                # 修复常见问题 1：多余的结尾括号
                # 尝试从最后一个 } 开始向前寻找能成功解析的最长子串
                parsed = None
                temp_json = clean_json
                while temp_json:
                    try:
                        # 修复 JSON 中的反斜杠转义
                        repaired_json = re.sub(r'(?<!\\)\\(?![nrtbf"\\/u])', r'\\\\', temp_json)
                        parsed = json.loads(repaired_json)
                        break
                    except json.JSONDecodeError:
                        last_brace = temp_json.rfind('}')
                        if last_brace <= 0:
                            break
                        temp_json = temp_json[:last_brace].strip()

                if isinstance(parsed, dict) and parsed:
                    # 修复常见问题 2：列表格式的代码片段（转回字符串）
                    for code_key in ["code_before", "code_after", "before", "after"]:
                        if code_key in parsed and isinstance(parsed[code_key], list):
                            parsed[code_key] = "\n".join(str(line) for line in parsed[code_key])

                    action_input = parsed
                    # 如果找到了包含关键字段的 JSON，则认为找到了正确的输入
                    if any(k in parsed for k in ["code_transformation", "before", "target_node_id", "node_id", "file_path", "target_file"]):
                        break
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

        return thought, action, action_input

    def _clean_code_snippet(self, code: str) -> str:
        """清理代码片段中的行号前缀和多余空白"""
        lines = code.splitlines()
        cleaned = []
        for line in lines:
            # 移除类似 " 123 | " 的前缀
            line = re.sub(r'^\s*\d+\s*\|\s*', '', line)
            cleaned.append(line.rstrip())
        return "\n".join(cleaned).strip()

    def _find_fuzzy_match(self, snippet: str, content: str, hint_line: Optional[int] = None) -> Optional[int]:
        """
        在内容中寻找代码片段的实际起始行号。
        支持高容错匹配，忽略空白、缩进差异。
        """
        def normalize(l):
            return re.sub(r'\s+', '', l).strip()

        snippet_lines = [normalize(l) for l in snippet.splitlines() if l.strip()]
        if not snippet_lines: return None

        content_lines = [normalize(l) for l in content.splitlines()]

        best_idx = -1
        max_score = 0
        n = len(snippet_lines)

        # 1. 如果有行号提示，优先检查提示位置附近的窗口 (+-10行)
        search_ranges = []
        if hint_line:
            h_idx = hint_line - 1
            search_ranges.append(range(max(0, h_idx - 10), min(len(content_lines), h_idx + 10)))

        # 全局扫描范围
        search_ranges.append(range(len(content_lines) - n + 1))

        for r in search_ranges:
            for i in r:
                score = 0
                for j in range(min(n, len(content_lines) - i)):
                    # 行内容完全一致
                    if snippet_lines[j] == content_lines[i+j]:
                        score += 1
                    # 包含关系（处理部分代码匹配）
                    elif snippet_lines[j] in content_lines[i+j] or content_lines[i+j] in snippet_lines[j]:
                        score += 0.5

                if score > max_score:
                    max_score = score
                    best_idx = i

                # 如果达到近乎完美的匹配，提前终止
                if score >= n - 0.1:
                    return i + 1

        # 2. 判定阈值：如果匹配得分超过 60%，则认为找到了正确位置
        if n > 0 and (max_score / n) >= 0.6:
            return best_idx + 1

        return None

    def _build_patch_system(self, file_path: str, before: str, after: str, start_line: int) -> str:
        """
        使用 SWE-Smith 风格的方法生成 patch。

        核心思路：直接修改文件，然后用 git diff 生成 patch，最后恢复文件。
        这样生成的 patch 保证格式正确且可以被 git apply 应用。

        Args:
            file_path: 目标文件路径（相对于仓库根目录）
            before: 原始代码（系统预读取）
            after: 修改后的代码（Agent 生成）
            start_line: 代码块在文件中的起始行号
        """
        import subprocess
        import tempfile

        if not self.repo_path:
            logger.warning("No repo_path set, falling back to difflib method")
            return self._build_patch_difflib(file_path, before, after, start_line)

        full_path = Path(self.repo_path) / file_path
        if not full_path.exists():
            logger.warning(f"File not found: {full_path}, falling back to difflib method")
            return self._build_patch_difflib(file_path, before, after, start_line)

        try:
            # 1. 读取完整文件内容
            with open(full_path, "r", encoding="utf-8") as f:
                original_lines = f.readlines()

            # 2. 计算结束行（基于 before 的行数）
            before_lines = before.splitlines(keepends=True)
            if not before.endswith('\n') and before_lines:
                before_lines[-1] += '\n'
            end_line = start_line + len(before_lines) - 1

            # 3. 准备替换内容
            after_lines = after.splitlines(keepends=True)
            if not after.endswith('\n') and after_lines:
                after_lines[-1] += '\n'

            # 4. 构建修改后的文件内容
            modified_lines = (
                original_lines[:start_line - 1] +
                after_lines +
                original_lines[end_line:]
            )

            # 5. 将当前文件状态（fixed baseline）加入 index 作为对比基线。
            try:
                subprocess.run(
                    ["git", "-C", self.repo_path, "add", file_path],
                    check=True,
                    capture_output=True
                )

                # 6. 写入修改后的内容（bug 注入后的 working tree）
                with open(full_path, "w", encoding="utf-8") as f:
                    f.writelines(modified_lines)

                # 获取 working tree diff（buggy vs fixed baseline）
                result = subprocess.run(
                    ["git", "-C", self.repo_path, "diff", "--", file_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                patch = result.stdout

            finally:
                # 7. 恢复文件（无论成功与否）
                subprocess.run(
                    ["git", "-C", self.repo_path, "restore", file_path],
                    capture_output=True
                )
                subprocess.run(
                    ["git", "-C", self.repo_path, "restore", "--staged", file_path],
                    capture_output=True
                )

            if patch and patch.strip():
                logger.info(f"SWE-Smith 风格 patch 生成成功: {len(patch)} bytes")
                return patch
            else:
                logger.warning("git diff 生成空 patch，代码可能没有变化")
                return ""

        except Exception as e:
            logger.error(f"SWE-Smith 风格 patch 生成失败: {e}")
            # 确保文件恢复
            try:
                subprocess.run(
                    ["git", "-C", self.repo_path, "restore", file_path],
                    capture_output=True
                )
            except OSError as e:
                logger.debug(f"git restore failed for {file_path}: {e}")
            # 回退到 difflib 方法
            return self._build_patch_difflib(file_path, before, after, start_line)

    def _build_patch_difflib(self, file_path: str, before: str, after: str, start_line: int) -> str:
        """
        使用 difflib 生成完整格式的 git patch（回退方法）。
        """
        import re

        # 补齐行尾换行符
        if not before.endswith('\n'): before += '\n'
        if not after.endswith('\n'): after += '\n'

        # 读取完整文件内容以获取正确的上下文
        full_before = before
        full_after = after

        if self.repo_path:
            full_path = Path(self.repo_path) / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        original_lines = f.readlines()

                    # 计算行范围
                    before_line_count = len(before.splitlines())
                    end_line = start_line + before_line_count - 1

                    # 添加上下文行（前后各3行）
                    ctx_start = max(1, start_line - 3)
                    ctx_end = min(len(original_lines), end_line + 3)

                    # 构建完整的 before（包括上下文）
                    before_with_ctx = original_lines[ctx_start-1:ctx_end]

                    # 构建完整的 after（包括上下文）
                    after_lines = after.splitlines(keepends=True)
                    if not after.endswith('\n') and after_lines:
                        after_lines[-1] += '\n'

                    after_with_ctx = (
                        original_lines[ctx_start-1:start_line-1] +
                        after_lines +
                        original_lines[end_line:ctx_end]
                    )

                    full_before = ''.join(before_with_ctx)
                    full_after = ''.join(after_with_ctx)
                    start_line = ctx_start  # 调整起始行号

                except Exception as e:
                    logger.warning(f"无法读取文件获取上下文: {e}")

        diff = difflib.unified_diff(
            full_before.splitlines(keepends=True),
            full_after.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=3
        )

        raw_patch = "".join(diff)
        if not raw_patch:
            return ""

        # 构建完整的 git patch 格式
        lines = raw_patch.splitlines()
        result_lines = [
            f"diff --git a/{file_path} b/{file_path}",
            f"--- a/{file_path}",
            f"+++ b/{file_path}",
        ]

        # 处理 hunk headers 和内容
        for line in lines:
            if line.startswith('---') or line.startswith('+++'):
                # 跳过 difflib 生成的这些行，我们已经添加了
                continue
            elif line.startswith('@@'):
                # 修正行号
                match = re.match(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@(.*)', line)
                if match:
                    orig_start = int(match.group(1))
                    orig_count = int(match.group(2))
                    new_start = int(match.group(3))
                    new_count = int(match.group(4))
                    context = match.group(5)  # 保留可能的上下文信息

                    # 计算实际文件行号
                    file_line_start = start_line + orig_start - 1

                    result_lines.append(f"@@ -{file_line_start},{orig_count} +{file_line_start},{new_count} @@{context}")
                else:
                    result_lines.append(line)
            else:
                result_lines.append(line)

        return "\n".join(result_lines) + "\n"

    def _read_node_code(self, node_id: str) -> str:
        """读取节点的代码"""
        try:
            logger.debug(f"Attempting to read code for node: {node_id}")

            # 检查图中是否有这个节点
            if self.graph and node_id in self.graph:
                node_data = self.graph.nodes[node_id]
                logger.debug(f"Node found in graph: {node_data}")
            else:
                logger.warning(f"Node {node_id} not found in graph")

            # 使用现有的ReadCodeTool逻辑
            tool = ReadCodeTool({"graph": self.graph, "repo_path": self.repo_path})
            result = tool.execute(node_id=node_id, include_line_numbers=False)

            if isinstance(result, dict):
                code = result.get("code", "")
            else:
                code = result

            logger.debug(f"Read code result length: {len(code)}")
            return code

        except Exception as e:
            logger.error(f"Failed to read code for {node_id}: {e}")
            return ""

    def _parse_structured_chain(self, chain_input: Any, seed_depth: int) -> List[Dict[str, Any]]:
        """
        将 Agent 提供的 proposed_chain 转换为带节点类型的结构化格式。

        根据 Seed 链路深度自动分配节点类型：
        - 第一个节点: symptom (测试/调用入口)
        - 最后一个节点: root_cause (Bug 注入位置)
        - 中间节点: intermediate
        """
        # 1. 标准化输入为列表
        if chain_input is None:
            return []

        if isinstance(chain_input, str):
            # 处理逗号分隔的字符串
            chain_list = [s.strip() for s in chain_input.split(",") if s.strip()]
        elif isinstance(chain_input, list):
            # 处理列表（可能是字符串列表或已结构化的字典列表）
            chain_list = []
            for item in chain_input:
                if isinstance(item, str):
                    chain_list.append(item.strip())
                elif isinstance(item, dict):
                    # 如果已经是结构化格式，直接使用
                    if "node_id" in item and "node_type" in item:
                        chain_list.append(item)
                    else:
                        # 尝试提取 node_id
                        node_id = item.get("node_id") or item.get("id") or str(item)
                        chain_list.append(node_id)
                else:
                    chain_list.append(str(item))
        else:
            chain_list = [str(chain_input)]

        # 2. 如果输入已经是结构化格式，直接返回
        if chain_list and isinstance(chain_list[0], dict) and "node_type" in chain_list[0]:
            return chain_list

        # 3. 截断或保留原长度（根据 seed_depth 调整）
        # 严格允许区间: [seed_depth, seed_depth+1]（delta=+1 实验证明更优）
        # 保底: seed_depth 至少为 2（symptom + root_cause 最小结构）
        effective_depth = max(seed_depth, 2)
        max_allowed = effective_depth + 1
        if len(chain_list) > max_allowed:
            logger.warning(
                f"proposed_chain 长度 ({len(chain_list)}) 超过允许上限 ({max_allowed})，"
                f"截断到 {max_allowed} 个节点"
            )
            chain_list = chain_list[:max_allowed]
        elif len(chain_list) < effective_depth and len(chain_list) > 0:
            logger.warning(
                f"proposed_chain 长度 ({len(chain_list)}) 少于 seed_depth ({effective_depth})，"
                f"链路可能不完整"
            )

        # 4. 构建结构化链路
        structured_chain = []
        for i, node_id in enumerate(chain_list):
            # 跳过已经是字典的项（不应该到达这里，但以防万一）
            if isinstance(node_id, dict):
                structured_chain.append(node_id)
                continue

            # 分配节点类型
            if i == 0:
                node_type = "symptom"
            elif i == len(chain_list) - 1:
                node_type = "root_cause"
            else:
                node_type = "intermediate"

            # 提取文件路径（如果节点 ID 包含文件信息）
            file_path = ""
            if ":" in node_id:
                file_path = node_id.split(":")[0]
            elif "/" in node_id and node_id.endswith(".py"):
                file_path = node_id

            structured_chain.append({
                "node_id": node_id,
                "node_type": node_type,
                "index": i + 1,
                "file_path": file_path,
            })

        return structured_chain

    def _get_recommended_tests(self, file_path: str) -> List[str]:
        """
        自动扫描目标文件附近的推荐回归测试
        """
        recommended = []
        try:
            p = Path(self.repo_path) / file_path
            # 1. 尝试在同级目录下找 test_*.py
            test_dir = p.parent
            test_files = list(test_dir.glob("test_*.py")) + list(test_dir.glob("tests.py"))

            # 2. 如果没找到，去上一级的 tests/ 目录下找
            if not test_files and test_dir.parent.joinpath("tests").exists():
                test_files = list(test_dir.parent.joinpath("tests").glob("*.py"))

            for tf in test_files[:3]: # 最多读3个文件防止过长
                with open(tf, "r", encoding="utf-8") as f:
                    content = f.read()
                    # 匹配 def test_...
                    matches = re.findall(r"def\s+(test_\w+)", content)
                    recommended.extend(matches[:5]) # 每个文件取前5个
        except Exception as e:
            logger.debug(f"推荐测试扫描失败: {e}")

        return list(set(recommended))[:10] # 总共返回前10个

    def _is_django_repo(self) -> bool:
        """Heuristic: detect Django repo for test-planning purposes."""
        if self.repo_profile is not None:
            return self.repo_profile.is_django
        repo = ""
        if self.current_seed and isinstance(self.current_seed.seed_metadata, dict):
            repo = (self.current_seed.seed_metadata.get("repo", "") or "").lower()
        return "django/django" in repo or repo.endswith("django/django") or repo.startswith("django/django")

    def _plan_validation_test_suite(self, file_path: str) -> Dict[str, Any]:
        """
        Pre-plan which existing test suite the validator is expected to run for this target.
        """
        if not self.repo_path:
            return {
                "is_django": self._is_django_repo(),
                "test_modules": [],
                "test_files": [],
                "test_cmd": "",
            }

        repo_root = Path(self.repo_path)

        # 优先使用 RepoProfile 的 plan_test_suite
        if self.repo_profile is not None:
            planned = self.repo_profile.plan_test_suite(file_path, repo_root)
            if not planned.get("test_files") and not planned.get("test_modules"):
                try:
                    test_files = self._get_nearby_test_files(file_path)
                    if test_files:
                        planned["test_files"] = sorted(test_files)[:20]
                        planned["test_modules"] = planned["test_files"]
                        planned["test_cmd"] = self.repo_profile.build_test_cmd(planned["test_files"])
                except (OSError, ValueError) as e:
                    logger.debug(f"Fallback test file discovery failed: {e}")
            return planned

        # Legacy fallback: 无 repo_profile 时使用旧逻辑
        planned: Dict[str, Any] = {
            "is_django": self._is_django_repo(),
            "test_modules": [],
            "test_files": [],
            "test_cmd": "",
        }
        try:
            test_files = self._get_nearby_test_files(file_path)
            if test_files:
                planned["test_files"] = sorted(test_files)[:20]
                planned["test_modules"] = planned["test_files"]
                planned["test_cmd"] = f"pytest {' '.join(planned['test_files'])} -v --tb=short"
        except (OSError, ValueError) as e:
            logger.debug(f"Legacy test file discovery failed: {e}")
        return planned

    def _get_nearby_test_files(self, file_path: str) -> List[str]:
        """获取目标文件附近的测试文件列表"""
        nearby_tests = []
        try:
            p = Path(self.repo_path) / file_path
            test_dir = p.parent
            # 搜寻同级和父级下的测试
            potential_dirs = [test_dir, test_dir.parent / "tests", test_dir.parent / "test"]
            for d in potential_dirs:
                if d.exists() and d.is_dir():
                    nearby_tests.extend([str(f.relative_to(self.repo_path)) for f in d.glob("test_*.py")])
                    nearby_tests.extend([str(f.relative_to(self.repo_path)) for f in d.glob("tests.py")])

            # 全局启发式：从仓库根目录的 tests/ 目录中按文件名/目录名搜索相关测试
            repo_root = Path(self.repo_path)
            stem = Path(file_path).stem
            if stem:
                for tests_root_name in ("tests", "test", "testing"):
                    tests_root = repo_root / tests_root_name
                    if not tests_root.exists():
                        continue

                    patterns = [
                        f"**/test_{stem}.py",
                        f"**/test_{stem}_*.py",
                        f"**/{stem}/tests.py",
                        f"**/{stem}/test*.py",
                        f"**/*{stem}*.py",
                    ]
                    for pat in patterns:
                        for tf in tests_root.glob(pat):
                            if tf.is_file():
                                nearby_tests.append(str(tf.relative_to(repo_root)))
        except (OSError, ValueError) as e:
            logger.debug(f"Nearby test file discovery error: {e}")
        return list(set(nearby_tests))[:50]

    def _collect_related_test_details(
        self,
        file_path: str,
        target_node_id: str,
        planned_test_files: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        收集与目标函数相关的测试用例信息，包括断言逻辑。
        """
        import ast

        related_tests: list[dict] = []
        if not self.repo_path:
            return related_tests

        func_name = target_node_id.split(":")[-1] if ":" in target_node_id else ""
        stem = Path(file_path).stem
        module_path = file_path.replace("/", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]

        keywords = [k for k in [func_name, stem, module_path] if k]
        if func_name and "." in func_name:
            for part in [p.strip() for p in func_name.split(".") if p.strip()]:
                keywords.append(part)

        target_markers: list[str] = []
        if func_name:
            target_markers.append(func_name)
            if "." in func_name:
                parts = [p.strip() for p in func_name.split(".") if p.strip()]
                for p in [parts[0], parts[-1]]:
                    if p:
                        target_markers.append(p)

        if func_name and func_name.endswith("_with_args"):
            base_name = func_name[: -len("_with_args")]
            if base_name:
                keywords.append(base_name)
                target_markers.append(base_name)

        # Loc-chain keywords
        try:
            seed_meta = self._get_seed_chain_meta()
            for nid in seed_meta.get("node_ids", []) or []:
                if not nid or not isinstance(nid, str):
                    continue
                nid = nid.strip()
                if not nid:
                    continue
                keywords.append(nid)
                if ":" in nid:
                    keywords.append(nid.split(":")[-1])
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Seed chain meta keyword extraction failed: {e}")
        candidate_files: list[str] = []
        candidate_files.extend(self._get_nearby_test_files(file_path))
        if planned_test_files:
            candidate_files.extend(planned_test_files)

        # 2) Extra hint: tests/<part>/... based on file path parts
        try:
            tests_dir = Path(self.repo_path) / "tests"
            if tests_dir.exists():
                for part in Path(file_path).parts:
                    test_subdir = tests_dir / part
                    if test_subdir.exists() and test_subdir.is_dir():
                        for tf in test_subdir.glob("*.py"):
                            candidate_files.append(str(tf.relative_to(self.repo_path)))
        except (OSError, ValueError) as e:
            logger.debug(f"Test subdir discovery error: {e}")
        repo_root = Path(self.repo_path)
        deduped: list[str] = []
        seen = set()
        for rel in candidate_files:
            if not rel or rel in seen:
                continue
            seen.add(rel)
            if not rel.endswith(".py"):
                continue
            if not (repo_root / rel).exists():
                continue
            deduped.append(rel)

        if not deduped:
            return related_tests

        # 3) Score files by keyword hits
        file_candidates: list[tuple[float, str, str, bool]] = []
        for rel in deduped[:120]:
            try:
                content = (repo_root / rel).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            score = 0.0
            for kw in keywords:
                if kw and kw in content:
                    score += min(content.count(kw), 20) * 1.0
            if stem and stem in rel:
                score += 2.0
            target_hit = any(m and m in content for m in target_markers)
            if target_hit:
                score += 6.0
            if score == 0.0 and planned_test_files and rel in planned_test_files:
                score = 0.5

            file_candidates.append((score, rel, content, target_hit))

        file_candidates.sort(key=lambda x: x[0], reverse=True)
        top_files = [t for t in file_candidates if t[0] > 0][:25]
        if not top_files:
            top_files = file_candidates[:10]

        def is_test_class_name(name: str) -> bool:
            if not name:
                return False
            return name.startswith("Test") or name.endswith("Tests") or ("Test" in name)

        # 4) Parse test cases and score them
        test_entries: list[tuple[float, dict]] = []
        for file_score, test_file, content, file_target_hit in top_files:
            try:
                tree = ast.parse(content)
            except Exception:
                continue

            # Collect marker-bound names
            marker_bound_names: set[str] = set()
            try:
                for top in getattr(tree, "body", []):
                    if not isinstance(top, (ast.Assign, ast.AnnAssign)):
                        continue
                    seg = ast.get_source_segment(content, top) or ""
                    if not seg:
                        continue
                    if not any(m and m in seg for m in target_markers):
                        continue

                    targets = []
                    if isinstance(top, ast.Assign):
                        targets = top.targets
                    elif isinstance(top, ast.AnnAssign) and top.target is not None:
                        targets = [top.target]
                    for tgt in targets:
                        if isinstance(tgt, ast.Name) and tgt.id:
                            marker_bound_names.add(tgt.id)
                        elif isinstance(tgt, (ast.Tuple, ast.List)):
                            for elt in getattr(tgt, "elts", []) or []:
                                if isinstance(elt, ast.Name) and elt.id:
                                    marker_bound_names.add(elt.id)
            except Exception:
                marker_bound_names = set()

            # Module-level functions
            for node in getattr(tree, "body", []):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    test_source = ast.get_source_segment(content, node) or ""
                    assertions = self._extract_assertions(node, content)
                    local_score = file_score + len(assertions) * 0.5
                    for kw in keywords:
                        if kw and kw in test_source:
                            local_score += 2.0
                    target_hit = False
                    if any(m and m in content for m in target_markers):
                        if any(m and m in test_source for m in target_markers):
                            target_hit = True
                        elif marker_bound_names:
                            for n in marker_bound_names:
                                if re.search(rf"\b{re.escape(n)}\b", test_source):
                                    target_hit = True
                                    break

                    if target_hit:
                        local_score += 8.0
                    test_entries.append(
                        (
                            local_score,
                            {
                                "test_file": test_file,
                                "test_class": "",
                                "test_method": node.name,
                                "full_path": f"{test_file}::{node.name}",
                                "source_preview": (test_source[:300] if test_source else ""),
                                "assertions": assertions[:5],
                                "target_hit": bool(target_hit),
                            },
                        )
                    )

                if not isinstance(node, ast.ClassDef):
                    continue

                if not is_test_class_name(node.name):
                    continue

                for item in node.body:
                    if not isinstance(item, ast.FunctionDef) or not item.name.startswith("test_"):
                        continue
                    test_source = ast.get_source_segment(content, item) or ""
                    assertions = self._extract_assertions(item, content)
                    local_score = file_score + len(assertions) * 0.5
                    for kw in keywords:
                        if kw and kw in test_source:
                            local_score += 2.0
                    target_hit = False
                    if any(m and m in content for m in target_markers):
                        if any(m and m in test_source for m in target_markers):
                            target_hit = True
                        elif marker_bound_names:
                            for n in marker_bound_names:
                                if re.search(rf"\b{re.escape(n)}\b", test_source):
                                    target_hit = True
                                    break
                    if target_hit:
                        local_score += 8.0
                    test_entries.append(
                        (
                            local_score,
                            {
                                "test_file": test_file,
                                "test_class": node.name,
                                "test_method": item.name,
                                "full_path": f"{test_file}::{node.name}::{item.name}",
                                "source_preview": (test_source[:300] if test_source else ""),
                                "assertions": assertions[:5],
                                "target_hit": bool(target_hit),
                            },
                        )
                    )

        # 5) Select top tests
        test_entries.sort(key=lambda x: x[0], reverse=True)
        seen_full_paths: set[str] = set()
        for _score, entry in test_entries:
            fp = entry.get("full_path", "")
            if not fp or fp in seen_full_paths:
                continue
            seen_full_paths.add(fp)
            related_tests.append(entry)
            if len(related_tests) >= 15:
                break

        return related_tests

    def _collect_test_cases_from_files(self, test_files: List[str], limit: int = 60) -> List[str]:
        """
        Collect pytest-style test case ids from a list of test files.
        """
        import ast

        if not self.repo_path or not test_files:
            return []

        repo_root = Path(self.repo_path)
        collected: list[str] = []
        seen: set[str] = set()

        def add(item: str) -> None:
            if not item or item in seen:
                return
            seen.add(item)
            collected.append(item)

        for rel in test_files:
            if len(collected) >= limit:
                break
            if not rel or not rel.endswith(".py"):
                continue
            path = repo_root / rel
            if not path.exists():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(content)
            except Exception:
                continue

            for node in getattr(tree, "body", []):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    assertions = self._extract_assertions(node, content)
                    if not assertions:
                        continue
                    add(f"{rel}::{node.name}")
                    if len(collected) >= limit:
                        break
                if isinstance(node, ast.ClassDef):
                    if node.name.startswith("Base") or node.name.endswith("Mixin"):
                        continue
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                            assertions = self._extract_assertions(item, content)
                            if not assertions:
                                continue
                            add(f"{rel}::{node.name}::{item.name}")
                            if len(collected) >= limit:
                                break

        return collected

    def _extract_assertions(self, func_node, content: str) -> List[str]:
        """从测试函数中提取断言语句。"""
        import ast
        assertions = []

        for stmt in ast.walk(func_node):
            try:
                # 处理 assert 语句
                if isinstance(stmt, ast.Assert):
                    assertion_src = ast.get_source_segment(content, stmt)
                    if assertion_src:
                        assertions.append(assertion_src.strip())
                # 处理 self.assert* 方法调用
                elif isinstance(stmt, ast.Call):
                    func = stmt.func
                    if isinstance(func, ast.Attribute):
                        if func.attr.startswith('assert'):
                            assertion_src = ast.get_source_segment(content, stmt)
                            if assertion_src:
                                assertions.append(assertion_src.strip())
                    elif isinstance(func, ast.Name):
                        if func.id.startswith('assert'):
                            assertion_src = ast.get_source_segment(content, stmt)
                            if assertion_src:
                                assertions.append(assertion_src.strip())
                # 处理 with self.assertRaises 等上下文管理器
                elif isinstance(stmt, ast.With):
                    for item in stmt.items:
                        if isinstance(item.context_expr, ast.Call):
                            call = item.context_expr
                            if isinstance(call.func, ast.Attribute):
                                if call.func.attr.startswith('assert'):
                                    assertion_src = ast.get_source_segment(content, item.context_expr)
                                    if assertion_src:
                                        assertions.append(assertion_src.strip())
            except Exception:
                continue

        return assertions

    def _generate_test_patch(
        self,
        test_code: str,
        target_node: str,
        file_path: str,
        instance_id: str,
        expected_failure_reason: str = "",
        bug_description: str = ""
    ) -> tuple:
        """
        将 Agent 生成的测试代码转换为 Django 测试格式的 patch。

        Returns:
            tuple: (test_patch, test_file_path, test_class_name, test_method_name)
        """
        if not test_code or not test_code.strip():
            test_code = self._generate_default_test_code(
                target_node, file_path, expected_failure_reason, bug_description
            )

        # 生成唯一的测试类名和方法名
        safe_instance_id = re.sub(r'[^a-zA-Z0-9]', '_', instance_id)
        test_class_name = f"SyntheticTest_{safe_instance_id[-20:]}"
        test_method_name = "test_synthetic_bug"

        # 确定测试文件路径
        test_file_path = self._determine_test_file_path(file_path)

        # 解析和清理测试代码
        parse_result = self._parse_and_clean_test_code(
            test_code, target_node, file_path, expected_failure_reason
        )

        # 处理返回值 (可能是 2 个或 3 个元素)
        if len(parse_result) == 3:
            imports, test_body, helper_code = parse_result
        else:
            imports, test_body = parse_result
            helper_code = ""

        # 生成完整的 Django 测试类
        full_test_content = self._generate_django_test_class(
            test_class_name=test_class_name,
            test_method_name=test_method_name,
            imports=imports,
            test_body=test_body,
            target_node=target_node,
            expected_failure_reason=expected_failure_reason,
            helper_code=helper_code
        )

        # 验证生成的代码是否有效
        if not self._validate_python_syntax(full_test_content):
            logger.warning("生成的测试代码语法无效，使用默认模板")
            full_test_content = self._generate_fallback_test(
                test_class_name, test_method_name, target_node,
                file_path, expected_failure_reason, bug_description
            )

        # 生成 patch (创建新文件)
        test_patch = self._generate_new_file_patch(test_file_path, full_test_content)

        return test_patch, test_file_path, test_class_name, test_method_name

    def _determine_test_file_path(self, file_path: str) -> str:
        """根据源文件路径确定测试文件路径。委托给 repo_profile。"""
        if self.repo_profile is not None:
            return self.repo_profile.get_test_file_path(file_path)
        return "tests/test_synthetic.py"

    def _generate_default_test_code(
        self,
        target_node: str,
        file_path: str,
        expected_failure_reason: str,
        bug_description: str
    ) -> str:
        """当没有测试代码时，生成一个默认的测试代码。"""
        func_name = target_node.split(":")[-1] if ":" in target_node else "target_function"
        module_path = file_path.replace("/", ".").replace(".py", "")

        return f'''
# 测试 {func_name} 的 bug
from {module_path} import {func_name}

# 预期失败: {expected_failure_reason or "AttributeError"}
result = {func_name}()
# 验证结果
assert result is not None
'''

    def _parse_and_clean_test_code(
        self,
        test_code: str,
        target_node: str,
        file_path: str,
        expected_failure_reason: str
    ) -> tuple:
        """
        解析并清理测试代码，提取导入语句和测试逻辑。

        Returns:
            tuple: (imports_list, test_body_str, helper_code_str)
        """
        import ast

        imports = set()
        helper_code_lines = []
        test_logic_lines = []

        # 添加基本的 Django 导入
        imports.add("from django.test import TestCase")

        # 从 file_path 推断需要的导入
        if file_path:
            module_path = file_path.replace("/", ".").replace(".py", "")
            func_name = target_node.split(":")[-1] if ":" in target_node else ""
            if func_name and not func_name.startswith("_"):
                imports.add(f"from {module_path} import {func_name}")

        # 尝试使用 AST 解析
        try:
            tree = ast.parse(test_code)

            for node in ast.iter_child_nodes(tree):
                node_source = ast.get_source_segment(test_code, node)
                if node_source is None:
                    continue

                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_line = node_source.strip()
                    if 'django.test' not in import_line:
                        imports.add(import_line)
                elif isinstance(node, ast.ClassDef):
                    helper_code_lines.append(node_source)
                elif isinstance(node, ast.FunctionDef):
                    helper_code_lines.append(node_source)
                elif isinstance(node, (ast.Expr, ast.Assign, ast.AugAssign)):
                    test_logic_lines.append(node_source)
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    test_logic_lines.append(node_source)
                elif isinstance(node, ast.Assert):
                    test_logic_lines.append(node_source)
                else:
                    test_logic_lines.append(node_source)

        except SyntaxError:
            logger.debug("AST 解析失败，使用行级解析")
            return self._parse_test_code_by_lines(
                test_code, target_node, file_path, expected_failure_reason
            )

        helper_code = '\n\n'.join(helper_code_lines) if helper_code_lines else ""

        if test_logic_lines:
            test_body = '\n'.join(test_logic_lines)
        else:
            func_name = target_node.split(":")[-1] if ":" in target_node else "target"
            test_body = f"# Test for {func_name}\npass"

        return list(imports), test_body, helper_code

    def _parse_test_code_by_lines(
        self,
        test_code: str,
        target_node: str,
        file_path: str,
        expected_failure_reason: str
    ) -> tuple:
        """
        行级解析测试代码（AST 失败时的后备方案）。

        Returns:
            tuple: (imports_list, test_body_str, helper_code_str)
        """
        imports = set()
        imports.add("from django.test import TestCase")

        if file_path:
            module_path = file_path.replace("/", ".").replace(".py", "")
            func_name = target_node.split(":")[-1] if ":" in target_node else ""
            if func_name and not func_name.startswith("_"):
                imports.add(f"from {module_path} import {func_name}")

        lines = test_code.strip().split('\n')
        helper_blocks = []
        test_lines = []
        current_block = []
        in_class_or_func = False
        block_indent = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip())

            if stripped.startswith('import ') or stripped.startswith('from '):
                if 'django.test' not in stripped:
                    imports.add(stripped)
                continue

            if stripped.startswith('class ') or stripped.startswith('def '):
                if in_class_or_func and current_block:
                    helper_blocks.append('\n'.join(current_block))
                    current_block = []

                in_class_or_func = True
                block_indent = current_indent
                current_block.append(line)
                continue

            if in_class_or_func:
                if stripped and current_indent <= block_indent:
                    if current_block:
                        helper_blocks.append('\n'.join(current_block))
                        current_block = []
                    in_class_or_func = False
                    if stripped:
                        test_lines.append(stripped)
                else:
                    current_block.append(line)
                continue

            if stripped:
                test_lines.append(stripped)

        if current_block:
            helper_blocks.append('\n'.join(current_block))

        helper_code = '\n\n'.join(helper_blocks) if helper_blocks else ""
        test_body = '\n'.join(test_lines) if test_lines else "pass"

        return list(imports), test_body, helper_code

    def _generate_django_test_class(
        self,
        test_class_name: str,
        test_method_name: str,
        imports: list,
        test_body: str,
        target_node: str,
        expected_failure_reason: str,
        helper_code: str = ""
    ) -> str:
        """
        生成测试类/函数。委托给 repo_profile.generate_test_class()。
        """
        if self.repo_profile is not None:
            return self.repo_profile.generate_test_class(
                test_class_name=test_class_name,
                test_method_name=test_method_name,
                imports=imports,
                test_body=test_body,
                target_node=target_node,
                expected_failure_reason=expected_failure_reason,
                helper_code=helper_code,
            )

        # Legacy fallback: Django TestCase 格式
        imports_str = '\n'.join(sorted(set(imports)))

        indented_body_lines = []
        for line in test_body.split('\n'):
            if line.strip():
                stripped = line.lstrip()
                orig_indent = len(line) - len(stripped)
                indented_body_lines.append('        ' + ' ' * orig_indent + stripped)
            else:
                indented_body_lines.append('')
        indented_body = '\n'.join(indented_body_lines)

        if not indented_body.strip() or indented_body.strip() == 'pass':
            func_name = target_node.split(":")[-1] if ":" in target_node else "target"
            indented_body = f'''        # Synthetic test for {func_name}
        # This should trigger a failure when the bug is present
        self.assertTrue(True, "Placeholder assertion - bug should cause failure")'''

        helper_section = ""
        if helper_code and helper_code.strip():
            helper_section = f"\n\n# Helper classes and functions for testing\n{helper_code}\n"

        return f'''"""
Synthetic test case generated by TraceGen.
Target: {target_node}
Expected failure: {expected_failure_reason}
"""
{imports_str}{helper_section}


class {test_class_name}(TestCase):
    """
    Synthetic test to verify bug injection.
    This test should FAIL when the bug is present.
    """

    def {test_method_name}(self):
        """
        Test that verifies the synthetic bug.
        Expected: {expected_failure_reason or 'Test should fail due to injected bug'}
        """
{indented_body}
'''

    def _generate_fallback_test(
        self,
        test_class_name: str,
        test_method_name: str,
        target_node: str,
        file_path: str,
        expected_failure_reason: str,
        bug_description: str = ""
    ) -> str:
        """生成一个保证语法正确的后备测试。"""
        func_name = target_node.split(":")[-1] if ":" in target_node else "target_function"
        module_path = file_path.replace("/", ".").replace(".py", "") if file_path else "module"
        failure_lower = (expected_failure_reason or "").lower()
        bug_lower = (bug_description or "").lower()

        def kw_check(kws):
            return any(kw in failure_lower or kw in bug_lower for kw in kws)

        if kw_check(["decorator", "__name__", "__wrapped__", "functools", "wraps", "attribute"]):
            test_body = (
                f'        from {module_path} import {func_name}\n'
                f'        def sample_func():\n'
                f'            return "test result"\n'
                f'        try:\n'
                f'            if callable({func_name}):\n'
                f'                result = {func_name}(sample_func)\n'
                f'                self.assertTrue(hasattr(result, "__name__"))\n'
                f'        except AttributeError as e:\n'
                f'            self.fail(f"Missing attribute: {{e}}")\n'
                f'        except Exception as e:\n'
                f'            self.fail(f"Unexpected error: {{e}}")'
            )
        elif kw_check(["type", "cast", "convert", "isinstance", "typeerror"]):
            test_body = (
                f'        from {module_path} import {func_name}\n'
                f'        for test_input in [None, "", 0]:\n'
                f'            try:\n'
                f'                if callable({func_name}):\n'
                f'                    result = {func_name}(test_input)\n'
                f'                    self.assertIsNotNone(result)\n'
                f'            except TypeError as e:\n'
                f'                self.fail(f"Type error: {{e}}")\n'
                f'            except Exception:\n'
                f'                pass'
            )
        elif kw_check(["boundary", "off-by-one", "index", "range", "<=", ">=", "<", ">"]):
            test_body = (
                f'        from {module_path} import {func_name}\n'
                f'        for value in [0, 1, -1, 100]:\n'
                f'            try:\n'
                f'                if callable({func_name}):\n'
                f'                    result = {func_name}(value)\n'
                f'                    self.assertIsNotNone(result)\n'
                f'            except (IndexError, ValueError) as e:\n'
                f'                self.fail(f"Boundary error: {{e}}")'
            )
        else:
            test_body = (
                f'        try:\n'
                f'            from {module_path} import {func_name}\n'
                f'        except ImportError:\n'
                f'            self.skipTest("Cannot import target")\n'
                f'            return\n'
                f'        try:\n'
                f'            if callable({func_name}):\n'
                f'                result = {func_name}\n'
                f'                if hasattr(result, "__name__"):\n'
                f'                    self.assertIsNotNone(result.__name__)\n'
                f'        except AttributeError as e:\n'
                f'            self.fail(f"Missing attribute: {{e}}")\n'
                f'        except Exception as e:\n'
                f'            self.fail(f"Error during test: {{e}}")'
            )

        return (
            f'"""\nSynthetic test case generated by TraceGen.\n'
            f'Target: {target_node}\nExpected failure: {expected_failure_reason}\n"""\n'
            f'from django.test import TestCase\n\n\n'
            f'class {test_class_name}(TestCase):\n'
            f'    def {test_method_name}(self):\n'
            f'{test_body}\n'
        )

    def _generate_new_file_patch(self, file_path: str, content: str) -> str:
        """生成创建新文件的 patch。"""
        lines = content.split('\n')
        diff_lines = [
            f"diff --git a/{file_path} b/{file_path}",
            "new file mode 100644",
            f"--- /dev/null",
            f"+++ b/{file_path}",
            f"@@ -0,0 +1,{len(lines)} @@",
        ]
        for line in lines:
            diff_lines.append(f"+{line}")
        return '\n'.join(diff_lines) + '\n'
