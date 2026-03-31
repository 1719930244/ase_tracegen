"""
Code Graph Builder - Core Logic
Extracted and refactored from LocAgent's dependency_graph module
"""
import ast
import hashlib
import os
from collections import defaultdict
from typing import List, Optional

import networkx as nx
from loguru import logger
from tqdm import tqdm

# Graph Configuration
VERSION = 'v2.3'
NODE_TYPE_DIRECTORY = 'directory'
NODE_TYPE_FILE = 'file'
NODE_TYPE_CLASS = 'class'
NODE_TYPE_FUNCTION = 'function'
EDGE_TYPE_CONTAINS = 'contains'
EDGE_TYPE_INHERITS = 'inherits'
EDGE_TYPE_INVOKES = 'invokes'
EDGE_TYPE_IMPORTS = 'imports'

VALID_NODE_TYPES = [NODE_TYPE_DIRECTORY, NODE_TYPE_FILE, NODE_TYPE_CLASS, NODE_TYPE_FUNCTION]
VALID_EDGE_TYPES = [EDGE_TYPE_CONTAINS, EDGE_TYPE_INHERITS, EDGE_TYPE_INVOKES, EDGE_TYPE_IMPORTS]

SKIP_DIRS = ['.github', '.git', '__pycache__', '.pytest_cache', 'node_modules']


def is_skip_dir(dirname: str) -> bool:
    """Check if directory should be skipped"""
    for skip_dir in SKIP_DIRS:
        if skip_dir in dirname:
            return True
    return False


class CodeAnalyzer(ast.NodeVisitor):
    """AST-based code analyzer to extract classes and functions"""
    _ast_cache = {}

    def __init__(self, filename: str):
        self.filename = filename
        self.nodes = []
        self.node_name_stack = []
        self.node_type_stack = []

    def visit_ClassDef(self, node):
        class_name = node.name
        full_class_name = '.'.join(self.node_name_stack + [class_name])
        self.nodes.append({
            'name': full_class_name,
            'type': NODE_TYPE_CLASS,
            'code': self._get_source_segment(node),
            'start_line': node.lineno,
            'end_line': node.end_lineno,
        })

        self.node_name_stack.append(class_name)
        self.node_type_stack.append(NODE_TYPE_CLASS)
        self.generic_visit(node)
        self.node_name_stack.pop()
        self.node_type_stack.pop()

    def visit_FunctionDef(self, node):
        if self.node_type_stack and self.node_type_stack[-1] == NODE_TYPE_CLASS and node.name == '__init__':
            return
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_func(node)

    def _visit_func(self, node):
        function_name = node.name
        full_function_name = '.'.join(self.node_name_stack + [function_name])
        self.nodes.append({
            'name': full_function_name,
            'parent_type': self.node_type_stack[-1] if self.node_type_stack else None,
            'type': NODE_TYPE_FUNCTION,
            'code': self._get_source_segment(node),
            'start_line': node.lineno,
            'end_line': node.end_lineno,
        })

        self.node_name_stack.append(function_name)
        self.node_type_stack.append(NODE_TYPE_FUNCTION)
        self.generic_visit(node)
        self.node_name_stack.pop()
        self.node_type_stack.pop()

    def _get_source_segment(self, node):
        try:
            with open(self.filename, 'r', encoding='utf-8') as file:
                source_code = file.read()
            return ast.get_source_segment(source_code, node)
        except (IndexError, TypeError):
            # 当 AST 节点的行号超出文件实际行数时返回 None
            return None


def analyze_file(filepath: str) -> List[dict]:
    """Parse file and extract classes/functions using AST"""
    # 每次都重新读取文件并解析，避免缓存导致的行号不匹配
    with open(filepath, 'r', encoding='utf-8') as file:
        code = file.read()
    try:
        tree = ast.parse(code, filename=filepath)
        CodeAnalyzer._ast_cache[filepath] = tree
    except SyntaxError:
        raise

    analyzer = CodeAnalyzer(filepath)
    try:
        analyzer.visit(tree)
    except (RecursionError, IndexError):
        # RecursionError: 深度嵌套的 AST
        # IndexError: get_source_segment 行号越界
        pass
    return analyzer.nodes


def find_imports(filepath: str, repo_path: str, tree=None) -> List[dict]:
    """Extract import statements from a file"""
    if tree is None:
        if filepath in CodeAnalyzer._ast_cache:
            tree = CodeAnalyzer._ast_cache[filepath]
        else:
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    tree = ast.parse(file.read(), filename=filepath)
                    CodeAnalyzer._ast_cache[filepath] = tree
            except (SyntaxError, OSError, UnicodeDecodeError):
                raise SyntaxError
        candidates = ast.walk(tree)
    else:
        candidates = ast.iter_child_nodes(tree)

    imports = []
    for node in candidates:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                asname = alias.asname
                imports.append({
                    "type": "import",
                    "module": module_name,
                    "alias": asname
                })
        elif isinstance(node, ast.ImportFrom):
            import_entities = []
            for alias in node.names:
                if alias.name == '*':
                    import_entities = [{'name': '*', 'alias': None}]
                    break
                else:
                    entity_name = alias.name
                    asname = alias.asname
                    import_entities.append({
                        "name": entity_name,
                        "alias": asname
                    })

            if node.level == 0:
                module_name = node.module
            else:
                rel_path = os.path.relpath(filepath, repo_path)
                package_parts = rel_path.split(os.sep)
                if len(package_parts) >= node.level:
                    package_parts = package_parts[:-node.level]
                else:
                    package_parts = []
                if node.module:
                    module_name = '.'.join(package_parts + [node.module])
                else:
                    module_name = '.'.join(package_parts)

            imports.append({
                "type": "from",
                "module": module_name,
                "entities": import_entities
            })
    return imports


def resolve_module(module_name: str, repo_path: str) -> Optional[str]:
    """Resolve module name to file path"""
    module_path = os.path.join(repo_path, module_name.replace('.', '/') + '.py')
    if os.path.isfile(module_path):
        return module_path

    init_path = os.path.join(repo_path, module_name.replace('.', '/'), '__init__.py')
    if os.path.isfile(init_path):
        return init_path

    return None


def add_imports(root_node: str, imports: List[dict], graph: nx.MultiDiGraph, repo_path: str):
    """Add import edges to graph"""
    for imp in imports:
        if imp['type'] == 'import':
            module_name = imp['module']
            module_path = resolve_module(module_name, repo_path)
            if module_path:
                imp_filename = os.path.relpath(module_path, repo_path)
                if graph.has_node(imp_filename):
                    graph.add_edge(root_node, imp_filename, type=EDGE_TYPE_IMPORTS, alias=imp['alias'])
        elif imp['type'] == 'from':
            module_name = imp['module']
            entities = imp['entities']

            if len(entities) == 1 and entities[0]['name'] == '*':
                module_path = resolve_module(module_name, repo_path)
                if module_path:
                    imp_filename = os.path.relpath(module_path, repo_path)
                    if graph.has_node(imp_filename):
                        graph.add_edge(root_node, imp_filename, type=EDGE_TYPE_IMPORTS, alias=None)
                continue

            for entity in entities:
                entity_name, entity_alias = entity['name'], entity['alias']
                entity_module_name = f"{module_name}.{entity_name}"
                entity_module_path = resolve_module(entity_module_name, repo_path)
                if entity_module_path:
                    entity_filename = os.path.relpath(entity_module_path, repo_path)
                    if graph.has_node(entity_filename):
                        graph.add_edge(root_node, entity_filename, type=EDGE_TYPE_IMPORTS, alias=entity_alias)
                else:
                    module_path = resolve_module(module_name, repo_path)
                    if module_path:
                        imp_filename = os.path.relpath(module_path, repo_path)
                        node = f"{imp_filename}:{entity_name}"
                        if graph.has_node(node):
                            graph.add_edge(root_node, node, type=EDGE_TYPE_IMPORTS, alias=entity_alias)
                        elif graph.has_node(imp_filename):
                            graph.add_edge(root_node, imp_filename, type=EDGE_TYPE_IMPORTS, alias=entity_alias)


def get_inner_nodes(query_node: str, src_node: str, graph: nx.MultiDiGraph) -> List[str]:
    """Get all inner nodes (contained within src_node)"""
    inner_nodes = []
    if not graph.has_node(src_node):
        return inner_nodes
    for _, dst_node, attr in graph.edges(src_node, data=True):
        if attr['type'] == EDGE_TYPE_CONTAINS and dst_node != query_node:
            inner_nodes.append(dst_node)
            dst_data = graph.nodes.get(dst_node)
            if dst_data and dst_data.get('type') == NODE_TYPE_CLASS:
                inner_nodes.extend(get_inner_nodes(query_node, dst_node, graph))
    return inner_nodes


def find_all_possible_callee(node: str, graph: nx.MultiDiGraph) -> tuple:
    """Find all possible callees from a given node"""
    callee_nodes, callee_alias = [], {}
    cur_node = node
    pre_node = node

    def find_parent(_cur_node):
        for predecessor in graph.predecessors(_cur_node):
            edge_data = graph.get_edge_data(predecessor, _cur_node)
            if edge_data is None:
                continue
            for key, attr in edge_data.items():
                if attr['type'] == EDGE_TYPE_CONTAINS:
                    return predecessor
        return None  # 显式返回 None

    while True:
        callee_nodes.extend(get_inner_nodes(pre_node, cur_node, graph))

        # 检查节点是否存在且有 type 属性
        node_data = graph.nodes.get(cur_node)
        if node_data is None or 'type' not in node_data:
            break  # 节点不存在或缺少属性，退出循环

        if node_data['type'] == NODE_TYPE_FILE:
            file_list = []
            file_stack = [cur_node]
            while len(file_stack) > 0:
                for _, dst_node, attr in graph.edges(file_stack.pop(), data=True):
                    if attr['type'] == EDGE_TYPE_IMPORTS and dst_node not in file_list + [cur_node]:
                        dst_data = graph.nodes.get(dst_node)
                        if dst_data and dst_data.get('type') == NODE_TYPE_FILE and dst_node.endswith('__init__.py'):
                            file_list.append(dst_node)
                            file_stack.append(dst_node)

            for file in file_list:
                callee_nodes.extend(get_inner_nodes(cur_node, file, graph))
                for _, dst_node, attr in graph.edges(file, data=True):
                    if attr['type'] == EDGE_TYPE_IMPORTS:
                        if attr['alias'] is not None:
                            callee_alias[attr['alias']] = dst_node
                        dst_data = graph.nodes.get(dst_node)
                        if dst_data:
                            dst_type = dst_data.get('type')
                            if dst_type in [NODE_TYPE_FILE, NODE_TYPE_CLASS]:
                                callee_nodes.extend(get_inner_nodes(file, dst_node, graph))
                            if dst_type in [NODE_TYPE_FUNCTION, NODE_TYPE_CLASS]:
                                callee_nodes.append(dst_node)

            for _, dst_node, attr in graph.edges(cur_node, data=True):
                if attr['type'] == EDGE_TYPE_IMPORTS:
                    if attr['alias'] is not None:
                        callee_alias[attr['alias']] = dst_node
                    dst_data = graph.nodes.get(dst_node)
                    if dst_data:
                        dst_type = dst_data.get('type')
                        if dst_type in [NODE_TYPE_FILE, NODE_TYPE_CLASS]:
                            callee_nodes.extend(get_inner_nodes(cur_node, dst_node, graph))
                        if dst_type in [NODE_TYPE_FUNCTION, NODE_TYPE_CLASS]:
                            callee_nodes.append(dst_node)

            break

        pre_node = cur_node
        cur_node = find_parent(cur_node)
        if cur_node is None:
            break  # 找不到父节点，退出循环

    return callee_nodes, callee_alias


def analyze_init(node: str, code_tree: ast.AST, graph: nx.MultiDiGraph, repo_path: str) -> tuple:
    """Analyze __init__ method of a class"""
    caller_name = node.split(':')[-1].split('.')[-1]
    file_path = os.path.join(repo_path, node.split(':')[0])

    invocations = []
    inheritances = []

    def add_invoke(func_name):
        invocations.append(func_name)

    def add_inheritance(class_name):
        inheritances.append(class_name)

    def process_decorator_node(_decorator_node):
        if isinstance(_decorator_node, ast.Name):
            add_invoke(_decorator_node.id)
        else:
            for _sub_node in ast.walk(_decorator_node):
                if isinstance(_sub_node, ast.Call) and isinstance(_sub_node.func, ast.Name):
                    add_invoke(_sub_node.func.id)
                elif isinstance(_sub_node, ast.Attribute):
                    add_invoke(_sub_node.attr)

    def process_inheritance_node(_inheritance_node):
        if isinstance(_inheritance_node, ast.Attribute):
            add_inheritance(_inheritance_node.attr)
        if isinstance(_inheritance_node, ast.Name):
            add_inheritance(_inheritance_node.id)

    for ast_node in ast.walk(code_tree):
        if isinstance(ast_node, ast.ClassDef) and ast_node.name == caller_name:
            imports = find_imports(file_path, repo_path, tree=ast_node)
            add_imports(node, imports, graph, repo_path)

            for inheritance_node in ast_node.bases:
                process_inheritance_node(inheritance_node)

            for decorator_node in ast_node.decorator_list:
                process_decorator_node(decorator_node)

            for body_item in ast_node.body:
                if isinstance(body_item, ast.FunctionDef) and body_item.name == '__init__':
                    imports = find_imports(file_path, repo_path, tree=body_item)
                    add_imports(node, imports, graph, repo_path)

                    for decorator_node in body_item.decorator_list:
                        process_decorator_node(decorator_node)

                    for sub_node in ast.walk(body_item):
                        if isinstance(sub_node, ast.Call):
                            if isinstance(sub_node.func, ast.Name):
                                add_invoke(sub_node.func.id)
                            if isinstance(sub_node.func, ast.Attribute):
                                add_invoke(sub_node.func.attr)
                    break
            break

    return invocations, inheritances


def analyze_invokes(node: str, code_tree: ast.AST, graph: nx.MultiDiGraph, repo_path: str) -> List[str]:
    """Analyze function invocations"""
    caller_name = node.split(':')[-1].split('.')[-1]
    file_path = os.path.join(repo_path, node.split(':')[0])

    invocations = []

    def add_invoke(func_name):
        invocations.append(func_name)

    def process_decorator_node(_decorator_node):
        if isinstance(_decorator_node, ast.Name):
            add_invoke(_decorator_node.id)
        else:
            for _sub_node in ast.walk(_decorator_node):
                if isinstance(_sub_node, ast.Call) and isinstance(_sub_node.func, ast.Name):
                    add_invoke(_sub_node.func.id)
                elif isinstance(_sub_node, ast.Attribute):
                    add_invoke(_sub_node.attr)

    def traverse_call(_node):
        for child in ast.iter_child_nodes(_node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    add_invoke(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    add_invoke(child.func.attr)
            traverse_call(child)

    for ast_node in ast.walk(code_tree):
        if (isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and ast_node.name == caller_name):
            imports = find_imports(file_path, repo_path, tree=ast_node)
            add_imports(node, imports, graph, repo_path)

            for decorator_node in ast_node.decorator_list:
                process_decorator_node(decorator_node)

            traverse_call(ast_node)
            break

    return invocations


def compute_hash(content: str) -> str:
    """Compute MD5 hash of string content"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def build_graph_from_repo(repo_path: str, fuzzy_search: bool = True, global_import: bool = False, base_graph: Optional[nx.MultiDiGraph] = None) -> nx.MultiDiGraph:
    """
    Build a code knowledge graph from a repository, optionally reusing nodes from a base graph.

    Args:
        repo_path: Path to the code repository
        fuzzy_search: Enable fuzzy matching for function calls
        global_import: Enable global name resolution
        base_graph: Previous graph to reuse nodes from (incremental build)

    Returns:
        NetworkX MultiDiGraph representing the code structure
    """
    graph = nx.MultiDiGraph()
    file_nodes = {}

    # Pre-process base graph for reuse
    reusable_nodes = {} # file_path -> list of node_ids
    file_hashes = {}    # file_path -> hash

    if base_graph:
        # Index nodes by file
        for node, data in base_graph.nodes(data=True):
            if data.get('type') == NODE_TYPE_FILE:
                # Store hash if available, or compute it from code
                f_hash = data.get('hash')
                if not f_hash and 'code' in data:
                    f_hash = compute_hash(data['code'])
                file_hashes[node] = f_hash

            # Map file path to all its contained nodes
            # Note: node names in graph are relative paths or "path:func"
            if ':' in node:
                f_path = node.split(':')[0]
            else:
                f_path = node

            if f_path not in reusable_nodes:
                reusable_nodes[f_path] = []
            reusable_nodes[f_path].append(node)

    # Add root directory node
    graph.add_node('/', type=NODE_TYPE_DIRECTORY)
    dir_stack: List[str] = []
    dir_include_stack: List[bool] = []

    file_count = 0
    reuse_count = 0

    for root, _, files in os.walk(repo_path):
        dirname = os.path.relpath(root, repo_path)
        if dirname == '.':
            dirname = '/'
        elif is_skip_dir(dirname):
            continue
        else:
            graph.add_node(dirname, type=NODE_TYPE_DIRECTORY)
            parent_dirname = os.path.dirname(dirname)
            if parent_dirname == '':
                parent_dirname = '/'
            graph.add_edge(parent_dirname, dirname, type=EDGE_TYPE_CONTAINS)

        while len(dir_stack) > 0 and len(dir_include_stack) > 0 and not dirname.startswith(dir_stack[-1]):
            if dir_include_stack[-1] is False:
                graph.remove_node(dir_stack[-1])
            dir_stack.pop()
            dir_include_stack.pop()
        if dirname != '/':
            dir_stack.append(dirname)
            dir_include_stack.append(False)

        dir_has_py = False
        for file in files:
            if file.endswith('.py'):
                dir_has_py = True
                file_count += 1

                try:
                    file_path = os.path.join(root, file)
                    filename = os.path.relpath(file_path, repo_path)

                    if os.path.islink(file_path):
                        continue

                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()

                    current_hash = compute_hash(file_content)

                    # Check for reuse
                    reused = False
                    if base_graph and filename in file_hashes:
                        if file_hashes[filename] == current_hash:
                            # REUSE STRATEGY: Copy nodes from base_graph
                            if filename in reusable_nodes:
                                for node_id in reusable_nodes[filename]:
                                    if base_graph.has_node(node_id):
                                        # Copy node and attributes (including 'code', 'type', etc.)
                                        attr = base_graph.nodes[node_id]
                                        # Ensure hash is set
                                        if attr.get('type') == NODE_TYPE_FILE:
                                            attr['hash'] = current_hash
                                        graph.add_node(node_id, **attr)

                                        # Re-establish internal containment edges (e.g. File -> Class)
                                        # We only copy internal structure edges here.
                                        # External edges (imports/invokes) are rebuilt later.
                                # Re-construct containment edges for reused file
                                # The original logic for new files:
                                # graph.add_edge(dirname, filename, type=EDGE_TYPE_CONTAINS)
                                # graph.add_edge(parent, child, ...)

                                # We need to reconstruct the containment tree for this file
                                # based on node names which follow "filename:parent.child"
                                graph.add_edge(dirname, filename, type=EDGE_TYPE_CONTAINS)

                                # Sort nodes by length to ensure parents come before children
                                sub_nodes = [n for n in reusable_nodes[filename] if n != filename]
                                sub_nodes.sort(key=lambda x: len(x))

                                for node_id in sub_nodes:
                                    # Infer parent from name
                                    # Name format: filename:class.func
                                    # Parent is filename OR filename:class
                                    parts = node_id.split(':')
                                    if len(parts) < 2:
                                        # Skip nodes without proper format (e.g., file nodes)
                                        continue
                                    local_name = parts[1]
                                    name_parts = local_name.split('.')
                                    if len(name_parts) == 1:
                                        parent = filename
                                    else:
                                        parent = f"{filename}:{'.'.join(name_parts[:-1])}"

                                    if graph.has_node(parent):
                                        graph.add_edge(parent, node_id, type=EDGE_TYPE_CONTAINS)

                                file_nodes[filename] = file_path # Needed for imports
                                reused = True
                                reuse_count += 1

                    if not reused:
                        # Process as new/modified file
                        graph.add_node(filename, type=NODE_TYPE_FILE, code=file_content, hash=current_hash)
                        file_nodes[filename] = file_path
                        nodes = analyze_file(file_path) # Parse AST

                        for node in nodes:
                            full_name = f'{filename}:{node["name"]}'
                            graph.add_node(full_name, type=node['type'], code=node['code'],
                                           start_line=node['start_line'], end_line=node['end_line'])

                        graph.add_edge(dirname, filename, type=EDGE_TYPE_CONTAINS)
                        for node in nodes:
                            full_name = f'{filename}:{node["name"]}'
                            name_list = node['name'].split('.')
                            if len(name_list) == 1:
                                graph.add_edge(filename, full_name, type=EDGE_TYPE_CONTAINS)
                            else:
                                parent_name = '.'.join(name_list[:-1])
                                full_parent_name = f'{filename}:{parent_name}'
                                graph.add_edge(full_parent_name, full_name, type=EDGE_TYPE_CONTAINS)

                except (UnicodeDecodeError, SyntaxError, IndexError):
                    continue

        if dir_has_py:
            for i in range(len(dir_include_stack)):
                dir_include_stack[i] = True

    while len(dir_stack) > 0 and len(dir_include_stack) > 0:
        if dir_include_stack[-1] is False:
            graph.remove_node(dir_stack[-1])
        dir_stack.pop()
        dir_include_stack.pop()

    if base_graph:
        logger.info(f"Incremental Build: Reused {reuse_count}/{file_count} files.")

    # Add import edges with progress bar
    logger.info("[1/3] Analyzing imports...")
    for filename, filepath in tqdm(file_nodes.items(), desc="Processing imports", unit="file"):
        try:
            imports = find_imports(filepath, repo_path)
        except SyntaxError:
            continue
        add_imports(filename, imports, graph, repo_path)

    global_name_dict = defaultdict(list)
    if global_import:
        for node in graph.nodes():
            node_name = node.split(':')[-1].split('.')[-1]
            global_name_dict[node_name].append(node)

    # Add invocation and inheritance edges with progress bar
    logger.info("[2/3] Analyzing function calls and inheritance...")
    class_func_nodes = [n for n, d in graph.nodes(data=True)
                        if d.get('type') in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION]]

    for node in tqdm(class_func_nodes, desc="Analyzing relationships", unit="node"):
        attributes = graph.nodes[node]
        if attributes.get('type') not in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION]:
            continue

        # 防御性检查：确保节点有 code 属性
        node_code = attributes.get('code')
        if not node_code:
            continue  # 跳过没有代码的节点

        try:
            caller_code_tree = ast.parse(node_code)
        except SyntaxError:
            continue  # 跳过语法错误的代码

        callee_nodes, callee_alias = find_all_possible_callee(node, graph)
        if fuzzy_search:
            callee_name_dict = defaultdict(list)
            for callee_node in set(callee_nodes):
                callee_name = callee_node.split(':')[-1].split('.')[-1]
                callee_name_dict[callee_name].append(callee_node)
            for alias, callee_node in callee_alias.items():
                callee_name_dict[alias].append(callee_node)
        else:
            callee_name_dict = {
                callee_node.split(':')[-1].split('.')[-1]: callee_node
                for callee_node in callee_nodes[::-1]
            }
            callee_name_dict.update(callee_alias)

        if attributes.get('type') == NODE_TYPE_CLASS:
            invocations, inheritances = analyze_init(node, caller_code_tree, graph, repo_path)
        else:
            invocations = analyze_invokes(node, caller_code_tree, graph, repo_path)
            inheritances = []

        for callee_name in set(invocations):
            callee_node = callee_name_dict.get(callee_name)
            if callee_node:
                if isinstance(callee_node, list):
                    for callee in callee_node:
                        graph.add_edge(node, callee, type=EDGE_TYPE_INVOKES)
                else:
                    graph.add_edge(node, callee_node, type=EDGE_TYPE_INVOKES)
            elif global_import:
                global_fuzzy_nodes = global_name_dict.get(callee_name)
                if global_fuzzy_nodes:
                    for global_fuzzy_node in global_fuzzy_nodes:
                        graph.add_edge(node, global_fuzzy_node, type=EDGE_TYPE_INVOKES)

        for callee_name in set(inheritances):
            callee_node = callee_name_dict.get(callee_name)
            if callee_node:
                if isinstance(callee_node, list):
                    for callee in callee_node:
                        graph.add_edge(node, callee, type=EDGE_TYPE_INHERITS)
                else:
                    graph.add_edge(node, callee_node, type=EDGE_TYPE_INHERITS)
            elif global_import:
                global_fuzzy_nodes = global_name_dict.get(callee_name)
                if global_fuzzy_nodes:
                    for global_fuzzy_node in global_fuzzy_nodes:
                        graph.add_edge(node, global_fuzzy_node, type=EDGE_TYPE_INHERITS)
    
    logger.info("[3/3] Graph construction complete!")
    return graph
