import json
import os
import sys
from pathlib import Path

def find_latest_usage_log(base_dir="outputs"):
    """寻找最新的包含 token 统计的目录或文件"""
    return Path(base_dir)

MODEL_PRICING = {
    "claude-3-opus": {"input": 15.0, "cache_write": 18.75, "cache_read": 1.50, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "cache_write": 3.75, "cache_read": 0.30, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "cache_write": 0.30, "cache_read": 0.03, "output": 1.25},
    "claude-3-5-sonnet": {"input": 3.0, "cache_write": 3.75, "cache_read": 0.30, "output": 15.0},
    "claude-3-5-haiku": {"input": 0.80, "cache_write": 1.0, "cache_read": 0.08, "output": 4.0},
    "claude-3.7-sonnet": {"input": 3.0, "cache_write": 3.75, "cache_read": 0.30, "output": 15.0},
    "claude-4-opus": {"input": 15.0, "cache_write": 18.75, "cache_read": 1.50, "output": 75.0},
    "claude-4-sonnet": {"input": 3.0, "cache_write": 3.75, "cache_read": 0.30, "output": 15.0},
    "claude-4.1-opus": {"input": 15.0, "cache_write": 18.75, "cache_read": 1.50, "output": 75.0},
    "claude-4.5-opus": {"input": 5.0, "cache_write": 6.25, "cache_read": 0.50, "output": 25.0},
    "claude-4.5-sonnet": {"input": 3.0, "cache_write": 3.75, "cache_read": 0.30, "output": 15.0},
    "claude-4.5-haiku": {"input": 1.0, "cache_write": 1.25, "cache_read": 0.10, "output": 5.0},
    "qwen-coder-plus": {"input": 0.2, "cache_write": 0.2, "cache_read": 0.02, "output": 0.6},
    "qwen3-coder-plus": {"input": 0.2, "cache_write": 0.2, "cache_read": 0.02, "output": 0.6},
}

def get_cost(entry):
    """根据单次请求的 entry 计算成本"""
    model = entry.get("model", "").lower()
    input_tokens = entry.get("input_tokens", 0)
    output_tokens = entry.get("output_tokens", 0)
    
    # 尝试模糊匹配已知模型
    matched_pricing = None
    for m_key, pricing in MODEL_PRICING.items():
        if m_key in model:
            matched_pricing = pricing
            break
            
    if matched_pricing:
        cache_write_tokens = entry.get("cache_creation_input_tokens", 0) or entry.get("cache_write_tokens", 0) or 0
        cache_read_tokens = entry.get("cache_read_input_tokens", 0) or entry.get("cache_hit_tokens", 0) or 0
        base_input_tokens = max(0, input_tokens - cache_write_tokens - cache_read_tokens)
        
        cost = (base_input_tokens / 1_000_000 * matched_pricing["input"]) + \
               (cache_write_tokens / 1_000_000 * matched_pricing["cache_write"]) + \
               (cache_read_tokens / 1_000_000 * matched_pricing["cache_read"]) + \
               (output_tokens / 1_000_000 * matched_pricing["output"])
        return cost
    else:
        # 默认回退逻辑
        in_rate, out_rate = 0.574, 2.294
        cost = (input_tokens / 1_000_000 * in_rate) + (output_tokens / 1_000_000 * out_rate)
        return cost

def extract_usage_entries(data):
    """递归地从各种 JSON 结构中提取包含 model 和 tokens 的字典"""
    entries = []
    if isinstance(data, list):
        for item in data:
            entries.extend(extract_usage_entries(item))
    elif isinstance(data, dict):
        # 识别 LLM 调用记录的特征键
        # 增加对 prompt_length 和 response_length 的支持 (常见于某些 trace 文件)
        has_model = "model" in data
        has_tokens = any(k in data for k in ["input_tokens", "prompt_tokens", "input", "prompt_length"])
        
        if has_model and has_tokens:
            entry = data.copy()
            # 字段归一化
            if "prompt_tokens" in entry and "input_tokens" not in entry:
                entry["input_tokens"] = entry["prompt_tokens"]
            if "completion_tokens" in entry and "output_tokens" not in entry:
                entry["output_tokens"] = entry["completion_tokens"]
            if "input" in entry and isinstance(entry["input"], int) and "input_tokens" not in entry:
                entry["input_tokens"] = entry["input"]
            if "output" in entry and isinstance(entry["output"], int) and "output_tokens" not in entry:
                entry["output_tokens"] = entry["output"]
            if "prompt_length" in entry and "input_tokens" not in entry:
                entry["input_tokens"] = entry["prompt_length"]
            if "response_length" in entry and "output_tokens" not in entry:
                entry["output_tokens"] = entry["response_length"]
            
            if "input_tokens" in entry:
                entries.append(entry)
        else:
            for key, value in data.items():
                if key in ["system_prompt", "code_before", "code_after", "fail_to_pass_code", "prompt", "response"]:
                    continue
                entries.extend(extract_usage_entries(value))
    return entries

def calculate_stats(path):
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"错误: 路径不存在 {path}")
        return

    if path_obj.is_dir():
        usage_files = list(path_obj.rglob("*.json"))
        print(f"正在分析目录: {path} (共检索 {len(usage_files)} 个 JSON 文件)")
    else:
        usage_files = [path_obj]

    try:
        total_input = 0
        total_output = 0
        call_count = 0
        total_cost = 0.0
        model_stats = {}
        
        processed_files_count = 0
        # 记录已处理的唯一条目，防止重复统计
        seen_entries = set()

        for file_path in usage_files:
            try:
                if file_path.name in ["final_dataset.json", "target_instances.json"]:
                    continue
                    
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content: continue
                    data = json.loads(content)
                
                entries = extract_usage_entries(data)
                if not entries: continue
                
                processed_files_count += 1
                for entry in entries:
                    # 使用特征字段创建唯一标识，防止同一条目在不同文件中重复统计
                    entry_id = f"{entry.get('model')}_{entry.get('input_tokens')}_{entry.get('output_tokens')}_{entry.get('timestamp', '')}"
                    if entry_id in seen_entries:
                        continue
                    seen_entries.add(entry_id)

                    model = str(entry.get("model", "unknown")).lower()
                    in_t = entry.get("input_tokens", 0)
                    out_t = entry.get("output_tokens", 0)
                    cost = get_cost(entry)
                    
                    total_input += in_t
                    total_output += out_t
                    total_cost += cost
                    call_count += 1
                    
                    if model not in model_stats:
                        model_stats[model] = {"input": 0, "output": 0, "cost": 0.0, "count": 0}
                    m_stat = model_stats[model]
                    m_stat["input"] += in_t
                    m_stat["output"] += out_t
                    m_stat["cost"] += cost
                    m_stat["count"] += 1
                    
            except Exception:
                continue 
            
        print("=" * 60)
        print(f"Token 使用及成本统计分析 (全量扫描模式)")
        print("=" * 60)
        
        print(f"{'模型名称':<25} {'调用数':>6} {'总Tokens':>12} {'成本($)':>10}")
        print("-" * 60)
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["cost"], reverse=True)
        for model_name, m_stat in sorted_models:
            m_total_tokens = m_stat["input"] + m_stat["output"]
            print(f"{model_name[:24]:<25} {m_stat['count']:>8} {m_total_tokens:>12,} {m_stat['cost']:>10.4f}")
        
        print("-" * 60)
        print(f"汇总统计:")
        print(f"有效数据文件数: {processed_files_count}")
        print(f"总调用次数:   {call_count}")
        print(f"总输入 Tokens: {total_input:,}")
        print(f"总输出 Tokens: {total_output:,}")
        print(f"总计 Tokens:   {total_input + total_output:,}")
        print("-" * 60)
        print(f"预估总成本:    ${total_cost:.4f}")
        print(f"折合人民币:    ¥{total_cost * 7.2:.2f} (汇率按7.2计算)")
        print("=" * 60)
        
    except Exception as e:
        print(f"统计过程出错: {e}")

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not log_path:
        root_dir = Path(__file__).parent.parent
        log_path = root_dir / "outputs"
        
    calculate_stats(log_path)
