#!/usr/bin/env python3
import sys
import re
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import argparse
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CallInfo:
    caller_func: str
    callee_func: str
    caller_file: str
    callee_file: str
    call_line: int
    caller_range: Tuple[int, int]
    callee_range: Tuple[int, int]

    @property
    def caller_component(self) -> str:
        match = re.search(r'OH5/[^/]+/[^/]+/([^/]+)/', self.caller_file)
        return match.group(1).upper() if match else "UNKNOWN"

    @property
    def callee_component(self) -> str:
        match = re.search(r'OH5/[^/]+/[^/]+/([^/]+)/', self.callee_file)
        return match.group(1).upper() if match else "UNKNOWN"


class AsyncComponentAnalyzer:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", use_local: bool = False):
        self.use_local = use_local
        self.base_url = base_url
        if not use_local:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = None
        self.semaphore = asyncio.Semaphore(100)

    def extract_function_block(self, file_path: str, start_line: int, end_line: int) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                start_idx = max(0, start_line - 2)
                end_idx = min(len(lines), end_line)
                function_lines = [line.lstrip() for line in lines[start_idx:end_idx]]
                return ''.join(function_lines)
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return ""

    # 组件交互下每个以*开头的5行为一个函数调用信息
    def parse_call_block(self, block: str) -> Optional[CallInfo]:
        try:
            lines = [line.strip() for line in block.strip().split('\n')]
            if len(lines) < 5:
                return None

            # 第一行：形如 "* caller_func calls callee_func"
            first_line = lines[0].removeprefix('*').strip()
            if ' calls ' not in first_line:
                return None
            caller_func, callee_func = first_line.split(' calls ', 1)

            from_match = re.search(r'from (.*?) line (\d+)', lines[1])
            if not from_match:
                return None
            caller_file = from_match.group(1)
            call_line = int(from_match.group(2))

            to_match = re.search(r'to (.*)', lines[2])
            if not to_match:
                return None
            callee_file = to_match.group(1)

            caller_match = re.search(r'caller range: (\d+)-(\d+)', lines[3])
            callee_match = re.search(r'callee range: (\d+)-(\d+)', lines[4])
            if not (caller_match and callee_match):
                return None

            caller_range = (int(caller_match.group(1)), int(caller_match.group(2)))
            callee_range = (int(callee_match.group(1)), int(callee_match.group(2)))

            return CallInfo(
                caller_func=caller_func.strip(),
                callee_func=callee_func.strip(),
                caller_file=caller_file.strip(),
                callee_file=callee_file.strip(),
                call_line=call_line,
                caller_range=caller_range,
                callee_range=callee_range
            )
        except Exception as e:
            logger.error(f"Error parsing block: {e}\nProblematic block:\n{block}")
            return None

    # 生成给大模型的 prompt
    def generate_llm_prompt(self, call_info: CallInfo) -> str:
        caller_code = self.extract_function_block(
            call_info.caller_file,
            call_info.caller_range[0],
            call_info.caller_range[1]
        )
        callee_code = self.extract_function_block(
            call_info.callee_file,
            call_info.callee_range[0],
            call_info.callee_range[1]
        )
        system_prompt = (
            "You are a specialized code analyzer that generates ONLY a single line of standardized log output. "
            "Do not provide any additional explanation or context.\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "[component_name] caller_function: action | target_component.callee_function [brief_details]\n"
            "FORMAT RULES:\n"
            "- component_name: Uppercase component name\n"
            "- action: Use present tense verbs (processes, handles, validates)\n"
            "- target_component: Uppercase component name\n"
            "- brief_details: Describe the core purpose in 5-7 words\n"
            "- Square brackets and colons must be exactly as shown\n"
            "- Must be exactly ONE line with no additional text"
        )
        return f"""Here are two related functions. Generate a standardized log message describing their interaction:

Caller Function ({call_info.caller_component}):
{caller_code}

Callee Function ({call_info.callee_component}):
{callee_code}

Call occurs when {call_info.caller_func} calls {call_info.callee_func} at line {call_info.call_line}"""

    # 调用大模型接口获取响应并包装日志
    async def get_llm_response(self, prompt: str, call_info: CallInfo) -> str:
        async with self.semaphore:
            system_prompt = (
                "You are a specialized code analyzer that generates ONLY a single line of standardized log output. "
                "Do not provide any additional explanation or context.\n"
                "REQUIRED OUTPUT FORMAT:\n"
                "[component_name] caller_function: action | target_component.callee_function [brief_details]\n"
                "FORMAT RULES:\n"
                "- component_name: Uppercase component name\n"
                "- action: Use present tense verbs (processes, handles, validates)\n"
                "- target_component: Uppercase component name\n"
                "- brief_details: Describe the core purpose in 5-7 words\n"
                "- Square brackets and colons must be exactly as shown\n"
                "- Must be exactly ONE line with no additional text"
            )
            if self.use_local:
                combined_prompt = system_prompt + "\n" + prompt
                try:
                    response = await asyncio.to_thread(
                        lambda: requests.post(self.base_url, json={
                            "model": "llama3.1:70b",
                            "prompt": combined_prompt,
                            "stream": False,
                        })
                    )
                    if response.status_code != 200:
                        logger.error(f"Local LLM API error {response.status_code}: {response.text}")
                        return f"Error generating log for {call_info.caller_func} -> {call_info.callee_func}"
                    json_response = response.json()
                    raw_response = json_response.get("response", "").strip()
                    cleaned_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
                    return f'HILOG_INFO("{cleaned_response}"); insert [{call_info.caller_file}:{call_info.call_line}]'
                except Exception as e:
                    logger.error(f"Error calling local LLM API: {e}")
                    return f"Error generating log for {call_info.caller_func} -> {call_info.callee_func}"
            else:
                try:
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        stream=False
                    )
                    log_message = response.choices[0].message.content.strip()
                    return f'HILOG_INFO("{log_message}"); insert [{call_info.caller_file}:{call_info.call_line}]'
                except Exception as e:
                    logger.error(f"Error calling LLM API: {e}")
                    return f"Error generating log for {call_info.caller_func} -> {call_info.callee_func}"

    # 提取交互块
    def extract_target_interaction(self, content: str, target: str) -> Optional[List[str]]:
        blocks = []
        lines = content.splitlines()
        # 指定 target 情况：只提取目标标题下的块
        if target:
            capturing = False
            current_block = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped == target + ':':
                    capturing = True
                    continue
                # 当遇到下一个 "->" 标题时结束采集
                if capturing and re.match(r'^.+? -> .+?:$', stripped):
                    break
                if capturing:
                    if stripped.startswith('*'):
                        if current_block:
                            blocks.append('\n'.join(current_block))
                        current_block = []
                    current_block.append(stripped)
                    if len(current_block) == 5:
                        blocks.append('\n'.join(current_block))
                        current_block = []
            if current_block and len(current_block) == 5:
                blocks.append('\n'.join(current_block))
            return blocks if blocks else None
        else:
            # 未指定 target，扫描所有形如 "组件A -> 组件B:" 的标题，并提取其下的调用日志块
            pattern = re.compile(r'^.+? -> .+?:$')
            current_block = []
            capturing = False
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if pattern.match(stripped):
                    # 遇到新的标题，先将已有 block 按 5 行一组保存
                    if current_block:
                        for i in range(0, len(current_block), 5):
                            group = current_block[i:i+5]
                            if len(group) == 5:
                                blocks.append("\n".join(group))
                        current_block = []
                    capturing = True
                    continue
                if capturing:
                    current_block.append(stripped)
            if current_block:
                for i in range(0, len(current_block), 5):
                    group = current_block[i:i+5]
                    if len(group) == 5:
                        blocks.append("\n".join(group))
            return blocks if blocks else None


# 显示带行号的预览
def preview_with_line_numbers(file_content: str) -> None:
    lines = file_content.splitlines()
    for idx, line in enumerate(lines, start=1):
        print(f"{idx:4}: {line}")


# 倒序插入日志行（便于多处插入时不影响后续行号）
def insert_multiple_logs_reverse(file_content: str, insertions: List[Tuple[int, str]], highlight: bool = False) -> str:
    lines = file_content.splitlines(keepends=True)
    prefix = "\033[1;32m" if highlight else ""
    suffix = "\033[0m" if highlight else ""
    insertions_sorted = sorted(insertions, key=lambda x: x[0], reverse=True)
    for line_number, log_line in insertions_sorted:
        index = max(0, line_number - 1)
        if highlight:
            log_line = f"{prefix}{log_line}{suffix}"
        lines.insert(index, log_line + "\n")
    return "".join(lines)


# 偏移插入日志行
def insert_multiple_logs_with_offset(file_content: str, insertions: List[Tuple[int, str]], highlight: bool = False) -> str:
    lines = file_content.splitlines(keepends=True)
    prefix = "\033[1;32m" if highlight else ""
    suffix = "\033[0m" if highlight else ""
    offset = 0
    for line_number, log_line in sorted(insertions, key=lambda x: x[0]):
        index = max(0, line_number - 1 + offset)
        if highlight:
            log_line = f"{prefix}{log_line}{suffix}"
        lines.insert(index, log_line + "\n")
        offset += 1
    return "".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description='分析组件交互并预览或实际插入日志'
    )
    parser.add_argument('api_key', nargs='?', default='', help='大模型 API 的 API key（使用本地大模型时可省略）')
    parser.add_argument('--target', '-t', required=False,
                        help='待分析的目标交互（例如："FatFs -> liteos_m"），若不指定则扫描报告中所有形如 "组件A -> 组件B:" 的部分')
    parser.add_argument('--actual-insert', action='store_true',
                        help='实际将日志插入到源文件（无行号和高亮）')
    parser.add_argument('--insert-mode', choices=['reverse', 'offset'], default='reverse',
                        help='多处插入时使用的策略，默认为 reverse')
    parser.add_argument('--base_url', default="https://api.deepseek.com",
                        help='大模型 API 的 Base URL')
    parser.add_argument('--local', action='store_true',
                        help='使用本地大模型（例如通过 http://localhost:11434/api/generate）')
    args = parser.parse_args()

    if not args.local and not args.api_key:
        parser.error("非本地模式时必须提供 api_key")
    if args.local and args.base_url == "https://api.deepseek.com":
        args.base_url = "http://localhost:11434/api/generate"

    analyzer = AsyncComponentAnalyzer(api_key=args.api_key, base_url=args.base_url, use_local=args.local)
    content = sys.stdin.read()

    # 如果提供 target，则只分析该部分；否则扫描所有 "组件A -> 组件B:" 部分
    if args.target:
        blocks = analyzer.extract_target_interaction(content, args.target)
    else:
        blocks = analyzer.extract_target_interaction(content, "")

    if not blocks:
        logger.error(f"No interaction found for: {args.target}" if args.target else "No interactions found")
        return

    print(f"\nAnalyzing interaction: {args.target if args.target else 'All Components'}")
    print("Found", len(blocks), "function calls to analyze")

    # 输出正在分析的交互信息
    interaction_title = args.target if args.target else "All Components"
    print(f"\nGenerated Logs for {interaction_title}:")

    tasks = []
    logs = []
    for block in blocks:
        call_info = analyzer.parse_call_block(block)
        if call_info:
            prompt = analyzer.generate_llm_prompt(call_info)
            task = analyzer.get_llm_response(prompt, call_info)
            tasks.append(task)

    # 使用 asyncio.as_completed 逐个输出结果
    if tasks:
        for future in asyncio.as_completed(tasks):
            log = await future
            logs.append(log)
            print(log)

    insertions_by_file: Dict[str, List[Tuple[int, str]]] = {}
    insertion_pattern = re.compile(r"insert \[(.*?):(\d+)\]$")
    for log in logs:
        match = insertion_pattern.search(log)
        if match:
            file_path = match.group(1)
            line_number = int(match.group(2))
            log_line = log[:match.start()].strip()
            insertions_by_file.setdefault(file_path, []).append((line_number, log_line))
        else:
            logger.warning(f"无法解析插入信息的日志: {log}")

    if args.actual_insert:
        for file_path, insertions in insertions_by_file.items():
            try:
                with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
                continue

            if args.insert_mode == "reverse":
                modified_content = insert_multiple_logs_reverse(file_content, insertions, highlight=False)
            else:
                modified_content = insert_multiple_logs_with_offset(file_content, insertions, highlight=False)

            try:
                bak_file_path = file_path + ".bak"
                with open(bak_file_path, "w", encoding='utf-8') as f:
                    f.write(modified_content)
                print(f"Backup file updated: {bak_file_path}")
            except Exception as e:
                logger.warning(f"Error writing to backup file {bak_file_path}: {e}")
    else:
        for file_path, insertions in insertions_by_file.items():
            try:
                with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
                continue

            if args.insert_mode == "reverse":
                modified_content = insert_multiple_logs_reverse(file_content, insertions, highlight=True)
            else:
                modified_content = insert_multiple_logs_with_offset(file_content, insertions, highlight=True)

            print(f"\n==== Preview of inserted file: {file_path} ====")
            preview_with_line_numbers(modified_content)


if __name__ == "__main__":
    asyncio.run(main())

