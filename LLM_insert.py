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
                start_idx = max(0, start_line - 10)
                end_idx = min(len(lines), end_line + 5)
                function_lines = [line.rstrip() for line in lines[start_idx:end_idx]]
                return '\n'.join(function_lines)
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return ""

    def parse_call_block(self, block: str) -> Optional[CallInfo]:
        try:
            lines = [line.strip() for line in block.strip().split('\n')]
            if len(lines) < 5:
                return None

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

    def generate_llm_prompt(self, call_info: CallInfo) -> str:
        caller_code = self.extract_function_block(
            call_info.caller_file,
            call_info.caller_range[0] - 10,
            call_info.caller_range[1] + 5
        )
        callee_code = self.extract_function_block(
            call_info.callee_file,
            call_info.callee_range[0],
            call_info.callee_range[1]
        )
        system_prompt = (
            "You are a specialized code analyzer that generates EXACTLY ONE LINE of valid C++ log statement using HILOG_<LEVEL>.\n"
            "Requirements:\n"
            "1. Choose LEVEL: DEBUG for condition checks, INFO for normal operations, WARN for potential issues, ERROR for failures\n"
            "2. Format with param: HILOG_<LEVEL>(\"[<CALLER_COMPONENT>] <operation> <target>, <param_name>=0x%x\", (UINTPTR)¶m)\n"
            "3. Format without param: HILOG_<LEVEL>(\"[<CALLER_COMPONENT>] <operation> <target>\")\n"
            "4. <operation> is a verb derived from caller/callee name (e.g., 'schedules', 'processes')\n"
            "5. <target> is a noun from callee (e.g., 'task', 'system')\n"
            "6. <param_name> handling:\n"
            "   - Use ONLY the exact parameter name if explicitly passed in the call (e.g., 'func(param)' uses 'param')\n"
            "   - If the call has no parameters and is DIRECTLY INSIDE an 'if' condition branch (e.g., 'if (var) { func(); }'), use the condition variable from the 'if' statement (e.g., 'var')\n"
            "   - If the call has no parameters and is NOT DIRECTLY INSIDE an 'if' condition branch, do NOT include any parameter\n"
            "7. Do NOT use function return values (e.g., 'isSwitch') or variables not explicitly passed or used as the condition in the immediately preceding 'if' statement\n"
            "8. Output must be a single line ending with semicolon, starting with HILOG_<LEVEL>\n"
            "Examples with Scenarios:\n"
            "- Scenario 1 (With explicit parameter):\n"
            "  Code: void ProcessData(int data) { HandleInput(data); }\n"
            "  Output: HILOG_INFO(\"[ARCH] processes task, data=0x%x\", (UINTPTR)&data);\n"
            "- Scenario 2 (No param, inside condition):\n"
            "  Code: void CheckSched() { if (g_condition) { LOS_Schedule(); } }\n"
            "  Output: HILOG_DEBUG(\"[ARCH] schedules task, g_condition=0x%x\", (UINTPTR)&g_condition);\n"
            "- Scenario 3 (No param, no condition):\n"
            "  Code: void InitSystem() { int x = 1; if (x) { y = 2; } StartTask(); }\n"
            "  Output: HILOG_INFO(\"[ARCH] starts task\");\n"
            "- Scenario 4 (No param, after condition):\n"
            "  Code: void RunTask() { if (flag) { x = 1; } TaskRun(); }\n"
            "  Output: HILOG_INFO(\"[ARCH] runs task\");\n"
        )
        return f"""Analyze and generate a log statement:

Caller Function ({call_info.caller_component}):
{caller_code}

Callee Function ({call_info.callee_component}):
{callee_code}

Call: {call_info.caller_func} calls {call_info.callee_func} at line {call_info.call_line}"""

    # 将单行HILOG包装为条件编译块
    def wrap_log_in_conditional(self, log_line: str) -> str:
        if not log_line.startswith('HILOG_') or not log_line.endswith(';'):
            return f'#ifdef HILOG_INFO\nHILOG_INFO("[{self.caller_component}] invokes {self.callee_component}");\n#else\nprintf("[{self.caller_component}] invokes {self.callee_component}\\n");\n#endif'
        
        # 提取 HILOG 级别和内容
        match = re.match(r'HILOG_(DEBUG|INFO|WARN|ERROR)\((.*)\);', log_line)
        if not match:
            return f'#ifdef HILOG_INFO\nHILOG_INFO("[{self.caller_component}] invokes {self.callee_component}");\n#else\nprintf("[{self.caller_component}] invokes {self.callee_component}\\n");\n#endif'
        
        level, content = match.groups()
        if "0x%x" in content:
            # 有参数的情况，换行符在格式字符串内
            printf_content = content.replace("0x%x", "0x%08x").replace('",', '\\n",')
        else:
            # 无参数的情况
            printf_content = content.replace('")', '\\n")')
        
        return f'#ifdef HILOG_INFO\n{log_line}\n#else\nprintf({printf_content});\n#endif'

    async def get_llm_response(self, prompt: str, call_info: CallInfo) -> str:
        async with self.semaphore:
            self.caller_component = call_info.caller_component
            self.callee_component = call_info.callee_component
            system_prompt = (
                "You are a specialized code analyzer that generates EXACTLY ONE LINE of valid C++ log statement using HILOG_<LEVEL>.\n"
                "Requirements:\n"
                "1. Choose LEVEL: DEBUG for condition checks, INFO for normal operations, WARN for potential issues, ERROR for failures\n"
                "2. Format with param: HILOG_<LEVEL>(\"[<CALLER_COMPONENT>] <operation> <target>, <param_name>=0x%x\", (UINTPTR)¶m)\n"
                "3. Format without param: HILOG_<LEVEL>(\"[<CALLER_COMPONENT>] <operation> <target>\")\n"
                "4. <operation> is a verb from caller/callee name (e.g., 'schedules', 'processes')\n"
                "5. <target> is a noun from callee (e.g., 'task', 'system')\n"
                "6. <param_name> is the exact parameter name ONLY if explicitly passed in the call (e.g., 'func(param)' uses 'param'); "
                "if no param is passed, use condition variable from 'if' statement ONLY if the call is directly inside an 'if' branch; "
                "if no param and no condition branch, do NOT include any parameter\n"
                "7. Do NOT use function return values (e.g., 'isSwitch') or variables not explicitly passed or conditioned in the 'if' statement\n"
                "8. Output must be a single line ending with semicolon, starting with HILOG_<LEVEL>\n"
                "Examples with Scenarios:\n"
                "- Scenario 1 (With explicit parameter):\n"
                "  Code: void ProcessData(int data) { HandleInput(data); }\n"
                "  Output: HILOG_INFO(\"[ARCH] processes task, data=0x%x\", (UINTPTR)&data);\n"
                "- Scenario 2 (No param, inside condition):\n"
                "  Code: void CheckSched() { if (g_condition) { LOS_Schedule(); } }\n"
                "  Output: HILOG_DEBUG(\"[ARCH] schedules task, g_condition=0x%x\", (UINTPTR)&g_condition);\n"
                "- Scenario 3 (No param, no condition):\n"
                "  Code: void InitSystem() { int x = 1; if (x) { y = 2; } StartTask(); }\n"
                "  Output: HILOG_INFO(\"[ARCH] starts task\");\n"
                "- Scenario 4 (No param, after condition):\n"
                "  Code: void RunTask() { if (flag) { x = 1; } TaskRun(); }\n"
                "  Output: HILOG_INFO(\"[ARCH] runs task\");\n"
            )
            if self.use_local:
                combined_prompt = system_prompt + "\n" + prompt
                try:
                    response = await asyncio.to_thread(
                        lambda: requests.post(self.base_url, json={
                            "model": "llama3.1:70b", # 需要修改本地大模型型号
                            "prompt": combined_prompt,
                            "stream": False,
                        })
                    )
                    if response.status_code != 200:
                        logger.error(f"Local LLM API error {response.status_code}: {response.text}")
                        return self.wrap_log_in_conditional(f'HILOG_ERROR("[{call_info.caller_component}] handles {call_info.callee_component}, error=0x%x", errno)') + f'\n insert [{call_info.caller_file}:{call_info.call_line}]'
                    json_response = response.json()
                    raw_response = json_response.get("response", "").strip()
                    log_line = next((line.strip() for line in raw_response.split('\n') if line.strip().startswith('HILOG_') and line.strip().endswith(';')), None)
                    if not log_line:
                        return self.wrap_log_in_conditional(f'HILOG_INFO("[{call_info.caller_component}] invokes {call_info.callee_component}")') + f'\n insert [{call_info.caller_file}:{call_info.call_line}]'
                    # 后处理：移除非显式参数和返回值
                    caller_code = self.extract_function_block(call_info.caller_file, call_info.caller_range[0] - 10, call_info.caller_range[1] + 5)
                    call_line_index = caller_code.find(call_info.callee_func)
                    preceding_lines = caller_code[:call_line_index].split('\n')[-5:]  # 检查前5行
                    in_if_branch = any('if' in line and '{' in line for line in preceding_lines) and call_line_index < caller_code.find('}', call_line_index)
                    has_param = re.search(rf'{call_info.callee_func}\s*\(\s*\w+\s*\)', caller_code)
                    if "0x%x" in log_line and not has_param and not in_if_branch:
                        log_line = f'HILOG_{log_line.split("_")[1].split("(")[0]}("[{call_info.caller_component}] {log_line.split("]")[1].split(",")[0].strip()}")'
                    return self.wrap_log_in_conditional(log_line) + f'\n insert [{call_info.caller_file}:{call_info.call_line}]'
                except Exception as e:
                    logger.error(f"Error calling local LLM API: {e}")
                    return self.wrap_log_in_conditional(f'HILOG_ERROR("[{call_info.caller_component}] handles {call_info.callee_component}, error=0x%x", errno)') + f'\n insert [{call_info.caller_file}:{call_info.call_line}]'
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
                    raw_response = response.choices[0].message.content.strip()
                    log_line = next((line.strip() for line in raw_response.split('\n') if line.strip().startswith('HILOG_') and line.strip().endswith(';')), None)
                    if not log_line:
                        return self.wrap_log_in_conditional(f'HILOG_INFO("[{call_info.caller_component}] invokes {call_info.callee_component}")') + f'\n insert [{call_info.caller_file}:{call_info.call_line}]'
                    # 后处理：移除非显式参数和返回值
                    caller_code = self.extract_function_block(call_info.caller_file, call_info.caller_range[0] - 10, call_info.caller_range[1] + 5)
                    call_line_index = caller_code.find(call_info.callee_func)
                    preceding_lines = caller_code[:call_line_index].split('\n')[-5:]  # 检查前5行
                    in_if_branch = any('if' in line and '{' in line for line in preceding_lines) and call_line_index < caller_code.find('}', call_line_index)
                    has_param = re.search(rf'{call_info.callee_func}\s*\(\s*\w+\s*\)', caller_code)
                    if "0x%x" in log_line and not has_param and not in_if_branch:
                        log_line = f'HILOG_{log_line.split("_")[1].split("(")[0]}("[{call_info.caller_component}] {log_line.split("]")[1].split(",")[0].strip()}")'
                    return self.wrap_log_in_conditional(log_line) + f'\n insert [{call_info.caller_file}:{call_info.call_line}]'
                except Exception as e:
                    logger.error(f"Error calling LLM API: {e}")
                    return self.wrap_log_in_conditional(f'HILOG_ERROR("[{call_info.caller_component}] handles {call_info.callee_component}, error=0x%x", errno)') + f'\n insert [{call_info.caller_file}:{call_info.call_line}]'

    def extract_target_interaction(self, content: str, target: str) -> Optional[List[str]]:
        blocks = []
        lines = content.splitlines()
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
            pattern = re.compile(r'^.+? -> .+?:$')
            current_block = []
            capturing = False
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if pattern.match(stripped):
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

def preview_with_line_numbers(file_content: str) -> None:
    lines = file_content.splitlines()
    for idx, line in enumerate(lines, start=1):
        print(f"{idx:4}: {line}")

def insert_multiple_logs_reverse(file_content: str, insertions: List[Tuple[int, str]], highlight: bool = False) -> str:
    lines = file_content.splitlines(keepends=True)
    prefix = "\033[1;32m" if highlight else ""
    suffix = "\033[0m" if highlight else ""
    insertions_sorted = sorted(insertions, key=lambda x: x[0], reverse=True)
    for line_number, log_line in insertions_sorted:
        index = max(0, line_number - 1)
        if highlight:
            log_line_colored = ""
            for line in log_line.split('\n'):
                log_line_colored += f"{prefix}{line}{suffix}\n"
            log_line = log_line_colored.rstrip('\n')
        lines.insert(index, log_line + "\n")
    return "".join(lines)

async def main():
    parser = argparse.ArgumentParser(
        description='分析组件交互并预览或直接插入日志到源文件'
    )
    parser.add_argument('api_key', nargs='?', default='', help='大模型 API 的 API key（使用本地大模型时可省略）')
    parser.add_argument('--target', '-t', required=False,
                        help='待分析的目标交互（例如："arch -> kernel"）')
    parser.add_argument('--actual-insert', action='store_true',
                        help='直接将日志插入到源文件')
    parser.add_argument('--base_url', default="https://api.deepseek.com",
                        help='大模型 API 的 Base URL')
    parser.add_argument('--local', action='store_true',
                        help='使用本地大模型')
    args = parser.parse_args()

    if not args.local and not args.api_key:
        parser.error("非本地模式时必须提供 api_key")
    if args.local and args.base_url == "https://api.deepseek.com":
        args.base_url = "http://localhost:11434/api/generate"

    analyzer = AsyncComponentAnalyzer(api_key=args.api_key, base_url=args.base_url, use_local=args.local)
    content = sys.stdin.read()

    blocks = analyzer.extract_target_interaction(content, args.target) if args.target else analyzer.extract_target_interaction(content, "")
    if not blocks:
        logger.error(f"No interaction found for: {args.target}" if args.target else "No interactions found")
        return

    print(f"\nAnalyzing interaction: {args.target if args.target else 'All Components'}")
    print("Found", len(blocks), "function calls to analyze")

    tasks = []
    logs = []
    for block in blocks:
        call_info = analyzer.parse_call_block(block)
        if call_info:
            prompt = analyzer.generate_llm_prompt(call_info)
            task = analyzer.get_llm_response(prompt, call_info)
            tasks.append(task)

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
                modified_content = insert_multiple_logs_reverse(file_content, insertions, highlight=False)
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(modified_content)
                print(f"Updated source file: {file_path}")
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
    else:
        for file_path, insertions in insertions_by_file.items():
            try:
                with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
                modified_content = insert_multiple_logs_reverse(file_content, insertions, highlight=True)
                print(f"\n==== Preview of inserted file: {file_path} ====")
                preview_with_line_numbers(modified_content)
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
