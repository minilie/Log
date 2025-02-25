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
import json

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
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", use_local: bool = False, use_huawei: bool = False, bearer_token: str = None):
        self.use_local = use_local
        self.use_huawei = use_huawei
        self.base_url = base_url
        self.bearer_token = bearer_token
        if not use_local and not use_huawei:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif use_huawei:
            self.session = requests.Session()
            self.huawei_url = "http://mlops.huawei.com/mlops-service/api/v1/agentService/v1/chat/completions"
            self.headers = {
                'Authorization': bearer_token or 'Bearer sk-eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhY2NvdW50SWQiOiJsMzAwMjI4ODUiLCJhY2NvdW50TmFtZSI6ImxpeXUifQ.OknarADgrBgirjdqiz-7U-Yu_KIlRU7ca_6KZSk-0e0'
            }
        else:
            self.client = None
        self.semaphore = asyncio.Semaphore(100)
        self.original_file_contents = {}  # 存储原始文件内容

    def extract_function_block(self, file_path: str, start_line: int, end_line: int, use_original: bool = False) -> str:
        try:
            if use_original and file_path in self.original_file_contents:
                content = self.original_file_contents[file_path]
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if use_original:
                    self.original_file_contents[file_path] = content
            lines = content.splitlines()
            start_idx = max(0, start_line - 1)  # 行号从 1 开始，索引从 0 开始
            end_idx = min(len(lines), end_line)
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
            call_info.caller_range[0] - 20,
            call_info.caller_range[1] + 10,
            use_original=True  # 使用原始文件内容
        )
        callee_code = self.extract_function_block(
            call_info.callee_file,
            call_info.callee_range[0],
            call_info.callee_range[1]
        )
        caller_lines = caller_code.splitlines()
        call_line_idx = call_info.call_line - (call_info.caller_range[0] - 20)
        call_line_code = caller_lines[call_line_idx].strip() if 0 <= call_line_idx < len(caller_lines) else "Unknown"

        system_prompt = (
            "You are a code analyzer that generates ONE LINE of C++ log using HILOG_<LEVEL> macros WITHOUT parameters.\n\n"
            "Rules:\n"
            "1. Analyze the 'Call at line ...' and its context in the caller function to determine the log level:\n"
            "   - DEBUG: Call is inside a conditional block (if/while/for).\n"
            "   - INFO: Call is standalone (no conditions or error checks).\n"
            "   - WARN: Call is followed by an error check (e.g., checking return value).\n"
            "   - ERROR: Call is within an explicit error handling block (e.g., try-catch).\n"
            "2. Log format: HILOG_<LEVEL>(\"[<COMPONENT>] <verb> <noun>\");\n"
            "3. Use a present-tense <verb> (e.g., 'reads', 'parses') and a <noun> matching the action (e.g., 'file', 'data').\n"
            "4. Do NOT include any parameters; focus only on level, verb, and noun.\n\n"
            "Examples:\n"
            "1. Standalone call:\n"
            "   Call: bool ret = HasSystemCapability(syscapString);\n"
            "   -> HILOG_INFO(\"[COMP] checks capability\");\n"
            "2. Conditional call:\n"
            "   Call: if (ptr) { Process(ptr); }\n"
            "   -> HILOG_DEBUG(\"[COMP] processes data\");\n"
            "3. Error-checked call:\n"
            "   Call: result = Allocate(size); if (!result) { ... }\n"
            "   -> HILOG_WARN(\"[COMP] allocates buffer\");\n\n"
            "Analyze the 'Call at line ...' and generate the log without parameters."
        )
        return f"""{system_prompt}

Caller Function ({call_info.caller_component}):
{caller_code}

Callee Function ({call_info.callee_component}):
{callee_code}

Call at line {call_info.call_line}: {call_line_code}
Caller: {call_info.caller_func}, Callee: {call_info.callee_func}
Component: Caller=[{call_info.caller_component}], Callee=[{call_info.callee_component}]
"""

    def wrap_log_in_conditional(self, log_line: str, param: Optional[str] = None) -> str:
        if not log_line.startswith('HILOG_') or not log_line.endswith(';'):
            base_log = f'HILOG_INFO("[{self.caller_component}] invokes {self.callee_component}")'
        else:
            base_log = log_line

        if param:
            base_log = base_log.replace('");', f' {param} = 0x%x", {param});')

        match = re.match(r'HILOG_(DEBUG|INFO|WARN|ERROR)\((.*)\);', base_log)
        if not match:
            return f'#ifdef HILOG_INFO\nHILOG_INFO("[{self.caller_component}] invokes {self.callee_component}");\n#else\nprintf("[{self.caller_component}] invokes {self.callee_component}\\n");\n#endif'

        level, content = match.groups()
        if "0x%x" in content:
            printf_content = content.replace("0x%x", "0x%08x").replace('",', '\\n",')
        else:
            printf_content = content.replace('")', '\\n")')

        return f'#ifdef HILOG_INFO\n{base_log}\n#else\nprintf({printf_content});\n#endif'

    async def get_llm_response(self, prompt: str, call_info: CallInfo) -> str:
        async with self.semaphore:
            self.caller_component = call_info.caller_component
            self.callee_component = call_info.callee_component

            # 使用原始文件内容提取调用行
            call_line_code = self.extract_function_block(
                call_info.caller_file,
                call_info.call_line - 1,
                call_info.call_line,
                use_original=True  # 使用插入前的原始文件内容
            ).strip()
            caller_context = self.extract_function_block(
                call_info.caller_file,
                call_info.call_line - 5,
                call_info.call_line + 1,
                use_original=True
            )

            if self.use_huawei:
                try:
                    json_data = {
                        "model": "meta-llama-3-1-70b-instruct-20241203161536",
                        "messages": [{"role": "user", "content": prompt}]
                    }
                    response = await asyncio.to_thread(
                        self.session.post,
                        self.huawei_url,
                        headers=self.headers,
                        json=json_data,
                        verify=False
                    )
                    dict0 = json.loads(response.text)
                    raw_response = dict0['choices'][0]['message']['content']
                    answer0 = json.loads(raw_response)
                    log_line = answer0 if isinstance(answer0, str) else f'HILOG_INFO("[{call_info.caller_component}] invokes {call_info.callee_component}")'
                except Exception as e:
                    logger.error(f"Error calling Huawei API: {e}")
                    log_line = f'HILOG_ERROR("[{call_info.caller_component}] handles {call_info.callee_component}")'
            elif self.use_local:
                try:
                    response = await asyncio.to_thread(
                        lambda: requests.post(self.base_url, json={
                            "model": "deepseek-r1:32b-qwen-distill-fp16",
                            "prompt": prompt,
                            "stream": False,
                        })
                    )
                    if response.status_code != 200:
                        logger.error(f"Local LLM API error {response.status_code}: {response.text}")
                        return self.wrap_log_in_conditional(f'HILOG_ERROR("[{call_info.caller_component}] handles {call_info.callee_component}")') + f'\n insert [{call_info.caller_file}:{call_info.call_line}]'

                    raw_text = response.text.strip()
                    try:
                        json_response = json.loads(raw_text)
                        raw_response = json_response.get("response") or json_response.get("text") or json_response.get("output") or ""
                        if not raw_response:
                            raw_response = raw_text
                    except json.JSONDecodeError:
                        logger.info("Local LLM returned plain text instead of JSON")
                        raw_response = raw_text

                    log_line = next(
                        (line.strip() for line in raw_response.split('\n') if line.strip().startswith('HILOG_') and line.strip().endswith(';')),
                        None
                    )
                    if not log_line:
                        logger.warning(f"Local LLM did not return a valid HILOG line, falling back to default: {raw_response}")
                        log_line = f'HILOG_INFO("[{call_info.caller_component}] invokes {call_info.callee_component}")'
                except Exception as e:
                    logger.error(f"Error calling local LLM API: {e}")
                    return self.wrap_log_in_conditional(f'HILOG_ERROR("[{call_info.caller_component}] handles {call_info.callee_component}")') + f'\n insert [{call_info.caller_file}:{call_info.call_line}]'
            else:
                try:
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": prompt}],
                        stream=False
                    )
                    raw_response = response.choices[0].message.content.strip()
                    log_line = next((line.strip() for line in raw_response.split('\n') if line.strip().startswith('HILOG_') and line.strip().endswith(';')), None)
                    if not log_line:
                        log_line = f'HILOG_INFO("[{call_info.caller_component}] invokes {call_info.callee_component}")'
                except Exception as e:
                    logger.error(f"Error calling LLM API: {e}")
                    return self.wrap_log_in_conditional(f'HILOG_ERROR("[{call_info.caller_component}] handles {call_info.callee_component}")') + f'\n insert [{call_info.caller_file}:{call_info.call_line}]'

            # 参数提取：基于原始调用行
            param = None
            call_match = re.search(r'{}\s*\((.*?)\)'.format(re.escape(call_info.callee_func)), call_line_code)
            if call_match:
                args = call_match.group(1).split(',')
                if args and args[0].strip():
                    param = args[0].strip().split()[-1].strip(' *')  # 提取第一个参数的变量名
            else:
                # Fallback: 提取括号中的第一个变量
                call_match = re.search(r'\(([^,)]+)', call_line_code)
                if call_match:
                    param = call_match.group(1).split()[-1].strip(' *')
                # 如果在条件语句中，提取条件变量并调整日志级别
                elif 'if (' in caller_context or 'while (' in caller_context or 'for (' in caller_context:
                    cond_match = re.search(r'(if|while|for)\s*\(([^)]+)', caller_context)
                    if cond_match:
                        cond = cond_match.group(2).split(' ')[0].strip(' !&')
                        param = cond
                        log_line = log_line.replace('HILOG_INFO', 'HILOG_DEBUG')

            # 清理参数名中的语法错误
            if param and '(' in param:
                param = param.split('(')[0].strip()

            return self.wrap_log_in_conditional(log_line, param) + f'\n insert [{call_info.caller_file}:{call_info.call_line}]'

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
        index = max(0, line_number - 1)  # 插入到调用行之前
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
                        help='待分析的目标交互（例如："js -> resource_management_lite"）')
    parser.add_argument('--actual-insert', action='store_true',
                        help='直接将日志插入到源文件')
    parser.add_argument('--base_url', default="http://localhost:11434/api/generate",
                        help='大模型 API 的 Base URL')
    parser.add_argument('--local', action='store_true',
                        help='使用本地大模型')
    parser.add_argument('--use-huawei', action='store_true', help='使用华为 MLOps API')
    parser.add_argument('--bearer-token', help='Bearer token for Huawei API authentication')
    args = parser.parse_args()

    if not args.local and not args.use_huawei and not args.api_key:
        parser.error("非本地模式或非华为模式时必须提供 api_key")
    if args.local and args.base_url == "https://api.deepseek.com":
        args.base_url = "http://localhost:11434/api/generate"

    analyzer = AsyncComponentAnalyzer(api_key=args.api_key, base_url=args.base_url, use_local=args.local, use_huawei=args.use_huawei, bearer_token=args.bearer_token)
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
