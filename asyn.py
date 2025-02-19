#!/usr/bin/env python3
import json
import os
import time
import asyncio
import argparse
from typing import List, Dict, Tuple
import logging
import yaml

# 加载批处理数量
def load_batch_size(config_path: str) -> int:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['batch_size']
    except Exception as e:
        logging.error(f"Failed to load config file: {str(e)}")
        return 20

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 提取必要编译选项
def filter_compile_options(command: str) -> List[str]:
    parts = command.split()
    relevant_opts = []
    seen_opts = set()
    skip_next = False

    for i, part in enumerate(parts):
        if skip_next:
            skip_next = False
            continue

        if part in seen_opts:
            continue
        seen_opts.add(part)

        if part.startswith(('-I', '-D', '-std=')):
            relevant_opts.append(part)
        elif any(part.endswith(ext) for ext in ('.cpp', '.c', '.cc')):
            relevant_opts.append(part)
        elif part in ('-O2', '-Os', '-march', '-mtune', '-mabi', '-static', '-o'):
            skip_next = True

    return relevant_opts

# 运行单一命令
def process_compile_command(entry: Dict) -> Dict:
    result = {}
    directory = entry.get('directory', '')

    if 'file' in entry:
        orig_file = entry['file']
        if os.path.isabs(entry['file']): # 绝对路径生成
            result['file'] = entry['file']
        else:
            result['file'] = os.path.abspath(os.path.join(directory, entry['file']))

    if 'command' in entry:
        opts = filter_compile_options(entry['command'])
        processed_opts = []

        for opt in opts:
            if opt.startswith('-I'):
                include_path = opt[2:]
                if not os.path.isabs(include_path):
                    abs_path = os.path.abspath(os.path.join(directory, include_path))
                    processed_opts.append(f'-I{abs_path}')
                else:
                    processed_opts.append(opt)
            elif opt == orig_file:  # 如果是源文件路径，使用绝对路径
                processed_opts.append(result['file'])
            else:
                processed_opts.append(opt)

        result['filtered_command'] = processed_opts

    return result

# 分析单个文件
async def analyze_file(analyzer_path: str, source_file: str, options: List[str], output_all_calls: bool = False, debug_mode: bool = False) -> Tuple[bool, str, str, List[str]]:
    cmd = [f"./{analyzer_path}"]
    if output_all_calls:
        cmd.append("--output-all-calls")
    if debug_mode:
        cmd.append("--debug-mode")
    cmd.extend([source_file, '--'] + options)

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode() if stdout else ""
        stderr_text = stderr.decode() if stderr else ""

        has_valid_output = bool(stdout_text and "Function" in stdout_text and "calls:" in stdout_text)

        # 返回码非0，有有效输出也视为成功
        if has_valid_output:
            return True, stdout_text, stderr_text, cmd
        elif process.returncode == 0:
            return True, "", "", cmd
        else:
            # 没有有效输出也返回非0，视为真正的错误
            return False, stdout_text, stderr_text, cmd

    except Exception as e:
        return False, "", str(e), cmd

# 批处理文件
async def process_files(files_to_process: List[Tuple[str, List[str]]], analyzer_path: str, output_file: str, output_all_calls: bool = False, debug_mode: bool = False, batch_size: int = 3) -> Tuple[int, int, int]:
    successful = 0
    successful_no_output = 0
    failed = 0
    total_files = len(files_to_process)
    start_time = time.time()

    results_dir = "analysis_results"
    os.makedirs(results_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f, \
         open(os.path.join(results_dir, "failed_analysis.log"), 'w', encoding='utf-8') as fail_f, \
         open(os.path.join(results_dir, "success_no_output.log"), 'w', encoding='utf-8') as no_output_f, \
         open(os.path.join(results_dir, "partial_success.log"), 'w', encoding='utf-8') as partial_f:

        f.write("=== Function Analysis Results ===\n")
        fail_f.write("=== Failed Analysis Details ===\n")
        no_output_f.write("=== Successfully Analyzed Files with No Output ===\n")
        partial_f.write("=== Partially Successful Analysis (with warnings) ===\n")

        for i in range(0, total_files, batch_size):
            batch = files_to_process[i:i + batch_size]
            tasks = []

            for source_file, options in batch:
                task = analyze_file(analyzer_path, source_file, options,
                                  output_all_calls=output_all_calls,
                                  debug_mode=debug_mode)
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            for idx, (success, stdout, stderr, cmd) in enumerate(results):
                source_file = batch[idx][0]

                if success:
                    if stdout:
                        successful += 1
                        f.write(f"\nAnalysis results for {source_file}:\n{stdout}\n")
                        if stderr:  # 有警告但分析成功
                            partial_f.write(f"\n{source_file}:\n")
                            partial_f.write(f"Warnings:\n{stderr}\n")
                            partial_f.write("-" * 80 + "\n")
                    else:
                        successful_no_output += 1
                        no_output_f.write(f"{source_file}\n")
                else:
                    failed += 1
                    fail_f.write(f"\nFailed to analyze {source_file}:\n")
                    fail_f.write(f"Command: {' '.join(cmd)}\n")
                    if stdout:
                        fail_f.write(f"Stdout:\n{stdout}\n")
                    if stderr:
                        fail_f.write(f"Stderr:\n{stderr}\n")
                    fail_f.write("-" * 80 + "\n")

            total_processed = i + len(batch)
            elapsed_time = time.time() - start_time
            speed = total_processed / elapsed_time if elapsed_time > 0 else 0

            print(f'\rProcessing: {total_processed}/{total_files} '
                  f'({(total_processed/total_files)*100:.1f}%) '
                  f'Speed: {speed:.1f} files/s', end='', flush=True)

    print("\nAnalysis Statistics:")
    print(f"Total files processed: {total_files}")
    print(f"Successfully analyzed with output: {successful}")
    print(f"Successfully analyzed without output: {successful_no_output}")
    print(f"Failed to analyze: {failed}")
    print(f"\nResults saved to directory: {results_dir}")
    print(f"- Analysis results: {output_file}")
    print(f"- Failed analysis log: {os.path.join(results_dir, 'failed_analysis.log')}")
    print(f"- No output log: {os.path.join(results_dir, 'success_no_output.log')}")
    print(f"- Partial success log: {os.path.join(results_dir, 'partial_success.log')}")

    return successful, successful_no_output, failed

async def main():
    parser = argparse.ArgumentParser(description='Run function analyzer on compilation database')
    parser.add_argument('compile_commands', help='Path to compile_commands.json')
    parser.add_argument('analyzer_path', help='Path to function-analyzer executable')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', default='analysis_results.txt',
                       help='Output file path')
    parser.add_argument('--output-all-calls', action='store_true',
                       help='Output all function call lines')
    parser.add_argument('--debug-mode', action='store_true',
                       help='Enable debug output')
    args = parser.parse_args()

    batch_size = load_batch_size(args.config)

    try:
        with open(args.compile_commands, 'r') as f:
            compile_commands = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read compile_commands.json: {str(e)}")
        return

    files_to_process = []
    for entry in compile_commands:
        try:
            processed = process_compile_command(entry)
            if 'file' in processed and 'filtered_command' in processed:
                files_to_process.append((
                    processed['file'],
                    processed['filtered_command']
                ))
        except Exception as e:
            logging.error(f"Failed to process entry: {str(e)}")

    successful, successful_no_output, failed = await process_files(
        files_to_process,
        args.analyzer_path,
        args.output,
        output_all_calls=args.output_all_calls,
        debug_mode=args.debug_mode,
        batch_size=batch_size
    )

    print("\nAnalysis completed!")
    print(f"Total files: {len(files_to_process)}")
    print(f"Successfully analyzed with output: {successful}")
    print(f"Successfully analyzed without output: {successful_no_output}")
    print(f"Failed to analyze: {failed}")
    print(f"Results saved to: {args.output}")

if __name__ == '__main__':
    asyncio.run(main())
