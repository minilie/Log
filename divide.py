#!/usr/bin/env python3
import re
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import sys
import yaml

class ComponentConfig:
    def __init__(self, config_file: str = "config.yaml"):
        self.config = self._load_config(config_file)
        self.component_pattern = self._build_component_pattern()

    def _load_config(self, config_file: str) -> dict:
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")
            return {
                "project_root": "OH5", 
                "component_level": 3,
                "filter_operator_calls": True
            }

    # 按设置的分级来构建正则表达式
    def _build_component_pattern(self) -> str:
        root = self.config["project_root"]
        level = self.config["component_level"]
        pattern = f"{root}/" + "/".join(["[^/]+" for _ in range(level-1)]) + "/([^/]+)/"
        return pattern

    def extract_component_name(self, filepath: str) -> str:
        match = re.search(self.component_pattern, filepath)
        return match.group(1) if match else None

class ComponentAnalyzer:
    def __init__(self, config_file: str = "config.yaml"):
        self.config = ComponentConfig(config_file)
        self.components = {}
        self.function_to_component = {}
        self.component_functions = defaultdict(set)
        self.function_calls = defaultdict(list)
        self.header_dependencies = defaultdict(lambda: defaultdict(set))
        self.header_component_map = {}
        self.function_ranges = {}
        self.function_locations = {}
        self.file_count = 0

    def is_system_header(self, filepath: str) -> bool:
        system_paths = [
            "/usr/include/",
            "/usr/local/include/",
            "/usr/lib/gcc/",
            "/usr/lib/llvm",
            "/usr/lib/clang"
        ]
        return any(path in filepath for path in system_paths)

    def is_source_file(self, filepath: str) -> bool:
        return filepath.endswith(('.c', '.cpp', '.cc', '.cxx'))

    def extract_component_name(self, filepath: str) -> str:
        return self.config.extract_component_name(filepath)

    def is_operator_function(self, func_name: str) -> bool:
        if func_name.startswith('operator'):
            return True
        if '::operator' in func_name:
            return True
        return False

    def update_function_info(self, function: str, file: str, start_line: int, end_line: int) -> bool:
        is_in_source = self.is_source_file(file)

        if function not in self.function_locations:
            self.function_locations[function] = file
            self.function_ranges[function] = (start_line, end_line)
            return True

        current_file = self.function_locations[function]
        current_is_source = self.is_source_file(current_file)

        if is_in_source and not current_is_source:
            self.function_locations[function] = file
            self.function_ranges[function] = (start_line, end_line)
            return True

        return False

    # 去重模板特化
    def deduplicate_template_calls(self, calls: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
        unique_calls = {}

        for called_func, caller_file, line, _ in calls:
            actual_file = self.function_locations.get(called_func)
            actual_range = self.function_ranges.get(called_func)

            if actual_file and actual_range:
                caller_range = self.function_ranges.get(called_func)
                callee_range = actual_range

                if caller_range:
                    line_num = int(line)
                    if not (caller_range[0] <= line_num <= caller_range[1]):
                        key = (called_func, actual_file,
                               f"{caller_range[0]}-{caller_range[1]}",
                               f"{callee_range[0]}-{callee_range[1]}")
                    else:
                        key = (called_func, caller_file, line)
                else:
                    key = (called_func, caller_file, line)

                unique_calls[key] = (called_func, caller_file, line, actual_file)
            else:
                key = (called_func, caller_file, line)
                unique_calls[key] = (called_func, caller_file, line, _)

        return list(unique_calls.values())

    def parse_function_analysis(self, data: str):
        current_file = None
        current_function = None
        current_header = None
        in_header_section = False

        for line in data.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('Header File: '):
                current_header = line.replace('Header File: ', '').strip()
                if not self.is_system_header(current_header):
                    component = self.extract_component_name(current_header)
                    if component:
                        self.header_component_map[current_header] = component
                in_header_section = True
                continue
            elif line.startswith('='):
                continue

            if line.startswith('Analysis results for'):
                current_file = line.replace('Analysis results for', '').strip(':').strip()
                if current_file:
                    self.file_count += 1
                component = self.extract_component_name(current_file)
                if component:
                    self.components[current_file] = component
                in_header_section = False
                continue

            if line.startswith('Function'):
                match = re.match(r'Function (.*?) from line (\d+) to line (\d+) calls:', line)
                if match:
                    current_function = match.group(1).strip()
                    start_line = int(match.group(2))
                    end_line = int(match.group(3))

                    file_to_use = current_file if not in_header_section else current_header
                    self.update_function_info(current_function, file_to_use, start_line, end_line)

                    if not in_header_section and self.components.get(current_file):
                        component = self.components[current_file]
                        self.function_to_component[current_function] = component
                        self.component_functions[component].add(current_function)
                    elif current_header in self.header_component_map:
                        component = self.header_component_map[current_header]
                        self.function_to_component[current_function] = component
                        self.component_functions[component].add(current_function)
                continue

            if line.startswith('- ') and current_function:
                base_match = re.match(r'-\s+(.*?)\s+(?:defined in|called at line)', line)
                if not base_match:
                    continue

                called_function = base_match.group(1).strip()
                if self.config.config.get('filter_operator_calls', True) and self.is_operator_function(called_function):
                    continue

                full_match = re.match(
                    r'-\s+(.*?)\s+defined in\s+(.*?)\s+from line\s+(\d+)\s+to line\s+(\d+)\s+called at line\s+(\d+)',
                    line
                )

                if full_match:
                    callee_file = full_match.group(2)
                    callee_start = int(full_match.group(3))
                    callee_end = int(full_match.group(4))
                    call_line = full_match.group(5)

                    self.update_function_info(called_function, callee_file, callee_start, callee_end)
                else:
                    simple_match = re.match(
                        r'-\s+(.*?)\s+defined in\s+(.*?)\s+called at line\s+(\d+)',
                        line
                    )
                    if simple_match:
                        callee_file = simple_match.group(2)
                        call_line = simple_match.group(3)
                    else:
                        basic_match = re.match(r'-\s+(.*?)\s+called at line\s+(\d+)', line)
                        if basic_match:
                            call_line = basic_match.group(2)
                            callee_file = "unknown location"

                # 不管是源文件还是头文件，都记录到 function_calls 中
                file_to_use = current_file if not in_header_section else current_header
                self.function_calls[current_function].append(
                    (called_function, file_to_use, call_line, callee_file)
                )

                # 头文件的依赖关系也单独记录
                if in_header_section and current_header and not self.is_system_header(current_header):
                    self.header_dependencies[current_header][current_function].add(
                        (called_function, call_line, callee_file)
                    )

    # 查找组件交互
    def find_component_interactions(self) -> List[Tuple[str, str, str, str, str, str, str, Tuple[int, int], Tuple[int, int]]]:
        interactions = []

        for caller_func, called_info in self.function_calls.items():
            caller_component = self.function_to_component.get(caller_func)
            if not caller_component:
                continue

            unique_calls = self.deduplicate_template_calls(called_info)

            for called_func, file, line, callee_file in unique_calls:
                callee_component = None
                if callee_file and callee_file != "unknown location":
                    callee_component = self.extract_component_name(callee_file)

                if not callee_component:
                    callee_component = self.function_to_component.get(called_func)

                if callee_component and caller_component != callee_component:
                    caller_range = self.function_ranges.get(caller_func, (None, None))
                    callee_range = self.function_ranges.get(called_func, (None, None))

                    interactions.append((
                        caller_component,
                        callee_component,
                        caller_func,
                        called_func,
                        file,
                        line,
                        callee_file,
                        caller_range,
                        callee_range
                    ))

        return interactions

    def generate_report(self) -> str:
        report = []
        report.append("Component Analysis Report")
        report.append("=" * 50)

        # 1. 组件函数统计
        report.append("\n1. Component Function Statistics:")
        for component in sorted(self.component_functions.keys()):
            num_functions = len(self.component_functions[component])
            num_outgoing_calls = len([f for f in self.function_calls.keys()
                                      if self.function_to_component.get(f) == component])
            report.append(f"   - {component}:")
            report.append(f"     * Number of defined functions: {num_functions}")
            report.append(f"     * Number of external calls: {num_outgoing_calls}")

        # 2. 组件调用依赖
        report.append("\n2. Component Call Dependencies:")
        interactions = self.find_component_interactions()
        interaction_map = defaultdict(list)

        for interaction in interactions:
            (caller_comp, callee_comp, caller_func, called_func,
             caller_file, line, callee_file, caller_range, callee_range) = interaction
            key = (caller_comp, callee_comp)
            interaction_map[key].append((
                caller_func, called_func, caller_file, line,
                callee_file, caller_range, callee_range
            ))

        for (caller_comp, callee_comp), calls in sorted(interaction_map.items()):
            report.append(f"\n   {caller_comp} -> {callee_comp}:")
            for (caller_func, called_func, caller_file, line,
                 callee_file, caller_range, callee_range) in sorted(set(calls)):
                report.append(f"     * {caller_func} calls {called_func}")
                report.append(f"       from {caller_file} line {line}")
                report.append(f"       to {callee_file}")
                if caller_range[0] is not None:
                    report.append(f"       caller range: {caller_range[0]}-{caller_range[1]}")
                if callee_range[0] is not None:
                    report.append(f"       callee range: {callee_range[0]}-{callee_range[1]}")

        # 3. 头文件依赖分析（只显示组件内调用）
        report.append("\n3. Header File Dependencies:")
        component_calls = defaultdict(list)
        for header, functions in sorted(self.header_dependencies.items()):
            if header in self.header_component_map:
                component = self.header_component_map[header]
                for caller, calls in functions.items():
                    unique_calls = self.deduplicate_template_calls(
                        [(c[0], header, c[1], c[2]) for c in calls]
                    )
                    for called_func, _, line, callee_file in sorted(unique_calls):
                        caller_component = self.function_to_component.get(caller)
                        called_component = None

                        if callee_file and callee_file != "unknown location":
                            called_component = self.extract_component_name(callee_file)

                        if not called_component:
                            called_component = self.function_to_component.get(called_func)

                        # 仅输出组件内调用
                        if called_component and caller_component and called_component != caller_component:
                            continue

                        caller_range = self.function_ranges.get(caller)
                        callee_range = self.function_ranges.get(called_func)
                        component_calls[component].append((
                            caller, called_func, header, line, callee_file,
                            caller_range, callee_range
                        ))

        for component in sorted(component_calls.keys()):
            report.append(f"\n   {component}:")
            for (caller, called_func, header, line, callee_file,
                 caller_range, callee_range) in sorted(set(component_calls[component])):
                report.append(f"     * {caller} calls {called_func}")
                report.append(f"       from {header} line {line}")
                report.append(f"       to {callee_file}")
                if caller_range:
                    report.append(f"       caller range: {caller_range[0]}-{caller_range[1]}")
                if callee_range:
                    report.append(f"       callee range: {callee_range[0]}-{callee_range[1]}")

        # 4. 总体统计
        report.append("\n4. Overall Statistics:")
        report.append(f"   - Total components: {len(self.component_functions)}")
        report.append(f"   - Total functions: {len(self.function_to_component)}")
        report.append(f"   - Total inter-component calls: {sum(len(calls) for calls in interaction_map.values())}")
        report.append(f"   - Total files analyzed: {self.file_count}")

        return "\n".join(report)

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py analysis_results.txt")
        sys.exit(1)

    try:
        # 读取分析结果文件
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            analysis_data = f.read()

        # 创建分析器并解析数据
        analyzer = ComponentAnalyzer()
        analyzer.parse_function_analysis(analysis_data)

        # 生成并输出报告
        report = analyzer.generate_report()
        print(report)

        # 保存报告到文件
        output_file = "component_analysis_report.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: File not found {sys.argv[1]}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
