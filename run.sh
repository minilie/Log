#!/bin/bash

# 编译数据库路径
COMPILE_COMMANDS="compile_commands.json"
# 分析器可执行文件路径
ANALYZER_PATH="./function-analyzer"                
# 配置文件路径
CONFIG_FILE="config.yaml"                 
# 输出文件路径
OUTPUT_FILE="analysis_results.txt"                  
# 是否开启调试模式
DEBUG_MODE=false                           
# 是否输出所有调用
OUTPUT_ALL_CALLS=false

CMD="python3 asyn.py"
DMC="python3 divide.py $OUTPUT_FILE"

CMD="$CMD $COMPILE_COMMANDS $ANALYZER_PATH"

CMD="$CMD --config $CONFIG_FILE"
CMD="$CMD --output $OUTPUT_FILE"

if [ "$DEBUG_MODE" = true ]; then
    CMD="$CMD --debug-mode"
fi

if [ "$OUTPUT_ALL_CALLS" = true ]; then
    CMD="$CMD --output-all-calls"
fi

OUTPUT1=$($CMD | tee /dev/tty)

ANALYZED_WITH_OUTPUT=$(echo "$OUTPUT1" | grep "Successfully analyzed with output:" | tail -n1 | awk '{print $5}')

OUTPUT2=$($DMC | tee /dev/tty)

TOTAL_FILES_ANALYZED=$(echo "$OUTPUT2" | grep "Total files analyzed:" | cut -d':' -f2 | tr -d ' ')

echo -e "\nChecking number consistency:"
if [ "$ANALYZED_WITH_OUTPUT" = "$TOTAL_FILES_ANALYZED" ]; then
    echo "✓ Numbers match: Both show $ANALYZED_WITH_OUTPUT files"
else
    echo "✗ Warning: Numbers don't match!"
    echo "Successfully analyzed with output: $ANALYZED_WITH_OUTPUT"
    echo "Total files analyzed: $TOTAL_FILES_ANALYZED"
fi
