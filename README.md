# 操作手册

## 1. 安装与配置

### 1.1 容器环境设置

使用提供的Dockerfile创建运行环境：

```bash
# 构建容器镜像
docker build -t function-analyzer .

# 运行容器，挂载代码目录
docker run -it --rm --network host -v $(pwd):/Log -v /home/minilie/ohos/OH5:/home/minilie/ohos/OH5 function-analyzer
#/home/minilie/ohos/OH5 为OpenHarmony源码所在目录
```

若不使用Docker

```bash
# Ubuntu 22.04
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    wget \
    git \
    llvm-14 \
    llvm-14-dev \
    clang-14 \
    libclang-14-dev
    
pip3 install --no-cache-dir \
    openai \
    pyyaml \
    requests \
    typing \
    asyncio
```



### 1.2 配置文件

系统使用YAML格式的配置文件(`config.yaml`)，包含以下主要配置项：

```yaml
# 项目根目录名称
project_root: "OH5"
# 组件分级层次（默认值为3）
component_level: 3
# 是否过滤运算符调用
filter_operator_calls: true
# 批处理大小
batch_size: 20
```

## 2. 使用流程

### 2.1 编译分析工具

首先需要编译分析工具，可以使用以下命令：

```bash
bash make_build.sh
```

或

```bash
rm -rf build
mkdir build
cd build/
cmake ..
make
cp function-analyzer ../
```

这里还需要将待插入的系统编译库拷贝到当前文件夹下

```bash
#在OpenHarmony源码目录下编译
hb build --gn-flags=--export-compile-commands # 编译导出编译数据库

cp /home/minilie/ohos/OH5/out/riscv32_virt/qemu_riscv_mini_system_demo/compile_commands.json .
```

### 2.2 运行代码分析

使用`asyn.py`处理编译命令并分析函数调用关系：

```bash
python3 asyn.py compile_commands.json function-analyzer --config config.yaml --output analysis_results.txt
```

参数说明：
- `compile_commands.json`: 编译数据库文件路径
- `function-analyzer`: 函数分析器可执行文件路径
- `--config`: 配置文件路径
- `--output`: 输出文件路径
- `--output-all-calls`（可选）: 输出所有函数调用行
- `--debug-mode`（可选）: 启用调试输出

### 2.3 组件交互分析

使用`divide.py`分析组件间的调用关系：

```bash
python3 divide.py analysis_results.txt
```

执行后将生成组件分析报告，包含：
- 组件函数统计
- 组件调用依赖
- 头文件依赖分析
- 总体统计信息

### 2.4 自动生成并插入日志

使用`LLM_insert.py`基于分析结果自动生成日志语句：

```bash
# 预览指定组件之间的交互日志
cat component_analysis_report.txt | python3 LLM_insert.py YOUR_API_KEY --target "arch -> kernel"

# 直接插入日志到源文件
cat component_analysis_report.txt | python3 LLM_insert.py YOUR_API_KEY --target "arch -> kernel" --actual-insert

# 使用本地LLM模型
cat component_analysis_report.txt | python3 LLM_insert.py --local --base_url "http://localhost:11434/api/generate"
```

参数说明：
- `YOUR_API_KEY`: 大语言模型API密钥（使用`--local`时可省略）
- `--target`: 指定要分析的组件交互（格式：`组件A -> 组件B`）具体组件可查看analysis_results.txt
- `--actual-insert`: 实际插入日志到源文件（不加此参数仅预览）
- `--base_url`: API基础URL（默认为`https://api.deepseek.com`）
  - 本地大语言模型URL（默认为`http://localhost:11434/api/generate`）

- `--local`: 使用本地大语言模型

## 3. 日志格式

系统生成的日志遵循以下格式：

```c
// 带参数的格式
HILOG_<LEVEL>("[<CALLER_COMPONENT>] <operation> <target>, <param_name>=0x%x", (UINTPTR)&param);

// 不带参数的格式
HILOG_<LEVEL>("[<CALLER_COMPONENT>] <operation> <target>");
```

日志级别(`<LEVEL>`)选择规则：
- `DEBUG`: 条件检查
- `INFO`: 正常操作
- `WARN`: 潜在问题
- `ERROR`: 失败情况

所有日志都会被包装在条件编译块中：

```c
#ifdef HILOG_INFO
HILOG_INFO("[COMPONENT] operation target");
#else
printf("[COMPONENT] operation target\n");
#endif
```

## 4. 输出文件说明

系统会生成以下输出文件：

- `analysis_results.txt`: 函数分析结果
- `component_analysis_report.txt`: 组件交互分析报告
- `analysis_results/failed_analysis.log`: 分析失败的文件记录
- `analysis_results/success_no_output.log`: 成功分析但无输出的文件
- `analysis_results/partial_success.log`: 部分成功（有警告）的分析记录

## 5. 高级配置

### 5.1 组件识别配置

在`config.yaml`中，可以调整组件识别方式：

```yaml
# 项目根目录名称
project_root: "OH5"
# 组件分级层次（值越大，组件粒度越细）
component_level: 3
```

对应的组件路径解析规则为：`OH5/level1/level2/[component_name]/...`

### 5.2 批处理配置

调整处理并发度：

```yaml
# 批处理大小（值越大并发度越高，但可能消耗更多资源）
batch_size: 20
```

## 6. 示例工作流

完整分析工作流示例：

```bash
# 1. 编译分析工具
bash make_build.sh

# 2. 运行函数分析
python3 asyn.py compile_commands.json function-analyzer --config config.yaml --output analysis_results.txt

# 3. 分析组件交互
python3 divide.py analysis_results.txt

# 2、3步骤可替换为
bash run.sh

# 4. 预览需要插入的日志
cat component_analysis_report.txt | python3 LLM_insert.py YOUR_API_KEY

# 5. 插入日志到特定组件交互
cat component_analysis_report.txt | python3 LLM_insert.py YOUR_API_KEY --target "arch -> kernel" --actual-insert

# 6. 插入日志整个系统
cat component_analysis_report.txt | python3 LLM_insert.py  --local --actual-insert （调用本地大模型）
```



通过以上步骤，系统会自动分析代码库中的组件交互，并在适当位置插入上下文相关的日志语句，提高代码的可调试性和可观测性。
