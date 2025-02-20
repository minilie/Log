/*************************************************************************
	> File Name: function-analyzer.cpp
	> Author:Xin 
	> Mail:minilie270@gmail.com
	> Created Time: Fri 17 Jan 2025 11:36:10 PM CST
 ************************************************************************/

#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <iostream>

using namespace clang;
using namespace clang::tooling;

static llvm::cl::opt<bool> OutputAllCalls(
    "output-all-calls", 
    llvm::cl::desc("Output all function call lines"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> DebugMode(
    "debug-mode",
    llvm::cl::desc("Enable debug output"),
    llvm::cl::init(false));

struct LocationInfo {
    std::string filename;
    unsigned startLine;
    unsigned endLine;
    bool isInHeader;
};

struct CallInfo {
    std::string callee;
    unsigned line;
    std::string callExpr;
    std::string filename;
};

struct HeaderInfo {
    std::string headerFile;
    std::map<std::string, std::vector<CallInfo> > dependencies;
    std::set<std::string> functions;
};

class QuietDiagnosticConsumer : public DiagnosticConsumer {
public:
    void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel, const Diagnostic &Info) override {
        if (DiagLevel == DiagnosticsEngine::Fatal) {
            DiagnosticConsumer::HandleDiagnostic(DiagLevel, Info);
        }
    }
};

class FuncDependencyVisitor : public RecursiveASTVisitor<FuncDependencyVisitor> {
private:
    std::map<std::string, std::vector<CallInfo> > sourceDependencies;
    std::vector<HeaderInfo> headerInfos;
    std::map<std::string, LocationInfo> functionLocations;
    std::set<std::string> processedCalls;
    std::string currentFunction;
    std::string currentFile;
    bool isCurrentFunctionInHeader;
    SourceManager& SM;
    LangOptions LangOpts;

public:
    explicit FuncDependencyVisitor(SourceManager& SM) : SM(SM) {
        LangOpts = LangOptions();
    }

    bool shouldProcessNode(SourceLocation loc) {
        if (loc.isInvalid()) return false;
        return !SM.isInSystemHeader(loc);
    }

    bool isHeaderFile(SourceLocation loc) {
        if (loc.isInvalid()) return false;
        StringRef filename = SM.getFilename(loc);
        return filename.endswith(".h") || filename.endswith(".hpp") || filename.endswith(".hxx") || filename.endswith(".hh") || filename.endswith(".inl");
    }

    std::string getFullFunctionName(const FunctionDecl* func) {
        if (!func) return "unknown_function";

        std::string name;
        if (const auto* method = dyn_cast<CXXMethodDecl>(func)) {
            if (const auto* parent = dyn_cast<CXXRecordDecl>(method->getParent())) {
                name = parent->getNameAsString() + "::";
            }
        }
        name += func->getNameAsString();
        return name;
    }

    bool isFirstDefinition(const std::string& funcName) {
        return functionLocations.find(funcName) == functionLocations.end();
    }

    void updateFunctionLocation(const std::string& funcName, const LocationInfo& info) {
        if (isFirstDefinition(funcName)) {
            functionLocations[funcName] = info;
            if (DebugMode) {
                std::cout << "\nRecorded first definition of " << funcName << ":\n" << "  File: " << info.filename << "\n" << "  Lines: " << info.startLine << "-" << info.endLine << "\n" << "  Is in header: " << (info.isInHeader ? "yes" : "no") << "\n";
            }
        }
    }

    bool VisitFunctionDecl(const FunctionDecl* func) {
        if (!func) return true;
    
        std::string funcName = getFullFunctionName(func);
        SourceLocation loc = func->getLocation();
        if (!shouldProcessNode(loc)) return true;
    
        SourceLocation expansionLoc = SM.getExpansionLoc(loc);
        std::string filename = SM.getFilename(expansionLoc).str();
        bool isHeader = isHeaderFile(expansionLoc);
        unsigned declLine = SM.getExpansionLineNumber(expansionLoc);
    
        LocationInfo locInfo;
        bool hasImplementation = false;
    
        // 检查函数实现
        if (func->hasBody()) {
            const Stmt* body = func->getBody();
            SourceLocation bodyStart = body->getBeginLoc();
            SourceLocation bodyEnd = body->getEndLoc();
    
            if (bodyStart.isValid() && bodyEnd.isValid()) {
                SourceLocation expStartLoc = SM.getExpansionLoc(bodyStart);
                SourceLocation expEndLoc = SM.getExpansionLoc(bodyEnd);
                std::string implFilename = SM.getFilename(expStartLoc).str();
    
                locInfo = {
                    implFilename,  // 实现的文件路径
                    SM.getExpansionLineNumber(expStartLoc),
                    SM.getExpansionLineNumber(expEndLoc),
                    isHeaderFile(expStartLoc)
                };
                hasImplementation = true;
    
                if (DebugMode) {
                    std::cout << "\nFound implementation of " << funcName << " at " << locInfo.filename << ":" << locInfo.startLine << "-" << locInfo.endLine << "\n";
                }
            }
        }
    
        // 没有实现，用声明的位置
        if (!hasImplementation) {
            locInfo = {
                filename,
                declLine,
                declLine,
                isHeader
            };
    
            if (DebugMode) {
                std::cout << "\nFound only declaration of " << funcName << " at " << locInfo.filename << ":" << locInfo.startLine << "\n";
            }
        }
    
        // 更新函数位置信息
        if (isFirstDefinition(funcName)) {
            if (DebugMode) {
                std::cout << "Recording " << (hasImplementation ? "implementation" : "declaration") << " of " << funcName << " at " << locInfo.filename << ":" << locInfo.startLine << "\n";
            }
            updateFunctionLocation(funcName, locInfo);
        }
    
        // 更新当前上下文
        currentFunction = funcName;
        currentFile = filename;
        isCurrentFunctionInHeader = isHeader;
    
        // 处理头文件相关信息
        if (isHeader) {
            auto it = std::find_if(headerInfos.begin(), headerInfos.end(),
                [&filename](const HeaderInfo& info) { return info.headerFile == filename; });
    
            if (it == headerInfos.end()) {
                HeaderInfo newHeader{filename, {}, {funcName}};
                headerInfos.push_back(newHeader);
            } else {
                it->functions.insert(funcName);
            }
        }
    
        return true;
    }

    bool VisitCallExpr(const CallExpr* call) {
        if (!call || currentFunction.empty()) return true;
    
        if (const FunctionDecl* func = call->getDirectCallee()) {
            SourceLocation callLoc = call->getBeginLoc();
            if (!shouldProcessNode(callLoc)) return true;
    
            if (DebugMode) {
                std::cout << "\n==== Call Expression Debug Info ====\n";
    
                std::cout << "Basic call info:\n";
                std::cout << "  Current function: " << currentFunction << "\n";
                std::cout << "  Called function: " << getFullFunctionName(func) << "\n";
    
                SourceLocation expansionLoc = SM.getExpansionLoc(callLoc);
                SourceLocation spellingLoc = SM.getSpellingLoc(callLoc);
    
                std::cout << "\nLocation info:\n";
                std::cout << "  Expansion line: " << SM.getExpansionLineNumber(callLoc) << "\n";
                std::cout << "  Spelling line: " << SM.getSpellingLineNumber(callLoc) << "\n";
                std::cout << "  Expansion file: " << SM.getFilename(expansionLoc).str() << "\n";
                std::cout << "  Spelling file: " << SM.getFilename(spellingLoc).str() << "\n";
    
                std::cout << "\nTemplate info:\n";
                if (const FunctionTemplateDecl* tpl = func->getDescribedFunctionTemplate()) {
                    std::cout << "  Is template definition\n";
                    SourceLocation templateLoc = tpl->getLocation();
                    if (templateLoc.isValid()) {
                        std::cout << "  Template definition at: " << SM.getFilename(templateLoc).str() << ":" << SM.getExpansionLineNumber(templateLoc) << "\n";
                    }
                }
                if (func->isTemplateInstantiation()) {
                    std::cout << "  Is template instantiation\n";
                    if (const FunctionDecl* pattern = func->getTemplateInstantiationPattern()) {
                        SourceLocation patternLoc = pattern->getLocation();
                        if (patternLoc.isValid()) {
                            std::cout << "  Pattern definition at: " << SM.getFilename(patternLoc).str() << ":" << SM.getExpansionLineNumber(patternLoc) << "\n";
                        }
                    }
                }
    
                std::string callExpr;
                SourceRange range = call->getSourceRange();
                if (range.isValid()) {
                    CharSourceRange charRange = CharSourceRange::getTokenRange(
                        SM.getExpansionLoc(range.getBegin()),
                        SM.getExpansionLoc(range.getEnd())
                    );
                    callExpr = Lexer::getSourceText(charRange, SM, LangOpts).str();
                    std::cout << "\nCall expression:\n  " << callExpr << "\n";
                }
    
                std::cout << "================================\n";
            }
    
            SourceLocation expansionLoc = SM.getExpansionLoc(callLoc);
            std::string callee = getFullFunctionName(func);
            unsigned line = SM.getExpansionLineNumber(callLoc);
            std::string callFilename = SM.getFilename(expansionLoc).str();
    
            std::string callExpr;
            SourceRange range = call->getSourceRange();
            if (range.isValid()) {
                CharSourceRange charRange = CharSourceRange::getTokenRange(
                    SM.getExpansionLoc(range.getBegin()),
                    SM.getExpansionLoc(range.getEnd())
                );
                callExpr = Lexer::getSourceText(charRange, SM, LangOpts).str();
            }
    
            SourceLocation funcDefLoc = func->getLocation();
            std::string defFilename;
            if (funcDefLoc.isValid()) {
                defFilename = SM.getFilename(SM.getExpansionLoc(funcDefLoc)).str();
            }

            std::string callKey = currentFunction + "::" + callee + "@" + std::to_string(line);
            if (processedCalls.find(callKey) != processedCalls.end()) {
                if (DebugMode) {
                    std::cout << "Skipping duplicate call: " << callKey << "\n";
                }
                return true;
            }
            processedCalls.insert(callKey);
    
            CallInfo info{callee, line, callExpr, defFilename};
            bool callInHeader = isHeaderFile(expansionLoc);
    
            if (callInHeader) {
                auto it = std::find_if(headerInfos.begin(), headerInfos.end(),
                    [&callFilename](const HeaderInfo& info) { return info.headerFile == callFilename; });
    
                if (it == headerInfos.end()) {
                    HeaderInfo newHeader{callFilename, {{currentFunction, {info}}}, {}};
                    headerInfos.push_back(newHeader);
                } else {
                    it->dependencies[currentFunction].push_back(info);
                }
            } else {
                sourceDependencies[currentFunction].push_back(info);
            }
        }
        return true;
    }

    void printDependencies() const {
        // 不是输出所有调用的话，记录已输出过
        std::map<std::string, std::set<std::string> > printedCalls;
    
        // 输出源文件依赖
        for (const auto& func : sourceDependencies) {
            auto funcLoc = functionLocations.find(func.first);
            if (funcLoc != functionLocations.end()) {
                const auto& info = funcLoc->second;
                std::cout << "\nFunction " << func.first;
                if (info.startLine > 0) {
                    std::cout << " from line " << info.startLine << " to line " << info.endLine;
                }
                std::cout << " calls:\n";
    
                for (const auto& call : func.second) {
                    if (!OutputAllCalls) {
                        auto& printed = printedCalls[func.first];
                        if (printed.find(call.callee) != printed.end()) {
                            continue;  // 跳过已经输出过的
                        }
                        printed.insert(call.callee);
                    }
    
                    auto calleeLoc = functionLocations.find(call.callee);
                    std::cout << "  - " << call.callee;
    
                    if (calleeLoc != functionLocations.end()) {
                        const auto& loc = calleeLoc->second;
                        std::cout << " defined in " << loc.filename << " from line " << loc.startLine << " to line " << loc.endLine;
                    } else {
                        if (!call.filename.empty()) {
                            std::cout << " defined in " << call.filename;
                        }
                    }
    
                    std::cout << " called at line " << call.line << "\n";
                    if (DebugMode && !call.callExpr.empty()) {
                        std::cout << "    Call expression: " << call.callExpr << "\n";
                    }
                }
            }
        }
    
        printedCalls.clear();
    
        for (const auto& header : headerInfos) {
            if (!header.dependencies.empty()) {
                std::cout << "\n==========================================\n";
                std::cout << "Header File: " << header.headerFile << "\n";
                std::cout << "==========================================\n";
    
                for (const auto& func : header.dependencies) {
                    auto funcLoc = functionLocations.find(func.first);
                    if (funcLoc != functionLocations.end()) {
                        const auto& info = funcLoc->second;
                        std::cout << "\nFunction " << func.first;
                        if (info.startLine > 0) {
                            std::cout << " from line " << info.startLine << " to line " << info.endLine;
                        }
                        std::cout << " calls:\n";
    
                        for (const auto& call : func.second) {
                            if (!OutputAllCalls) {
                                auto& printed = printedCalls[func.first];
                                if (printed.find(call.callee) != printed.end()) {
                                    continue;
                                }
                                printed.insert(call.callee);
                            }
    
                            auto calleeLoc = functionLocations.find(call.callee);
                            std::cout << "  - " << call.callee;
    
                            if (calleeLoc != functionLocations.end()) {
                                const auto& loc = calleeLoc->second;
                                std::cout << " defined in " << loc.filename << " from line " << loc.startLine << " to line " << loc.endLine;
                            } else {
                                if (!call.filename.empty()) {
                                    std::cout << " defined in " << call.filename;
                                }
                            }
    
                            std::cout << " called at line " << call.line << "\n";
                            if (DebugMode && !call.callExpr.empty()) {
                                std::cout << "    Call expression: " << call.callExpr << "\n";
                            }
                        }
                    }
                }
            }
        }
    }

};

class FuncDependencyASTConsumer : public ASTConsumer {
private:
    CompilerInstance &CI;

public:
    explicit FuncDependencyASTConsumer(CompilerInstance &CI) : CI(CI) {}

    void HandleTranslationUnit(ASTContext &Context) override {
        FuncDependencyVisitor visitor(CI.getSourceManager());
        visitor.TraverseDecl(Context.getTranslationUnitDecl());
        visitor.printDependencies();
    }
};

class FuncDependencyAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(
        CompilerInstance &CI, StringRef file) override {
        setupCompilerInstance(CI);
        return std::make_unique<FuncDependencyASTConsumer>(CI);
    }

private:
    void setupCompilerInstance(CompilerInstance &CI) {
        CI.getDiagnostics().setClient(new QuietDiagnosticConsumer, true);
        auto &PP = CI.getPreprocessor();
        PP.SetSuppressIncludeNotFoundError(true);
        auto &diags = CI.getDiagnostics();
        diags.setSuppressAllDiagnostics(true);
        diags.setIgnoreAllWarnings(true);
    }
};

static llvm::cl::OptionCategory ToolCategory("Function Analyzer Options");

int main(int argc, const char **argv) {
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, ToolCategory);
    if (!ExpectedParser) {
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }

    CommonOptionsParser& OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

    std::vector<std::string> compiler_args{"-fsyntax-only", "-w", "-Wno-everything"};

    Tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster(compiler_args, ArgumentInsertPosition::BEGIN));

    Tool.setDiagnosticConsumer(new QuietDiagnosticConsumer());

    return Tool.run(newFrontendActionFactory<FuncDependencyAction>().get());
}
