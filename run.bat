@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   HireInsight-Agent 一键启动脚本
echo ============================================
echo.

REM ---- 清理残留进程，释放端口 ----
echo [0/4] 清理残留 Streamlit 进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8502" ^| findstr "LISTENING" 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo ✅ 端口 8502 已释放
echo.

REM ---- 检查 venv ----
if not exist "venv\Scripts\python.exe" (
    echo [1/4] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ 创建虚拟环境失败！
        pause
        exit /b 1
    )
) else (
    echo [1/4] ✅ 虚拟环境已存在
)

REM ---- 安装依赖 ----
echo [2/4] 安装/更新依赖...
venv\Scripts\python.exe -m pip install -r requirements.txt --quiet --disable-pip-version-check --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
if errorlevel 1 (
    echo ⚠️ 依赖安装可能存在问题，尝试继续...
)
echo ✅ 依赖就绪
echo.

REM ---- 测试 RAG 模块 ----
echo [3/4] 验证 RAG 模块...
venv\Scripts\python.exe -c "from utils.rag_loader import init_store, load_mock_experiences, query_experiences; s=init_store(); load_mock_experiences(s); r=query_experiences('测试',1,s); print(f'  RAG 就绪 (文档数: {s.count()})')" 2>nul
if errorlevel 1 (
    echo ⚠️ RAG 模块验证异常，但继续启动...
) else (
    echo ✅ 验证通过
)
echo.

REM ---- 启动 Streamlit ----
echo [4/4] 启动 Streamlit 数据大屏...
echo.
echo 🌐 浏览器将自动打开 http://localhost:8502
echo 📊 按 Ctrl+C 停止服务
echo.
venv\Scripts\python.exe -m streamlit run app.py --server.port 8502

pause
