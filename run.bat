@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   HireInsight-Agent 一键启动脚本
echo ============================================
echo.

REM 检查 venv 是否存在
if not exist "venv\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ 创建虚拟环境失败！
        pause
        exit /b 1
    )
) else (
    echo [1/3] ✅ 虚拟环境已存在
)

REM 安装依赖
echo [2/3] 安装/更新依赖...
venv\Scripts\python.exe -m pip install -r requirements.txt --quiet --disable-pip-version-check --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
if errorlevel 1 (
    echo ⚠️ 依赖安装可能存在问题，尝试继续...
)

REM 启动 Streamlit
echo [3/3] 启动 Streamlit 数据大屏...
echo.
echo 🌐 浏览器将自动打开 http://localhost:8502
echo 📊 按 Ctrl+C 停止服务
echo.
venv\Scripts\python.exe -m streamlit run app.py --server.port 8502

pause
