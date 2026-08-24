@echo off
setlocal EnableDelayedExpansion
title Hades RL Agent Launcher

rem ============================================================
rem  Configuration - edit these if paths change
rem ============================================================
set "PROJ=C:\Users\ericb\game-playing-agent"
set "HADES_EXE=D:\Games\Hades.v1.38290\Hades.v1.38290\x64\Hades.exe"
set "VL_URL=http://127.0.0.1:8080"
set "VL_LAUNCHER=%USERPROFILE%\Desktop\launch-vl.bat"
set "MODE=train"
set "EPISODES=10"

echo ============================================================
echo   Hades RL Agent Launcher
echo ============================================================

rem ---- Admin check: F12 pause hotkey needs an elevated terminal ----
net session >nul 2>&1
if errorlevel 1 (
    echo [WARN] Not elevated - F12 pause hotkey may not register.
    echo        Right-click this .bat and "Run as administrator" to fix.
) else (
    echo [OK] Elevated terminal detected.
)

rem ---- 1. Hades: already loaded? otherwise launch and wait ----
tasklist /FI "IMAGENAME eq Hades.exe" 2>nul | find /I "Hades.exe" >nul
if not errorlevel 1 (
    echo [OK] Hades is already loaded - attaching to it.
    goto hades_done
)
if not exist "%HADES_EXE%" (
    echo [ERR] Hades.exe not found at:
    echo       %HADES_EXE%
    echo       Fix HADES_EXE above or start Hades via Steam, then rerun.
    pause
    exit /b 1
)
echo [..] Hades not running - starting it...
start "" "%HADES_EXE%"
:hades_wait
timeout /t 3 /nobreak >nul
tasklist /FI "IMAGENAME eq Hades.exe" 2>nul | find /I "Hades.exe" >nul
if errorlevel 1 goto hades_wait
echo [OK] Hades process is up ^(window keeps loading; agent waits for it^).
:hades_done

rem ---- 2. Qwen VL server: already up? otherwise start and wait ----
curl -s -o nul --max-time 3 "%VL_URL%/props" >nul 2>&1
if not errorlevel 1 (
    echo [OK] Qwen VL server already running at %VL_URL%
    goto vl_done
)
if not exist "%VL_LAUNCHER%" (
    echo [WARN] VL server down and no launcher found at:
    echo        %VL_LAUNCHER%
    echo        Continuing with pixel-only perception.
    goto vl_done
)
echo [..] Starting VL server: %VL_LAUNCHER%
start "qwen-vl-server" /MIN cmd /c ""%VL_LAUNCHER%""
set /a tries=0
:vl_wait
timeout /t 2 /nobreak >nul
curl -s -o nul --max-time 3 "%VL_URL%/props" >nul 2>&1
if not errorlevel 1 goto vl_up
set /a tries+=1
if !tries! lss 45 goto vl_wait
echo [WARN] VL server did not respond after 90s - continuing pixel-only.
goto vl_done
:vl_up
echo [OK] Qwen VL server is up.
:vl_done

rem ---- 3. Run the agent (it attaches to whichever Hades is running) ----
cd /d "%PROJ%"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
echo ============================================================
echo   Starting agent: mode=%MODE% episodes=%EPISODES%
echo   F12 toggles pause  ^|  Ctrl+C emergency stop
echo ============================================================
python scripts\main.py --mode %MODE% --episodes %EPISODES%

echo.
echo Agent exited.
pause
