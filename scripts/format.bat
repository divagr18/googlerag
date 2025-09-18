@echo off
rem Move to project root (one level up from scripts/)
pushd "%~dp0\.."

rem Check if ruff exists on PATH
where ruff >nul 2>&1
if %ERRORLEVEL%==0 goto :use_ruff

rem ruff not found, try python -m ruff
echo ruff not found on PATH, trying python -m ruff
python -m ruff format .
if %ERRORLEVEL% NEQ 0 goto :python_ruff_failed
goto :success

:use_ruff
echo Using ruff from PATH
ruff format .
if %ERRORLEVEL% NEQ 0 goto :ruff_failed
goto :success

:ruff_failed
echo ruff failed with error %ERRORLEVEL%.
popd
exit /b 1

:python_ruff_failed
echo.
echo ERROR: Failed to run ruff. Install ruff (pip install ruff) or add it to PATH.
popd
exit /b 1

:success
echo Formatting completed successfully.
popd
exit /b 0