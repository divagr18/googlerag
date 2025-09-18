@echo off
pushd "%~dp0\.."

rem Check if ruff exists on PATH
where ruff >nul 2>&1
if %ERRORLEVEL%==0 goto :use_ruff

rem ruff not found, try python -m ruff
echo ruff not found on PATH, trying python -m ruff
python -m ruff check .
if %ERRORLEVEL% NEQ 0 goto :python_ruff_failed
goto :success

:use_ruff
echo Using ruff from PATH
ruff check .
if %ERRORLEVEL% NEQ 0 goto :ruff_failed
goto :success

:ruff_failed
echo ruff check completed with issues (exit code %ERRORLEVEL%).
popd
exit /b %ERRORLEVEL%

:python_ruff_failed
if %ERRORLEVEL%==127 goto :ruff_not_installed
echo ruff check completed with issues (exit code %ERRORLEVEL%).
popd
exit /b %ERRORLEVEL%

:ruff_not_installed
echo.
echo ERROR: Failed to run ruff. Install ruff (pip install ruff) or add it to PATH.
echo.
popd
exit /b 1

:success
echo ruff check completed successfully - no issues found.
popd
exit /b 0