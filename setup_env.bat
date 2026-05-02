@echo off
REM setup_env.bat
REM Windows Environment Variable Configuration Script
REM
REM Usage:
REM   setup_env.bat
REM
REM Note:
REM   This script only affects the current command prompt session

echo ============================================================
echo Environment Variable Configuration
echo ============================================================

REM ============================================================
REM Core Path Configuration
REM ============================================================

REM Please modify to your actual path
REM Note: Use forward slashes / or double backslashes \\
REM Examples:
REM   D:/Projects/diffraction-metrology
REM   D:\\Projects\\diffraction-metrology

set FILAMENT_PROJECT_ROOT=D:/A-Elsevier-Computer/Computer Physics Communications/publish

REM If you are already in the publish directory, you can use current directory:
REM set FILAMENT_PROJECT_ROOT=%CD%

REM ============================================================
REM Experiment Configuration
REM ============================================================

set FILAMENT_ID=FIL001
set EVAL_FILAMENT_ID=FIL001
set FOCAL_MM=75
set GT_DIAMETER_UM=100.2

REM ============================================================
REM Training Parameters
REM ============================================================

set SEED=124
set NUM_EPOCHS=60
set ALLOW_TRAIN=1

REM ============================================================
REM Advanced Configuration
REM ============================================================

set EARLY_PAIR_ONLY_EPOCHS=10
set LOSS_GRADUAL_INTRO_ENABLE=1
set LOSS_GRADUAL_INTRO_EPOCHS=10
set LOSS_GRADUAL_INTRO_MODE=cosine
set LOSS_GRADUAL_INTRO_TARGETS=phys,weak,vic
set LOSS_CONDITIONAL_ENABLE=1
set LOSS_CONDITIONAL_PAIR_THRESHOLD=0.025
set LOSS_CONDITIONAL_STABLE_EPOCHS=2
set W_PHYS_REDUCED=0.5
set LOSS_ADAPTIVE_PAIR_BOOST=1
set LOSS_ADAPTIVE_PAIR_THRESHOLD=0.030
set LOSS_ADAPTIVE_PAIR_MAX_BOOST=2.5

REM ============================================================
REM Output Paths
REM ============================================================

set RUN_ROOT=%FILAMENT_PROJECT_ROOT%/runs/train_%FILAMENT_ID%/focal_%FOCAL_MM%mm
set REPORT_ROOT=%FILAMENT_PROJECT_ROOT%/reports/%FILAMENT_ID%

REM ============================================================
REM Display Configuration
REM ============================================================

echo.
echo Core Paths:
echo   FILAMENT_PROJECT_ROOT: %FILAMENT_PROJECT_ROOT%
echo.
echo Experiment Configuration:
echo   FILAMENT_ID:           %FILAMENT_ID%
echo   EVAL_FILAMENT_ID:      %EVAL_FILAMENT_ID%
echo   FOCAL_MM:              %FOCAL_MM%
echo   GT_DIAMETER_UM:        %GT_DIAMETER_UM%
echo.
echo Training Parameters:
echo   SEED:                  %SEED%
echo   NUM_EPOCHS:            %NUM_EPOCHS%
echo   ALLOW_TRAIN:           %ALLOW_TRAIN%
echo.
echo Output Paths:
echo   RUN_ROOT:              %RUN_ROOT%
echo   REPORT_ROOT:           %REPORT_ROOT%
echo.
echo ============================================================
echo Environment Variables Set
echo ============================================================
echo.
echo Next Steps:
echo   1. Verify paths: python verify_paths.py
echo   2. Start training: cd src\Code_75\experiments ^&^& python run_all_experiments.py
echo ============================================================
