@echo off
REM Activate Anaconda
CALL "C:\Users\EarthSR\anaconda3\Scripts\activate.bat"

REM Activate your conda environment named 'pytorch'
CALL conda activate pytorch

REM Change directory to your script folder
cd /d "C:\Users\EarthSR\AI and API Stock Project\Preproces"

REM Run your Python script
python autoscript.py

REM Optional: keep the command window open
PAUSE
