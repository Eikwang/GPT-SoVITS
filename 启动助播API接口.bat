SET PYTHON_PATH=%cd%\py312\
rem overriding default python env vars in order not to interfere with any system python installation
SET PYTHONHOME=
SET PYTHONPATH=
SET PYTHONEXECUTABLE=%PYTHON_PATH%\python.exe
SET PYTHONWEXECUTABLE=%PYTHON_PATH%pythonw.exe
SET PYTHON_EXECUTABLE=%PYTHON_PATH%\python.exe
SET PYTHONW_EXECUTABLE=%PYTHON_PATH%pythonw.exe
SET PYTHON_BIN_PATH=%PYTHON_EXECUTABLE%
SET PYTHON_LIB_PATH=%PYTHON_PATH%\Lib\site-packages
SET FFMPEG_PATH=%cd%\runtime\ffmpeg\bin
SET PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts;%FFMPEG_PATH%;%PATH%
"%PYTHON_EXECUTABLE%" api_v2.py -c GPT_SoVITS/configs/female_config.yaml -p 9881
pause
