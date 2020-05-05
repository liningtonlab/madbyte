@echo on
set root=%LocalAppData%\Continuum\anaconda3
call %root%\Scripts\activate.bat
call conda activate madbyte
call python %userprofile%\Downloads\madbyte\madbyte_gui.py
pause