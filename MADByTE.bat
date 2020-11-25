@echo on
set root1=%userprofile%\anaconda3
set root2=%LocalAppData%\Continuum\anaconda3

IF EXIST %root1% (
echo Anaconda Python found!
call %root1%\Scripts\activate.bat
) ELSE (
IF EXIST %root2% (
echo Anaconda Python found!
call %root2%\Scripts\activate.bat
) ELSE (
echo Anaconda Python could not be found!
echo Please make sure you have Anaconda installed in the default location or report and issue to the MADByTE team.
TIMEOUT 10
exit /b
)
)
call conda activate madbyte
call python %cd%\madbyte_gui.py
TIMEOUT 5