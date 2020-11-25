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


echo Setting up virtual environment and installing dependencies for MADByTE!
call conda env create -f environment.yml
echo Done installing MADByTE!
TIMEOUT 10

