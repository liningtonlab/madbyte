@echo on
set root=%LocalAppData%\Continuum\anaconda3
call %root%\Scripts\activate.bat

echo "Setting up virtual environment and installing dependencies for MADByTE!"
call conda env create -f environment.yml
echo "Done installing MADByTE!"
pause

