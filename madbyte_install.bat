@echo on
set root=%LocalAppData%\Continuum\anaconda3
call %root%\Scripts\activate.bat

call conda create -y -n madbyte python=3.7
call conda install -y -n madbyte -c conda-forge libspatialindex rtree joblib networkx bokeh^>=1.0 pandas matplotlib tqdm pyqt pyqtgraph scipy"
call conda activate madbyte
call pip install nmrglue
echo "Done installing MADByTE!"
pause

