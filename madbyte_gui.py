#!/usr/bin/env python
# coding: utf-8
import logging
import os
import subprocess
import nmrglue as ng
import pandas as pd
# import pyqtgraph as pg
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (QApplication, QColorDialog, QFileDialog,
                             QMainWindow, QMessageBox,QTableWidgetItem)
from pyqtgraph import InfiniteLine

import madbyte as MADByTE
from madbyte.gui import Worker
from madbyte.logging import setup_logging

BASE = os.path.dirname(__file__)
DEFAULT_NETWORKS = os.path.join(BASE, "Networks")
LOGO_PATH = os.path.join(BASE, "static", "MADByTE_LOGO.png")
Banner_Path = os.path.join(BASE,"static","MADByTE_Banner_2.png")


class MADByTE_Main(QMainWindow):
    def __init__(self):
        __version__ = 'GUI Version 4.4'
        super(MADByTE_Main, self).__init__()
        uic.loadUi(os.path.join(BASE, 'static','MADByTE_GUI_v4_2.ui'),self)

        # setup threadpool for processing
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        ### setup window details
        self.Version_Label.setText(__version__)
        self.setWindowIcon(QIcon(LOGO_PATH))
        self.Hppm_Input.setText('0.05')
        self.Cppm_Input.setText('0.40')
        self.Hppm_Input_2.setText('0.03')
        self.Cppm_Input_2.setText('0.40')
        self.Consensus_Error_Input.setText('0.03')
        self.Similarity_Ratio_Input.setText('0.50')
        Banner_Pixmap = QPixmap(Banner_Path)
        Logo_Pixmap = QPixmap(LOGO_PATH)
        self.Logo_Space.setPixmap(Logo_Pixmap.scaled(121,101,Qt.KeepAspectRatio,Qt.SmoothTransformation))
        self.Banner_Space.setPixmap(Banner_Pixmap.scaled(681,121,Qt.KeepAspectRatio,Qt.SmoothTransformation))
        self.Extract_Node_Size_Box.setText('15')
        self.Feature_Node_Size_Box.setText('10')
        self.Spin_Max_Size.setText('20')
        self.Dereplicate_Button.setEnabled(False)
        self.SMART_Export_Button.setEnabled(False)
        self.Export_Derep_File_Button.setEnabled(False)
        self.Plot_Proton_Button.setEnabled(False)
        self.VIEWHSQC_2.setEnabled(False)
        self.VIEWTOCSY_2.setEnabled(False)
        self.MADByTE_Button_2.setEnabled(False)
        self.TOCSY_Net_Button_2.setEnabled(False)
        self.Multiplet_Merger_Checkbox.setChecked(True)
        for NMR_Datatype in ['Bruker','Mestrenova']:#,'JOEL','Agilent','NMRPipe','Peak Lists]:
            self.NMR_Data_Type_Combo_Box.addItem(NMR_Datatype)
        self.Network_Filename_Input.setText("MADByTE")
        ### Bioactivity Layering values ###
        self.High_Activity_Box.setText('1.0')
        self.Mild_Activity_Box.setText('0.66')
        self.Low_Activity_Box.setText('0.33')
        self.Generate_Bioactivity_Plot_Button.clicked.connect(self.Bioactivity_Plotting_Fx)
        self.Select_Bioactivity_File_Button.clicked.connect(self.Select_Bioactivity_File_Fx)
        self.Select_Network_To_Layer_Button.clicked.connect(self.Select_Network_To_Layer_Fx)
        self.Bioactivity_Network_Name_Box.setText("Bioactivity_Network")
        self.SMART_Export_Button.clicked.connect(self.SMART_Export_Fx)
        self.Export_Derep_File_Button.clicked.connect(self.Export_Derep_File)
        ###Connect Buttons to Fx ###
        self.NMR_Data_Directory_Select.clicked.connect(self.Select_NMR_Data_Directory)
        self.Select_Project_Directory.clicked.connect(self.Select_Project_Directory_Fx)
        self.RemoveSampleFromListButton.clicked.connect(self.Remove_From_Sample_List)
        self.MADByTE_Button_2.clicked.connect(self.prompt_MADByTE)
        self.ViewNetwork.clicked.connect(self.ViewNetwork_launch)
        self.TOCSY_Net_Button_2.clicked.connect(self.MADByTE_Networking_Launch)
        self.actionDocumentation.triggered.connect(self.Launch_Documentation)
        self.actionExamples.triggered.connect(self.Launch_Example)
        self.Plot_Proton_Button.clicked.connect(self.View_1D_Data)
        self.VIEWHSQC_2.clicked.connect(self.View_HSQC_Data)
        self.VIEWTOCSY_2.clicked.connect(self.View_TOCSY_Data)
        self.Update_Log.clicked.connect(self.Update_Log_Fx)
        self.Dereplicate_Button.clicked.connect(self.Dereplication_Report)
        self.Extract_Node_Color_Button.clicked.connect(self.Select_Extract_Color)
        self.Spin_Node_Color_Button.clicked.connect(self.Select_Spin_Color)
        self.Load_Parameters_Button.clicked.connect(self.Load_Parameters)
        ###Create the Plotting Window for the NMR Data####
        Plotted = self.plot
        global vLine
        global hLine
        vLine = InfiniteLine(angle=90, movable=False)
        hLine = InfiniteLine(angle=0, movable=False)
        Plotted.addItem(vLine, ignoreBounds=True)
        Plotted.addItem(hLine, ignoreBounds=True)
        Plotted.setMouseTracking(True)
        Plotted.showGrid(x=True,y=True,alpha=0.75)
        Plotted.scene().sigMouseMoved.connect(self.mouseMoved)
        ###Default Values for colors for networking###
        global Spin_color
        Spin_color = "#009999"
        global Extract_color
        Extract_color = "#009900"
        # Load sample networks if there...
        if not os.path.isdir(DEFAULT_NETWORKS):
            os.mkdir(DEFAULT_NETWORKS)

        for Network in os.listdir(DEFAULT_NETWORKS):
            if 'html' in Network:
                self.Drop_Down_List_Networks.addItem(Network)

    ###Functions####
    def Launch_Documentation(self):
        subprocess.Popen([os.path.join('Documentation','MADByTE_User_Guide.pdf')],shell=True)
    def Launch_Example(self):
        subprocess.Popen([os.path.join('Documentation','MADByTE_Example.pdf')],shell=True)
        
    def Load_Existing_Networks(self,MasterOutput):
        for Network in os.listdir(DEFAULT_NETWORKS):
            if 'html' in Network:
                self.Drop_Down_List_Networks.addItem(Network)
        try:
            for Network in os.listdir(os.path.join(MasterOutput)):
                if 'html' in Network:
                    self.Drop_Down_List_Networks.addItem(Network)
        except:
            pass

    def Load_Parameters(self):
        ID = 'temp'
        global Entity
        Entity = "Extract"
        global Hppm_Error
        Hppm_Error = float(self.Hppm_Input.text())
        global Cppm_Error
        Cppm_Error = float(self.Cppm_Input.text())
        global Tocsy_Error
        Tocsy_Error = float(self.Consensus_Error_Input.text())
        if self.Multiplet_Merger_Checkbox.isChecked()== True:
            Multiplet_Merger = True
        elif self.Multiplet_Merger_Checkbox.isChecked() == False:
            Multiplet_Merger = False
        Similarity_Cutoff = float(self.Similarity_Ratio_Input.text())
        PopUP("Parameters Loaded","MADByTE Parameters Loaded.")

    def Select_Extract_Color(self):
        global Extract_color
        Extract_color = QColorDialog.getColor()
        Extract_color = Extract_color.name()
        self.Extract_Node_Color_Button.setStyleSheet(str("background-color:"+Extract_color+';'))
        return Extract_color

    def Select_Spin_Color(self):
        global Spin_color
        Spin_color = QColorDialog.getColor()
        Spin_color = Spin_color.name()
        self.Spin_Node_Color_Button.setStyleSheet(str("background-color:"+Spin_color+';'))
        return Spin_color

    def Select_NMR_Data_Directory(self):
        Directory_Location = QFileDialog.getExistingDirectory()
        global DataDirectory
        DataDirectory = Directory_Location
        self.BatchSamplesList.clear()
        for item in os.listdir(DataDirectory):
            self.BatchSamplesList.addItem(item)
        for NMR_Dataset in os.listdir(DataDirectory):
            self.NMR_Data_View_Selector.addItem(NMR_Dataset)
            self.Plot_Proton_Button.setEnabled(True)
        return DataDirectory #Raw Data Directory (analogous to input_dir)

    def Select_Project_Directory_Fx(self):
        Directory_Location = QFileDialog.getExistingDirectory(self)
        global MasterOutput
        MasterOutput = os.path.join(Directory_Location)
        for Processed_Dataset in os.listdir(MasterOutput):
            self.Dereplication_Report_Sample_Select.addItem(Processed_Dataset)
        if len(os.listdir(MasterOutput))>0:
            self.Dereplicate_Button.setEnabled(True)
            self.SMART_Export_Button.setEnabled(True)
            self.Export_Derep_File_Button.setEnabled(True)
        for Network in os.listdir(MasterOutput):
            if 'html' in Network:
                self.Drop_Down_List_Networks.addItem(Network)
        self.TOCSY_Net_Button_2.setEnabled(True)
        self.MADByTE_Button_2.setEnabled(True)
        self.VIEWHSQC_2.setEnabled(True)
        self.VIEWTOCSY_2.setEnabled(True)
        return MasterOutput #Output Directory

    def Remove_From_Sample_List(self):
        Item_List = self.BatchSamplesList.selectedItems()
        if not Item_List: return
        for item in Item_List:
            self.BatchSamplesList.takeItem(self.BatchSamplesList.row(item)) #removes selected sample from list

    def openFileNameDialog(self):
        fileName,_ = QFileDialog.getOpenFileName(self)
        return fileName

    def MADByTE_Networking_Launch(self):
        self.MADByTE_Networking(Spin_color,Extract_color)

    def MADByTE_Networking(self,Spin_color,Extract_color):
        # Generates Network - allows for regen of network without reprocessing of files (updates size/colors)
        # Relevant Values: Colors and Sizes
        Extract_Node_Size = int(self.Extract_Node_Size_Box.text())
        Feature_Node_Size = int(self.Feature_Node_Size_Box.text())
        Filename = self.Network_Filename_Input.text() or "MADByTE" # Default if nothing entered
        Similarity_Cutoff = float(self.Similarity_Ratio_Input.text())
        Max_Spin_Size = int(self.Spin_Max_Size.text())
        colors = {'spin':Spin_color,'extract':Extract_color,'standard':"#0ffbff"}

        MADByTE.generate_network(
            MasterOutput,
            Similarity_Cutoff,
            Filename,
            Cppm_Error,
            Hppm_Error,
            colors,
            Extract_Node_Size,
            Feature_Node_Size,
            Max_Spin_Size
        )
        self.Load_Existing_Networks(MasterOutput)
        PopUP("Networking Completed","MADByTE Networking completed. Please select the network from the drop down list to view it.")
        self.Update_Log_Fx()


    #################################################################
    ## The Dereplication Report was made to do dereplication through HSQC pattern matching

    def Dereplication_Report(self):
        ID = self.Dereplication_Report_Sample_Select.currentText()
        Hppm_Error = float(self.Hppm_Input_2.text())
        Cppm_Error = float(self.Cppm_Input_2.text())
        list_of_Database_compounds = os.listdir(os.path.join('Dereplication_Database'))
        self.Dereplication_Report_Table.setColumnCount(2)
        self.Dereplication_Report_Table.setRowCount(len(list_of_Database_compounds)+1)
        self.Dereplication_Report_Table.setHorizontalHeaderLabels(["Compound","Matching Ratio"])
        Sample_Dataset = pd.read_json(os.path.join(MasterOutput,ID,ID+'_HSQC_Preprocessed.json')).drop(["Intensity"],axis=1)
        compound_number = 0
        for i in list_of_Database_compounds:
            compound_number +=1
            Database_Sample_ID = i[4:-5]
            self.Dereplication_Report_Table.setItem(compound_number,0, QTableWidgetItem(str(list_of_Database_compounds[compound_number-1][4:-5])))
            Database_Sample = pd.read_json(os.path.join('Dereplication_Database',i))
            Number_Of_Resonances_Sample = len(Sample_Dataset)
            Number_Of_Resonances_Database_Item = len(Database_Sample)
            Match_Counter = 0
            def point_comparison(observed_value, expectedVal, tolerance):
                observed_value = float(observed_value)
                expectedVal = float(expectedVal)
                if (expectedVal - tolerance < observed_value) & (expectedVal + tolerance > observed_value):
                    return True
                return False
            for i in range(Number_Of_Resonances_Database_Item):
                Database_Proton = Database_Sample.iloc[i-1].H_PPM
                Database_Carbon = Database_Sample.iloc[i-1].C_PPM
                for i in range(Number_Of_Resonances_Sample):
                    Sample_Proton = Sample_Dataset.iloc[i-1].H_PPM
                    Sample_Carbon = Sample_Dataset.iloc[i-1].C_PPM
                    if point_comparison(Database_Proton,Sample_Proton,Hppm_Error)==True:
                        if point_comparison(Database_Carbon,Sample_Carbon,Cppm_Error)==True:
                            Match_Counter+=1
            Compound_Match_Ratio = str(Match_Counter)+'/'+str(Number_Of_Resonances_Database_Item)
            self.Dereplication_Report_Table.setItem(compound_number,1, QTableWidgetItem(Compound_Match_Ratio))
    def SMART_Export_Fx(self):
        ID = self.Dereplication_Report_Sample_Select.currentText()
        Sample_Dataset = pd.read_json(os.path.join(MasterOutput,ID,ID+'_HSQC_Preprocessed.json')).drop(["Intensity"],axis=1)
        Sample_Dataset.columns = ['1H','13C']
        Sample_Dataset = Sample_Dataset.sort_values(by=['1H'],ascending = True).round({'1H':2,'13C':1})
        Sample_Dataset.to_csv(os.path.join(MasterOutput,ID,ID+'_SMART_Peak_List.csv'))
        PopUP('Dataset Exported',str('The HSQC Data for'+ID+' has been converted to a CSV formatted for direct import into SMART. Go to SMART.ucsd.edu to search this dataset against over 40k HSQC spectra'))
    def Export_Derep_File(self):
        ID = self.Dereplication_Report_Sample_Select.currentText()
        Sample_Dataset = pd.read_json(os.path.join(MasterOutput,ID,ID+'_HSQC_Preprocessed.json')).drop(["Intensity"],axis=1)
        Sample_Dataset["Identity"] = ID 
        Sample_Dataset.to_json(os.path.join('Dereplication_Database','DDF_'+ID+'.json'))
    def ViewNetwork_launch(self):
        self.window2=QMainWindow()
        self.ui = Network_Viewer()
        self.ui.show()

    def Update_Log_Fx(self):
        try:
            with open(os.path.join(MasterOutput, "MADByTE_Log.txt"), "r") as f:
                contents = f.read()
            self.Madbyte_Log_Viewer.setText(contents)
        except:
            PopUP('Log file not found.',"The MADByTE log file was not found. Please ensure you have selected a project directory to load the file.")

    def prompt_MADByTE(self):
        PopUP("Begining MADByTE","Now Begining MADByTE Analysis. \n Based on how many samples were submitted, this may take a while. Please hit 'ok'. ")
        ID = 'temp'
        global Entity
        Entity = "Extract"
        global Hppm_Error
        Hppm_Error = float(self.Hppm_Input.text())
        global Cppm_Error
        Cppm_Error = float(self.Cppm_Input.text())
        global Tocsy_Error
        Tocsy_Error = float(self.Consensus_Error_Input.text())
        # The new code (MBv8) may not use the multiplet merger function and instead do it by default through alignment
        if self.Multiplet_Merger_Checkbox.isChecked()== True:
            Multiplet_Merger = True
        elif self.Multiplet_Merger_Checkbox.isChecked() == False:
            Multiplet_Merger = False
        Similarity_Cutoff = float(self.Similarity_Ratio_Input.text())
        global nmr_data_type
        nmr_data_type = self.NMR_Data_Type_Combo_Box.currentText()

        self.run_MADByTE(DataDirectory, Entity, Hppm_Error, Cppm_Error,
                Tocsy_Error, MasterOutput, Multiplet_Merger, Similarity_Cutoff,
                nmr_data_type)

    def run_MADByTE(
        self,
        DataDirectory,
        Entity,
        Hppm_Error,
        Cppm_Error,
        Tocsy_Error,
        MasterOutput,
        Multiplet_Merger,
        Similarity_Cutoff,
        nmr_data_type
    ):
        Sample_List = []
        for x in range(self.BatchSamplesList.count()):
            Sample_List.append(self.BatchSamplesList.item(x).text())

        setup_logging("MADByTE_Log.txt", fpath=MasterOutput, level=logging.DEBUG)
        # Define workers to start processing
        ss_worker = Worker(
            fn=MADByTE.spin_system_construction,
            sample_list=Sample_List,
            input_dir=DataDirectory,
            project_dir=MasterOutput,
            nmr_data_type=nmr_data_type,
            entity=Entity,
            hppm_error=Hppm_Error,
            tocsy_error=Tocsy_Error,
            merge_multiplets=Multiplet_Merger,
        )
        corr_worker = Worker(
            fn=MADByTE.correlation_matrix_generation,
            project_dir=MasterOutput,
            hppm_error=Hppm_Error,
            cppm_error=Cppm_Error,
        )
        def ss_complete():
            for s in Sample_List:
                self.Dereplication_Report_Sample_Select.addItem(s)
            # Trigger correlation network
            self.threadpool.start(corr_worker)
        def corr_complete():
            PopUP('MADByTE Analysis Completed',
                "MADByTE Analysis and Correlation Matrix Generation has completed on these datasets."
            )
            self.Update_Log_Fx()

        # Tell workers to execute functions when complete
        ss_worker.signals.finished.connect(ss_complete)
        corr_worker.signals.finished.connect(corr_complete)
        # Execute
        self.threadpool.start(ss_worker)


    ###Plotting Functions###
    def mouseMoved(self,evt):
        pos = evt
        if self.plot.sceneBoundingRect().contains(pos):
            mousePoint = self.plot.plotItem.vb.mapSceneToView(pos)
            self.mousecoordinatesdisplay.setText("<span style='font-size: 15pt'>X=%0.01f, <span style='color: black'>Y=%0.01f</span>" % (mousePoint.x(),mousePoint.y()))
        vLine.setPos(mousePoint.x())
        hLine.setPos(mousePoint.y())

    ###How to view 1D NMR Data###
    def View_1D_Data(self):
        try:
            self.plot.clear()
            ID = self.NMR_Data_View_Selector.currentText()
            path_ = os.path.join(DataDirectory,ID)
            PROTON_DIR ="undefined"
            for directory in os.listdir(path_):
                with open(os.path.join(path_,directory,'pulseprogram')) as f:
                    content = f.readlines(1)
                    content = [x.strip() for x in content]
                    content = [x.rsplit('pp/')[1] for x in content]
                    if content == ['zg"']:
                        PROTON_DIR = os.path.join(path_,directory,'pdata',"1")

            dic, data = ng.bruker.read_pdata(PROTON_DIR)

            udic = ng.bruker.guess_udic(dic, data) #needed to convert from points to PPM
            uc = ng.fileiobase.uc_from_udic(udic)
            ppm_scale = uc.ppm_scale()
            self.plot.plot(ppm_scale,data)
            self.plot.invertX(True)
            self.plot.invertY(False)
        except:
            PopUP('Data not found','Selected Dataset not found. Please process in Topspin Prior to running MADByTE. ')

    def View_HSQC_Data(self):
        try:
            self.plot.clear()
            ID = self.NMR_Data_View_Selector.currentText()
            HSQC_DATA = pd.read_json(os.path.join(MasterOutput,ID,str(ID+"_HSQC_Preprocessed.json")))
            self.plot.plot(HSQC_DATA.H_PPM,HSQC_DATA.C_PPM,pen=None,symbol = "o")
            self.plot.invertX(True)
            self.plot.invertY(True)
        except:
            PopUP('Data not found','Selected Dataset not found. Please process in Topspin Prior to running MADByTE. For 2D Datasets, the displayed data is derived from peak picking lists.')

    def View_TOCSY_Data(self):
        try:
            self.plot.clear()
            ID = self.NMR_Data_View_Selector.currentText()
            TOCSY_DATA = pd.read_json(os.path.join(MasterOutput,ID,str(ID+"_TOCSY_Data.json")))
            self.plot.plot(TOCSY_DATA.Ha,TOCSY_DATA.Hb,pen=None,symbol="o" )
            self.plot.invertX(True)
            self.plot.invertY(True)
            vLine = InfiniteLine(angle=45, movable=False)
            self.plot.addItem(vLine)
        except:
            PopUP('Data not found','Selected Dataset not found. Please process in Topspin Prior to running MADByTE. For 2D Datasets, the displayed data is derived from peak picking lists.')
    def Bioactivity_Plotting_Fx(self):
        try:
            Bioactivity_Low = float(self.Low_Activity_Box.text())
            Bioactivity_Med = float(self.Mild_Activity_Box.text())
            Bioactivity_High = float(self.High_Activity_Box.text())
            fname = self.Bioactivity_Network_Name_Box.text() 
            title = fname
            MADByTE.plotting.Bioactivity_plot(MasterOutput,Network_In_Path,Bioactivity_Data_In,title,fname,Bioactivity_Low,Bioactivity_Med,Bioactivity_High)
            PopUP('Network successfully created','The Bioactivity Network has been created successfully.')
        except:
            PopUP('Something went wrong',"Please ensure you've correctly set the values for the bioactivity cutoffs and that the bioactivity data is in the correct format.")
    def Select_Network_To_Layer_Fx(self):
        global Network_In_Path
        Network_In_Path = self.openFileNameDialog()
        if '.graphml' not in Network_In_Path:
            PopUP('Incorrect data type','Please select the .graphml version of the network, the HTML file will not work.')
        
    def Select_Bioactivity_File_Fx(self):
        global Bioactivity_Data_In
        Bioactivity_Data_In = self.openFileNameDialog()
        if '.csv' not in Bioactivity_Data_In:
            PopUP('Incorrect data type','Please select a CSV file.')
        
        
        
        
class Network_Viewer(QMainWindow):
    def __init__(self):
        super(Network_Viewer, self).__init__()
        uic.loadUi(os.path.join(BASE, 'static','Network_View.ui'), self)
        self.setWindowTitle("MADByTE Networking View")
        self.setWindowIcon(QIcon(LOGO_PATH))
        try:
            Network_Location = os.path.join(DEFAULT_NETWORKS,str(window.Drop_Down_List_Networks.currentText()))
            view = QWebEngineView()
            f = open(Network_Location, 'r')
        except:
            Network_Location = os.path.join(MasterOutput,str(window.Drop_Down_List_Networks.currentText()))
            view = QWebEngineView()
            f = open(Network_Location, 'r')
        html = f.read()
        f.close()
        self.Network_View_Plot_Area.setHtml(html)#load(Network_Location)#(QtCore.QUrl.fromLocalFile(Network_Location))
        self.Network_View_Plot_Area.show()


def PopUP(Type,message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(message)
    msg.setWindowTitle(Type)
    msg.exec_()


# from mplwidget import MplWidget
# from pyqtgraph import PlotWidget
    
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MADByTE_Main()
    window.setWindowIcon(QIcon(LOGO_PATH))
    if os.name == "nt":
        import ctypes
        myappid = u'MADByTE.MADByTE_NMR.v8' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid) #sets the tray icon for windows 10.
    window.show()
    sys.exit(app.exec_())

