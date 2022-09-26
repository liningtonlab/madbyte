#!/usr/bin/env python
# coding: utf-8
import logging
import os
import subprocess
import nmrglue as ng
import pandas as pd
import json
# import pyqtgraph as pg
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (QApplication, QColorDialog, QFileDialog,
                             QMainWindow, QMessageBox,QTableWidgetItem, QDialog)
from pyqtgraph import InfiniteLine

import madbyte as MADByTE
from madbyte.gui import Worker
from madbyte.logging import setup_logging
from madbyte import utils
import itertools

BASE = os.path.dirname(__file__)
DEFAULT_NETWORKS = os.path.join(BASE, "Networks")
LOGO_PATH = os.path.join(BASE, "static", "MADByTE_LOGO.png")
Banner_Path = os.path.join(BASE,"static","MADByTE_Banner_2.png")
Dereplication_Database = 'Dereplication_Database'


class MADByTE_Main(QMainWindow):
    def __init__(self):
        __version__ = '1.3.0'
        super(MADByTE_Main, self).__init__()
        uic.loadUi(os.path.join(BASE, 'static','MADByTE_GUI.ui'),self)

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
        self.Overlap_Score_lineEdit.setText('0.30')
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
        for NMR_Datatype in ['Bruker','Mestrenova','CSV']:#,'JOEL','Agilent','NMRPipe','Peak Lists]:
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
        self.Export_Derep_Button.clicked.connect(self.Export_Derep_Results)
        self.Load_Derep_Library_Button.clicked.connect(self.Select_Dereplication_Library)
        ###Create the Plotting Window for the NMR Data####
        Plotted = self.plot
        global vLine
        global hLine
        vLine = InfiniteLine(angle=90, movable=False)
        hLine = InfiniteLine(angle=0, movable=False)
        Plotted.enableAutoRange(True)
        Plotted.addItem(vLine, ignoreBounds=True)
        Plotted.addItem(hLine, ignoreBounds=True)
        Plotted.setMouseTracking(True)
        Plotted.showGrid(x=True,y=True,alpha=0.75)
        Plotted.scene().sigMouseMoved.connect(self.mouseMoved)
        self.Solvent_comboBox.addItems(['DMSO-D6','MeOD','CDCl3','D2O'])
        ###Default Values for colors for networking###
        global Spin_color
        Spin_color = "#009999"
        global Extract_color
        Extract_color = "#ff3333"
        # Load sample networks if there...
        if not os.path.isdir(DEFAULT_NETWORKS):
            os.mkdir(DEFAULT_NETWORKS)

        for Network in os.listdir(DEFAULT_NETWORKS):
            if 'html' in Network:
                self.Drop_Down_List_Networks.addItem(Network)

    ###Functions####
    def Launch_Documentation(self):
        try: 
            subprocess.call(["open", os.path.join('Documentation','MADByTE_User_Manual.pdf')])
        except: 
            subprocess.Popen([os.path.join('Documentation','MADByTE_User_Manual.pdf')],shell=True)
    def Launch_Example(self):
        try: 
            subprocess.call(["open", os.path.join('Documentation','MADByTE_Quick_Start_Tutorial.pdf')])
        except:
            subprocess.Popen([os.path.join('Documentation','MADByTE_Quick_Start_Tutorial.pdf')],shell=True)

    def Load_Existing_Networks(self,MasterOutput):
        try:
            for Network in os.listdir(os.path.join(MasterOutput)):
                if 'html' in Network:
                    self.Drop_Down_List_Networks.addItem(Network)
        except:
            pass
        for Network in os.listdir(DEFAULT_NETWORKS):
            if 'html' in Network:
                self.Drop_Down_List_Networks.addItem(Network)

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
        
        try:
            MasterOutput
            DataDirectory
            self.MADByTE_Button_2.setEnabled(True)
        except:
            try: 
                DataDirectory
            except: 
                PopUP('Please Select NMR Data Directory','Please select an NMR data directory before proceeding.','Error')
            try:
                MasterOutput
            except:
                PopUP('Please Select Project Directory','Please select a project directory before proceeding.','Error')
         
        if MasterOutput == DataDirectory:
                PopUP('Please Differentiate Directories','The NMR data directory is where the processed NMR datafiles are, and the project directory is the MADByTE output location. They must be different.','Error')
        else:
            PopUP("Parameters Loaded","MADByTE parameters Loaded.","Info")

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
        self.Ready_Check()
        return DataDirectory #Raw Data Directory (analogous to input_dir)

    def Select_Project_Directory_Fx(self):
        Directory_Location = QFileDialog.getExistingDirectory(self)
        if Directory_Location == '':
            PopUP('Please Select A Project Directory', 'Please select a directory to store the project results in. It is recommended to create a new project directory for each experiment processing or batch of samples.','Error')
            return
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
        self.VIEWHSQC_2.setEnabled(True)
        self.VIEWTOCSY_2.setEnabled(True)
        self.Ready_Check()
        if 'correlation_matrix.json' in os.listdir(MasterOutput): 
            self.TOCSY_Net_Button_2.setEnabled(True)
        return MasterOutput #Output Directory

    def Remove_From_Sample_List(self):
        Item_List = self.BatchSamplesList.selectedItems()
        if not Item_List: return
        for item in Item_List:
            self.BatchSamplesList.takeItem(self.BatchSamplesList.row(item)) #removes selected sample from list

    def openFileNameDialog(self):
        try:
            fileName,_ = QFileDialog.getOpenFileName(self)
            return fileName
        except:
            PopUP('Select Directory',"Please select a directory.",'Error')

    def MADByTE_Networking_Launch(self):
        self.MADByTE_Networking(Spin_color,Extract_color)

    def MADByTE_Networking(self,Spin_color,Extract_color):
        # Generates Network - allows for regen of network without reprocessing of files (updates size/colors)
        # Relevant Values: Colors and Sizes
        self.Drop_Down_List_Networks.clear()
        Extract_Node_Size = int(self.Extract_Node_Size_Box.text())
        Feature_Node_Size = int(self.Feature_Node_Size_Box.text())
        Filename = self.Network_Filename_Input.text() or "MADByTE" # Default if nothing entered
        Similarity_Cutoff = float(self.Similarity_Ratio_Input.text())
        Max_Spin_Size = int(self.Spin_Max_Size.text())
        colors = {'spin':Spin_color,'extract':Extract_color,'standard':"#0ffbff"}
        try: 
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
            PopUP("Networking Completed","MADByTE networking completed. Please select the network from the drop down list to view it.",'Info')
            self.Update_Log_Fx()
        except:
            try: 
                Cppm_Error
            except:
                PopUP('Load Parameters Before Proceeding','Please load the MADByTE parameters before generating a network.','Error')
            PopUP('Networking Error','Network constructin could not be completed due to an error.','Error')


    #################################################################
    ## The Dereplication Report was made to do dereplication through HSQC pattern matching ##
    ## The HSQC matching is only done when one of the spin systems has been found in the sample ## 

    def Dereplication_Report(self):
        print('Comparing sample against the dereplication library... ')
        def point_comparison(observed_value, expectedVal, tolerance):
            observed_value = float(observed_value)
            expectedVal = float(expectedVal)
            if (expectedVal - tolerance < observed_value) & (expectedVal + tolerance > observed_value):
                return True
            return False
        def HSQC_Scoring(Database_Sample_ID,Sample_Analyzed,MasterOutput,ID,result_dict):
            if Database_Sample_ID not in Sample_Analyzed: 
                Sample_Dataset = pd.read_json(os.path.join(MasterOutput,ID,ID+'_HSQC_Preprocessed.json')).drop(["Intensity"],axis=1)
                Database_Sample = pd.read_json(os.path.join(Dereplication_Database,Database_Sample_ID,"DDF_"+Database_Sample_ID+'_HSQC.json'))
                Number_Of_Resonances_Sample = len(Sample_Dataset)
                Number_Of_Resonances_Database_Item = len(Database_Sample)
                Match_Counter = 0
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
                if Match_Counter >= Number_Of_Resonances_Database_Item:
                    Compound_Match_Ratio = '1'
                Sample_Analyzed.append(Database_Sample_ID)
                result_dict[Database_Sample_ID]=Compound_Match_Ratio
                return result_dict
        ID = self.Dereplication_Report_Sample_Select.currentText()
        Hppm_Error = float(self.Hppm_Input_2.text())
        Cppm_Error = float(self.Cppm_Input_2.text())
        list_of_Database_compounds = os.listdir(os.path.join(Dereplication_Database))
        self.Dereplication_Report_Table.setColumnCount(2)
        self.Dereplication_Report_Table.setRowCount(len(list_of_Database_compounds)+1)
        self.Dereplication_Report_Table.setHorizontalHeaderLabels(["Compound","Matching Ratio"])
        Spin_System_Confirmed = False
        Sample_Analyzed = list()
        Not_Detected = list()
        result_dict=dict()
        if self.Require_Spin_System_checkBox.isChecked()==True: 
            with open(os.path.join(os.path.join(MasterOutput,ID,ID+'_spin_systems.json'))) as f: 
                Sample_Spin_Systems = json.load(f)
                df_sample = pd.DataFrame([{"ID": k, "H_ppm": x[0], "C_ppm": x[1]} for k,v in Sample_Spin_Systems.items() for x in v])
                df_DDFs = utils.load_spin_systems(os.path.join(Dereplication_Database))
                df = pd.concat([df_sample,df_DDFs])
                idxs = df_sample["ID"].unique()
                idys = df_DDFs["ID"].unique()
                for idx,idy in itertools.product(idxs,idys):
                        ratio = utils.ratio_two_systems(idx, idy, df, Hppm_Error, Cppm_Error)
                        if 'HND_' in idy: 
                            Database_Sample_ID = str("_".join(idy.split("_")[1:-1]))
                        elif 'HND_' not in idy: 
                            Database_Sample_ID= str("_".join(idy.split("_")[:-1]))
                        if ratio>float(self.Overlap_Score_lineEdit.text()):
                            HSQC_Scoring(Database_Sample_ID,Sample_Analyzed,MasterOutput,ID,result_dict)
                        if ratio <=float(self.Overlap_Score_lineEdit.text()):
                            Not_Detected.append(Database_Sample_ID)
                            if Database_Sample_ID not in Sample_Analyzed:
                                result_dict[Database_Sample_ID]='Not Detected'
        elif self.Require_Spin_System_checkBox.isChecked()==False:
            for Database_Sample_ID in os.listdir(Dereplication_Database): 
                result_dict = HSQC_Scoring(Database_Sample_ID,Sample_Analyzed,MasterOutput,ID,result_dict)
        compound_number=0
        for Database_Value in result_dict:
            self.Dereplication_Report_Table.setItem(compound_number,0, QTableWidgetItem(str(Database_Value)))
            self.Dereplication_Report_Table.setItem(compound_number,1, QTableWidgetItem(result_dict[Database_Value]))
            compound_number+=1
        print('Completed.')
    def SMART_Export_Fx(self):
        '''
        
        Generates a SMARTNMR compatable output of the HSQC data. Designed originally for SMART 2.0 (http://smart.ucsd.edu/classic) for the drag and drop function. 
        SMART reference DOI: 10.1021/jacs.9b13786
        Thanks to the SMART team for all the help along the way.  
        
        '''
        ID = self.Dereplication_Report_Sample_Select.currentText()
        Sample_Dataset = pd.read_json(os.path.join(MasterOutput,ID,ID+'_HSQC_Preprocessed.json')).drop(["Intensity"],axis=1)
        Sample_Dataset.columns = ['1H','13C']
        Sample_Dataset = Sample_Dataset.sort_values(by=['1H'],ascending = True).round({'1H':2,'13C':1})
        Sample_Dataset.to_csv(os.path.join(MasterOutput,ID,ID+'_SMART_Peak_List.csv'))
        PopUP('Dataset Exported',str('The HSQC Data for'+ID+' has been converted to a CSV formatted for direct import into SMART. Go to SMART.ucsd.edu to search this dataset against over 40k HSQC spectra'),'Info')
    def Export_Derep_File(self):
        from shutil import copyfile
        ID = self.Dereplication_Report_Sample_Select.currentText()
        Sample_Dataset = pd.read_json(os.path.join(MasterOutput,ID,ID+'_HSQC_Preprocessed.json')).drop(["Intensity"],axis=1)
        Sample_Dataset["Identity"] = ID
        if 'HND_' in ID: 
            ID2 = ID
            ID = ID.replace("HND_","")
        os.mkdir(os.path.join('Dereplication_Database',ID))
        try:
            ID2 
            copyfile(os.path.join(MasterOutput,ID2,ID2+'_spin_systems.json'),os.path.join('Dereplication_Database',ID,ID+'_spin_systems.json'))
        except: 
            copyfile(os.path.join(MasterOutput,ID,ID+'_spin_systems.json'),os.path.join('Dereplication_Database',ID,ID+'_spin_systems.json'))
        Sample_Dataset.to_json(os.path.join('Dereplication_Database',ID,'DDF_'+ID+'_HSQC.json'))
    def Export_Derep_Results(self):
        '''
        Creates the output table for the dereplication function as a CSV. 
        
        '''
        import csv
        path = QFileDialog.getSaveFileName(
                self, 'Save File', '', 'CSV(*.csv)')[0]
        with open(path, 'w',newline='') as stream:
            writer = csv.writer(stream)
            headers = []
            for column in range(self.Dereplication_Report_Table.columnCount()):
                header = self.Dereplication_Report_Table.horizontalHeaderItem(column)
                if header is not None:
                        headers.append(header.text())
                else:
                    headers.append("Column " + str(column))
            writer.writerow(headers)
            for row in range(self.Dereplication_Report_Table.rowCount()):
                rowdata = []
                for column in range(self.Dereplication_Report_Table.columnCount()):
                    item = self.Dereplication_Report_Table.item(row, column)
                    if item is not None:
                        rowdata.append(str(item.text()))
                        print(str(item.text()))
                    else:
                        pass
                writer.writerow(rowdata)

    def Select_Dereplication_Library(self):
        global Dereplication_Database
        Dereplication_Database = QFileDialog.getExistingDirectory(self)
        print('Custom dereplication library loaded.')
        return Dereplication_Database
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
            PopUP('Log file not found.',"The MADByTE log file was not found. Please ensure you have selected a project directory to load the file.",'Error')

    def prompt_MADByTE(self):
        '''

        Generates a pop up for the user informing them the MADByTE analysis has begun. 

        '''
        PopUP("Begining MADByTE","Now Begining MADByTE Analysis. \n Based on how many samples were submitted, this may take a while. Please hit 'ok'. ",'Info')
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
        '''
        
        Runs MADByTE analysis using inputs from the GUI. 
        
        '''
        
        Sample_List = []
        for x in range(self.BatchSamplesList.count()):
            Sample_List.append(self.BatchSamplesList.item(x).text())

        setup_logging("MADByTE_Log.txt", fpath=MasterOutput, level=logging.DEBUG)
        # Define workers to start processing
        Solvent = self.Solvent_comboBox.currentText()
        Restart_Flag = False
        if 'correlation_matrix.json' in os.listdir(MasterOutput): 
            DF_Dialog = Data_Found_Dialog()
            if DF_Dialog.exec_():
                print('Reprocessing Data')
                Restart_Flag = True
            else:
                print('Reprocessing Canceled')
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
            restart = Restart_Flag,
            solvent=Solvent,
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
            PopUP('MADByTE Analysis Completed',"MADByTE Analysis and Correlation Matrix Generation has completed on these datasets.","Info")
            self.Update_Log_Fx()
        self.TOCSY_Net_Button_2.setEnabled(True)
        self.Dereplicate_Button.setEnabled(True)
        self.SMART_Export_Button.setEnabled(True)
        self.Export_Derep_File_Button.setEnabled(True)

        # Tell workers to execute functions when complete
        ss_worker.signals.finished.connect(ss_complete)
        corr_worker.signals.finished.connect(corr_complete)
        # Execute
        self.threadpool.start(ss_worker)
        

    ###Plotting Functions###
    def mouseMoved(self,evt):
        '''
        Tracks mouse movement on the NMR_Viewer_Plot
        '''
        pos = evt
        if self.plot.sceneBoundingRect().contains(pos):
            mousePoint = self.plot.plotItem.vb.mapSceneToView(pos)
            self.mousecoordinatesdisplay.setText("<span style='font-size: 15pt'>X=%0.01f, <span style='color: black'>Y=%0.01f</span>" % (mousePoint.x(),mousePoint.y()))
        vLine.setPos(mousePoint.x())
        hLine.setPos(mousePoint.y())

    ###How to view 1D NMR Data###
    def View_1D_Data(self):
        '''
        Plots 1D NMR data in the NMR_Data_View plot. 
        Expecting either zg or zg30 pulse sequence, which must be declared in the first or second line for comments in the pulse sequence. 
        
        '''
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
                    elif content == ['zg30']:
                        PROTON_DIR = os.path.join(path_,directory,'pdata',"1")

            dic, data = ng.bruker.read_pdata(PROTON_DIR)

            udic = ng.bruker.guess_udic(dic, data) #needed to convert from points to PPM
            uc = ng.fileiobase.uc_from_udic(udic)
            ppm_scale = uc.ppm_scale()
            self.plot.plot(ppm_scale,data)
            self.plot.invertX(True)
            self.plot.invertY(False)
        except:
            try: 
                self.plot.clear()
                ID = self.NMR_Data_View_Selector.currentText()
                path_ = os.path.join(DataDirectory,ID)
                PROTON_DIR ="undefined"
                for directory in os.listdir(path_):
                    with open(os.path.join(path_,directory,'pulseprogram')) as f:
                        content = f.readlines(0)
                        content = [x.strip() for x in content]
                        content = [x.rsplit('pp/')[1] for x in content]
                        if content == ['zg']:
                            PROTON_DIR = os.path.join(path_,directory,'pdata',"1")
                        elif content == ['zg30']:
                            PROTON_DIR = os.path.join(path_,directory,'pdata',"1")

                dic, data = ng.bruker.read_pdata(PROTON_DIR)

                udic = ng.bruker.guess_udic(dic, data) #needed to convert from points to PPM
                uc = ng.fileiobase.uc_from_udic(udic)
                ppm_scale = uc.ppm_scale()
                self.plot.plot(ppm_scale,data)
                self.plot.invertX(True)
                self.plot.invertY(False)
            except: 
                PopUP('Data Not Found','1H data not found. \n This may be for a few reasons: \n \n * MADByTE can only display data processed by Topspin.\n * The FID is corrupted and cannot be read. \n  * The pulse program file is missing or corrupted.','Error')

    def View_HSQC_Data(self):
        '''
        Plots HSQC data in the NMR_Viewer_Plot. 
        Utilizes the MADByTE Processed data - raw data not supported at this time. 
        
        '''
        try:
            self.plot.clear()
            ID = self.NMR_Data_View_Selector.currentText()
            HSQC_DATA = pd.read_json(os.path.join(MasterOutput,ID,str(ID+"_HSQC_Preprocessed.json")))
            self.plot.plot(HSQC_DATA.H_PPM,HSQC_DATA.C_PPM,pen=None,symbol = "o")
            self.plot.invertX(True)
            self.plot.invertY(True)
        except:
            PopUP('Data not found','Selected Dataset not found. Please process in Topspin Prior to running MADByTE. For 2D Datasets, the displayed data is derived from peak picking lists.','Error')

    def View_TOCSY_Data(self):
        '''
        Plot TOCSY data in the NMR_Viewer_Plot. 
        Utilizes the MADByTE Processed data - raw data not supported at this time. 
        
        '''
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
            PopUP('Data not found','Selected Dataset not found. Please process in Topspin Prior to running MADByTE. For 2D Datasets, the displayed data is derived from peak picking lists.','Error')
    def Bioactivity_Plotting_Fx(self):
        try:
            Bioactivity_Low = float(self.Low_Activity_Box.text())
            Bioactivity_Med = float(self.Mild_Activity_Box.text())
            Bioactivity_High = float(self.High_Activity_Box.text())
            fname = self.Bioactivity_Network_Name_Box.text()
            title = fname
            MADByTE.plotting.Bioactivity_plot(MasterOutput,Network_In_Path,Bioactivity_Data_In,title,fname,Bioactivity_Low,Bioactivity_Med,Bioactivity_High)
            PopUP('Network successfully created','The Bioactivity Network has been created successfully.','Info')
        except:
            PopUP('Something went wrong',"Please ensure you've correctly set the values for the bioactivity cutoffs and that the bioactivity data is in the correct format.",'Error')
    def Select_Network_To_Layer_Fx(self):
        global Network_In_Path
        Network_In_Path = self.openFileNameDialog()
        if '.graphml' not in Network_In_Path:
            PopUP('Incorrect data type','Please select the .graphml version of the network, the HTML file will not work.','Error')

    def Select_Bioactivity_File_Fx(self):
        global Bioactivity_Data_In
        Bioactivity_Data_In = self.openFileNameDialog()
        if Directory_Location == '':
            PopUP('Please Select A Bioactivity File', 'Please select a bioactivity file to continue.','Error')
            return
        if '.csv' not in Bioactivity_Data_In:
            PopUP('Incorrect data type','Please select a CSV file.','Error')
    
    def Ready_Check(self):
        try:
            MasterOutput
        except:
            self.MADByTE_Button_2.setToolTip('Select a project directory before proceeding.')
        try:
            DataDirectory
        except:
            self.MADByTE_Button_2.setToolTip('Select an NMR data directory.')
        try:
            MasterOutput
            DataDirectory
            self.MADByTE_Button_2.setToolTip('Ready to run MADByTE Analysis.')
            self.MADByTE_Button_2.setEnabled(True)
        except:
            return
        




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
        self.Network_View_Plot_Area.setHtml(html)
        self.Network_View_Plot_Area.show()


def PopUP(title,message,type):
    msg = QMessageBox()
    if type =='Error':
        msg.setIcon(QMessageBox.Critical)
    elif type == "Info":
        msg.setIcon(QMessageBox.Information)
    msg.setWindowIcon(QIcon(LOGO_PATH))
    msg.setText(message)
    msg.setWindowTitle(title)
    
    msg.exec_()

class Data_Found_Dialog(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(BASE,'static','Re_Run.ui'),self)
        self.ReRun_Button.clicked.connect(self.accept)
        self.Cancel_Button.clicked.connect(self.reject)
        self.setWindowIcon(QIcon(LOGO_PATH))

if __name__ == "__main__":
    print("MADByTE is loading...")
    import sys
    app = QApplication(sys.argv)
    app.setStyle('Windows')
    window = MADByTE_Main()
    window.setWindowIcon(QIcon(LOGO_PATH))
    if os.name == "nt":
        import ctypes
        myappid = u'MADByTE.MADByTE_NMR' 
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid) 
    window.show()
    sys.exit(app.exec_())




