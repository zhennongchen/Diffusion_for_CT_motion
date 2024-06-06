import numpy as np
import os
import pandas as pd


class Build():
    def __init__(self,file_list):
        self.a = 1
        self.file_list = file_list
        self.data = pd.read_excel(file_list, dtype = {'PatientID': str, 'AccessionNumber': str})

    def __build__(self,batch_list):
        for b in range(len(batch_list)):
            cases = self.data.loc[self.data['batch'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])

        batch_list = np.asarray(c['batch'])
        patient_id_list = np.asarray(c['PatientID'])
        patient_subid_list = np.asarray(c['AccessionNumber'])
        random_name_list = np.asarray(c['MotionName'])
        start_slice_list = np.asarray(c['PAR_start_slice'])
        end_slice_list = np.asarray(c['PAR_end_slice'])
        motion_free_file_list = np.asarray(c['MotionFree_File'])
        motion_free_ds_file_list = np.asarray(c['MotionFreeDs_File'])
        motion_file_list = np.asarray(c['Motion_File'])
        motion_ds_file_list = np.asarray(c['MotionDs_File'])
        motion_param_list = np.asarray(c['MotionParam'])
        
        return batch_list, patient_id_list, patient_subid_list, random_name_list, start_slice_list, end_slice_list, motion_free_file_list, motion_free_ds_file_list, motion_file_list, motion_ds_file_list, motion_param_list, 1,1
       
