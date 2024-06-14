load('infoData.mat'); % infoData
%%

myTable = table(strings(0,1), strings(0,1),strings(0,1),strings(0,1),strings(0,1),strings(0,1),strings(0,1),strings(0,1),strings(0,1),strings(0,1),strings(0,1), ...
    strings(0,1), strings(0,1),strings(0,1),strings(0,1),strings(0,1),strings(0,1),strings(0,1),strings(0,1),strings(0,1),strings(0,1),...
    'VariableNames', {'OurID', 'PatientID', 'AccessionNumber','StudyID', 'PatientGivenName', 'PatientFamilyName', 'PatientFullName', 'PatientBirthdate','PatientSex', 'PatientAge'...
    'Institution', 'InstitutionAddress', 'StationName', 'Manufacturer', 'StudyDate', 'StudyTime', 'SeriesDate', 'SeriesTime','AcquisitionDate', 'SOPClassUID', 'SOPInstanceUID'});

for ii = 1: length(infoData)
    
    thisStruct = infoData{ii};
    
    thisPatientID =  thisStruct.PatientID;
    thisAccessionNumber = thisStruct.AccessionNumber;
    thisStudyID = thisStruct.StudyID;
    
    name = thisStruct.PatientName;
    thisGivenName = name.GivenName;
    thisFamilyName = name.FamilyName;
    thisFullName = [name.GivenName, ',', name.FamilyName];
    thisBirthDate = thisStruct.PatientBirthDate;
    thisPatientSex = thisStruct.PatientSex;
    thisPatientAge = thisStruct.PatientAge;
    
    thisInstitution = thisStruct.InstitutionName;
    if isfield(thisStruct, 'InstitutionAddress') == 1
        thisInstitutionAddress = thisStruct.InstitutionAddress;
    else
        thisInstitutionAddress = '';
    end
  
    thisStationName = thisStruct.StationName;
    thisManufacturer =  thisStruct.Manufacturer;
    
    thisStudyDate = thisStruct.StudyDate;
    thisStudyTime = thisStruct.StudyTime;
    thisSeriesDate = thisStruct.SeriesDate;
    thisSeriesTime =  thisStruct.SeriesTime;
    thisAcquisitionDate = thisStruct.AcquisitionDate;
    
    thisSOPClassUID = thisStruct.SOPClassUID;
    thisSOPInstanceUID = thisStruct.SOPInstanceUID;
    
%     thisSlicethickness = num2str(thisStruct.SliceThickness);
    
    % Add a new row to the table with the extracted values
    myTable = [myTable; {num2str(ii),thisPatientID, thisAccessionNumber,thisStudyID, thisGivenName, thisFamilyName, thisFullName, thisBirthDate,...
        thisPatientSex, thisPatientAge,thisInstitution,  thisInstitutionAddress, thisStationName, thisManufacturer,...
        thisStudyDate,thisStudyTime, thisSeriesDate, thisSeriesTime, thisAcquisitionDate, thisSOPClassUID, thisSOPInstanceUID}];
end
    

writetable(myTable, 'HFpEF_ID_correspondence.xlsx');



























