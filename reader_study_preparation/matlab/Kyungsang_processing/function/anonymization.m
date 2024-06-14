function metadata = anonymization(metadata)

    attr_name = {'PatientBirthDate','PerformingPhysicianName','PhysicianReadingStudy',...
        'PatientName','PatientID','OtherPatientID','PatientAge','PatientAddress','RequestingPhysician'};
    attr_value = {'19000000','Anonymous','Anonymous','Anonymous','00000000',...
        '00000000','0','Anonymous','Anonymous'};
    for i = 1:length(attr_name)
        if(isfield(metadata,attr_name{i}))
            metadata = setfield(metadata,attr_name{i},attr_value{i});
        end
    end
        
end