% reorganize and anonymize files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   inroot:     root of input file
%   folder1:    folder name of input file
%   outro:      root of output file
%   outfolder:  folder name of output file
%
%
% function anonymization is used to do anonymization
%
% Author:   Zhiling Zhou
%
% Date:     26th Jan, 2019
%
% Version:  1.0
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function res = reorganize_dcm(list,outfolder,Seq)
res = 0;
count = 0;
no_ins_count = 0;

sublist = dir([list.folder,'/',list.name]);
sublist = sublist(3:end);
try
    for j = 1:length(sublist)
        clc;disp(['Converting ',list.name,' ## ',num2str(j,'%.4d'),'...']);
        subfolder = sublist(j).name;
        
        if(~strcmp(subfolder(end-2:end),'xml'))
            subsublist = dir([sublist(j).folder,'/',sublist(j).name]);
            subsublist = subsublist(3:end);
            
            for k = 1:length(subsublist)
                dicom_name = subsublist(k).name;
                if(strcmp(dicom_name(end-2:end),'dcm'))
                    path = [subsublist(k).folder,'/',dicom_name];
                    metadata = dicominfo(path);
                    metadata = rmfield( metadata , 'TransferSyntaxUID' ) ;
                    if(isfield(metadata,'SOPClassUID'))
                        metadata = rmfield( metadata , 'SOPClassUID' ) ;
                    end
                    %                     metadata = anonymization(metadata);
                    
                    
                    if(isfield(metadata,'SeriesDescription'))
                        SeriesDes = regexprep(metadata.SeriesDescription,'/|\|<','_');
                    else
                        SeriesDes = 'No Series Description';
                    end
                    
                    if(isfield(metadata,'SeriesNumber'))
                        num = metadata.SeriesNumber;
                        SeriesNum = num2str(num,'%.4d');
                    else
                        SeriesNum = 'No Series Number';
                    end
                    
                    if(isfield(metadata,'InstanceNumber'))
                        num = metadata.InstanceNumber;
                        InstanceNum = num2str(num,'%.4d');
                    else
                        no_ins_count = no_ins_count + 1;
                        InstanceNum = num2str(no_ins_count,'%.4d');
                    end
                    outpath = regexprep([outfolder,'/',SeriesDes,'_Series',SeriesNum,'/'],'*|<|>','');
                    
                    for iseq = 1:length(Seq)
                        if strcmp(Seq{iseq},SeriesDes)==1
                            if(~exist(outpath,'dir')) mkdir(outpath); end
                            outname = [outpath,'Image_',InstanceNum,'.dcm'];
                            
                            count = count + 1;
                            X = dicomread(path);
                            dicomwrite(X, outname,metadata);
                        end
                    end
                    
                end
            end
        end
    end
catch
    disp(['Cannot process this: ',subfolder]);
end
    res = 1;
end
