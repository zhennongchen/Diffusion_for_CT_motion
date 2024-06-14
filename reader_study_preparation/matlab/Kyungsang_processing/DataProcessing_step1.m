%% Folder Path Setting
FolderPath = [
    {'E:\image\HF supplement case'},
    {'E:\image\supplement case 1029'},
    {'E:\MGH 1CMR\mi2b2-workbench-win\files\ImageLibrary\hr893_121218162432332296\Feb 1'},
    {'E:\MGH 1CMR\mi2b2-workbench-win\files\ImageLibrary\hr893_121218162432332296\Feb 6'},
    {'E:\MGH 1CMR\mi2b2-workbench-win\files\ImageLibrary\hr893_121218162432332296\Feb 6_2'},
    {'E:\MGH 1CMR\mi2b2-workbench-win\files\ImageLibrary\hr893_121218162432332296\Feb 6_3'},
    {'E:\MGH 1CMR\mi2b2-workbench-win\files\ImageLibrary\hr893_121218162432332296\Feb 6_4'},
    {'E:\MGH 1CMR\mi2b2-workbench-win\files\ImageLibrary\hr893_121218162432332296\Feb 6_5'},
    {'E:\MGH 1CMR\mi2b2-workbench-win\files\ImageLibrary\hr893_121218162432332296\Feb 6_6'},
    {'E:\MGH 1CMR\mi2b2-workbench-win\files\ImageLibrary\hr893_121218162432332296\Feb 12'},
    {'E:\MGH 1CMR\mi2b2-workbench-win\files\ImageLibrary\hr893_121218162432332296\Feb 19'},
    {'E:\MGH 1CMR\mi2b2-workbench-win\files\ImageLibrary\hr893_121218162432332296\Jan28'},
    {'E:\MGH 1CMR\mi2b2-workbench-win\files\ImageLibrary\hr893_121218162432332296\March 1'},
    {'E:\MGH CMR HF v2 (n=521)\1120'},
    {'E:\MGH CMR HF v2 (n=521)\DM\DM'},
    {'E:\MGH CMR HF v2 (n=521)\supplement case 1029'},
    {'E:\MGH CMR HF v2 (n=521)\Supplement case 1029 (2)'},
    {'E:\MGH CMR HF v2 (n=521)\supplement case 1030'},
    {'E:\MGH CMR HF v2 (n=521)\supplement case 1031'},
    {'E:\MGH CMR HF v2 (n=521)\supplement case 1101'},
    {'E:\MGH CMR HF v2 (n=521)\supplement case 1102'},
    {'E:\Repeated'}
    ];

OutputPath = 'C:\Hui';


Ref_Frame = 12; % Calculate deformation field

%% Skip (if already reorganized)
SkipReorganizeDicoms = 0;


Seq = [{'SHORT AXIS TRUE FISP'}, 
    {'CINE IR SHORT STEP'}, 
    {'SAX FIESTA'}, 
    {'SA T2 FSE'}
    ];


%%
addpath('function');
makedir(OutputPath);
NumFolder = size(FolderPath,1);

%%

ProcessingPath = OutputPath;
save('ProcessingPath.mat','ProcessingPath');

infoData= [ ];
nid=1;
%%
for iFolder = 1:NumFolder
    
    list = dir(FolderPath{iFolder});
    list = list(3:end);
    
    for ii = 1:length(list)
        if list(ii).isdir == 1
            target_folder = [OutputPath,'/PreProcessing/','ID_',num2str(nid,'%.4d')];
            mkdir(target_folder);
            if SkipReorganizeDicoms ~= 1
                result = reorganize_dcm(list(ii),target_folder,Seq);
            else 
                result = 1;
            end
            
            for iSeq = 1:length(Seq)
                
                FolderSquence = dir([target_folder,'/',Seq{iSeq},'*']);
                if size(FolderSquence,1) >= 8
                    disp(['We use the following sequence: ',Seq{iSeq}]);
                    break;
                end
            end
            
            
            try
                
                FolderMain = [OutputPath,'/Processing/','ID_',num2str(nid,'%.4d')];
                makedir(FolderMain);
                
                if size(FolderSquence,1) >= 8  % 8 is the minimum number of slice for registration
                                        
                    % Get maximal and minimal SliceLocation
                    MaxSliceLocation = -Inf;
                    MinSliceLocation = Inf;
                    for slice = 1:length(FolderSquence)
                        SubFolder = [FolderSquence(slice).folder,'/',FolderSquence(slice).name];
                        SubDicoms = dir([SubFolder,'/*.dcm']);
                        for ii = 1:length(SubDicoms)
                            copyfile([SubFolder,'/',SubDicoms(ii).name],[FolderMain,'/S',num2str(slice),'_',SubDicoms(ii).name]);
                            tmpinfo = dicominfo([FolderMain,'/S',num2str(slice),'_',SubDicoms(ii).name]);
                            if tmpinfo.SliceLocation > MaxSliceLocation
                                MaxSliceLocation = tmpinfo.SliceLocation;
                            end
                            if tmpinfo.SliceLocation < MinSliceLocation
                                MinSliceLocation = tmpinfo.SliceLocation;
                            end
                        end
                    end
                    
                    
                    tmpinfo = dicominfo([FolderMain,'/S',num2str(1),'_',SubDicoms(1).name]);
                    
                    infoData{nid} = tmpinfo;
                    
                    
                    % Rows and Columes is the size of current image
                    nx = tmpinfo.Rows;
                    ny = tmpinfo.Columns;
                    % nz is the number of slice
                    nz = length(FolderSquence);
                    % nt is the number of timeframe
                    nt = length(SubDicoms);
                    
                    dx = tmpinfo.PixelSpacing(1);
                    dy = tmpinfo.PixelSpacing(2);
                    dz = tmpinfo.SliceThickness;
                    
                    img4D = zeros(nx, ny, nz, nt, 'uint16');
                    
                    % SliceLocation can be used to determine the location of each slice
                    for iz = 1:nz
                        for it = 1:nt
                            File = dir([FolderMain,'/S',num2str(iz),'_*',num2str(it,'%.4d'),'.dcm']);
                            tmp = dicomread([FolderMain,'/',File.name]);
                            tmpinfo = dicominfo([FolderMain,'/',File.name]);
                            iz_id = int32((tmpinfo.SliceLocation-MinSliceLocation)/dz + 1);
                            img4D(:,:,iz_id,it) = tmp;
                        end
                    end
                                        
                    % the data in Data4D is in original size, different resolution, which is
                    % recorded in [dx, dy, dz];
                    for it = 1:nt
                        nrrdWriter([FolderMain,'/Org3D_frame',num2str(it),'.nrrd'],img4D(:,:,:,it),[dx, dy, dz],[ 0 0 0],'raw');
                        
                        if it== Ref_Frame
                            makedir([OutputPath,'/Processing/ForMaskDrawing']);
                            nrrdWriter([OutputPath,'/Processing/ForMaskDrawing','/ID_',num2str(nid,'%.4d'),'_frame',num2str(it),'.nrrd'],img4D(:,:,:,it),[dx, dy, dz],[ 0 0 0],'raw');
                        end
                        
                    end
                    
                    delete([FolderMain,'/S*']);
                    
                    if result == 1
                        nid = nid+1;
                        save( [OutputPath,'/Processing/infoData.mat'], 'infoData');
                    end
                else
                    
                    rmdir(target_folder,'s');
                    rmdir(FolderMain,'s');
                end
                
            catch
                rmdir(target_folder,'s');
                rmdir(FolderMain,'s');
            end
            
            
        end
    end
    
end

clc; disp('Done..');








