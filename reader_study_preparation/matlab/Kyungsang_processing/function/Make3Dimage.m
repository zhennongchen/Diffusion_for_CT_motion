function [  ] = Make3Dimage( Path, image ,nx, ny, nz, dx, dy, dz)

type = class(image);

if strcmp(type,'uint8') || strcmp(type,'int8') 
    N = 1;
end

if strcmp(type,'uint16') || strcmp(type,'int16') 
    N = 2;
end

if strcmp(type,'uint32') || strcmp(type,'int32') || strcmp(type,'single') 
    N = 4;
end

if strcmp(type,'uint64') || strcmp(type,'int64') || strcmp(type,'double') 
    N = 8;
end

if strcmp(type,'single') 
    type = 'float';
end

fid = fopen(Path,'w');
fwrite(fid, single(image), type);
fclose(fid);

fid = fopen([Path,'.hdr'],'w');
fprintf(fid,'!INTERFILE := \n');
fprintf(fid,'data format := image \n');
fprintf(fid,['number format := ',type,' \n']);
fprintf(fid,['number of bytes per pixel := ',num2str(N),' \n']);
fprintf(fid,'number of dimensions := 3 \n');
fprintf(fid,'matrix size [1] := %d \n',nx);
fprintf(fid,'matrix size [2] := %d \n',ny);
fprintf(fid,'matrix size [3] := %d \n',nz);
fprintf(fid,'scaling factor (mm/pixel) [1] := %f \n', dx);
fprintf(fid,'scaling factor (mm/pixel) [2] := %f \n', dy);
fprintf(fid,'scaling factor (mm/pixel) [3] := %f \n', dz);
fclose(fid);            
            
end

