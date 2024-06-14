function Mesh = CPD_registration(Mesh,info,opts)

% Defining a time vector without the chosen template phase for registration
time_vector = info.timeframes; time_vector(info.timeframes == info.template) = [];

%Initializing template CPD to template mesh
Mesh(info.template).CPD = Mesh(info.template).crop_verts;

for j = time_vector %Time frame loop
    
    % Ceherent point drift (CPD) algorithm: aling template vertices to the
    % vertices in other time frames
    if info.matlab
        [T,C] = pcregistercpd(Mesh(j).vertices,Mesh(1).vertices,'Transform','Nonrigid','OutlierRatio',opts.outliers,'MaxIterations',opts.max_it,...
            'Tolerance',opts.tol,'InteractionSigma',opts.beta,'SmoothingWeight',opts.lambda);
    else
        [T,C] = cpd_register(Mesh(j).vertices(Mesh(j).indxs,:),Mesh(info.template).vertices(Mesh(info.template).indxs,:),opts);
    end
    
    % Initializing the registered meshes
    Mesh(j).CPD = zeros(size(Mesh(info.template).vertices));
    % NaN-ing vertices belonging to planes and initializing the rest to the output of CPD
    Mesh(j).CPD(~Mesh(info.template).indxs,:) = NaN; Mesh(j).CPD(Mesh(info.template).indxs,:) = T.Y;
    
    dummy = find(Mesh(j).indxs);
    
    %Initializiing the correspondence matrix (it shows that each LV vertex
    %in template Y corresponds to which (index of) vertex in X.
    
    Mesh(j).Correspondence = zeros(size(Mesh(info.template).vertices,1),1);
    % NaN-ing vertices belonging to planes and initializing the rest to the output correspondence of CPD
    Mesh(j).Correspondence(~Mesh(info.template).indxs) = NaN; 
    
    % dummy saves all the LV vertices in current time frame
    % C shows that one vertex in Y (template) is corresponding to which(the
    % index of) vertex in this time frame.
    % e.g. if dummy = [1 3 4 5 6], C = [1 1 2 3 5], the resultant
    % correspondance matrix is [1 1 3 4 6], showing that the first vertex
    % in Y (template) corresponds to the first vertex in X, and the fifth
    % vertex in Y corresponds to the sixth vertex in X.
    % the reason we need to do this is that C is the indexes of LV-only
    % vertices in X, not the real indexes in X.
    Mesh(j).Correspondence(Mesh(info.template).indxs) = dummy(C);
    
    clear dummy
    
end

if info.smooth_verts
    
    th_sort = linspace(0,2*pi,length(info.timeframes)+1);
    nHARMO = 3; %no. of modes

    % Tpeak period of my signal
    tI = linspace(th_sort(1),th_sort(1)+2*pi,length(info.timeframes)+1)';
    
    verts_vect = 1:length(Mesh(info.template).CPD);
    verts_vect = verts_vect(Mesh(info.template).indxs);

    for j1 = verts_vect

        for j = 1:length(info.timeframes)

            X(1,j) = Mesh(info.timeframes(j)).CPD(j1,1);
            X(2,j) = Mesh(info.timeframes(j)).CPD(j1,2);
            X(3,j) = Mesh(info.timeframes(j)).CPD(j1,3);

        end
        
        % Enforcing periodicity
        X = [X X(:,1)];
        
        % Average value
        Avg_X(:,1) = trapz(th_sort,X(1,:))./(2*pi).*ones(length(info.timeframes)+1,1);
        Avg_X(:,2) = trapz(th_sort,X(2,:))./(2*pi).*ones(length(info.timeframes)+1,1);
        Avg_X(:,3) = trapz(th_sort,X(3,:))./(2*pi).*ones(length(info.timeframes)+1,1);

        for i = 1:nHARMO
            Si = sin(i*th_sort);
            Ci = cos(i*th_sort);

            Avg_X(:,1) = Avg_X(:,1) + trapz(th_sort,X(1,:).*Si)./trapz(th_sort,Si.^2).*...
                sin(i*tI);
            Avg_X(:,1) = Avg_X(:,1) + trapz(th_sort,X(1,:).*Ci)./trapz(th_sort,Ci.^2).*...
               cos(i*tI);

            Avg_X(:,2) = Avg_X(:,2) + trapz(th_sort,X(2,:).*Si)./trapz(th_sort,Si.^2).*...
                sin(i*tI);
            Avg_X(:,2) = Avg_X(:,2) + trapz(th_sort,X(2,:).*Ci)./trapz(th_sort,Ci.^2).*...
               cos(i*tI);

            Avg_X(:,3) = Avg_X(:,3) + trapz(th_sort,X(3,:).*Si)./trapz(th_sort,Si.^2).*...
                sin(i*tI);
            Avg_X(:,3) = Avg_X(:,3) + trapz(th_sort,X(3,:).*Ci)./trapz(th_sort,Ci.^2).*...
               cos(i*tI);
        end

        for j = 1:length(info.timeframes)

            Mesh(info.timeframes(j)).CPD(j1,1) = Avg_X(j,1);
            Mesh(info.timeframes(j)).CPD(j1,2) = Avg_X(j,2);
            Mesh(info.timeframes(j)).CPD(j1,3) = Avg_X(j,3);

        end

        clear Avg_X X

    end
end    