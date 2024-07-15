clc
clear variables;
close all;

%save files as .xls, then convert to .xlsx with converter macro.

%% == CHANGE based on experiment!!!!!!! == %%
R_um = input('Probe radius (um): '); %mm
R_mm = R_um/1000; %m
R_m = R_mm/1000; %um
conversion = 1; %to convert to uN

t_mm = input('Sample thickness (mm): '); %mm
t_um = t_mm*1000; %um

v=0.5; % Assumed poisson's ratio of the substrate
%% What is this angle and why use cosine approximation?
angle = 38.*(pi./180); %rad 
cos_approx = 1-angle.^2./2;
saa_d = R_um-cos_approx.*R_um;

% outputting the probe radius, Poisson's ratio, and maximum indentation depth to stay within the small angle approximation
fprintf('v: %0.1f\nMax indentation depth (SAA): %0.2f um\n', v, saa_d);

%% == opening folder with data files == %%
pathName = uigetdir('*.*','Select Data Folder'); % the user selects the folder with the desired files
allFiles = dir(fullfile(pathName,'*.xlsx')); % selecting all the files in the folder with the .xlsx format
allFiles = natsortfiles(allFiles); % sorting the files so they are in numerical order (uses a function that needed to be downloaded)

%% == CHANGE based on experiment == %%
startLoop = 1;
numFiles = length(allFiles); % counting the number of files in the folder
%numFiles = 2;

%% == initializing structures == %%
indent_struct = struct([]);
indent_struct_shift = struct([]);
indent_struct_crop = struct([]);
indent_fit_struct = struct([]);
FadArray = zeros(numFiles, 1);
WadArray = zeros(numFiles, 1);
FmaxArray = zeros(numFiles,1);

%% == defining where to save structure files so I don't have to do it by hand == %%  
%% defining file location to save structures
saveName = input('\nTest name: ', 's');
splitting = split(pathName, "\"); %splitting the path name up 
size_path = length(splitting); %determining the number of folders in the path
savePath = char(join(splitting(1:(size_path-1)),"\")); %rejoining the path to exclude the last folder
%savePath = uigetdir('*.*','Select Data Folder to Save Outputs'); % the user selects the folder with the desired files

%% == INDENT_STRUCT: looping through all the data to store in a structure == %%
for k = startLoop:numFiles
    %% reading the data from the cycle files
    fileName = allFiles(k).name; % iterating through all the sorted filenames in the folder
    filePath = fullfile(pathName,fileName); % total path name is created
    [my_data,txt,raw]=xlsread(filePath);    % reads data from selected cycle data file
    
    %% reading the spring constant of the cantilever
    %nonnans = find(~isnan(my_data(:,1)));   % finds indices of all numbers in column 5
    %startpoint = nonnans(1);            % index of first data point in column 5 that has a number
    %clear nonnans

    Kn = my_data(11,2); % normal calibration constant [µN/um]=[N/m]
    %Kt = my_data(startpoint+1,5);      % tangential calibration constant [µN/um] 

    %% === Read important columns into individual arrays === %% 
    nonnans = find(~isnan(my_data(:,1)));   % finds indices of all numbers in column 1
    startpoint = nonnans(1);            % index of first data point in column 5 that has a number
    clear nonnans

    time = my_data(startpoint:end,1);    % time [seconds]
    Fn = my_data(startpoint:end,2);      % normal force [µN]
    d_indentation = my_data(startpoint:end,3)./1000;% indentation column converted from nm to um.
    d_cantilever = my_data(startpoint: end, 4) ./1000; %um
    d_piezo = my_data(startpoint:end, 5) ./1000;%um
    z_stage = d_piezo - d_cantilever; %absolute probe position of the Pavone
    
    
    %% compliance correcting the data
    Tipdisp=Fn./Kn; %deflection of cantilever under load. [uN/N/m] = uM
    penetration=z_stage; %- Tipdisp; cantilever data is read directly in the pavone, further correction incorrect
    
    %% creating a structure to store all the data 
    indent_struct(k).name = fileName;                                            % filename
    indent_struct(k).full_curve = [z_stage, Fn];                                 % full curve (not compliance corrected) (units: um and uN)
    indent_struct(k).full_curve_compliance = [penetration, Fn];                  % full curve compliance corrected (units: um and uN)
end

fprintf('Kn: %0.1f uN/um\nKf: %0.1f uN/um\n', Kn);

%% == FIGURE 1: plotting full indentation curves (compliance corrected) == %%
fig1 = figure('Name', 'Full indentation curve (compliance corrected)');
for i = startLoop:numFiles
    scatter(indent_struct(i).full_curve_compliance(:,1),indent_struct(i).full_curve_compliance(:,2));
    %cycle = [startLoop:numFiles];
    %legend(strcat('Cycle ', num2str(cycle')));
    xlabel('penetration (um)');
    ylabel('normal force (uN)');    
    hold on
end
saveas(gcf, fullfile(savePath, saveName), 'png') %% automatically saving the full indentation curves as a png

min_depth = input('\nCut-off indentation depth (um): '); % cutting off the excess data at the beginning of the figure

%% == FIGURE 2 and INDENT_STRUCT_SHIFT: plotting approach curve to determine contact point and also cropping off excess data == %%
fprintf('Choose the contact point (on approach) and pull-off point (on retraction) on the graph and then press ''Enter''\n');
for j = startLoop:numFiles
    %% cropping the full curve 
    full_curve_index = find(indent_struct(j).full_curve_compliance(:,1) > min_depth);
    full_curve_crop = [indent_struct(j).full_curve_compliance(full_curve_index,1),indent_struct(j).full_curve_compliance(full_curve_index,2)];   
      
    %% splitting up the curve into approach and retraction 
    [d_value, d_apex] = max(full_curve_crop(:,1));
    [F_value, F_apex] = max(full_curve_crop(:,2));
    
    lowpoint = 1;
    highpoint = min(d_apex,F_apex);
    
    approach_penetration = full_curve_crop(lowpoint:highpoint,1);
    approach_Fn = full_curve_crop(lowpoint:highpoint,2);
    retraction_penetration = full_curve_crop(highpoint+1:end,1);
    retraction_Fn = full_curve_crop(highpoint+1:end,2); 
    
    approach_curve_crop = [approach_penetration, approach_Fn];             % approach curve compliance corrected (units: um and uN)
    retraction_curve_crop = [retraction_penetration, retraction_Fn];       % retraction curve compliance corrected (units: um and uN)  
    
    fig2 = figure('Name', indent_struct(j).name);
    scatter(approach_curve_crop(:,1),approach_curve_crop(:,2));
    xlim([(min_depth - 10) inf ])
    hold on
    xline(0);
    yline(0);
    hold off;
    
    %% enabling the data cursor mode on the figure to choose the contact point on the plot 
    dcm_obj = datacursormode(fig2); % enables data cursor mode on figure
    set(dcm_obj, 'DisplayStyle','datatip','SnapToDataVertex','off','Enable','on') %click range of data to be analyze 
    pause;
    c_info = getCursorInfo(dcm_obj); % gets x and y values selected x = Position(1) y = Position(2)
    limit(1) = find(approach_curve_crop(:,1)==c_info(1,1).Position(1) & approach_curve_crop(:,2)==c_info(1,1).Position(2));
    lowpoint = limit(find(limit==min(limit)));  % index of low point selected 

    %% shifting the full curve data
    full_penetration_shift = full_curve_crop(:,1)-full_curve_crop(lowpoint,1);
    full_force_shift = full_curve_crop(:,2)-full_curve_crop(lowpoint,2);
    %% shifting the approach curve data
    approach_penetration_shift = approach_curve_crop(:,1)-approach_curve_crop(lowpoint,1);
    approach_force_shift = approach_curve_crop(:,2)-approach_curve_crop(lowpoint,2);
    
    %% creating new structure with shifted data
    indent_struct_shift(j).name = indent_struct(j).name;
    indent_struct_shift(j).full_curve_shift = [full_penetration_shift, full_force_shift];
    indent_struct_shift(j).approach_shift = [approach_penetration_shift, approach_force_shift]; 
    indent_struct_shift(j).retraction = retraction_curve_crop; 
    
    close(fig2);

    %% Define points for calculating the work of adhesion
    fig2 = figure('Name', indent_struct(j).name);
    scatter(retraction_curve_crop(:,1),retraction_curve_crop(:,2));
    xlim([(min_depth - 10) inf ])
    hold on
    xline(0);
    yline(0);
    hold off;
    
    %% enabling the data cursor mode on the figure to choose the contact point on the plot 
    dcm_obj = datacursormode(fig2); % enables data cursor mode on figure
    set(dcm_obj, 'DisplayStyle','datatip','SnapToDataVertex','off','Enable','on') %click range of data to be analyze 
    pause;
    c_info = getCursorInfo(dcm_obj); % gets x and y values selected x = Position(1) y = Position(2)
    limit(1) = find(retraction_curve_crop(:,1)==c_info(1,1).Position(1) & retraction_curve_crop(:,2)==c_info(1,1).Position(2));

    %% Calculating force of adhesion, work of adhesion, max indent force
    Fad = min(retraction_curve_crop(:,2)); %Saves minimum force of retraction as force of adhesion in micronewtons
    Fmax = max(approach_force_shift);
    adhesionLowPoint = limit(find(limit==min(limit)));  % index of low z adhesion cutoff
    adhesionHighPointArray = find(abs(retraction_curve_crop(:,2))<5); %find all points where d = 0, returns array
    adhesionHighPoint = min(adhesionHighPointArray); % Item in the array with the greatest d
    adhesionCurve = [retraction_curve_crop(adhesionHighPoint:adhesionLowPoint, 1) retraction_curve_crop(adhesionHighPoint:adhesionLowPoint, 2)];
    Wad = trapz(adhesionCurve(:,1), adhesionCurve(:,2)); %Integrates work of adhesion for the current file.
    Wad = (Wad * 10^-6); %Units of W_ad were um*uN or 10^-12J. This line converts to uJ (10^-6 J)
    WadArray(j, 1) = Wad; %Add the current Wad to the Wad storage array
    FadArray(j, 1) = Fad;
    FmaxArray(j, 1) = Fmax;

end
 
%% == FIGURE 3: plotting shifted approach curve == %%
fig3 = figure('Name', 'Approach curve shifted');
for m = startLoop:numFiles
    scatter(indent_struct_shift(m).approach_shift(:,1),indent_struct_shift(m).approach_shift(:,2));
    xlabel('indentation depth (um)');
    ylabel('normal force (uN)');    
    hold on
end
xline(0);
yline(0);

%% == CHOOSING TO CONTINUE TO FIT THE DATA TO A CONTACT MECHANICS MODEL OR STOPPING THE PROGRAM == %%
choice = input('Continue with contact mechanics model fitting (y or n)?: ', 's');
if choice == 'n'
    return %skips to the end of the code 
end
shg %to get plot to pop up

%% == FIGURE 4 and INDENT_STRUCT_CROP: choosing the max force and plotting the data == %%
max_force = input('Max force (uN): ');
fig4 = figure('Name', 'Approach curve cropped');
for p = startLoop:numFiles
    index = find(indent_struct_shift(p).approach_shift(:,2) <= max_force);
    indent_struct_crop(p).name = indent_struct(p).name;
    indent_struct_crop(p).approach_crop = [indent_struct_shift(p).approach_shift(index,1),indent_struct_shift(p).approach_shift(index,2)];  
    
    % Figure 4 
    scatter(indent_struct_crop(p).approach_crop(:,1),indent_struct_crop(p).approach_crop(:,2));
    xlabel('indentation depth (um)');
    ylabel('normal force (uN)');   
    hold on
end
xline(0);
yline(0);

%% ==== CONTACT MECHANICS MODEL FITTING ==== %%
% defining values
modulus_guess = input('\nModulus guess (kPa): ');
shg %to get plot to pop up again
t_guess = input('Surface gel layer guess (larger than max indentation depth) (um): ');
guess = [modulus_guess, t_guess];
lb = [0,0];

for n = startLoop:numFiles
    % finding the start point 
    zero_index = find(indent_struct_crop(n).approach_crop(:,1) == 0);   
    d_p = indent_struct_crop(n).approach_crop(zero_index:end, 1);
    F_p = indent_struct_crop(n).approach_crop(zero_index:end, 2);
    
    % getting rid of negative numbers left in the array
    p_index = find(d_p >= 0 & F_p >= 0);
    d_positive = d_p(p_index); %um
    F_positive = F_p(p_index);
    
    % calculating pressure
    a = sqrt(R_um.*d_positive); %contact area radius (um)
    pressure = (((3.*F_positive)./(2.*pi.*a.^2)))*1000; %kPa
    max_pressure = pressure(end);
    max_contact_radius = a(end);
    
    %% == INDENT_FIT_STRUCT: setting up structure that will store all the fit values   
    indent_fit_struct(n).name = indent_struct(n).name;
    indent_fit_struct(n).probe_radius = R_mm; %mm
    indent_fit_struct(n).force = max(F_positive); %uN
    indent_fit_struct(n).pressure = max_pressure; %kPa
    indent_fit_struct(n).contact_radius = max_contact_radius; %um
    
    %% Long model
    d_max = max(d_positive); %um
    strain = d_max/t_um;
    ratio = R_mm/t_mm;
    if ratio > 2 %for the 2 <= R/t <= 12.7 range
        alpha = 9.5;
        beta = 4.212; %frictionless
    else %for the 0.5 < = R/t < 2 range
        alpha = 10.05-0.63*sqrt(t_mm/R_mm)*(3.1+t_mm^2/R_mm^2);
        beta = 4.8-((4.23*t_mm^2)/R_mm^2);
    end
    omega = ((R_um*d_max)/t_um^2)^(3/2);
    correction = (1+2.3*omega)/(1+1.15*omega^(1/3)+alpha*omega+beta*omega^2);

    %% Hertz        
    Hertz_fun = @(E_H, d_positive) ((4*R_m.^0.5.*d_positive.^(3/2).*E_H)./3)./conversion; 
    [E_Hertz, H_error] = lsqcurvefit(Hertz_fun,modulus_guess,d_positive,F_positive,0);   
    Hertz_force = Hertz_fun(E_Hertz,d_positive); %[uN]

    %% Long model
    E_Long = E_Hertz*correction; %kPa

    %% Winkler (two-parameter fit)
    %W(1) = modulus (kPa)
    %W(2) = thickness (mm)
%     Winkler_fun = @(W,d_positive)((pi.*R_m.*d_positive.^2.*W(1))./W(2))./conversion;
%     [W, W_error] = lsqcurvefit(Winkler_fun,guess,d_positive,F_positive,lb);
%     Winkler_force = Winkler_fun(W,d_positive); %jellyfish [N] + brushy PAAm [nN] 
%     E_Winkler = W(1); %kPa
%     t_Winkler = W(2); %mm
    
   % one parameter fit
    %Winkler_fun = @(E_W, d_positive) ((pi.*R_m.*d_positive.^2.*E_W)./t_mm)./conversion;
    %[E_Winkler, W_error] = lsqcurvefit(Winkler_fun,modulus_guess,d_positive,F_positive,0);
    %Winkler_force = Winkler_fun(E_Winkler,d_positive); %[uN]
    %t_Winkler = t_mm; 

    %% Hu 
    a_mm = sqrt(R_mm.*(d_positive/1000)); %contact area radius (mm) (the 1000 converts the indentation depth from um to mm)
    Hu_fun=@(E_Hu,d_positive) (((4.*E_Hu.*d_positive.*a_mm)./3).*((2.36.*(a_mm./t_um).^2)+(0.82.*(a_mm./t_um)+0.46))./((a_mm./t_um)+0.46))./conversion;
    [E_Hu, Hu_error] = lsqcurvefit(Hu_fun,modulus_guess,d_positive,F_positive,0);
    Hu_force = Hu_fun(E_Hu,d_positive); %[uN]

    %% Linear (our porohyperelastic model) (two-parameter fit)
    %Linear(1) = modulus (kPa)
    %Linear(2) = thickness (um)
    Linear_fun = @(Linear, d_positive) ((8.*pi.*R_mm.*Linear(2).*Linear(1))./(5.*(Linear(2)-d_positive))).*((Linear(2).^(5/4)./((Linear(2) - d_positive).^(1/4))) - Linear(2) + d_positive);
    [Linear, Linear_error] = lsqcurvefit(Linear_fun,guess,d_positive,F_positive,lb);
    Linear_force = Linear_fun(Linear,d_positive); %[uN] 
    E_Linear = Linear(1); %kPa
    t_Linear = Linear(2); %um
    
%     one parameter fit
%     Linear_fun = @(E_L, d_positive) ((8.*pi.*R_mm.*t_um.*E_L)./(5.*(t_um-d_positive))).*((t_um.^(5/4)./((t_um - d_positive).^(1/4))) - t_um + d_positive);
%     [E_Linear, Linear_error] = lsqcurvefit(Linear_fun,modulus_guess,d_positive,F_positive,0);
%     Linear_force = Linear_fun(E_Linear,d_positive); %[uN]

    %% compiling data into the INDENT_FIT_STRUCT
    indent_fit_struct(n).modulus_Hertz = E_Hertz; %kPa
    indent_fit_struct(n).modulus_Long = E_Long; %kPa
    %indent_fit_struct(n).modulus_Winkler = E_Winkler; %kPa
    indent_fit_struct(n).modulus_Hu = E_Hu; %kPa
    indent_fit_struct(n).modulus_Linear = E_Linear; %kPa
    %indent_fit_struct(n).thickness_Winkler = t_Winkler*1000; %um
    indent_fit_struct(n).thickness_Linear = t_Linear; %um
    %indent_fit_struct(n).fits = real([d_positive F_positive Hertz_force Winkler_force Hu_force Linear_force]); %uN
    indent_fit_struct(n).error_Hertz = H_error;
    %indent_fit_struct(n).error_Winkler = W_error;
    indent_fit_struct(n).error_Hu = Hu_error;
    indent_fit_struct(n).error_Linear = Linear_error;

    %% print    
    fprintf('\nHertz\n');
    fprintf('Composite Elastic Modulus: %0.1f kPa\n', E_Hertz);
    fprintf('Error: %0.3e\n', H_error);
    fprintf('\nLong\n'); 
    fprintf('Composite Elastic Modulus: %0.1f kPa\n', E_Long);
    %fprintf('\nWinkler\n');
    %fprintf('Composite Elastic Modulus: %0.1f kPa\n', E_Winkler);
    %fprintf('Surface Gel Thickness: %0.2f um\n', t_Winkler*1000);
    %fprintf('Error: %0.3e\n', W_error);
    fprintf('\nHu\n');
    fprintf('Composite Elastic Modulus: %0.1f kPa\n', E_Hu);
    fprintf('Error: %0.3e\n', Hu_error);
    fprintf('\nLinear (porohyperelastic)\n');
    fprintf('Composite Elastic Modulus: %0.1f kPa\n', E_Linear);
    fprintf('Surface Gel Thickness: %0.2f um\n', t_Linear);
    fprintf('Error: %0.3e\n', Linear_error); 
    fprintf('\nMax Force: %0.1f uN\n', max(F_positive));
    fprintf('Max Pressure: %0.2f kPa\n', max_pressure); 
    fprintf('Max contact radius: %0.2f um\n', max_contact_radius);
    

    % FIGURE 5: approach curve with contact mechanics model fit
    fig5=figure('Name', 'Contact Mechanics Model Fits');
    scatter(d_positive,F_positive,'LineWidth', 2);
    hold on
    title(n);
    xlabel('indentation depth (um)')
    ylabel('force (uN)')
    plot(d_positive, Hertz_force, 'LineWidth', 2);
    %plot(d_positive, Winkler_force, 'LineWidth', 2);
    %plot(d_positive, Hu_force, 'LineWidth', 2);
    plot(d_positive, Linear_force, 'LineWidth', 2);
    legend('Data', 'Hertz', 'Linear', 'Location', 'northwest')  
    %legend('Data', 'Hertz', 'Winkler', 'Hu', 'Linear', 'Location', 'northwest')  
end



%% == saving structure files so I don't have to do it by hand == %%
%savePath = uigetdir('*.*','Select Data Folder to Save Outputs'); % the user selects the folder with the desired files
%saveName = input('\nTest name: ', 's');
force = string(max_force);
saveFileName1 = strcat(saveName, '_indent_fit_struct_', force, 'uN.mat');
saveFileName2 = strcat(saveName, '_indent_struct_', force, 'uN.mat');
saveFileName3 = strcat(saveName, '_indent_struct_shift_', force, 'uN.mat');

save(fullfile(savePath,saveFileName1), 'indent_fit_struct'); 
save(fullfile(savePath,saveFileName2), 'indent_struct');
save(fullfile(savePath,saveFileName3), 'indent_struct_shift'); 

writematrix(WadArray, 'AdhesionWork.csv')