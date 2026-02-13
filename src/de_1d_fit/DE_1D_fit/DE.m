clear; clc; close all;
format long;

% Parameter settings
global wavelength Diameter_init screen_distance ccd_cell_size;
wavelength = 0.78; % Wavelength (unit: micrometer)
screen_distance = 75000; % Distance from screen to slit (unit: micrometer)
ccd_cell_size = 4.8;


% ---------------------------- Data loading and preprocessing ---------------------------- %
% Specify the image folder and file name
folder = 'D:\task_diffraction\RealImages\Diameter_100.2_75mm';
image_files = dir(fullfile(folder, '*.BMP')); % List all image files with the '.BMP' extension
result = [];

% for k = 1:length(image_files)
% fileName = sprintf('lens_focal_length_120 (1).BMP', k);
fileName = 'foc_75_(4).BMP';
% Build the full file path
filePath = fullfile(folder, fileName);
% Read the image
inputImage = imread(filePath);
% Convert to grayscale if the input image is RGB
if size(inputImage, 3) == 3
    grayImage = rgb2gray(inputImage);
else
    grayImage = inputImage;
end

% Grayscale processing at the image borders
outputImage = double(grayImage(3:end-2,3:end-2));

% ---------------------------- Extract 1D data at the main maximum center ---------------------------- %
[row, col] = size(outputImage);
x = 1:col;
y = 1:row;
[X, Y] = meshgrid(x, y);

% Project the 2D data along the X direction (onto the Z-Y plane)
Intensities = squeeze(max(outputImage, [], 2)); % Take the maximum along the 2nd dimension (Y) to obtain the projected data

% ---------------------------- Wavelet packet denoising ---------------------------- %
% Wavelet packet decomposition
wpt = wpdec(Intensities, 4, 'sym8', 'shannon');
thr = wthrmngr('wp1ddenoGBL', 'bal_sn', wpt); % Threshold setting
signal_rec = wpdencmp(Intensities,'s', 4 ,'sym8','shannon',thr,1); % Signal reconstruction 
figure;
plot(signal_rec);

% Compute frequency information of the denoised signal
Fs = 1/ccd_cell_size;  % Spatial sampling rate
L = length(signal_rec);
Y = fft(signal_rec);
p2 = abs(Y/L);
p1 = p2(1:floor(L/2)+1);
p1(2:end-1) = 2*p1(2:end-1);
f = Fs/L*(0:floor(L/2));
[~,ind] = findpeaks(p1);
figure(2);
grid on;
plot(f,p1);

[peaks1,ind1] = findpeaks(signal_rec, 'MinPeakDistance', 40);
% ind1: peak indices
figure(4);
plot(signal_rec);
hold on;
scatter(ind1,peaks1);
[~,ind2] = findpeaks(-signal_rec, 'MinPeakDistance', 40);
% ind2: trough indices
if signal_rec(1) < signal_rec(2)
    if signal_rec(1) / max(signal_rec) < 0.2
        ind2 = [1, ind2];
    end
end
M = min(length(ind2),length(ind1));
for i = 1:M
    if abs(ind2(i) - ind1(i)) <= 35
        if  (signal_rec(ind2(i)) - mean(signal_rec(ind2)))/mean(signal_rec(ind2)) > 0.20
            ind2(i) = [];
            M = M - 1;
        elseif (signal_rec(ind1(i)) - mean(signal_rec(ind1)))/mean(signal_rec(ind1)) > 0.20
            ind1(i) = [];
            M = M - 1;
        end
    end
end
figure(4);
plot(ind2,signal_rec(ind2),'og');
title('Denoised signal');
E_point = length(ind1); % Number of peaks
length_of_signal = length(Intensities);

low = min(signal_rec(ind2));
signal_rec = (signal_rec-low)/max(signal_rec-low);
Intensities = (Intensities-low)/max(Intensities-low);
figure(1);
plot(Intensities);
legend('Measured signal');
grid on;

% Compute frequency information of the signal
Fs = 1/ccd_cell_size;  % Spatial sampling rate
L = length(Intensities);
Y = fft(Intensities);
p2 = abs(Y/L);
p1 = p2(1:floor(L/2)+1);
p1(2:end-1) = 2*p1(2:end-1);
f = Fs/L*(0:floor(L/2));
[~,ind] = findpeaks(p1);
Diameter_init = f(ind(1))*wavelength*screen_distance;
fprintf('Initial guess: %d\n',Diameter_init);

%% Parameter settings
real_diameter = 100.2;

popsize = 200; % Population size
maxGenerantion = 1000; % Maximum number of iterations
F = 0.8; 
CR = 0.9; 
w = 0.1; % Inertia weight
w_damp = 0.3; % Inertia weight damping factor
Lambda = 1.5; % Smaller values make the search more global

dim = 4;
num_int = 1;
num_continous = 3;

lowerboundint = 5;
upboundint = 30;
lowerboundcontinous = [-pi/1000, -pi/1000, Diameter_init * 0.95];
upboundcontinous = [pi/1000, pi/1000, Diameter_init * 1.05];

pop = zeros(popsize, num_continous + num_int);
pop(:,1) = randi([lowerboundint, upboundint], [popsize, 1]);

for j = 1:num_continous
    pop(:, num_int + j) = lowerboundcontinous(j) + (upboundcontinous(j) - lowerboundcontinous(j)) * rand([popsize, 1]);
end

pop_m_end = pop(:,1) + E_point;  % pop(:,1) is m_start for all individuals (starting bright-fringe order)

% ind1: peak indices, ind2: trough indices
if ind1(1) < ind2(1) && ind1(end) > ind2(end) % Start near a bright fringe and end near a bright fringe
    pop_theta_start = asin( ( 2*pop(:,1)-1 )*wavelength ./ (2 * pop(:,4) ));
    pop_theta_end = asin( ( 2*(pop_m_end - 1)-1 )*wavelength ./ (2 * pop(:,4) ));

elseif ind1(1) < ind2(1) && ind1(end) < ind2(end) % Start near a bright fringe and end near a dark fringe
    pop_theta_start = asin( ( 2*pop(:,1)-1 )*wavelength ./ (2 * pop(:,4) ));
    pop_theta_end = asin( pop_m_end * wavelength ./ pop(:,4) );

elseif ind1(1) > ind2(1) && ind1(end) > ind2(end) % Start near a dark fringe and end near a bright fringe
    pop_theta_start = asin( pop(:,1) * wavelength ./ pop(:,4) );
    pop_theta_end = asin( (pop_m_end - 1) * wavelength ./ pop(:,4) );    

elseif ind1(1) > ind2(1) && ind1(end) < ind2(end) % Start near a dark fringe and end near a dark fringe
    pop_theta_start = asin( pop(:,1) * wavelength ./ pop(:,4) );
    pop_theta_end = asin( pop_m_end * wavelength ./ pop(:,4) );    
end

Information = cell(popsize,1);
fitness = zeros(popsize,1);
signal_rec = signal_rec';
for i = 1:popsize
    % pop(i,2) and pop(i,3) are small radian perturbations; pop(i,4) is the diameter; 
    Information{i,1} = calculate(pop(i,4), pop(i,1), pop(i,1) + E_point, pop(i,2), pop(i,3));
    fitness(i) = obj(signal_rec, Information{i,1});
end

for gen = 1:maxGenerantion
    for i = 1:popsize
        % Select three random and distinct individuals
        indices = randperm(popsize, 3);
        while any(indices == i)
            indices = randperm(popsize, 3);
        end
        a = pop(indices(1), :);
        b = pop(indices(2), :);
        c = pop(indices(3), :);

        % Mutation
        mutant = a + F * (b - c);

        % Keep within bounds
        mutant(1) = min(max(round(mutant(1)), lowerboundint), upboundint);
        mutant(2:end) = min(max(mutant(2:end), lowerboundcontinous), upboundcontinous);

        % Crossover
        trial = pop(i,:);
        crossPoint = rand(1,dim) < CR;
        trial(crossPoint) = mutant(crossPoint);

        % Keep within bounds
        trial(1) = min(max(round(trial(1)), lowerboundint), upboundint);
        trial(2:end) = min(max(trial(2:end), lowerboundcontinous), upboundcontinous);

        trial_m_end = trial(1) + E_point;
        if ind1(1) < ind2(1) && ind1(end) > ind2(end) % Start near a bright fringe and end near a bright fringe
            trial_theta_start = asin(( 2*trial(1) - 1 )*wavelength / (2 * trial(4)));
            trial_theta_end = asin(( 2*( trial_m_end-1 ) - 1 )*wavelength / (2 * trial(4)));

        elseif ind1(1) < ind2(1) && ind1(end) < ind2(end) % Start near a bright fringe and end near a dark fringe
            trial_theta_start = asin( ( 2*trial(1)-1 )*wavelength ./ (2 * trial(4) ));
            trial_theta_end = asin( trial_m_end * wavelength ./ trial(end) );

        elseif ind1(1) > ind2(1) && ind1(end) > ind2(end) % Start near a dark fringe and end near a bright fringe
            trial_theta_start = asin( trial(1) * wavelength ./ trial(4) );
            trial_theta_end = asin( (trial_m_end - 1) * wavelength ./ trial(4) );

        elseif ind1(1) > ind2(1) && ind1(end) < ind2(end) % Start near a dark fringe and end near a dark fringe
            trial_theta_start = asin( trial(1) * wavelength ./ trial(4) );
            trial_theta_end = asin( trial_m_end * wavelength ./ trial(4) );
        end
        
        % Evaluate individual fitness
        Information_trial = calculate(trial(4), trial(1), trial(1) + E_point, trial(2), trial(3));
        trialFitness = obj(signal_rec, Information_trial);

        % Selection
        if trialFitness < fitness(i)
            pop(i,:) = trial;
            fitness(i) = trialFitness;
            Information{i,1} = Information_trial; % Update individual information
        end
    end

    % Print the current best solution
    [bestFitness, bestIndex] = min(fitness);
    bestSolution = pop(bestIndex, :);
    bestIntensities = Information{bestIndex,1};
    fprintf('Generation %d: Best Fitness = %.4f\n', gen, bestFitness);
    fprintf('Generation %d: Best m_start = %d\n', gen, bestSolution(1));
end

N = min(length(signal_rec), length(bestIntensities));
figure(5)
plot(Intensities(1:N));
hold on
plot(bestIntensities(1:N)/max(bestIntensities(1:N)));

if ind1(1) < ind2(1) && ind1(end) > ind2(end) % Start near a bright fringe and end near a bright fringe
    theta_start = asin( ( 2*pop(bestIndex,1)-1 )*wavelength ./ (2 * pop(bestIndex,4) ));
    theta_end = asin( ( 2*(pop(bestIndex,1) + E_point - 1)-1 )*wavelength ./ (2 * pop(bestIndex,4) ));

elseif ind1(1) < ind2(1) && ind1(end) < ind2(end) % Start near a bright fringe and end near a dark fringe
    theta_start = asin( ( 2*pop(bestIndex,1)-1 )*wavelength ./ (2 * pop(bestIndex,4) ));
    theta_end = asin( pop(:,1) + E_point * wavelength ./ pop(bestIndex,4) );

elseif ind1(1) > ind2(1) && ind1(end) > ind2(end) % Start near a dark fringe and end near a bright fringe
    theta_start = asin( pop(bestIndex,1) * wavelength ./ pop(bestIndex,4) );
    theta_end = asin( (pop(bestIndex,1) + E_point - 1) * wavelength ./ pop(bestIndex,4) );    

elseif ind1(1) > ind2(1) && ind1(end) < ind2(end) % Start near a dark fringe and end near a dark fringe
    theta_start = asin( pop(bestIndex,1) * wavelength ./ pop(bestIndex,4) );
    theta_end = asin( pop(bestIndex,1) + E_point * wavelength ./ pop(bestIndex,4) );    
end

theta_start = theta_start + bestSolution(2);
theta_end = theta_end + bestSolution(3);
rad2deg(theta_start)
r = tan(theta_start)*screen_distance / 1000