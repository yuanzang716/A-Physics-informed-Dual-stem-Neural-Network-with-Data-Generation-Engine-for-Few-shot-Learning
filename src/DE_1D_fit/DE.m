% DE_1D_fit/DE.m
% Coarse diameter estimation via Differential Evolution (DE)
% This is Stage 1 of the pipeline: uses FFT peak detection to obtain
% an initial diameter guess, then refines it with DE optimization.
%
% Usage: edit the 'folder' and 'fileName' variables below, then run in MATLAB.
% Output: estimated diameter (um), fringe angles, and best-fit curve.

clear; clc; close all;
format long;

% ========== Parameter settings ==========
global wavelength Diameter_init screen_distance ccd_cell_size;
wavelength = 0.78;          % Wavelength (unit: micrometer)
screen_distance = 75000;      % Screen-to-slit distance (unit: micrometer)
ccd_cell_size = 4.8;         % CCD pixel size (um)

% ========== 1. Data loading and preprocessing ==========
% Edit these paths to point to your raw BMP images
folder = '../data/filaments/FIL001/focal_75mm/raw';
image_files = dir(fullfile(folder, '*.BMP'));

% Process a single image:
fileName = 'foc_75_(4).BMP';
filePath = fullfile(folder, fileName);
inputImage = imread(filePath);

% Convert to grayscale if RGB
if size(inputImage, 3) == 3
    grayImage = rgb2gray(inputImage);
else
    grayImage = inputImage;
end

% Crop image borders
outputImage = double(grayImage(3:end-2, 3:end-2));

% ========== 2. Extract 1D intensity profile ==========
[row, col] = size(outputImage);
x = 1:col;
y = 1:row;
[X, Y] = meshgrid(x, y);

% Project 2D data onto X axis (max along Y)
Intensities = squeeze(max(outputImage, [], 2));

% ========== 3. Wavelet denoising ==========
wpt = wpdec(Intensities, 4, 'sym8', 'shannon');
thr = wthrmngr('wp1ddenoGBL', 'bal_sn', wpt);
signal_rec = wpdencmp(Intensities, 's', 4, 'sym8', 'shannon', thr, 1);

figure; plot(signal_rec);

% ========== 4. FFT peak detection for initial guess ==========
Fs = 1/ccd_cell_size;  % Spatial sampling rate
L = length(signal_rec);
Y = fft(signal_rec);
p2 = abs(Y/L);
p1 = p2(1:floor(L/2)+1);
p1(2:end-1) = 2*p1(2:end-1);
f = Fs/L*(0:floor(L/2));
[~, ind] = findpeaks(p1);

Diameter_init = f(ind(1)) * wavelength * screen_distance;
fprintf('Initial guess: %d um\n', Diameter_init);

% ========== 5. Peak/trough detection ==========
[peaks1, ind1] = findpeaks(signal_rec, 'MinPeakDistance', 40);
figure; plot(signal_rec); hold on; scatter(ind1, peaks1);

[~, ind2] = findpeaks(-signal_rec, 'MinPeakDistance', 40);

% Adjust start index if first point is a trough near zero
if signal_rec(1) < signal_rec(2)
    if signal_rec(1) / max(signal_rec) < 0.2
        ind2 = [1, ind2];
    end
end

% Filter out outliers
M = min(length(ind2), length(ind1));
for i = 1:M
    if abs(ind2(i) - ind1(i)) <= 35
        if (signal_rec(ind2(i)) - mean(signal_rec(ind2)))/mean(signal_rec(ind2)) > 0.20
            ind2(i) = [];
            M = M - 1;
        elseif (signal_rec(ind1(i)) - mean(signal_rec(ind1)))/mean(signal_rec(ind1)) > 0.20
            ind1(i) = [];
            M = M - 1;
        end
    end
end

figure; plot(ind2, signal_rec(ind2), 'og');
title('Denoised signal');

E_point = length(ind1);  % Number of peaks
length_of_signal = length(Intensities);

% Normalize signals
low = min(signal_rec(ind2));
signal_rec = (signal_rec - low) / max(signal_rec - low);
Intensities = (Intensities - low) / max(Intensities - low);

figure; plot(Intensities); legend('Measured signal'); grid on;

% ========== 6. FFT-based initial diameter ==========
Fs = 1/ccd_cell_size;
L = length(Intensities);
Y = fft(Intensities);
p2 = abs(Y/L);
p1 = p2(1:floor(L/2)+1);
p1(2:end-1) = 2*p1(2:end-1);
f = Fs/L*(0:floor(L/2));
[~, ind] = findpeaks(p1);
Diameter_init = f(ind(1)) * wavelength * screen_distance;

fprintf('Initial guess: %d um\n', Diameter_init);

% ========== 7. Differential Evolution setup ==========
real_diameter = 100.2;  % Ground-truth diameter (for reference only)

popsize = 200;           % Population size
maxGeneration = 1000;     % Maximum iterations
F = 0.8;                % DE mutation scale factor
CR = 0.9;               % Crossover probability
w = 0.1;                % Inertia weight
w_damp = 0.3;           % Inertia damping
Lambda = 1.5;            % Search locality (smaller = more global)

dim = 4;
num_int = 1;              % Integer variable: starting fringe order m_start
num_continous = 3;        % Continuous variables: theta1, theta2, diameter

% Bounds: [m_start, theta1, theta2, diameter]
lowerboundint = 5;
upboundint = 30;
lowerboundcontinous = [-pi/1000, -pi/1000, Diameter_init * 0.95];
upboundcontinous = [pi/1000, pi/1000, Diameter_init * 1.05];

% Initialize population
pop = zeros(popsize, num_continous + num_int);
pop(:,1) = randi([lowerboundint, upboundint], [popsize, 1]);

for j = 1:num_continous
    pop(:, num_int + j) = lowerboundcontinous(j) + ...
        (upboundcontinous(j) - lowerboundcontinous(j)) * rand([popsize, 1]);
end

pop_m_end = pop(:,1) + E_point;

% Precompute theta bounds based on fringe type
if ind1(1) < ind2(1) && ind1(end) > ind2(end)  % Bright-bright
    pop_theta_start = asin((2*pop(:,1)-1) * wavelength ./ (2 * pop(:,4)));
    pop_theta_end   = asin((2*(pop_m_end-1)-1) * wavelength ./ (2 * pop(:,4)));
elseif ind1(1) < ind2(1) && ind1(end) < ind2(end)  % Bright-dark
    pop_theta_start = asin((2*pop(:,1)-1) * wavelength ./ (2 * pop(:,4)));
    pop_theta_end   = asin(pop_m_end * wavelength ./ pop(:,4));
elseif ind1(1) > ind2(1) && ind1(end) > ind2(end)  % Dark-bright
    pop_theta_start = asin(pop(:,1) * wavelength ./ pop(:,4));
    pop_theta_end   = asin((pop_m_end-1) * wavelength ./ pop(:,4));
elseif ind1(1) > ind2(1) && ind1(end) < ind2(end)  % Dark-dark
    pop_theta_start = asin(pop(:,1) * wavelength ./ pop(:,4));
    pop_theta_end   = asin(pop_m_end * wavelength ./ pop(:,4));
end

% ========== 8. Evaluate initial population ==========
Information = cell(popsize, 1);
fitness = zeros(popsize, 1);
signal_rec = signal_rec';

for i = 1:popsize
    Information{i,1} = calculate(pop(i,4), pop(i,1), ...
        pop(i,1) + E_point, pop(i,2), pop(i,3));
    fitness(i) = obj(signal_rec, Information{i,1});
end

% ========== 9. DE main loop ==========
for gen = 1:maxGeneration
    for i = 1:popsize
        % Select three distinct random individuals
        indices = randperm(popsize, 3);
        while any(indices == i)
            indices = randperm(popsize, 3);
        end
        a = pop(indices(1), :);
        b = pop(indices(2), :);
        c = pop(indices(3), :);

        % Mutation
        mutant = a + F * (b - c);

        % Clamp to bounds
        mutant(1) = min(max(round(mutant(1)), lowerboundint), upboundint);
        mutant(2:end) = min(max(mutant(2:end), lowerboundcontinous), upboundcontinous);

        % Crossover
        trial = pop(i,:);
        crossPoint = rand(1, dim) < CR;
        trial(crossPoint) = mutant(crossPoint);

        % Clamp trial to bounds
        trial(1) = min(max(round(trial(1)), lowerboundint), upboundint);
        trial(2:end) = min(max(trial(2:end), lowerboundcontinous), upboundcontinous);

        trial_m_end = trial(1) + E_point;

        % Compute theta bounds for trial
        if ind1(1) < ind2(1) && ind1(end) > ind2(end)
            trial_theta_start = asin((2*trial(1)-1) * wavelength / (2 * trial(4)));
            trial_theta_end   = asin((2*(trial_m_end-1)-1) * wavelength / (2 * trial(4)));
        elseif ind1(1) < ind2(1) && ind1(end) < ind2(end)
            trial_theta_start = asin((2*trial(1)-1) * wavelength / (2 * trial(4)));
            trial_theta_end   = asin(trial_m_end * wavelength / trial(4));
        elseif ind1(1) > ind2(1) && ind1(end) > ind2(end)
            trial_theta_start = asin(trial(1) * wavelength / trial(4));
            trial_theta_end   = asin((trial_m_end-1) * wavelength / trial(4));
        elseif ind1(1) > ind2(1) && ind1(end) < ind2(end)
            trial_theta_start = asin(trial(1) * wavelength / trial(4));
            trial_theta_end   = asin(trial_m_end * wavelength / trial(4));
        end

        % Evaluate trial fitness
        Information_trial = calculate(trial(4), trial(1), ...
            trial(1) + E_point, trial(2), trial(3));
        trialFitness = obj(signal_rec, Information_trial);

        % Selection
        if trialFitness < fitness(i)
            pop(i,:) = trial;
            fitness(i) = trialFitness;
            Information{i,1} = Information_trial;
        end
    end

    % Print best solution
    [bestFitness, bestIndex] = min(fitness);
    bestSolution = pop(bestIndex, :);
    fprintf('Generation %d: Best Fitness = %.4f\n', gen, bestFitness);
    fprintf('Generation %d: Best m_start = %d\n', gen, bestSolution(1));
end

% ========== 10. Output and visualization ==========
N = min(length(signal_rec), length(Information{bestIndex,1}));
figure; plot(Intensities(1:N)); hold on;
plot(Information{bestIndex,1}(1:N) / max(Information{bestIndex,1}(1:N)));

% Compute estimated diameter from best solution
if ind1(1) < ind2(1) && ind1(end) > ind2(end)
    theta_start = asin((2*bestSolution(1)-1) * wavelength / (2 * bestSolution(4)));
    theta_end   = asin((2*(bestSolution(1)+E_point-1)-1) * wavelength / (2 * bestSolution(4)));
elseif ind1(1) < ind2(1) && ind1(end) < ind2(end)
    theta_start = asin((2*bestSolution(1)-1) * wavelength / (2 * bestSolution(4)));
    theta_end   = asin(bestSolution(1)+E_point * wavelength / bestSolution(4));
elseif ind1(1) > ind2(1) && ind1(end) > ind2(end)
    theta_start = asin(bestSolution(1) * wavelength / bestSolution(4));
    theta_end   = asin((bestSolution(1)+E_point-1) * wavelength / bestSolution(4));
elseif ind1(1) > ind2(1) && ind1(end) < ind2(end)
    theta_start = asin(bestSolution(1) * wavelength / bestSolution(4));
    theta_end   = asin(bestSolution(1)+E_point * wavelength / bestSolution(4));
end

theta_start = theta_start + bestSolution(2);
theta_end   = theta_end   + bestSolution(3);
estimated_diameter = rad2deg(theta_start)
r_um = tan(theta_start) * screen_distance / 1000;
fprintf('Estimated diameter: %.2f um\n', r_um);
