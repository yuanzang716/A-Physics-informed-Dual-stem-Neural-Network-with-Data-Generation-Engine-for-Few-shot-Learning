function [Intensities_simulate,theta] = calculate(Diameter,m_start,m_end,Dtheta1,Dtheta2)
% Single-slit Fraunhofer diffraction intensity simulation - only within the specified dark-fringe order range
global wavelength ccd_cell_size screen_distance;
% Specified dark-fringe order range
% m_start = 5; % Starting dark-fringe order
% m_end = 12; % Ending dark-fringe order

% Compute the angular range (radians)
theta_start = asin(m_start * wavelength / Diameter);
theta_end = asin(m_end * wavelength / Diameter);
deltatheta = atan(ccd_cell_size/screen_distance);
theta = Dtheta1+theta_start:deltatheta:Dtheta2 + theta_end; % Generate angles only within the specified range

% Compute the intensity distribution
beta = (pi * Diameter / wavelength) * sin(theta);
Intensities_simulate = (sin(beta)./beta).^2;
Intensities_simulate = Intensities_simulate/max(Intensities_simulate - min(Intensities_simulate));
end