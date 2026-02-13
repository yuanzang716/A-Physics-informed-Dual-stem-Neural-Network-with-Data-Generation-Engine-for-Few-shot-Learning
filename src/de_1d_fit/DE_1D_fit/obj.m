function fitness = obj(Intensities, Pre_Intensities)
    % Error loss
    N = min(length(Intensities), length(Pre_Intensities));
    fitness = sum( (Intensities(1:N) - Pre_Intensities(1:N)) .^2);
end
