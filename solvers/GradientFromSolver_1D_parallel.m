 function [ Grs_and_Effs ] = GradientFromSolver_1D_parallel( imgs, wavelengths, desired_angles)

disp(size(wavelengths))
imgs = squeeze(imgs);
N = length(wavelengths);
Grs = zeros(size(imgs));
Effs = zeros(N, 1);
tic
parfor n = 1:N
	wavelength = wavelengths(n);
	desired_angle = desired_angles(n);
	img = imgs(n, :);
	[Grs(n, :), Effs(n)]  = GradientFromSolver_1D(img, wavelength, desired_angle);
end
toc

Grs_and_Effs = [Effs, Grs];
end