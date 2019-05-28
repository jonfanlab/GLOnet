function [ Effs ] = Eval_Eff_1D_parallel( imgs, wavelengths, angles)

N = length(wavelengths);
Effs = zeros(1, N);

tic
parfor n = 1:N
	wavelength = wavelengths(n);
	angle = angles(n);
	img = imgs(n, :);
	Effs(n) = Eval_Eff_1D(img, wavelength, angle);
end
toc
end