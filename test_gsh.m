% GSH Numerical Test Script
% This script tests the Generalized Simplex Hessian (GSH) implementation
% by varying h1 (outer step size) and keeping h2 fixed.

fun = @(x) x(1)^2 + 3*x(2)^2;     % Test function: f(x) = x^2 + 3y^2
x0 = [1; 2];                       % Point to estimate Hessian at
S = [1 0; 0 1];                    % Outer directions (identity basis)
Ti = [1 0; 0 1];                   % Inner directions (identity basis)
h2 = 1e-4;                         % Fixed small step size for gradient estimation

h1_values = 10.^(-(0:16));        % h1 goes from 1 to 1e-16
abs_errors = zeros(size(h1_values));
rel_errors = zeros(size(h1_values));

fprintf('      h1\t\t\tAbsolute Error\tRelative Error\n');
fprintf('---------------------------------------------\n');

for i = 1:length(h1_values)
    h1 = h1_values(i);
    [~, info] = gsh(fun, x0, S, Ti, h1, h2);

    if isfield(info, 'AbsError') && isfield(info, 'RelError')
        abs_errors(i) = info.AbsError;
        rel_errors(i) = info.RelError;
        fprintf('%e\t%.5e\t%.5e\n', h1, info.AbsError, info.RelError);
    else
        fprintf('%e\tN/A\t\tN/A\n', h1);
    end
end

% Plot errors vs h1
figure;
loglog(h1_values, abs_errors, '-o', 'DisplayName', 'Absolute Error');
hold on;
loglog(h1_values, rel_errors, '-x', 'DisplayName', 'Relative Error');
xlabel('h1 (outer step size)');
ylabel('Error');
title('GSH Error vs. h1');
grid on;
legend;
