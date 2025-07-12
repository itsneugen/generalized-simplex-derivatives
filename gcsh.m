%Author: Gabriel Jarry-Bolduc
%Date created: May 11, 2025.
%Last update: May 11, 2025.
%
%This function returns an approximation of the hessian of f at x0.
%The approximation technique is called the generalized centered simplex
%Hessian.
%
%Input:fun: function handle from R^n to R.
%      x0: point of interest x0. Column vector in R^n
%       S: set of directions in matrix form  n \times m to add to x0
%       Ti: second sets of directions in matrix forms to add to each point
%       x0, x0+s_i. Provided as one matrix if all Ti are equal. Otherwise,
%       the input is a cell array containing m+1 matrices of dimensions n \times k_i:
%       {T1; \dots;Tm}
%       h1: parameter to shrink S. Positive number
%       h2: parameter to shrink all Ti. Positive number
%Output: CSHessValue:  Generalized Centered Simplex Hessian value at x0
%        info: info.AbsError: Absolute error between the true Hessian and
%                             the GCSH          
%              info.RelError: relative error
%              info.Hessian_x0: True Hessian at x0

function [CSHessValue, info] = gcsh(fun,x0,S,Ti,h1,h2)
%% Set h1=1 and h2=1 if it is not specified
if (nargin<4)
    error('gcsh must have at least three inputs');
elseif (nargin==4)
  h1=1;
  h2=1;
elseif (nargin==5)
   h2=1; 
end
%% Calculate CSHessValue
if iscell(Ti)==1
    Ti_neg = cellfun(@(x) x * -1, Ti,'UniformOutput',false);
else
    Ti_neg=-Ti;
end
CSHessValue=0.5*(gsh(fun,x0,S,Ti,h1,h2)+gsh(fun,x0,-S,Ti_neg,h1,h2));
%% Calculate Absolute Error and relative error
if nargout==2
    n=length(x0);
    x=sym('x',[n,1]);
    Fun=fun(x);
    Hessian=hessian(Fun,x);
    for iter=1:n
        CellArrayVar{1,iter}=x(iter);
        CellArray{1,iter}=x0(iter); %  Create a cell since gradfun does not accept a vector as an input
    end
    Hessian_x0=double(subs(Hessian, CellArrayVar, CellArray));
    info.hessian=Hessian_x0;
    info.AbsError=double((norm(CSHessValue-Hessian_x0)));
    Norm=norm(Hessian_x0);
    if Norm>1e-16
        info.RelError=double(norm(CSHessValue-Hessian_x0)/Norm);
    end
end
end
