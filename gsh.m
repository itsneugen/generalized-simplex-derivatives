%Author: Gabriel Jarry-Bolduc
%Date created: October 31, 2022.
%Last update: May 11, 2025.
%
%This function returns an approximation of the hessian of f at x^0 using
%the sets S and Ti. This technique is called the generalized simplex
%Hessian
%
%Input:fun: function handle from R^n to R
%      x0: point of interest x0. Column vector in R^n
%       S: set of directions in matrix form  n \times m to add to x0
%       Ti: second sets of directions in matrix forms to add to each point
%       x0, x0+s_i. Provided as one matrix if all Ti are equal. Otherwise,
%       the input is a cell array containing m+1 matrices of dimensions n \times k_i:
%       {T1; \dots;Tm}
%       h1: parameter to shrink S. Positive number
%       h2: parameter to shrink all Ti. Positive number
%Output: SHessValue: Simplex Hessian value at x_0
%        info: info.AbsError: Absolute error between the true Hessian and
%                             the GSH          
%              info.RelError: relative error
%              info.Hessian_x0: True Hessian at x0

function [SHessValue, info] = gsh(fun,x0,S,Ti,h1,h2)
%% Set h1=1 and h2=1 if it is not specified
if (nargin==4)
  h1=1;
  h2=1;
elseif (nargin==5)
   h2=1; 
end
%% Validate that h1 and h2 are positive real numbers
classes = {'double'};
attributes = {'size',[1,1],'>',0};
validateattributes(h1,classes,attributes); 
validateattributes(h2,classes,attributes); 
%% Check the form of the input Ti
[n,m]=size(S); %Dimension of S
TF=iscell(Ti); %Check if Ti is a cell array or not
if TF==1 %true
    LengthTi=length(Ti);
else %Ti should be a matrix  with n rows
    [nT,~]=size(Ti);   
end   
%% Size matrix S
if TF==0
    if nT~=n
        error('number of rows in Ti should be the same as the number of rows in S')
    end
elseif m~=LengthTi
   error('number of T_i should be the same as the number of columns in S') 
end
%% Create the matrix S_h1 and T_h2
   S_hone=h1*S;
   if TF==0
        Ti_htwo=h2*Ti;
   else %Ti is a cell array
       for i=1:LengthTi
           Ti_htwo{i}=h2*Ti{i};
       end
   end
%% Compute \nabla_s f(x^0+s^i;Ti); for all i 
sgMat=zeros(n,m); %initialize
if TF==0 %All Ti are equal
    for i=1:m
      sgMat(:,i)=gsg(fun,x0+S_hone(:,i),Ti_htwo); 
    end
    sgX0T=gsg(fun,x0,Ti_htwo);
else %Ti are provided as a cell array
    sgMat=cell(m,2);
    for i=1:m
      sgMat{i,1}=gsg(fun,x0+S_hone(:,i),Ti_htwo{i}); 
      sgMat{i,2}=gsg(fun,x0,Ti_htwo{i});
    end
end 
%% Create the matrix delta_{\nabla_s f} (row i :\nabla_s f(x^0+s^i;Ti)-\nabla_s f(x^0;Ti))
if TF==0
 for row=1:m
     delta_s(row,:)=(sgMat(:,row)-sgX0T)';
 end
else
    for row=1:m
        delta_s(row,:)=(sgMat{row,1}-sgMat{row,2})';
    end
end
%% Calculate SGradValue
SHessValue=transpose(pinv(S_hone))*delta_s;
%% Calculate Absolute Error
if nargout==2
x=sym('x',[n,1]);
Fun=fun(x);
Hessian=hessian(Fun,x);
for iter=1:n
        CellArrayVar{1,iter}=x(iter);
        CellArray{1,iter}=x0(iter); %  Create a cell since gradfun does not accept a vector as an input
end
Hessian_x0=double(subs(Hessian, CellArrayVar, CellArray));
info.hessian=Hessian_x0;
info.AbsError=double((norm(SHessValue-Hessian_x0)));
Norm=norm(Hessian_x0);
if Norm>1e-16
    info.RelError=double(norm(SHessValue-Hessian_x0)/Norm);
end
end
end
