"""
Implementation of "Scalable Kernel Methods via Doubly Stochastic Gradients" by Dail et al., 2015
Author: Dr. Fayyaz Minhas
Contact: afsar at pieas do edu dot pk
Web: http://faculty.pieas.edu.pk
"""
function xrand(i::Int,d::Int,g::Float64)
  srand(i)
  return sqrt(2*g)*randn(d)
end
function phi(x::Vector{Float64},w::Vector{Float64},T::Int64)
  z = dot(x,w)
  return [cos(z) sin(z)]/sqrt(T)
end
function predict(x::Vector{Float64},alpha::Array{Float64},g::Float64,T::Int64,k::Int64)
  z = 0
  d = length(x)
  for i=1:k
    if (alpha[1,i]!=0 || alpha[2,i]!=0)
      w=xrand(i,d,g)
      p = phi(x,w,T)
      z+=dot(alpha[:,i],p)
    end
  end
  return z
end

function genData(n)
  srand()
  X = randn((2,2*n))
  X[:,1:n]+=1.5
  X[:,n+1:2*n]-=1.5
  y = ones(2*n)
  y[n+1:2*n]=-1
  return X,y
end

function dsgd(X,y,g,T,reg_param)
  N = length(y)
  d = size(X)[1]
  alpha = zeros(2,T)
  step_size0 = 1.0
  step_size1 = reg_param
  vv = 0
  for i=1:T
    srand()
    idx = rand(1:N)
    x = X[:,idx]
    w = xrand(i,d,g)
    p = phi(x,w,T)
    z = predict(x,alpha,g,T,i-1)
    g_i = step_size0 / (1+step_size1 * i);
    if ((y[idx]*z)<1)
      vv+=1
      gp = g_i*y[idx]
      alpha[:,i]=gp.*p
    end
    if (reg_param>1e-6)
      alpha[:,1:i-1].*=(1-g_i*reg_param)
    end
  end
  print(vv)
  return alpha
end

##### TRAINING
X,y = genData(100)
g = 1.0
reg_param = 1e-2
T = 2000
alpha = dsgd(X,y,g,T,reg_param)
#### TESTING
X,y = genData(100)
N = length(y)
S = zeros(N)
for i=1:N
  S[i]=predict(X[:,i],alpha,g,T,T)
end
print(sum(y.==2*(S.>0)-1)/N)

#pidx = find(y.==1)
#nidx = find(y.==-1)
using Plots
scatter([X[1,pidx],X[1,nidx]],[X[2,pidx],X[2,nidx]])
#scatter(X[pidx,1],X[pidx,2],color="red")
#scatter(X[nidx,1],X[nidx,2],color="blue")
