n_samples = 1000   # Number of samples
n_features = 50    # Number of features
sparsity = 0.8     # Proportion of zeros

X = rand(n_samples, n_features)

mask = rand(n_samples, n_features) .> sparsity
X[mask] .= 0

y = rand(0:1, n_samples)

column_names = [Symbol("feature_", i) for i in 1:n_features]
df = DataFrame(X, column_names)
df[!, :target] = y

df
#df[!, :target] = replace(df[!, :target], 0 => -1);

y = Array(select(df, "target"))
X_df = select(df, Not("target"))
X = Array(select(df, Not("target")));
y = vec(y);

using SubsetSelection, JuMP, Gurobi, CPLEX, MathOptInterface, LinearAlgebra, GLMNet

#import Pkg; Pkg.add("Compat")
#import Compat.String
#include("inner_op.jl")

getthreads() = haskey(ENV, "SLURM_JOB_CPUS_PER_NODE") ? parse(Int, ENV["SLURM_JOB_CPUS_PER_NODE"]) : 0

using SubsetSelection

#Inner Opti Functions

function inner_op(ℓ::LossFunction, Y, X, s, γ; stochastic=false)
    if stochastic
        inner_op_stochastic(ℓ, Y, X, s, γ)
    else
        inner_op_plain(ℓ, Y, X, s, γ)
    end
end


function inner_op_plain(ℓ::LossFunction, Y, X, s, γ)
  indices = findall(s .> .5); k = length(indices)
  n,p = size(X)

  # Compute optimal dual parameter
  α = sparse_inverse(ℓ, Y, X[:, indices], γ)
  c = SubsetSelection.value_dual(ℓ, Y, X, α, indices, k, γ)

  ∇c = zeros(p)
  for j in 1:p
    ∇c[j] = -γ/2*dot(X[:,j],α)^2
  end
  return c, ∇c
end


function inner_op_stochastic(ℓ::LossFunction, Y, X, s, γ; B=10, bSize=max(0.1,2*sum(s)/size(X,1)) )
  indices = findall(s .> .5); k = length(indices)
  n,p = size(X)

  # Compute optimal dual parameter
  w = zeros(k)
  for b in 1:B
    subset = rand(n) .< bSize
    w .+= SubsetSelection.recover_primal(ℓ, Y[subset], X[subset,indices], γ)
  end
  w ./= B
  α = start_primal(ℓ, Y, X[:,indices], γ)
  c = SubsetSelection.value_dual(ℓ, Y, X, α, indices, k, γ)

  ∇c = zeros(p)
  for j in 1:p
    ∇c[j] = -γ/2*dot(X[:,j],α)^2
  end
  return c, ∇c
end


using LIBLINEAR


#Optimal Dual using Matrix Inversion Lemma

function sparse_inverse(ℓ::Classification, Y, X, γ; valueThreshold=1e-8, maxIter=1e3)
    n,k = size(X)
    indices = collect(1:k); n_indices = k
    cache = SubsetSelection.Cache(n, k)

    α = start_primal(ℓ, Y, X, γ)
    value = SubsetSelection.value_dual(ℓ, Y, X, α, indices, k, γ)

    for iter in 1:maxIter
        ∇ = SubsetSelection.grad_dual(ℓ, Y, X, α, indices, n_indices, γ, cache) #Compute gradient

        if norm(∇, 1) <= 1e-14
          break
        end
        if norm(∇, 1) == Inf
            α[findall(∇ .== Inf)] .= -Y[findall(∇ .== Inf)]*1e-14
            α[findall(∇ .== -Inf)] .= -Y[findall(∇ .== -Inf)]*(1-1e-14)

            ∇[findall(∇ .== Inf)] .= 0.
            ∇[findall(∇ .== -Inf)] .= 0.
        end

        learningRate = 2/norm(∇, 1)
        α1 = α[:]
        newValue = value - 1.

        while newValue < value  #Divide step sie by two as long as f decreases
          learningRate /= 2
          α1[:] = α .+ learningRate*∇       #Compute new alpha
          SubsetSelection.proj_dual!(ℓ, Y, α1)    #Project
          newValue = SubsetSelection.value_dual(ℓ, Y, X, α1, indices, k, γ)  #Compute new f(alpha, s)
        end

        value_gap = 2*(newValue-value)/(value+newValue)
        α = α1[:]; value = newValue
        if abs(value_gap) < valueThreshold
          break
        end
    end

    return α
end

function start_primal(ℓ::Classification, Y::Array, X::Array, γ::Real)
  n,k = size(X)
  w = SubsetSelection.recover_primal(ℓ, Y, X, γ)
  α = primal2dual(ℓ, Y, X, w)
  return α
end

function primal2dual(ℓ::LogReg, Y, X, w)
  n = size(X, 1)
  return [-Y[i]*exp(-Y[i]*dot(X[i,:], w))/(1+exp(-Y[i]*dot(X[i,:], w))) for i in 1:n] #LR
end

function recover_primal(ℓ::LogReg, Y, Z, γ)
    solverNumber = LibLinearSolver(ℓ)

    # Call linear_train without the init_sol parameter
    model = linear_train(Y, Z'; verbose=false, C=γ, solver_type=Cint(solverNumber), bias=1.0)
    
    return Y[1]*model.w
end
function LibLinearSolver(ℓ::LogReg)
  return 0 # L2R_LR solver
end

function oa_formulation(ℓ::LossFunction, Y, X, k::Int, γ;
          indices0=findall(rand(size(X,2)) .< k/size(X,2)),
          ΔT_max=60, verbose=false, Gap=0e-3, solver::Symbol=:Gurobi,
          rootnode::Bool=true, rootCuts::Int=20, stochastic::Bool=false)

  n,p = size(X)

  miop = (solver == :Gurobi) ? Model(Gurobi.Optimizer) : Model(CPLEX.Optimizer)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "TimeLimit" : "CPX_PARAM_TILIM", ΔT_max)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "OutputFlag" : "CPX_PARAM_SCRIND", 1*verbose)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "MIPGap" : "CPX_PARAM_EPGAP", Gap)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "Threads" : "CPXPARAM_Threads", getthreads())

  s0 = zeros(p); s0[indices0] .= 1.
  c0, ∇c0 = inner_op(ℓ, Y, X, s0, γ, stochastic=stochastic)

  # Optimization variables
  @variable(miop, s[j=1:p], Bin, start=s0[j])
  @variable(miop, t>=0, start=1.005*c0)

  for j in 1:p
    JuMP.set_start_value(s[j], s0[j])
  end
  JuMP.set_start_value(t, 1.005*c0)

  # Objective
  @objective(miop, Min, t)

  # Constraints
  @constraint(miop, sum(s) <= k)

  #Root node analysis
  cutCount=1; bestObj=sum(s0)<= k ? c0 : +Inf; bestSolution=sum(s0)<= k ? s0[:] : [] ;
  @constraint(miop, t>= c0 + dot(∇c0, s-s0))

  if rootnode
    s1 = zeros(p)
    l1 = isa(ℓ, SubsetSelection.Classification) ? glmnet(X, convert(Matrix{Float64}, [(Y.<= 0) (Y.>0)]), GLMNet.Binomial(), dfmax=k, intercept=false) : glmnet(X, Y, dfmax=k, intercept=false)
    for  i in size(l1.betas, 2):-1:max(size(l1.betas, 2)-rootCuts,1) #Add first rootCuts cuts from Lasso path
      ind = findall(abs.(l1.betas[:, i]) .> 1e-8); s1[ind] .= 1.
      c1, ∇c1 = inner_op(ℓ, Y, X, s1, γ, stochastic=stochastic)
      @constraint(miop, t>= c1 + dot(∇c1, s-s1))
      cutCount += 1; s1 .= 0.
    end
  end

  # Outer approximation method for Convex Integer Optimization (CIO)
  function outer_approximation(cb_data)
    s_val = [callback_value(cb_data, s[j]) for j in 1:p] #vectorized version of callback_value is not currently offered in JuMP
    # if maximum(s_val.*(1 .- s_val)) < 10*eps()
      s_val = 1.0 .* (rand(p) .< s_val) #JuMP updates calls Lazy Callbacks at fractional solutions as well

      c, ∇c = inner_op(ℓ, Y, X, s_val, γ, stochastic=stochastic)
      if stochastic && callback_value(cb_data, t) > c #If stochastic version and did not cut the solution
          c, ∇c = inner_op(ℓ, Y, X, s_val, γ, stochastic=false)
      end
      if sum(s_val)<=k && c<bestObj #if feasible and best value
        bestObj = c; bestSolution=s_val[:]
      end

      con = @build_constraint(t >= c + dot(∇c, s-s_val))
      MOI.submit(miop, MOI.LazyConstraint(cb_data), con)
      cutCount += 1
    # end
  end
  MOI.set(miop, MOI.LazyConstraintCallback(), outer_approximation)

  # # Feed warmstart
  # if (solver == :Gurobi)
  #   wsCount = 0
  #   function warm_start(cb_data)
  #     if wsCount == 0
  #       MOI.submit(miop, MOI.HeuristicSolution(cb_data), [s[j] for j in 1:p], floor.(Int, s0))
  #       MOI.submit(miop, MOI.HeuristicSolution(cb_data), [t], [c0])
  #       wsCount += 1
  #     end
  #   end
  #   MOI.set(miop, MOI.HeuristicCallback(), warm_start)
  # end

  t0 = time()
  optimize!(miop)
  status = termination_status(miop)
  Δt = JuMP.solve_time(miop)

  Gap = 1 - JuMP.objective_bound(miop) /  abs(JuMP.objective_value(miop))
  if JuMP.objective_bound(miop) < bestObj
    bestSolution = value.(s)[:]
  end

  # Find selected regressors and run a standard linear regression with Tikhonov regularization
  indices = findall(bestSolution .> .5)
  w = SubsetSelection.recover_primal(ℓ, Y, X[:, indices], γ)

  return indices, w, Δt, status, Gap, cutCount
end

function oa_formulation_old(ℓ::LossFunction, Y, X, k::Int, γ;
          indices0=findall(rand(size(X,2)) .< k/size(X,2)),
          ΔT_max=60, verbose=false, Gap=0e-3, solver::Symbol=:Gurobi,
          rootnode::Bool=true, rootCuts::Int=20, stochastic::Bool=false)

  n,p = size(X)

  miop = (solver == :Gurobi) ? Model(Gurobi.Optimizer) : Model(CPLEX.Optimizer)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "TimeLimit" : "CPX_PARAM_TILIM", ΔT_max)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "OutputFlag" : "CPX_PARAM_SCRIND", 1*verbose)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "MIPGap" : "CPX_PARAM_EPGAP", Gap)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "Threads" : "CPXPARAM_Threads", getthreads())

  s0 = zeros(p); s0[indices0] .= 1.
  c0, ∇c0 = inner_op(ℓ, Y, X, s0, γ, stochastic=stochastic)

  # Optimization variables
  @variable(miop, s[j=1:p], Bin, start=s0[j])
  @variable(miop, t>=0, start=1.005*c0)

  for j in 1:p
    JuMP.set_start_value(s[j], s0[j])
  end
  JuMP.set_start_value(t, 1.005*c0)

  # Objective
  @objective(miop, Min, t)

  # Constraints
  @constraint(miop, sum(s) <= k)

  #Root node analysis
  cutCount=1; bestObj=sum(s0)<= k ? c0 : +Inf; bestSolution=sum(s0)<= k ? s0[:] : [] ;
  @constraint(miop, t>= c0 + dot(∇c0, s-s0))

  if rootnode
    s1 = zeros(p)
    l1 = isa(ℓ, SubsetSelection.Classification) ? glmnet(X, convert(Matrix{Float64}, [(Y.<= 0) (Y.>0)]), GLMNet.Binomial(), dfmax=k, intercept=false) : glmnet(X, Y, dfmax=k, intercept=false)
    for  i in size(l1.betas, 2):-1:max(size(l1.betas, 2)-rootCuts,1) #Add first rootCuts cuts from Lasso path
      ind = findall(abs.(l1.betas[:, i]) .> 1e-8); s1[ind] .= 1.
      c1, ∇c1 = inner_op(ℓ, Y, X, s1, γ, stochastic=stochastic)
      @constraint(miop, t>= c1 + dot(∇c1, s-s1))
      cutCount += 1; s1 .= 0.
    end
  end

  # Outer approximation method for Convex Integer Optimization (CIO)
  function outer_approximation(cb_data)
    s_val = [callback_value(cb_data, s[j]) for j in 1:p] #vectorized version of callback_value is not currently offered in JuMP
    # if maximum(s_val.*(1 .- s_val)) < 10*eps()
      s_val = 1.0 .* (rand(p) .< s_val) #JuMP updates calls Lazy Callbacks at fractional solutions as well

      c, ∇c = inner_op(ℓ, Y, X, s_val, γ, stochastic=stochastic)
      if stochastic && callback_value(cb_data, t) > c #If stochastic version and did not cut the solution
          c, ∇c = inner_op(ℓ, Y, X, s_val, γ, stochastic=false)
      end
      if sum(s_val)<=k && c<bestObj #if feasible and best value
        bestObj = c; bestSolution=s_val[:]
      end

      con = @build_constraint(t >= c + dot(∇c, s-s_val))
      MOI.submit(miop, MOI.LazyConstraint(cb_data), con)
      cutCount += 1
    # end
  end
  MOI.set(miop, MOI.LazyConstraintCallback(), outer_approximation)

  # # Feed warmstart
  # if (solver == :Gurobi)
  #   wsCount = 0
  #   function warm_start(cb_data)
  #     if wsCount == 0
  #       MOI.submit(miop, MOI.HeuristicSolution(cb_data), [s[j] for j in 1:p], floor.(Int, s0))
  #       MOI.submit(miop, MOI.HeuristicSolution(cb_data), [t], [c0])
  #       wsCount += 1
  #     end
  #   end
  #   MOI.set(miop, MOI.HeuristicCallback(), warm_start)
  # end

  t0 = time()
  optimize!(miop)
  status = termination_status(miop)
  Δt = JuMP.solve_time(miop)

  Gap = 1 - JuMP.objective_bound(miop) /  abs(JuMP.objective_value(miop))
  if JuMP.objective_bound(miop) < bestObj
    bestSolution = value.(s)[:]
  end

  # Find selected regressors and run a standard linear regression with Tikhonov regularization
    indices = findall(bestSolution .> .5)
    w = SubsetSelection.recover_primal(ℓ, Y, X[:, indices], γ)
    return indices, w, Δt, status, Gap, cutCount
end

ℓ = SubsetSelection.LogReg()

indices, w, Δt, status, Gap, cutCount = oa_formulation(
    ℓ, 
    y, 
    X, 
    30, 
    0.1;
    indices0=findall(rand(size(X,2)) .< 30/size(X,2)),
    ΔT_max=60, 
    verbose=false, 
    Gap=0e-3, 
    solver=:Gurobi,
    rootnode=true, 
    rootCuts=20, 
    stochastic=false
)

y = Array(y);

#cross validation
using GLMNet
cv_model = glmnetcv(X, y)
best_lambda = cv_model.lambda[argmin(cv_model.meanloss)]
best_model = glmnet(X, y, lambda = [best_lambda])
coefficients_cv = best_model.betas

indices_cv = []
for i in 1:length(coefficients_cv)
    if coefficients_cv[i] != 0
        push!(indices_cv, i)
    end
end

indices_cv;

using DataFrames
selected_columns = [names(df)[indices_cv]; names(df)[end]]
new_df_cv = select(df, selected_columns)

#no cross validation
model = glmnet(X,y, lambda = [0.01])
coefficients_no_cv = model.betas
indices_no_cv = []
for i in 1:length(coefficients_no_cv)
    if coefficients_no_cv[i] != 0
        push!(indices_no_cv, i)
    end
end

indices_no_cv;

using DataFrames
selected_columns = [names(df)[indices_no_cv]; names(df)[end]]
new_df_no_cv = select(df, selected_columns)

#save the new dataset as csv (with reduced dimensionality)

#Given X and y (df and vector)

#Create train, val and test set 
seed = 15095
(X2, y2), (X_test, y_test) = IAI.split_data(:classification, X, y, seed=seed, train_proportion=0.8)
(X_train, y_train), (X_valid, y_valid) =  IAI.split_data(:classification, X2, y2, seed=seed, train_proportion=0.8);

cart = IAI.OptimalTreeClassifier(random_seed=1, 
                    localsearch=false, 
                    criterion=:gini,
                    max_categoric_levels_before_warning = 100)
grid_cart = IAI.GridSearch(cart, max_depth=2:6, minbucket=5:30)
IAI.fit_cv!(grid_cart, X2, Array(y2), validation_criterion = :auc, n_folds=5)
cart = IAI.get_learner(grid_cart)

#Find CART out of sample auc & accuracy
test_AUC_cart = IAI.score(cart,X_test, y_test,criterion=:auc)
test_acc_cart = IAI.score(cart, X_test, y_test, criterion=:accuracy, positive_label=1)

println("CART")

println("Test AUC = $test_AUC_cart")
println("Test accuracy = $test_acc_cart")

rf = IAI.RandomForestClassifier(
        random_seed=seed,
        max_categoric_levels_before_warning = 100
    )
grid_rf = IAI.GridSearch(rf, max_depth=2:6, num_trees = 30:70)
IAI.fit!(grid_rf, X2, Array(y2), validation_criterion = :auc)
rf = IAI.get_learner(grid_rf)

#Find Random Forest training and testing auc & accuracy
test_AUC_rf = IAI.score(rf,X_test, y_test,criterion=:auc)
test_acc_rf = IAI.score(rf, X_test, y_test, criterion=:accuracy, positive_label=1)
println("RANDOM FOREST")
println("Test accuracy = $test_acc_rf")
println("Test AUC = $test_AUC_rf")

xgb = IAI.XGBoostClassifier(
        random_seed=seed, max_categoric_levels_before_warning = 100
    )
grid_xgb = IAI.GridSearch(xgb, max_depth=2:6, num_estimators = 30:70)
IAI.fit!(grid_xgb, X2, Array(y2), validation_criterion = :auc)
xgb = IAI.get_learner(grid_xgb)

#Find xgboost training and testing auc & accuracy
test_AUC_xgb = IAI.score(xgb,X_test, y_test,criterion=:auc)
test_acc_xgb = IAI.score(xgb,X_train, y_train,criterion=:accuracy, positive_label=1)
println("XGBOOST")
println("Test accuracy = $test_acc_xgb")
println("Test AUC = $test_AUC_xgb")
