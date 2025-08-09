# IAI Models (merged)

using CSV, DataFrames, Statistics, Random, StatsBase, Distances, JuMP, Gurobi, Clustering, Distributions, ScikitLearn, DecisionTree, MLDataUtils, CSV, DataFrames, CategoricalArrays

#using sparse version of train and test
X2 = CSV.read("short_X_train", DataFrame);
X_test = CSV.read("short_X_test", DataFrame);
y2 = CSV.read("y_train 09.40.55.csv", DataFrame)
y_test = CSV.read("y_test 09.40.55.csv", DataFrame);

#standardizing both X_train and X_test
stan_X_train = copy(X_train)
stan_X_test = copy(X_test)
for col in names(stan_X_train)
    # Check if the column is numeric and not binary
    if eltype(stan_X_train[!, col]) <: Number && length(unique(stan_X_train[!, col])) > 2 #no standardization of binary variables
        col_mean = mean(stan_X_train[!, col])
        col_std = std(stan_X_train[!, col])

        # Standardize the column
        stan_X_train[!, col] = (stan_X_train[!, col] .- col_mean) ./ col_std
    end
end

for col in names(stan_X_test)
    # Check if the column is numeric and not binary
    if eltype(stan_X_test[!, col]) <: Number && length(unique(stan_X_test[!, col])) > 2 #no standardization of binary variables
        col_mean = mean(stan_X_test[!, col])
        col_std = std(stan_X_test[!, col])

        # Standardize the column
        stan_X_test[!, col] = (stan_X_test[!, col] .- col_mean) ./ col_std
    end
end

X2 = Matrix(X2)
y2 = vec(y2);

#Create train, val and test set 
seed = 15095
(X_train, y_train), (X_valid, y_valid) =  IAI.split_data(:classification, X2, y2, seed=seed, train_proportion=0.8);

cart = IAI.OptimalTreeClassifier(random_seed=1, 
                    localsearch=false, 
                    criterion=:gini,
                    max_categoric_levels_before_warning = 100)
grid_cart = IAI.GridSearch(cart, max_depth=2:6, minbucket=5:30)
IAI.fit_cv!(grid_cart, X2, Array(y2), validation_criterion = :auc, n_folds=5)
cart = IAI.get_learner(grid_cart)

#Find CART in and out of sample auc & accuracy
train_AUC_cart = IAI.score(cart,X2, y2,criterion=:auc)
train_acc_cart = IAI.score(cart, X2, y2, criterion=:accuracy, positive_label=1)

test_AUC_cart = IAI.score(cart,X_test, y_test,criterion=:auc)
test_acc_cart = IAI.score(cart, X_test, y_test, criterion=:accuracy, positive_label=1)

println("CART")

println("Train AUC = $train_AUC_cart")
println("Train accuracy = $train_acc_cart")
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
train_AUC_rf = IAI.score(rf,X2, y2,criterion=:auc)
train_acc_rf = IAI.score(rf, X2, y2, criterion=:accuracy, positive_label=1)

test_AUC_rf = IAI.score(rf,X_test, y_test,criterion=:auc)
test_acc_rf = IAI.score(rf, X_test, y_test, criterion=:accuracy, positive_label=1)

println("Random Forest")

println("Train AUC = $train_AUC_rf")
println("Train accuracy = $train_acc_rf")
println("Test AUC = $test_AUC_rf")
println("Test accuracy = $test_acc_rf")

xgb = IAI.XGBoostClassifier(
        random_seed=seed, max_categoric_levels_before_warning = 100
    )
grid_xgb = IAI.GridSearch(xgb, max_depth=2:6, num_estimators = 30:70)
IAI.fit!(grid_xgb, X2, Array(y2), validation_criterion = :auc)
xgb = IAI.get_learner(grid_xgb)

#Find XGBoost training and testing auc & accuracy
train_AUC_xgb = IAI.score(xgb,X2, y2,criterion=:auc)
train_acc_xgb = IAI.score(xgb, X2, y2, criterion=:accuracy, positive_label=1)

test_AUC_xgb = IAI.score(xgb,X_test, y_test,criterion=:auc)
test_acc_xgb = IAI.score(xgb, X_test, y_test, criterion=:accuracy, positive_label=1)

println("XGBoost")

println("Train AUC = $train_AUC_xgb")
println("Train accuracy = $train_acc_xgb")
println("Test AUC = $test_AUC_xgb")
println("Test accuracy = $test_acc_xgb")
#takes about 5 mins to run with this data

oct = IAI.OptimalTreeClassifier(random_seed=1, 
                    criterion=:gini,
                    max_categoric_levels_before_warning = 100)

grid_oct = IAI.GridSearch(oct, max_depth=[3,5,7], minbucket=[5,10,15,20,30])
IAI.fit_cv!(grid_oct, X2, Array(y2), validation_criterion = :auc, n_folds=5)
oct = IAI.get_learner(grid_oct)

#Find OCT training and testing auc & accuracy
train_AUC_oct = IAI.score(oct,X2, y2,criterion=:auc)
train_acc_oct = IAI.score(oct, X2, y2, criterion=:accuracy, positive_label=1)

test_AUC_oct = IAI.score(oct,X_test, y_test,criterion=:auc)
test_acc_oct = IAI.score(oct, X_test, y_test, criterion=:accuracy, positive_label=1)

println("OCT")

println("Train AUC = $train_AUC_oct")
println("Train accuracy = $train_acc_oct")
println("Test AUC = $test_AUC_oct")
println("Test accuracy = $test_acc_oct")

#add to total df a vector of your probabilities
X2[!,"probabilities"] = IAI.predict_proba(xgb, X2)[!,2]
X_test[!,"probabilities"] = IAI.predict_proba(xgb, X_test)[!,2]
CSV.write("X2.csv", X2)
CSV.write("X_test.csv", X_test)

X_proba = vcat(X2, X_test);
y_tot = vcat(y2, y_test)
df = hcat (X_proba, y_tot)

#standardize df
stan_df = copy(df)
for col in names(stan_df)
    # Check if the column is numeric and not binary
    if eltype(stan_df[!, col]) <: Number && length(unique(stan_df[!, col])) > 2 #no standardization of binary variables
        col_mean = mean(stan_df[!, col])
        col_std = std(stan_df[!, col])

        # Standardize the column
        stan_df[!, col] = (stan_df[!, col] .- col_mean) ./ col_std
    end
end

#cols_to_join = names(X_proba)[1:end-1]
#merged_df = leftjoin(X_proba, df, on=cols_to_join)

#stan_merged_df = copy(merged_df)
#for col in names(stan_merged_df)
    # Check if the column is numeric and not binary
    #if eltype(stan_merged_df[!, col]) <: Number && length(unique(stan_merged_df[!, col])) > 2 #no standardization of binary variables
        #col_mean = mean(stan_merged_df[!, col])
        #col_std = std(stan_merged_df[!, col])

        # Standardize the column
        #stan_merged_df[!, col] = (stan_merged_df[!, col] .- col_mean) ./ col_std
    #end
#end

#code to use predictions in test set as output
#X2[!,"target_cluster"] = y2 #take 
#X_test[!,"target_cluster"] = xgb_predictions
#cluster_df = vcat(X2, X_test)

#df_cluster = filter(row -> row[:target] == 1, df)
X = select(df_cluster, Not("target"));

stan_X = copy(X)
for col in names(stan_X)
    # Check if the column is numeric and not binary
    if eltype(stan_X[!, col]) <: Number && length(unique(stan_X[!, col])) > 2 #no standardization of binary variables
        col_mean = mean(stan_X[!, col])
        col_std = std(stan_X[!, col])

        # Standardize the column
        stan_X[!, col] = (stan_X[!, col] .- col_mean) ./ col_std
    end
end

CSV.write("hclust_scree_df", stan_X)

#using for normal Julia to run scree plot
using CSV, DataFrames, Distances, Clustering, Plots

stan_X = CSV.read("hclust_scree_df", DataFrame)
#hierarchical clustering
data_matrix = Matrix(stan_X);
distance_matrix = pairwise(Euclidean(), data_matrix, data_matrix; dims=1);
distance_matrix = (distance_matrix + transpose(distance_matrix)) / 2;
hclust_result = hclust(distance_matrix, linkage = :ward);

# scree plot to use elbow rule to decide the number of clusters
hc_dissim = [(k = length(hclust_result.height) - i, 
              dissimilarity = hclust_result.height[i]) for i in 1:length(hclust_result.height)];

#plot([x.k for x in hc_dissim], [x.dissimilarity for x in hc_dissim], line=:path, xlims = (0,40), xlabel="k", ylabel="Dissimilarity")
nclust = 9 #set number of clusters as result of on elbow rule

clusters = cutree(hclust_result, k = nclust)
stan_X.clusters = clusters;
X.clusters = clusters;

CSV.write("stan_X", stan_X)
CSV.write("X",X);

using CSV, DataFrames, Statistics, Random, StatsBase, Distances, JuMP, Gurobi, Clustering, Distributions, ScikitLearn, DecisionTree, MLDataUtils, CSV, DataFrames, CategoricalArrays

X = CSV.read("X", DataFrame)
y = X[!, :clusters]
X_data = X[:,1:end-1];

lnr = IAI.OptimalTreeClassifier(random_seed=1, 
                    criterion=:misclassification,
                    max_depth=3,
                    cp = 0.001,
                    max_categoric_levels_before_warning = 100,
                    minbucket=5)
IAI.fit!(lnr, X_data, y)

#take only rows where target == 1
merged_df1 = filter(row -> row[:target] == 1, merged_df)
cols_to_join = names(X)[1:end-1];
new_df = outerjoin(X, merged_df1, on=cols_to_join)
new_df = new_df[:,1:end-1]

grouped_df = groupby(new_df, :clusters)

summary_stats = combine(grouped_df, :probabilities => mean, 
                                   :probabilities => median, 
                                   :probabilities => std)

using CSV, DataFrames, Statistics, Random, StatsPlots, StatsBase, Distances, JuMP, Gurobi, Clustering, Distributions, ScikitLearn, DecisionTree, MLDataUtils, CategoricalArrays, Serialization

#using sparse version of train and test
X_train = CSV.read("/Users/nataliechuang/Documents/MIT/Coursework/Machine Learning/Project/ML_project/X_train_imputed.csv", DataFrame) # from Natalie impute.ipynb
X_test = CSV.read("/Users/nataliechuang/Documents/MIT/Coursework/Machine Learning/Project/ML_project/X_test_imputed.csv", DataFrame); # from Natalie impute.ipynb
y_train = CSV.read("/Users/nataliechuang/Documents/MIT/Coursework/Machine Learning/Project/ML_project/y_train_sparse.csv", DataFrame) # from preprocessing.ipynb
y_test = CSV.read("/Users/nataliechuang/Documents/MIT/Coursework/Machine Learning/Project/ML_project/y_test_sparse.csv", DataFrame) # from preprocessing.ipynb
w_train = CSV.read("/Users/nataliechuang/Documents/MIT/Coursework/Machine Learning/Project/ML_project/w_train_sparse.csv", DataFrame)
w_test = CSV.read("/Users/nataliechuang/Documents/MIT/Coursework/Machine Learning/Project/ML_project/w_test_sparse.csv", DataFrame);

# balance dataset
train_df = hcat(X_train, y_train, w_train)

df_0 = filter(row -> row[:DIABETE3] == 0, train_df)
df_1 = filter(row -> row[:DIABETE3] == 1, train_df)

min_count = min(nrow(df_0), nrow(df_1))

if nrow(df_0) > min_count
    df_0 = df_0[shuffle(1:nrow(df_0))[1:min_count], :]
else
    df_1 = df_1[shuffle(1:nrow(df_1))[1:min_count], :]
end

balanced_df = vcat(df_0, df_1);

#from 300k to 90k observations

# convert three classes to binary
balanced_df[balanced_df.DIABETE3 .== 2, :DIABETE3] .= 1
y_test[y_test.DIABETE3 .== 2, :DIABETE3] .= 1;

# split back into X, y, and w
y_train_balanced = select(balanced_df, "DIABETE3")
w_train_balanced = select(balanced_df, "WEIGHT")
X_train_balanced = balanced_df[!,1:(end - 2)];

X_train = Matrix(X_train_balanced)
y_train = vec(Array(y_train_balanced))
X_test = Matrix(X_test)
y_test = vec(Array(y_test));

#Create train, val and test set 
seed = 15095
(X_train, y_train), (X_valid, y_valid) =  IAI.split_data(:classification, X_train, y_train, seed=seed, train_proportion=0.8);

cart = IAI.OptimalTreeClassifier(random_seed=1, 
                    localsearch=false, 
                    criterion=:gini,
                    max_categoric_levels_before_warning = 100)
grid_cart = IAI.GridSearch(cart, max_depth=2:6, minbucket=5:30)
IAI.fit_cv!(grid_cart, X_train, y_train, positive_label = 1, validation_criterion = :auc, n_folds=5)
cart = IAI.get_learner(grid_cart)

#Find CART in and out of sample auc & accuracy
train_AUC_cart = IAI.score(cart, X_train, y_train, criterion=:auc)
train_acc_cart = IAI.score(cart, X_train, y_train, criterion=:accuracy, positive_label=1)

test_AUC_cart = IAI.score(cart, X_test, y_test, criterion=:auc)
test_acc_cart = IAI.score(cart, X_test, y_test, criterion=:accuracy, positive_label=1)

println("CART")

println("Train AUC = $train_AUC_cart")
println("Train accuracy = $train_acc_cart")
println("Test AUC = $test_AUC_cart")
println("Test accuracy = $test_acc_cart")

# serialize and save CART model
model_filename = "cart_model.jls"
open(model_filename, "w") do file
    serialize(file, cart)
end

rf = IAI.RandomForestClassifier(
        random_seed=seed,
        max_categoric_levels_before_warning = 100
    )
grid_rf = IAI.GridSearch(rf, max_depth=2:6, num_trees = 30:70)
IAI.fit!(grid_rf, X_train, y_train, validation_criterion = :auc)
rf = IAI.get_learner(grid_rf)

#Find Random Forest training and testing auc & accuracy
train_AUC_rf = IAI.score(rf, X_train, y_train, criterion=:auc)
train_acc_rf = IAI.score(rf, X_train, y_train, criterion=:accuracy, positive_label=1)

test_AUC_rf = IAI.score(rf, X_test, y_test,criterion=:auc)
test_acc_rf = IAI.score(rf, X_test, y_test, criterion=:accuracy, positive_label=1)

println("Random Forest")

println("Train AUC = $train_AUC_rf")
println("Train accuracy = $train_acc_rf")
println("Test AUC = $test_AUC_rf")
println("Test accuracy = $test_acc_rf")

# serialize and save RF model
model_filename = "rf_model.jls"
open(model_filename, "w") do file
    serialize(file, rf)
end

xgb = IAI.XGBoostClassifier(
        random_seed=seed, max_categoric_levels_before_warning = 100
    )
grid_xgb = IAI.GridSearch(xgb, max_depth=2:6, num_estimators = 30:70)
IAI.fit!(grid_xgb, X_train, y_train, validation_criterion = :auc)
xgb = IAI.get_learner(grid_xgb)

#Find XGBoost training and testing auc & accuracy
train_AUC_xgb = IAI.score(xgb, X_train, y_train,criterion=:auc)
train_acc_xgb = IAI.score(xgb, X_train, y_train, criterion=:accuracy, positive_label=1)

test_AUC_xgb = IAI.score(xgb, X_test, y_test, criterion=:auc)
test_acc_xgb = IAI.score(xgb, X_test, y_test, criterion=:accuracy, positive_label=1)

println("XGBoost")

println("Train AUC = $train_AUC_xgb")
println("Train accuracy = $train_acc_xgb")
println("Test AUC = $test_AUC_xgb")
println("Test accuracy = $test_acc_xgb")
#takes about 5 mins to run with this data

# serialize and save XBoost model
model_filename = "xgb_model.jls"
open(model_filename, "w") do file
    serialize(file, xgb)
end

loaded_xgb = nothing
open("xgb_model.jls", "r") do file
    loaded_xgb = deserialize(file)
end

oct = IAI.OptimalTreeClassifier(random_seed=1, 
                    criterion=:gini,
                    max_categoric_levels_before_warning = 100)

grid_oct = IAI.GridSearch(oct, max_depth=[3,5,7], minbucket=[5,10,15,20,30])
IAI.fit_cv!(grid_oct, X_train, y_train, validation_criterion = :auc, n_folds=5)
oct = IAI.get_learner(grid_oct)

#Find OCT training and testing auc & accuracy
train_AUC_oct = IAI.score(oct, X_train, y_train, criterion=:auc)
train_acc_oct = IAI.score(oct, X_train, y_train, criterion=:accuracy, positive_label=1)

test_AUC_oct = IAI.score(oct, X_test, y_test,criterion=:auc)
test_acc_oct = IAI.score(oct, X_test, y_test, criterion=:accuracy, positive_label=1)

println("OCT")

println("Train AUC = $train_AUC_oct")
println("Train accuracy = $train_acc_oct")
println("Test AUC = $test_AUC_oct")
println("Test accuracy = $test_acc_oct")

# serialize and save OCT model
model_filename = "oct_model.jls"
open(model_filename, "w") do file
    serialize(file, oct)
end

#add to total df a vector of your probabilities

model = xgb

X_train_df = DataFrame(X_train, :auto)
X_test_df = DataFrame(X_test, :auto)

X_train_df[!,"probabilities"] = IAI.predict_proba(model, X_train)[!,2]
X_test_df[!,"probabilities"] = IAI.predict_proba(model, X_test)[!,2]
CSV.write("X_train_w_prob.csv", X_train_df)
CSV.write("X_test_w_prob.csv", X_test_df)

X_tot = vcat(X_train_df, X_test_df);
y_tot = vcat(y_train, y_test)
df = hcat(X_tot, y_tot, makeunique=true);

# rename columns in new df with original features names
rename!(df, Dict("x1" => "GENHLTH", "x2" => "PHYSHLTH", "x3" => "MAXVO2_", "x4" => "FC60_", "x5" => "EMPLOY1", "x6" => "BPHIGH4",
        "x7" => "TOLDHI2", "x8" => "CVDINFR4", "x9" => "CHOLCHK", "x10" => "_BMI5CAT", "x11" => "_AGE80", "x1_1" => "DIABETES"));

rename!(X_tot, Dict("x1" => "GENHLTH", "x2" => "PHYSHLTH", "x3" => "MAXVO2_", "x4" => "FC60_", "x5" => "EMPLOY1", "x6" => "BPHIGH4",
        "x7" => "TOLDHI2", "x8" => "CVDINFR4", "x9" => "CHOLCHK", "x10" => "_BMI5CAT", "x11" => "_AGE80"))

CSV.write("data_w_probs.csv", df)

#using for normal Julia to run scree plot / if Plots work in IAI run "using Plots"
using CSV, DataFrames, Distances, Clustering, Plots, Statistics

#X2_df = CSV.read("X2_clustering.csv", DataFrame)
#X_test_df = CSV.read("X_test_clustering.csv", DataFrame);
#X_df = vcat(X2_df, X_test_df);
#y2 = CSV.read("y_train 09.40.55.csv", DataFrame)
#y_test = CSV.read("y_test 09.40.55.csv", DataFrame);

#adding weight to probabilities in df
weight = 3  
X_tot[:, "probabilities"] *= weight;

#standardize AFTER having applied the weight to keep quantities comparable
stan_X = copy(X_tot)
for col in names(stan_X)
    if eltype(stan_X[!, col]) <: Number && length(unique(stan_X[!, col])) > 2 #no standardization of binary variables
        col_mean = mean(stan_X[!, col])
        col_std = std(stan_X[!, col])
        stan_X[!, col] = (stan_X[!, col] .- col_mean) ./ col_std
    end
end

# convert X_tot to matrix
stan_X_matrix = Matrix(stan_X);

# scree plot for different cluster size
Random.seed!(10)
ss = []
for k in 1:20
    result = kmeans(stan_X_matrix', k)
    push!(ss, result.totalcost)
end

plot(1:20, ss, xlabel="Clusters", ylabel="Sum of Squared Distances")

# try 5 clusters
Random.seed!(10)
result = kmeans(stan_X_matrix', 6)
clusters = assignments(result)
centroids = result.centers;

# train OCT to predict cluster
lnr = IAI.OptimalTreeClassifier(
    random_seed = 10,
    criterion = :misclassification,
    max_depth = 5, 
    cp = 0.05)
IAI.fit!(lnr, X_tot, clusters)

X_tot.clusters = clusters;

grouped_df = groupby(X_tot, :clusters)

summary_stats = combine(grouped_df, :probabilities => mean, 
                                   :probabilities => median, 
                                   :probabilities => std)

#plot the different median and deviation (boxplots) across clusters
boxplot(X_tot.clusters, X_tot.probabilities, 
        label="Probabilities",
        xlabel="Clusters",
        ylabel="Probability",
        title="Probability Distribution Across Clusters",
        whisker_width=0.5,
        legend=:topright)
