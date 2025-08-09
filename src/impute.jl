using CSV, DataFrames, Statistics

# import train and test sets from brfss_preprocessing.ipynb
X_train = CSV.read("X_train_sparse.csv", DataFrame)
y_train = CSV.read("y_train_sparse.csv", DataFrame)
X_test = CSV.read("X_test_sparse.csv", DataFrame)
y_test = CSV.read("y_test_sparse.csv", DataFrame)
w_train = CSV.read("w_train_sparse.csv", DataFrame)
w_test = CSV.read("w_test_sparse.csv", DataFrame);

# see how much data is missing in each column
println(DataFrame(col=propertynames(X_train), 
    missing_values=[sum(ismissing.(col)) for col in eachcol(X_train)]))

# split X_train and X_test into two parts to perform opt_knn imputation
X_train_1 = X_train[!, [:GENHLTH, :PHYSHLTH, :MAXVO2_, :FC60_, :EMPLOY1]]
X_train_2 = X_train[!, [:BPHIGH4, :TOLDHI2, :CVDINFR4, :CHOLCHK, :_BMI5CAT]]

X_test_1 = X_test[!, [:GENHLTH, :PHYSHLTH, :MAXVO2_, :FC60_, :EMPLOY1]]
X_test_2 = X_test[!, [:BPHIGH4, :TOLDHI2, :CVDINFR4, :CHOLCHK, :_BMI5CAT]];

# initiate opt_knn learner
lnr = IAI.ImputationLearner(method=:opt_knn, random_seed=1)

# perform opt_knn imputation on training and test data
X_train_1_imp = IAI.fit_transform!(lnr, X_train_1)
X_test_1_imp = IAI.transform(lnr, X_test_1)

# transform testing data
X_train_2_imp = IAI.fit_transform!(lnr, X_train_2)
X_test_2_imp = IAI.transform(lnr, X_test_2)

# merge all training data and test data
X_train_imputed = hcat(X_train_1_imp, X_train_2_imp, X_train[!, [:_AGE80]])
X_test_imputed = hcat(X_test_1_imp, X_test_2_imp, X_test[!, [:_AGE80]])

# make sure all values except weight column are ints, round if not
for col in names(X_train_imputed)
    if col != "WEIGHT"
        X_train_imputed[!, col] .= round.(Int, X_train_imputed[!, col])
    end
end

for col in names(X_test_imputed)
    if col != "WEIGHT"
        X_test_imputed[!, col] .= round.(Int, X_test_imputed[!, col])
    end
end

# export imputed train and test data to csv
CSV.write("X_train_imputed.csv", X_train_imputed, delim=",")
CSV.write("X_test_imputed.csv", X_test_imputed, delim=",")
