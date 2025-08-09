# Main entry point
# Auto-generated on 2025-08-09
# Usage examples:
#   julia --project=. main.jl --stage=preprocess --input data/raw.csv --output data/clean.csv
#   julia --project=. main.jl --stage=train --config configs/default.yaml

using ArgParse

function build_parser()
    s = ArgParseSettings(description="Project runner")
    @add_arg_table s begin
        "--stage"
            help = "Which pipeline stage to run: preprocess | impute | train | evaluate"
            arg_type = String
            required = true
        "--input"
            help = "Input file (CSV/Parquet)"
            arg_type = String
            default = ""
        "--output"
            help = "Output file path"
            arg_type = String
            default = ""
        "--config"
            help = "Path to a config file (YAML/TOML)"
            arg_type = String
            default = ""
        "--seed"
            help = "Random seed for reproducibility"
            arg_type = Int
            default = 42
    end
    return s
end

# Includes (expecting functions to exist within)
include("src/brfss_preprocessing.jl")
include("src/impute.jl")
include("src/models_iai.jl")
include("src/sparse_log_reg.jl")

function main()
    parser = build_parser()
    args = parse_args(parser)
    println("Args: ", args)

    # Set seed if Random is used inside modules
    try
        using Random
        Random.seed!(args["seed"])
    catch e
        @warn "Random not available or seed not set: " * string(e)
    end

    stage = args["stage"]
    if stage == "preprocess"
        # TODO: Replace with the actual function from brfss_preprocessing.jl
        if args["input"] == "" || args["output"] == ""
            error("For preprocess, please provide --input and --output")
        end
        println("Running preprocessing on ", args["input"], " -> ", args["output"])
        # e.g., clean_brfss(args["input"], args["output"])

    elseif stage == "impute"
        println("Running imputation…")
        # TODO: call imputation entrypoint
        # e.g., run_imputation(args["input"], args["output"], args["config"])

    elseif stage == "train"
        println("Training models…")
        # TODO: call training entrypoint from models_iai.jl or sparse_log_reg.jl
        # e.g., train_models(args["config"])

    elseif stage == "evaluate"
        println("Evaluating models…")
        # TODO: call evaluation entrypoint
        # e.g., evaluate_models(args["config"])

    else
        error("Unknown stage: " * stage)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end