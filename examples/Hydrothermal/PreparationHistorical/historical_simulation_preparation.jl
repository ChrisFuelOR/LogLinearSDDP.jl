""" From 79 years of total data, construct 70 sample paths of 10 years each, i.e. [1:10], [2:11], ..., [70:79].
The data is stored in a txt file with 120 (10 years with 12 month) rows and 70 columns."""

function prepare_historical_simulation()

    # FILE PATH COMPONENTS
    directory_name = "historical_data"
    file_names = ["hist1.csv", "hist2.csv", "hist3.csv", "hist4.csv"]
    system_names = ["SE", "S", "NE", "N"]
    output_directory = "historical_simulation_data"

    for system_number in 1:4
        system_name = system_names[system_number]
        file_name = directory_name * "/" * file_names[system_number]
        output_file_name = output_directory * "/" * String(system_name) * ".txt"   
        
        # READING AND PREPARING THE DATA
        #######################################################################################
        # Get raw data data as dataframe (columns = months)
        df = read_raw_data(file_name)

        df_historical = DataFrames.DataFrame()

        for i in 1:70
            df_historical[!, Symbol(i)] = data_frame_to_vector(df[i:i+9,:])
        end

        # Output to file
        println("Model output for ", output_file_name)
        f = open(output_file_name, "w")
        for row in DataFrames.eachrow(df_historical)
            for i in 1:70
                print(f, row[Symbol(i)], ";")
            end
            println(f)
        end
        close(f)

    end

end