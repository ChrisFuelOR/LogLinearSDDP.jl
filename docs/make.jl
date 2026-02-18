import Literate
import Documenter
import LogLinearSDDP

const INPUT_DIR = joinpath(@__DIR__, "src")
const OUTPUT_DIR = joinpath(@__DIR__, "output")

input_file = "introduction.jl"
# output_dir = ""


_sorted_files(dir, ext) = sort(filter(f -> endswith(f, ext), readdir(dir)))

function list_of_sorted_files(prefix, dir, ext = ".md")
    return Any["$(prefix)/$(file)" for file in _sorted_files(dir, ext)]
end

function _link_example(content, filename)
    title_line = findfirst(r"\n# .+?\n", content)
    line = content[title_line]
    ipynb = filename[1:end-3] * ".ipynb"
    new_title = string(
        line,
        "\n",
        "_This tutorial was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl)._\n",
        "[_Download the source as a `.jl` file_]($filename).\n",
        "[_Download the source as a `.ipynb` file_]($ipynb).\n",
    )
    contennt = replace(content, "nothing #hide" => ""),
    return replace(content, line => new_title)
end


for filename_jl in list_of_sorted_files(INPUT_DIR, INPUT_DIR, ".jl")
    filename = replace(filename_jl, dirname(filename_jl) * "/" => "")

    Literate.markdown(
        filename_jl,
        OUTPUT_DIR;
        documenter = true,
        # postprocess = content -> _link_example(content, filename),
        # Turn off the footer. We manually add a modified one.
        # credit = false,
    )

    # Literate.notebook(jl_filename, dir; execute = false, credit = false)
end

# Literate.markdown(input_file, output_dir; flavor = Literate.DocumenterFlavor())

#Documenter.makedocs(
#    sitename = "My Project",
#    pages = [
#        "Home" => "introduction.md",
#    ]
#)