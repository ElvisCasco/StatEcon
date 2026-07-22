"""
    stata_label_list(label_dict; name="")

Stata `label list <name>`. Prints the integer ⇒ string mapping in the
two-column Stata format. `name` is the label-name header (e.g.
`"intmode"`); pass an empty string to suppress it.
"""
function stata_label_list(label_dict::AbstractDict; name::AbstractString = "")
    isempty(name) || println(name, ":")
    for k in sort(collect(keys(label_dict)))
        Printf.@printf("%12d %s\n", k, label_dict[k])
    end
    return nothing
end
