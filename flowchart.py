from simulate import wfirst_sim


def print_flowcart_dot_language():
    print("digraph {")
    cls = wfirst_sim
    print("subgraph cluster_{} {{".format("wfirst_sim"))
    print("style=filled;")
    print("color=lightgrey;")
    for out in cls.outputs.values():
        print('"{0}/{1}" [label="{1}"];'.format(cls.name, out.replace("{rank}", "*")))
    for out in cls.outputs.values():
        print('"{0}"->"{0}/{1}";'.format(cls.name, out.replace("{rank}", "*")))
    print('label = "{}"'.format(cls.name))
    print("}")
    for inp in cls.inputs.values():
        section, name = inp
        print('"{0}/{1}"->"{2}";'.format(section, name, cls.name))
    print("}")


if __name__ == "__main__":
    print_flowcart_dot_language()
