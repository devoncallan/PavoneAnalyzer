class PavoneKey:

    # Keys for raw Pavone data
    time = "Time (s)"
    load = "Load (uN)"
    indent = "Indentation (nm)"
    cantilever = "Cantilever (nm)"
    piezo = "Piezo (nm)"
    auxiliary = "Auxiliary"

    # Keys for processed Pavone data
    displacement = "Displacement (um)"
    force = "Force (uN)"
    force_gradient = "Force Gradient (uN/s)"


reduced_columns = [PavoneKey.time, PavoneKey.displacement, PavoneKey.force]
