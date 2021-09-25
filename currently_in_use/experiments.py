import driver

def main():
    node_sizes = ["500", "1000", "2000", "5000"] # ["150", "500", "1000", "2000"]
    possible_k = ["10","20","50","100"] # ["5","10","20","50"]
    possible_criticalities = ["0.5", "0.75"] # ["0.5", "0.5", "0.5", "0.5"]

    do_remove_cycles = "False"
    do_assumption_1 = "False"

    runs = 25

    for criticality in possible_criticalities:
        for num_nodes in node_sizes:
            for k in possible_k:
                for i in range(runs):
                    print("RUN " + str(i) + ": " + str(num_nodes) + " " + str(k) + " " + str(criticality))
                    driver.test_new_file(num_nodes, k, criticality, do_remove_cycles, do_assumption_1)

def test():
    graph_types = ["BA"]
    offset = [1]
    node_size = ["150"]
    do_remove_cycles = "False"
    do_assumption_1 = "False"

    for graph, ID_off in zip(graph_types, offset):
        for size_offset, size in enumerate(node_size):
            size_offset = 0
            for i in range(50):
                filename = graph + "/" + size + "/" + str((size_offset * 150) + (i * 3) + ID_off)
                # if filename[:3] != "BA/":
                driver.retest_old_file(filename, do_remove_cycles, do_assumption_1)

def practice():
    num_nodes = "500"
    k = "10"
    criticality = "0.5"
    do_remove_cycles = "False"
    do_assumption_1 = "False"

    for i in range(6):
        print("RUN " + str(i) + ": " + str(num_nodes) + " " + str(k) + " " + str(criticality))
        driver.test_new_file(num_nodes, k, criticality, do_remove_cycles, do_assumption_1)
def retest_all_ER_WS():
    graph_types = ["ER","WS"]
    offset = [2,3]
    node_size = ["150", "500", "1000", "2000"]
    do_remove_cycles = "False"
    do_assumption_1 = "False"

    for graph, ID_off in zip(graph_types, offset):
        for size_offset, size in enumerate(node_size):
            for i in range(50):
                filename = graph + "/" + size + "/" + str((size_offset * 150) + (i * 3) + ID_off)
                # if filename[:3] != "BA/":
                driver.retest_old_file(filename, do_remove_cycles, do_assumption_1)

def retest_all_files():
    graph_types = ["BA","ER","WS"]
    offset = [1,2,3]
    node_size = ["150", "500", "1000", "2000"]
    do_remove_cycles = "False"
    do_assumption_1 = "False"

    for graph, ID_off in zip(graph_types, offset):
        for size_offset, size in enumerate(node_size):
            for i in range(50):
                filename = graph + "/" + size + "/" + str((size_offset * 150) + (i * 3) + ID_off)
                # if filename[:3] != "BA/":
                driver.retest_old_file(filename, do_remove_cycles, do_assumption_1)

def retest_BA():
    graph_types = ["BA"]
    offset = [1]
    node_size = ["150", "500", "1000", "2000"]
    do_remove_cycles = "False"
    do_assumption_1 = "False"

    for graph, ID_off in zip(graph_types, offset):
        for size_offset, size in enumerate(node_size):
            for i in range(50):
                filename = graph + "/" + size + "/" + str((size_offset * 150) + (i * 3) + ID_off)
                # if filename[:3] != "BA/":
                driver.retest_old_file(filename, do_remove_cycles, do_assumption_1)

# practice()
main()