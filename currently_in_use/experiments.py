import driver

def main():
    node_sizes = ["500", "1000", "2000"] # ["150", "500", "1000", "2000"]
    possible_k = ["10","20","50"] # ["5","10","20","50"]
    possible_criticalities = ["0.5", "0.5", "0.5"] # ["0.5", "0.5", "0.5", "0.5"]
    do_remove_cycles = "True"
    do_assumption_1 = "True"

    runs = 50

    for num_nodes, k, criticality in zip(node_sizes, possible_k, possible_criticalities):
        for i in range(runs):
            print("RUN " + str(i) + ": " + str(num_nodes) + " " + str(k) + " " + str(criticality))
            driver.test_new_file(num_nodes, k, criticality, do_remove_cycles, do_assumption_1)

def test():
    graph_types = ["BA"]
    offset = [1]
    node_size = ["2000"]
    do_remove_cycles = "False"
    do_assumption_1 = "False"

    for graph, ID_off in zip(graph_types, offset):
        for size_offset, size in enumerate(node_size):
            size_offset = 3
            for i in range(50):
                if i >= 9:
                    filename = graph + "/" + size + "/" + str((size_offset * 150) + (i * 3) + ID_off)
                    # if filename[:3] != "BA/":
                    driver.retest_old_file(filename, do_remove_cycles, do_assumption_1)

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

retest_all_ER_WS()