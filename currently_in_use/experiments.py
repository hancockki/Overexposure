import driver

def main():
    node_sizes = ["150", "500", "1000", "2000"] # ["150", "500", "1000", "2000"]
    possible_k = ["5","10","20","50"] # ["5","10","20","50"]
    possible_criticalities = ["0.5", "0.5", "0.5", "0.5"]
    do_remove_cycles = "True"
    do_assumption_1 = "True"

    runs = 50

    for num_nodes, k, criticality in zip(node_sizes, possible_k, possible_criticalities):
        for i in range(runs):
            print("RUN " + str(i) + ": " + str(num_nodes) + " " + str(k) + " " + str(criticality))
            driver.test_new_file(num_nodes, k, criticality, do_remove_cycles, do_assumption_1)

def test_BA_150():
    for i in range(302):
        if i % 3 == 1:
            filename = "BA/150/" + str(i)
            driver.retest_old_file(filename)

main()