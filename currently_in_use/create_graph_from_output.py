"""
Authors: Laura Friel
Last updated: March 1st, 2021
"""
import re

'''
Used to recreate code of a randomly generated graph, so that you can repeat testing on the same graph multiple times in a row.
Enter the name of the file that contains the info for the graph you want to recreate, aswell as the name for the file that will
write the generated code
'''
def main():
    # take user input for filename (or use default) to open and read file. Additionally create file for the code to be generated
    filename = input("enter filename for graph info to be turned into code. Hit enter to use default file. Use .txt suffix:\n")
    if (filename == ''):
        filename = "last_generated_graph.txt"
    file = open("graph_code_generation\\" + filename,"r")
    if file.mode == 'r':
        contents = file.read()
    filename = filename.replace('.txt','')
    output = open("graph_code_generation\\" + filename + "_code.txt","w+")

    # split the data into raw data that can be more easily parsed
    raw = re.split("\{'weight': | 'visited': |,| |\n|\{|\}|\(|\)|\[|\]|:", contents)
    node_info = []
    edge_info = []
    in_edge_info = False
    # place the raw data into two categories, either edge information or node information
    for word in raw:
        if word != '': 
            if in_edge_info:
                edge_info.append(word)
            elif word == 'G.edges.data':
                in_edge_info = True
            else:
               node_info.append(word)
    node_info.pop(0) 

    # store node weight and visited information into seperate arrays
    # note that node id's are not stored, because each id is simply the index in the array
    node_weight = []
    node_visited = []

    for i in range(len(node_info)-1):
        if i%3==1:
            node_weight.append(node_info[i])
        elif i%3==2:
            node_visited.append(node_info[i])

    num_nodes = len(node_weight)

    print("node weight:", node_weight)
    print("node visited:", node_visited)
    print("num nodes:", num_nodes)

    # store edge id's and edge weight and data attributes into seperate arrays
    # since the data attribute is not universal, this code is slightly more complicated
    edge_tuple = []
    edge_weight = []
    edge_data = []
    
    data_tracker = 0 # data tracker used because 'data' attribuite not always present in graph. Used to fiddle with modulous operator
    for i in range(len(edge_info)-1):
        if data_tracker%5==0:
            edge_tuple.append((edge_info[i], edge_info[i+1]))
        elif data_tracker%5==2:
            edge_weight.append(edge_info[i])
        elif data_tracker%5==3:
            if edge_info[i] == "'data'":
                edge_data.append((len(edge_tuple)-1,edge_info[i+1])) #keep track of which tuples have data attribute associated with them
                data_tracker = -2
            else:
                data_tracker = 0
                edge_tuple.append((edge_info[i], edge_info[i+1]))
        data_tracker += 1

    print("edge tuple:", edge_tuple)
    print("edge weight:", edge_weight)
    print("edge date:", edge_data)

    # generate the actual code:
    output.write("\tG = nx.Graph()\n\n")
    # add nodes to G
    output.write("\tG.add_nodes_from(range(" + str(num_nodes) + "))\n\n")

    # add weight attribute to nodes in G
    for i, weight in enumerate(node_weight):
        output.write("\tG.nodes[" + str(i) + "]['weight'] = " + str(weight) + "\n")
    output.write("\n")

    # add visited attribute to nodes in G
    for i, visited in enumerate(node_visited):
        output.write("\tG.nodes[" + str(i) + "]['visited'] = " + str(visited) + "\n")
    output.write("\n")

    # add edges to G and weight attribute to each edge
    for i,edge in enumerate(edge_tuple):
        edge_str = "(" + str(edge[0]) + "," + str(edge[1]) + ")"
        output.write("\tG.add_edge" + edge_str +"\n")
        output.write("\tG.edges[" + edge_str +"]['weight'] = " + str(edge_weight[i]) + "\n")
    output.write("\n")

    # add data attribute to edges that have data
    for data in edge_data:
        edge = edge_tuple[data[0]]
        edge_str = "(" + str(edge[0]) + "," + str(edge[1]) + ")"
        output.write("\tG.edges[" + edge_str +"]['data'] = {" + str(data[1]) + "}\n")

    output.write("\treturn G")

    output.close()

    print("Node:",node_info)
    print("Edge:",edge_info)
    
if __name__ == "__main__":
    main()