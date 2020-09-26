import graph_tool

def graph_from_pandas_edgelist(df, source='source', target='target', weight='weight', directed=True, remove_empty=True):
    network = graph_tool.Graph(directed=directed)
    n_vertices = max(df[source].max(), df[target].max()) + 1
    vertex_list = network.add_vertex(n_vertices)
    
    if weight in df.columns:
        if remove_empty:
            df = df[df[weight] > 0]
        weight_prop = network.new_edge_property('double')
        network.add_edge_list(df[[source, target, weight]].values, eprops=[weight_prop])
        return network, weight_prop
    else:
        network.add_edge_list(df[[source, target]].values)
        return network