import pandapower as pp
import pandapower.topology as top
import networkx as nx
import random
from gridfm_datakit.network import load_net_from_file, load_net_from_pglib
import random

import numpy as np
import pandas as pd
import argparse
import os


NUM_SUBGRAPHS = 1000
MAX_LENGTH = 1000
MIN_LENGTH = 100
MH = True

def random_walk_step(G, current_node):
    """Perform one step of a normal random walk on graph G."""
    neighbors = list(G.neighbors(current_node))
    if not neighbors:
        return current_node  # No move if no neighbors
    return random.choice(neighbors)

def mh_random_walk_step(G, current_node):
    """Perform one step of a Metropolis-Hastings random walk on graph G."""
    neighbors = list(G.neighbors(current_node))
    if not neighbors:
        return current_node  # No move if no neighbors
    
    candidate = random.choice(neighbors)
    deg_current = G.degree[current_node]
    deg_candidate = G.degree[candidate]
    
    # Acceptance probability min(1, degree_current / degree_candidate)
    acceptance_prob = deg_current / deg_candidate
    
    if random.random() < acceptance_prob:
        return candidate
    else:
        return current_node
    
def generate_walk(G, net):
    buses = list(G.nodes())

    # Set of Slack bus indices
    slack_buses = set(net.ext_grid['bus'].values)

    # Start from a non-random Slack bus
    while True:
        start_node = random.choice(buses)
        if start_node not in slack_buses:
            break
    
    # Perform the random walk
    walk = [start_node]
    visited = {start_node}
    current_node = start_node

    length = random.randint(MIN_LENGTH, MAX_LENGTH)
    while len(walk) < length:
        if MH:
            current_node = mh_random_walk_step(G, current_node)
        else:
            current_node = random_walk_step(G, current_node)
        if current_node not in visited:
            walk.append(current_node)
            visited.add(current_node)
    if slack_buses & set(walk):
        print("SAMPLE DISCARDED: slack bus appears in the random walk")
        return None
    return walk

def create_pp_network(walk, net):
    # Create an empty pandapower network
    sub_net = pp.create_empty_network(sn_mva=net.sn_mva)

    # Copy only the buses in the walk
    node_mapping = {}
    for bus in walk:
        bus_data = net.bus.loc[bus]
        new_bus = pp.create_bus(sub_net, vn_kv=bus_data.vn_kv, name=bus_data.name, min_vm_pu=bus_data.min_vm_pu, max_vm_pu=bus_data.max_vm_pu)
        node_mapping[bus] = new_bus

    elements = ['load', 'sgen', 'gen', 'storage', 'shunt']

    for el in elements:
        for idx, row in net[el].iterrows():
            if row['bus'] in node_mapping:
                data = row.drop(labels=['name'], errors='ignore').to_dict()
                data['bus'] = node_mapping[row['bus']]
                new_idx = getattr(pp, f"create_{el}")(sub_net, **data)

                if el in ['gen', 'sgen']:
                    cost_row = net.poly_cost[(net.poly_cost['element'] == idx) & (net.poly_cost['et'] == el)]
                    if not cost_row.empty:
                        for _, cost in cost_row.iterrows():
                            cost_data = cost.drop(labels=["element", "et"]).to_dict()
                            pp.create_poly_cost(
                                sub_net,
                                element=new_idx,
                                et=el,
                                **cost_data
                            )
    for _, row in net.line.iterrows():
        if row.from_bus in node_mapping and row.to_bus in node_mapping:
            line_data = row.drop(labels=["from_bus", "to_bus", "std_type"]).to_dict()
            pp.create_line_from_parameters(
                sub_net,
                from_bus=node_mapping[row.from_bus],
                to_bus=node_mapping[row.to_bus],
                **line_data
            )
    
    for _, row in net.trafo.iterrows():
        if row.hv_bus in node_mapping and row.lv_bus in node_mapping:
            trafo_data = row.drop(labels=["hv_bus", "lv_bus"]).to_dict()
            pp.create_transformer_from_parameters(
                sub_net,
                hv_bus=node_mapping[row.hv_bus],
                lv_bus=node_mapping[row.lv_bus],
                **trafo_data
            )

    slack_bus = pp.create_bus(sub_net, vn_kv=net.bus.loc[walk[0]].vn_kv, name="VirtualSlack")
    pp.create_ext_grid(sub_net, bus=slack_bus, vm_pu=1.0, name="VirtualSlack")

    # Add border lines (edges between walk and outside)
    border_lines = []
    for idx, row in net.line.iterrows():
        in_walk = row.from_bus in walk
        out_walk = row.to_bus in walk
        if in_walk != out_walk:
            border_lines.append((idx, row))

    # For each border line, connect from inside to the new slack
    for idx, row in border_lines:
        if row.from_bus in node_mapping:
            from_bus = node_mapping[row.from_bus]
            to_bus = slack_bus
        else:
            from_bus = slack_bus
            to_bus = node_mapping[row.to_bus]

        line_data = row.drop(labels=["from_bus", "to_bus", "std_type"]).to_dict()

        pp.create_line_from_parameters(
            sub_net,
            from_bus=from_bus,
            to_bus=to_bus,
            **line_data
        )
    
    try:
        pp.runopp(sub_net)
        pp.runpp(sub_net)
    except Exception as e:
        print(f"ERROR OCCURRED DURING GENERATION: {e}")
        return None
    return sub_net

def main():
    parser = argparse.ArgumentParser(description="Generate subgraphs from a PGLIB grid")
    parser.add_argument('--grid_name', type=str, required=True, help="Name of the PGLIB grid to load")
    args = parser.parse_args()
    grid_name = args.grid_name
    print("Start generating subgraphs...")
    net = load_net_from_pglib(grid_name)

    # Create a networkx graph from the pandapower network
    G = top.create_nxgraph(net, include_switches=False, multi=False)
    
    output_dir = f"subnets_{grid_name}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(NUM_SUBGRAPHS):
        walk = generate_walk(G, net)

        if walk is None:
            continue

        sub_net = create_pp_network(walk, net)

        if sub_net is None:
            continue

        filename = f"{output_dir}/subnet_{grid_name}_{i}.json"
        pp.to_json(sub_net, filename)
        print(f"[{i}] New subnet of dim={len(walk)} saved to {filename}")




if __name__ == "__main__":
    main()