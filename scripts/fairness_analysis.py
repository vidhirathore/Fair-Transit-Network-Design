import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import pickle
import numpy as np
import pulp
import warnings

warnings.filterwarnings("ignore", message="Signature .* for <class 'numpy.longdouble'> does not match any known type")

DATA_PATH = '../data/'
GRAPHS_DIR = '../visuals/graphs'

os.makedirs(GRAPHS_DIR, exist_ok=True)

GRAPH_FILE = os.path.join(GRAPHS_DIR, "osaka_transit_graph.pickle")

PARAMETERS = {
    'NUM_SCENARIOS': 10,
    'SUBSET_SIZE': 5,
    'ALPHA': 1.5,
    'GAMMA': 0.7,
    'BUDGET': 1000000,
    'BETA': 3,
    'EPSILON': 0.05,
    'ENABLE_EQUITY_CONSTRAINT': True,
    'ENABLE_BUDGET_CONSTRAINT': True,
    'USE_SLACK_VARIABLES': True,
    'FAIRNESS_CLASSES': ['school_children', 'people_no_private_transport']
}

HYPERPARAMETER_GRID = {
    'NUM_SCENARIOS': [1, 2, 3, 5, 10],
    'SUBSET_SIZE': [3, 5, 7],
    'BUDGET': [500000, 700000, 1000000],
    'BETA': [1, 2, 3]
}

MAX_EDGES = 50
MAX_OD_PAIRS_PER_SCENARIO = 15
MAX_FREQUENCY = 10000

MAX_ITERATIONS = 10

def load_transit_graph(graph_file):
    """
    Load the transit graph from a pickle file.

    Parameters
    ----------
    graph_file : str
        The file path to the pickle file containing the transit graph.

    Returns
    -------
    G : networkx.Graph
        The loaded transit graph.

    Raises
    ------
    FileNotFoundError
        If the specified graph_file does not exist.
    """
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)
    print("Transit graph loaded successfully.")
    return G

def generate_od_matrix(G, num_scenarios, subset_size):
    """
    Generate Origin-Destination (OD) matrices for multiple scenarios based on the transit graph.

    Parameters
    ----------
    G : networkx.Graph
        The transit graph.
    num_scenarios : int
        The number of scenarios for which to generate OD matrices.
    subset_size : int
        The number of stations to include in the subset for OD matrix generation.

    Returns
    -------
    stations_subset : list
        A list of station names included in the subset.
    od_matrices : list of pandas.DataFrame
        A list of OD matrices, one for each scenario. Each DataFrame has stations_subset as both its index and columns,
        with demand values as entries.

    Notes
    -----
    - The function prunes the graph to include only the top MAX_EDGES based on edge betweenness centrality.
    - For each scenario, it randomly selects up to MAX_OD_PAIRS_PER_SCENARIO origin-destination pairs and assigns demands.
    """
    edge_centrality = nx.edge_betweenness_centrality(G)
    sorted_edges = sorted(edge_centrality, key=edge_centrality.get, reverse=True)
    pruned_G = G.edge_subgraph(sorted_edges[:MAX_EDGES]).copy()
    stations_subset = list(pruned_G.nodes())[:subset_size]

    od_matrices = []
    for s in range(num_scenarios):
        od_df = pd.DataFrame(0, index=stations_subset, columns=stations_subset)
        potential_od_pairs = [(o, d) for o in stations_subset for d in stations_subset if o != d]
        selected_od_pairs = random.sample(potential_od_pairs, min(MAX_OD_PAIRS_PER_SCENARIO, len(potential_od_pairs)))
        active_demands = 0
        for origin, destination in selected_od_pairs:
            if origin != destination:
                origin_data = G.nodes[origin]
                destination_data = G.nodes[destination]
                base_demand = random.randint(1, 5)
                demand = base_demand + \
                         origin_data.get('school_children', 0) * 0.01 + \
                         origin_data.get('people_no_private_transport', 0) * 0.01
                demand = max(int(demand), 1)
                od_df.loc[origin, destination] = demand
                if demand > 0:
                    active_demands += 1
        od_matrices.append(od_df)
        print(f"Scenario {s+1}/{num_scenarios} OD matrix generated with {len(selected_od_pairs)} active demands.")
    return stations_subset, od_matrices

def gini_coefficient(data):
    """
    Calculate the Gini coefficient for a given dataset.

    Parameters
    ----------
    data : array-like
        The data for which to calculate the Gini coefficient.

    Returns
    -------
    float
        The Gini coefficient, a measure of inequality where 0 represents perfect equality and 1 represents maximal inequality.

    Notes
    -----
    - Handles negative values by shifting the data to be non-negative.
    - Adds a small constant to avoid division by zero.
    """
    array = np.array(data)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = array.astype(np.float64)
    array += 1e-10
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((2 * np.sum(index * array)) / (n * np.sum(array))) - (n + 1) / n

def define_milp_model_pulp(G, stations_subset, od_matrices, alpha, gamma, budget, beta, epsilon, enable_equity=True, enable_budget=True, use_slack=False):
    """
    Define a Mixed-Integer Linear Programming (MILP) model for optimizing the transit network.

    Parameters
    ----------
    G : networkx.Graph
        The transit graph.
    stations_subset : list
        A list of station names included in the subset for OD matrix generation.
    od_matrices : list of pandas.DataFrame
        A list of OD matrices for each scenario.
    alpha : float
        Parameter for the MILP model.
    gamma : float
        Parameter for the MILP model.
    budget : float
        The total budget available for installation and operation costs.
    beta : float
        Parameter controlling the equity constraint.
    epsilon : float
        Parameter for the MILP model.
    enable_equity : bool, optional
        Whether to include the equity constraint in the model, by default True.
    enable_budget : bool, optional
        Whether to include the budget constraint in the model, by default True.
    use_slack : bool, optional
        Whether to use slack variables for constraints, by default False.

    Returns
    -------
    prob : pulp.LpProblem
        The defined MILP problem.
    edge_vars : dict
        Dictionary of edge binary variables.
    frequency_vars : dict
        Dictionary of frequency continuous variables.
    flow_vars : dict
        Dictionary of flow continuous variables.
    None
        Placeholder for future use.
    equity_slack : pulp.LpVariable or None
        The equity slack variable if used, else None.

    Notes
    -----
    - Variables:
        - edge_vars: Binary variables indicating whether an edge is installed.
        - frequency_vars: Continuous variables indicating the frequency of service on each edge.
        - flow_vars: Continuous variables representing the flow for each OD pair across edges.
    - Constraints include budget, equity, flow conservation, and frequency limits.
    - Objective is to minimize the total cost, including penalties for slack variables if used.
    """
    prob = pulp.LpProblem("Fair_Transit_Design", pulp.LpMinimize)

    edge_vars = {}
    for u, v in G.edges():
        var_name = f"edge_{u}_{v}"
        edge_vars[(u, v)] = pulp.LpVariable(var_name, cat='Binary')

    frequency_vars = {}
    for u, v in G.edges():
        var_name = f"freq_{u}_{v}"
        frequency_vars[(u, v)] = pulp.LpVariable(var_name, lowBound=0, cat='Continuous')

    flow_vars = {}
    for s, od_df in enumerate(od_matrices):
        for origin in od_df.index:
            for destination in od_df.columns:
                if origin != destination and od_df.loc[origin, destination] > 0:
                    for u, v in G.edges():
                        var_name = f"flow_s{s}_o{origin}_d{destination}_e{u}_{v}"
                        flow_vars[(s, origin, destination, u, v)] = pulp.LpVariable(var_name, lowBound=0, cat='Continuous')

    budget_slack = None
    equity_slack = None
    if enable_budget and use_slack:
        budget_slack = pulp.LpVariable("budget_slack", lowBound=0, cat='Continuous')
    if enable_equity and use_slack:
        equity_slack = pulp.LpVariable("equity_slack", lowBound=0, cat='Continuous')

    for u, v in G.edges():
        if 'length' not in G[u][v]:
            G[u][v]['length'] = 1

    installation_cost = pulp.lpSum([G[u][v]['length'] * edge_vars[(u, v)] for u, v in G.edges()])
    operational_cost = pulp.lpSum([frequency_vars[(u, v)] for u, v in G.edges()])
    total_cost = installation_cost + operational_cost

    penalty_budget = 1000 if enable_budget and use_slack else 0
    penalty_equity = 1000 if enable_equity and use_slack else 0

    if enable_budget and use_slack and enable_equity and use_slack:
        prob += total_cost + penalty_budget * budget_slack + penalty_equity * equity_slack, "Total_Cost"
    elif enable_budget and use_slack:
        prob += total_cost + penalty_budget * budget_slack, "Total_Cost"
    elif enable_equity and use_slack:
        prob += total_cost + penalty_equity * equity_slack, "Total_Cost"
    else:
        prob += total_cost, "Total_Cost"

    if enable_budget:
        if use_slack:
            prob += (installation_cost + operational_cost) <= budget + budget_slack, "Budget_Constraint_With_Slack"
        else:
            prob += (installation_cost + operational_cost) <= budget, "Budget_Constraint"

    if enable_equity:
        sorted_stations = sorted(stations_subset, key=lambda x: G.nodes[x].get('people_no_private_transport', 0))
        num_top = max(1, len(sorted_stations) // 5)
        num_bottom = max(1, len(sorted_stations) // 5)
        bottom_20 = sorted_stations[:num_bottom]
        top_20 = sorted_stations[-num_top:]

        service_bottom = pulp.lpSum([frequency_vars[(u, v)] for u, v in G.edges() if u in bottom_20 or v in bottom_20])
        service_top = pulp.lpSum([frequency_vars[(u, v)] for u, v in G.edges() if u in top_20 or v in top_20])
        avg_service_top = service_top / max(num_top, 1)
        avg_service_bottom = service_bottom / max(num_bottom, 1)

        if use_slack and enable_equity:
            prob += (avg_service_top - avg_service_bottom) <= beta + equity_slack, "Equity_Constraint_With_Slack"
        else:
            prob += (avg_service_top - avg_service_bottom) <= beta, "Equity_Constraint"

    for u, v in G.edges():
        prob += frequency_vars[(u, v)] <= edge_vars[(u, v)] * MAX_FREQUENCY, f"Freq_Install_{u}_{v}"

    for s, od_df in enumerate(od_matrices):
        for origin in od_df.index:
            for destination in od_df.columns:
                if origin != destination and od_df.loc[origin, destination] > 0:
                    demand = od_df.loc[origin, destination]
                    for node in stations_subset:
                        inflow = pulp.lpSum([
                            flow_vars.get((s, origin, destination, u, node), 0)
                            for u, v_data in G.nodes(data=True)
                            if (s, origin, destination, u, node) in flow_vars
                        ])
                        outflow = pulp.lpSum([
                            flow_vars.get((s, origin, destination, node, v), 0)
                            for v, v_data in G.nodes(data=True)
                            if (s, origin, destination, node, v) in flow_vars
                        ])
                        if node == origin:
                            prob += (outflow - inflow == demand), f"Flow_Conservation_s{s}_o{origin}_d{destination}_node{node}"
                        elif node == destination:
                            prob += (inflow - outflow == demand), f"Flow_Conservation_s{s}_o{origin}_d{destination}_node{node}"
                        else:
                            prob += (inflow - outflow == 0), f"Flow_Conservation_s{s}_o{origin}_d{destination}_node{node}"

    for u, v in G.edges():
        total_flow = pulp.lpSum([
            flow_vars.get((s, origin, destination, u, v), 0)
            for s in range(len(od_matrices))
            for origin in od_matrices[s].index
            for destination in od_matrices[s].columns
            if (s, origin, destination, u, v) in flow_vars
        ])
        prob += total_flow <= frequency_vars[(u, v)], f"Flow_Link_Freq_Link_{u}_{v}"

    return prob, edge_vars, frequency_vars, flow_vars, None, equity_slack

def compute_baseline_metrics(G, stations_subset):
    """
    Compute baseline service levels and the corresponding Gini coefficient.

    Parameters
    ----------
    G : networkx.Graph
        The transit graph.
    stations_subset : list
        A list of station names included in the subset.

    Returns
    -------
    service_levels : dict
        A dictionary mapping each station to its baseline service level.
    gini : float
        The Gini coefficient of the baseline service levels.

    Notes
    -----
    - Baseline service levels are randomly generated for each station based on its index and random variation.
    """
    service_levels = {}
    for idx, station in enumerate(stations_subset):
        service_levels[station] = idx * 10 + random.randint(1, 5)

    gini = gini_coefficient(list(service_levels.values()))

    print(f"Baseline Gini Coefficient: {gini:.4f}")

    return service_levels, gini

def compute_optimized_metrics_pulp(prob, edge_vars, frequency_vars, G, stations_subset, equity_slack):
    """
    Extract and compute optimized service levels and Gini coefficient after solving the MILP model.

    Parameters
    ----------
    prob : pulp.LpProblem
        The solved MILP problem.
    edge_vars : dict
        Dictionary of edge binary variables.
    frequency_vars : dict
        Dictionary of frequency continuous variables.
    G : networkx.Graph
        The transit graph.
    stations_subset : list
        A list of station names included in the subset.
    equity_slack : pulp.LpVariable or None
        The equity slack variable if used, else None.

    Returns
    -------
    service_levels : dict
        A dictionary mapping each station to its optimized service level.
    gini : float
        The Gini coefficient of the optimized service levels.
    optimized_frequencies : dict
        A dictionary mapping each edge to its optimized frequency.
    equity_slack_value : float
        The value of the equity slack variable used in the optimization.

    Notes
    -----
    - Service levels are calculated as the sum of incoming and outgoing frequencies for each station.
    """
    optimized_frequencies = {}
    for (u, v), var in frequency_vars.items():
        optimized_frequencies[(u, v)] = var.varValue if var.varValue is not None else 0

    service_levels = {}
    for station in stations_subset:
        outgoing = sum([optimized_frequencies.get((station, v), 0) for v in G.successors(station) if (station, v) in optimized_frequencies])
        incoming = sum([optimized_frequencies.get((u, station), 0) for u in G.predecessors(station) if (u, station) in optimized_frequencies])
        service_levels[station] = outgoing + incoming

    gini = gini_coefficient(list(service_levels.values()))

    equity_slack_value = equity_slack.varValue if equity_slack else 0

    print(f"Optimized Gini Coefficient: {gini:.4f}")
    if equity_slack:
        print(f"Equity Slack Used: {equity_slack_value:.4f}")

    return service_levels, gini, optimized_frequencies, equity_slack_value

def plot_service_levels(baseline, optimized, stations_subset, save_path):
    """
    Plot and save a comparison of baseline and optimized service levels for each station.

    Parameters
    ----------
    baseline : dict
        Dictionary of baseline service levels per station.
    optimized : dict
        Dictionary of optimized service levels per station.
    stations_subset : list
        A list of station names included in the subset.
    save_path : str
        The file path where the plot image will be saved.

    Notes
    -----
    - The plot displays two bars per station: one for baseline and one for optimized service levels.
    - Service levels are annotated above each bar.
    """
    baseline_values = [baseline.get(station, 0) for station in stations_subset]
    optimized_values = [optimized.get(station, 0) for station in stations_subset]

    x = np.arange(len(stations_subset))
    width = 0.35

    plt.figure(figsize=(14, 8))
    bars1 = plt.bar(x - width/2, baseline_values, width, label='Baseline', color='skyblue')
    bars2 = plt.bar(x + width/2, optimized_values, width, label='Optimized', color='salmon')

    plt.xlabel('Stations', fontsize=14)
    plt.ylabel('Service Level (frequency)', fontsize=14)
    plt.title('Baseline vs Optimized Service Levels per Station', fontsize=16)
    plt.xticks(x, stations_subset, rotation=45, ha='right')
    plt.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Service levels comparison plot saved as '{save_path}'.")

def plot_gini_comparison(baseline_gini, optimized_gini, save_path):
    """
    Plot and save a comparison of baseline and optimized Gini coefficients.

    Parameters
    ----------
    baseline_gini : float
        The baseline Gini coefficient.
    optimized_gini : float
        The optimized Gini coefficient.
    save_path : str
        The file path where the plot image will be saved.

    Notes
    -----
    - The plot displays two bars: one for baseline and one for optimized Gini coefficients.
    - Gini values are annotated above each bar.
    """
    labels = ['Baseline', 'Optimized']
    gini_values = [baseline_gini, optimized_gini]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, gini_values, color=['blue', 'green'])
    plt.ylabel('Gini Coefficient', fontsize=14)
    plt.title('Gini Coefficient Comparison', fontsize=16)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=12)
    plt.ylim(0, max(gini_values) + 0.1)
    plt.savefig(save_path)
    plt.close()
    print(f"Gini coefficient comparison plot saved as '{save_path}'.")

def plot_optimized_network(G, optimized_frequencies, stations_subset, save_path):
    """
    Plot and save the optimized transit network with edge frequencies.

    Parameters
    ----------
    G : networkx.Graph
        The transit graph.
    optimized_frequencies : dict
        Dictionary mapping each edge to its optimized frequency.
    stations_subset : list
        A list of station names included in the subset.
    save_path : str
        The file path where the plot image will be saved.

    Notes
    -----
    - Nodes are positioned using a spring layout if not already specified.
    - Edge widths are scaled based on frequency values.
    - The plot is saved as a high-resolution image.
    """
    if not all('pos' in data for node, data in G.nodes(data=True)):
        pos = nx.spring_layout(G, seed=42)
        for node, p in pos.items():
            G.nodes[node]['pos'] = p
    else:
        pos = {node: (data['pos'][0], data['pos'][1]) for node, data in G.nodes(data=True)}

    plt.figure(figsize=(35, 35))
    ax = plt.gca()

    frequencies = [optimized_frequencies.get((u, v), 0) for u, v in G.edges()]
    if frequencies and max(frequencies) > 0:
        max_freq = max(frequencies)
        min_freq = min(frequencies)
        if max_freq == min_freq:
            widths = [2 for _ in frequencies]
        else:
            widths = [2 + 4 * (freq - min_freq) / (max_freq - min_freq) for freq in frequencies]
    else:
        widths = [2 for _ in frequencies]

    nx.draw_networkx_edges(G, pos, edge_color='black', alpha=0.7, width=widths, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=10)

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='black', alpha=0.9, ax=ax)

    plt.title("Optimized Osaka Transit Network with Edge Frequencies", fontsize=20)
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Optimized transit network plot saved as '{save_path}'.")

def main():
    """
    Main function to execute the transit network optimization workflow.

    Workflow:
    1. Load the transit graph.
    2. Convert the graph to a directed graph.
    3. Initialize parameters for the optimization.
    4. Iterate up to MAX_ITERATIONS to find a feasible solution:
        a. Generate OD matrices for the current scenario.
        b. Compute baseline metrics.
        c. Define and solve the MILP model.
        d. If infeasible, adjust parameters and retry.
    5. If a feasible solution is found:
        a. Compute optimized metrics.
        b. Generate and save comparison plots.
        c. Print a summary of metrics.
    6. If no feasible solution is found after maximum iterations, notify the user.

    Notes
    -----
    - The function uses PuLP's CBC solver with a time limit of 600 seconds per iteration.
    - Adjusts the BETA and BUDGET parameters if the model is infeasible.
    """
    G = load_transit_graph(GRAPH_FILE)
    G = G.to_directed()

    num_scenarios = PARAMETERS['NUM_SCENARIOS']
    subset_size = PARAMETERS['SUBSET_SIZE']
    budget = PARAMETERS['BUDGET']
    beta = PARAMETERS['BETA']

    iteration = 0
    feasible = False

    while iteration < MAX_ITERATIONS and not feasible:
        print(f"\n--- Iteration {iteration + 1} ---")
        print(f"Testing with NUM_SCENARIOS={num_scenarios}, SUBSET_SIZE={subset_size}, BUDGET={budget}, BETA={beta}")

        stations_subset, od_matrices = generate_od_matrix(G, num_scenarios, subset_size)

        print("\nComputing baseline metrics...")
        baseline_service_levels, baseline_gini = compute_baseline_metrics(G, stations_subset)

        print("\nDefining the MILP model...")
        prob, edge_vars, frequency_vars, flow_vars, _, equity_slack = define_milp_model_pulp(
            G, stations_subset, od_matrices, PARAMETERS['ALPHA'], PARAMETERS['GAMMA'],
            budget, beta, PARAMETERS['EPSILON'],
            enable_equity=PARAMETERS['ENABLE_EQUITY_CONSTRAINT'],
            enable_budget=PARAMETERS['ENABLE_BUDGET_CONSTRAINT'],
            use_slack=PARAMETERS['USE_SLACK_VARIABLES']
        )

        print("Solving the MILP model with CBC solver...")
        prob_status = prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=600))

        if pulp.LpStatus[prob_status] == 'Optimal':
            print("\nOptimization was successful.")
            feasible = True
        elif pulp.LpStatus[prob_status] in ['Infeasible', 'Undefined']:
            print("Optimization was not successful. Model is infeasible or undefined.")
            print("Attempting to relax constraints...")
            if beta > 1:
                beta -= 1
                print(f"Decreasing BETA to {beta}.")
            elif budget < 2000000:
                budget += 500000
                print(f"Increasing BUDGET to {budget}.")
            else:
                print("Maximum adjustments reached. Exiting.")
                break
        else:
            print(f"Optimization was not successful. Status: {pulp.LpStatus[prob_status]}")
            break

        iteration += 1

    if feasible:
        print("\nComputing optimized metrics...")
        optimized_service_levels, optimized_gini, optimized_frequencies, equity_slack_used = compute_optimized_metrics_pulp(
            prob, edge_vars, frequency_vars, G, stations_subset, equity_slack
        )

        print("\nPlotting service levels comparison...")
        service_levels_path = os.path.join(GRAPHS_DIR, "service_levels_comparison.png")
        plot_service_levels(baseline_service_levels, optimized_service_levels, stations_subset, service_levels_path)

        print("\nPlotting Gini coefficient comparison...")
        gini_comparison_path = os.path.join(GRAPHS_DIR, "gini_comparison.png")
        plot_gini_comparison(baseline_gini, optimized_gini, gini_comparison_path)

        print("\nPlotting optimized transit network...")
        optimized_network_path = os.path.join(GRAPHS_DIR, "optimized_transit_network.png")
        plot_optimized_network(G, optimized_frequencies, stations_subset, optimized_network_path)

        print("\n--- Metrics Summary ---")
        print(f"Baseline Gini Coefficient: {baseline_gini:.4f}")
        print(f"Optimized Gini Coefficient: {optimized_gini:.4f}")
        improvement = baseline_gini - optimized_gini
        print(f"Improvement in Gini Coefficient: {improvement:.4f}")
        if equity_slack_used > 0:
            print(f"Equity Slack Utilized: {equity_slack_used:.4f}")
    else:
        print("\nFailed to find a feasible solution after multiple iterations.")

if __name__ == "__main__":
    main()
