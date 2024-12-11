import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
import matplotlib.colors as mcolors
import pickle
from shapely import wkt
from shapely.geometry import LineString, MultiLineString
import numpy as np

random.seed(42)
np.random.seed(42)

DATA_PATH = '../data/'
GRAPHS_DIR = '../visuals/graphs'

if not os.path.exists(GRAPHS_DIR):
    os.makedirs(GRAPHS_DIR)

print("Loading data...")

cities = pd.read_csv(os.path.join(DATA_PATH, "cities.csv"))
stations = pd.read_csv(os.path.join(DATA_PATH, "stations.csv"))
tracks = pd.read_csv(os.path.join(DATA_PATH, "tracks.csv"))
lines = pd.read_csv(os.path.join(DATA_PATH, "lines.csv"))
track_lines = pd.read_csv(os.path.join(DATA_PATH, "track_lines.csv"))
station_lines = pd.read_csv(os.path.join(DATA_PATH, "station_lines.csv"))

print("Data loaded successfully.")

osaka_city_id = 91
stations_osaka = stations[stations['city_id'] == osaka_city_id].copy()
tracks_osaka = tracks[tracks['city_id'] == osaka_city_id].copy()
track_lines_osaka = track_lines[track_lines['city_id'] == osaka_city_id].copy()
lines_osaka = lines[lines['city_id'] == osaka_city_id].copy()
station_lines_osaka = station_lines[station_lines['city_id'] == osaka_city_id].copy()

print(f"Osaka has {len(stations_osaka)} stations and {len(tracks_osaka)} tracks.")

def parse_point(geometry_str):
    try:
        point = wkt.loads(geometry_str)
        return point.x, point.y
    except Exception as e:
        print(f"Error parsing station geometry '{geometry_str}': {e}")
        return None, None

stations_osaka[['Long', 'Lat']] = stations_osaka['geometry'].apply(
    lambda x: pd.Series(parse_point(x))
)

initial_count = len(stations_osaka)
stations_osaka = stations_osaka.dropna(subset=['Long', 'Lat'])
final_count = len(stations_osaka)
dropped = initial_count - final_count
if dropped > 0:
    print(f"Dropped {dropped} stations due to invalid/missing coordinates.")

stations_osaka = stations_osaka.rename(columns={'id': 'station_id', 'name': 'station_name'})

stations_osaka['school_children'] = stations_osaka['station_id'].apply(lambda x: random.randint(100, 1000))
stations_osaka['people_no_private_transport'] = stations_osaka['station_id'].apply(lambda x: random.randint(50, 500))

def parse_track_geometry(geometry_str):
    try:
        geometry_str = geometry_str.strip('"')
        geom = wkt.loads(geometry_str)
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            if len(coords) < 2:
                print(f"LineString has less than 2 points: {geometry_str}")
                return None, None, None, None
            start_lon, start_lat = coords[0]
            end_lon, end_lat = coords[-1]
            return start_lat, start_lon, end_lat, end_lon
        elif isinstance(geom, MultiLineString):
            geom = geom[0]
            coords = list(geom.coords)
            if len(coords) < 2:
                print(f"First LineString in MultiLineString has less than 2 points: {geometry_str}")
                return None, None, None, None
            start_lon, start_lat = coords[0]
            end_lon, end_lat = coords[-1]
            return start_lat, start_lon, end_lat, end_lon
        else:
            print(f"Unsupported geometry type: {type(geom)} in '{geometry_str}'")
            return None, None, None, None
    except Exception as e:
        print(f"Error parsing track geometry '{geometry_str}': {e}")
        return None, None, None, None

tracks_osaka[['start_lat', 'start_long', 'end_lat', 'end_long']] = tracks_osaka['geometry'].apply(
    lambda x: pd.Series(parse_track_geometry(x))
)

print("Sample parsed track coordinates:")
print(tracks_osaka[['start_lat', 'start_long', 'end_lat', 'end_long']].head(5))

initial_tracks = len(tracks_osaka)
tracks_osaka = tracks_osaka.dropna(subset=['start_lat', 'start_long', 'end_lat', 'end_long'])
final_tracks = len(tracks_osaka)
dropped_tracks = initial_tracks - final_tracks
if dropped_tracks > 0:
    print(f"Dropped {dropped_tracks} tracks due to invalid/missing geometries.")

station_coords = stations_osaka.set_index('station_id')[['Lat', 'Long']]

def find_nearest_station(lat, long, stations_df):
    try:
        distances = ((stations_df['Lat'] - lat) ** 2 + (stations_df['Long'] - long) ** 2) ** 0.5
        nearest_station_id = distances.idxmin()
        return nearest_station_id
    except Exception as e:
        print(f"Error finding nearest station for ({lat}, {long}): {e}")
        return None

tracks_osaka['start_station'] = tracks_osaka.apply(
    lambda row: find_nearest_station(row['start_lat'], row['start_long'], station_coords) 
    if pd.notnull(row['start_lat']) and pd.notnull(row['start_long']) else None, axis=1)

tracks_osaka['end_station'] = tracks_osaka.apply(
    lambda row: find_nearest_station(row['end_lat'], row['end_long'], station_coords) 
    if pd.notnull(row['end_lat']) and pd.notnull(row['end_long']) else None, axis=1)

tracks_osaka = tracks_osaka.dropna(subset=['start_station', 'end_station'])
tracks_osaka = tracks_osaka[tracks_osaka['start_station'] != tracks_osaka['end_station']]

edges = tracks_osaka[['start_station', 'end_station', 'length']].drop_duplicates()

print(f"Identified {len(edges)} edges connecting stations.")

if len(edges) > 0:
    print("Sample 'edges' DataFrame:")
    print(edges.head(5))
else:
    print("No edges identified. Please check the geometry parsing and station mapping.")
    exit()

G = nx.Graph()

for _, row in stations_osaka.iterrows():
    G.add_node(row['station_id'], 
               name=row['station_name'], 
               pos=(row['Long'], row['Lat']),
               school_children=row['school_children'],
               people_no_private_transport=row['people_no_private_transport'])

for _, row in edges.iterrows():
    congestion = random.randint(1, 10)
    G.add_edge(row['start_station'], row['end_station'], 
               length=row['length'], 
               congestion=congestion)

print("Graph constructed with nodes and edges.")

try:
    nx.write_gpickle(G, os.path.join(GRAPHS_DIR, "osaka_transit_graph.gpickle"))
    print("Graph data saved as 'osaka_transit_graph.gpickle'.")
except AttributeError:
    with open(os.path.join(GRAPHS_DIR, "osaka_transit_graph.pickle"), "wb") as f:
        pickle.dump(G, f)
    print("Graph data saved as 'osaka_transit_graph.pickle' using pickle as a fallback.")

def connect_components(G):
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        print("Graph is already fully connected.")
        return G

    print(f"Number of connected components before connecting: {len(components)}")

    while len(components) > 1:
        comp1 = components[0]
        comp2 = components[1]

        min_distance = float('inf')
        node_pair = (None, None)

        for node1 in comp1:
            lat1, lon1 = G.nodes[node1]['pos'][1], G.nodes[node1]['pos'][0]
            for node2 in comp2:
                lat2, lon2 = G.nodes[node2]['pos'][1], G.nodes[node2]['pos'][0]
                distance = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                if distance < min_distance:
                    min_distance = distance
                    node_pair = (node1, node2)

        if node_pair[0] is not None and node_pair[1] is not None:
            congestion = random.randint(1, 10)
            length = min_distance * 1000
            G.add_edge(node_pair[0], node_pair[1], length=length, congestion=congestion)
            print(f"Connected station {node_pair[0]} with station {node_pair[1]} (Distance: {min_distance:.4f})")
        else:
            print("Could not find a pair of nodes to connect.")
            break

        components = list(nx.connected_components(G))

    print(f"Number of connected components after connecting: {len(components)}")
    return G

print("Checking for isolated stations and connecting them...")
initial_components = list(nx.connected_components(G))
initial_num_components = len(initial_components)
print(f"Number of connected components before connecting: {initial_num_components}")

if initial_num_components > 1:
    G = connect_components(G)
    final_components = list(nx.connected_components(G))
    final_num_components = len(final_components)
    print(f"Number of connected components after connecting: {final_num_components}")
else:
    print("Graph is already fully connected.")

if len(edges) > 0:
    fig, ax = plt.subplots(figsize=(15, 15))
    pos = {node: (data['pos'][0], data['pos'][1]) for node, data in G.nodes(data=True)}
    
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_size=[data['school_children'] / 5 for _, data in G.nodes(data=True)], 
        node_color=[data['people_no_private_transport'] for _, data in G.nodes(data=True)],
        cmap='viridis', 
        alpha=0.8, 
        ax=ax
    )
    
    edges_data = G.edges(data=True)
    congestion_values = [edge_attr['congestion'] for _, _, edge_attr in edges_data]
    
    norm_congestion = mcolors.Normalize(vmin=min(congestion_values), vmax=max(congestion_values))
    cmap_congestion = plt.cm.Reds
    
    edge_colors = [cmap_congestion(norm_congestion(congestion)) for congestion in congestion_values]
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=G.edges(),
        width=1,
        alpha=0.7,
        edge_color=edge_colors,
        ax=ax
    )
    
    sm_congestion = plt.cm.ScalarMappable(cmap=cmap_congestion, norm=norm_congestion)
    sm_congestion.set_array(congestion_values)
    cbar_congestion = fig.colorbar(sm_congestion, ax=ax, shrink=0.5)
    cbar_congestion.set_label('Synthetic Congestion Level', rotation=270, labelpad=15)
    
    sm_transport = plt.cm.ScalarMappable(cmap='viridis', norm=mcolors.Normalize(
        vmin=stations_osaka['people_no_private_transport'].min(),
        vmax=stations_osaka['people_no_private_transport'].max()))
    sm_transport.set_array(stations_osaka['people_no_private_transport'])
    cbar_transport = fig.colorbar(sm_transport, ax=ax, shrink=0.5)
    cbar_transport.set_label('People Without Private Transport', rotation=270, labelpad=15)
    
    ax.set_title("Osaka Transit Network Graph with Synthetic Congestion and Demographics", fontsize=20)
    ax.axis('off')
    
    plt.savefig(os.path.join(GRAPHS_DIR, "osaka_transit_graph.png"), bbox_inches='tight')
    plt.close()
    print("Static graph visualization saved as 'osaka_transit_graph.png'.")
else:
    print("No edges to visualize in static plot.")

if len(edges) > 0:
    edge_x = []
    edge_y = []
    edge_congestion = []
    for node1, node2, edge_data in G.edges(data=True):
        start = G.nodes[node1]['pos']
        end = G.nodes[node2]['pos']
        edge_x += [start[0], end[0], None]
        edge_y += [start[1], end[1], None]
        edge_congestion.append(edge_data['congestion'])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    school_children = []
    people_no_private_transport = []
    node_text = []
    for node, data in G.nodes(data=True):
        node_x.append(data['pos'][0])
        node_y.append(data['pos'][1])
        school_children.append(data['school_children'])
        people_no_private_transport.append(data['people_no_private_transport'])
        node_text.append(f"Station: {data['name']}<br>"
                         f"School Children: {data['school_children']}<br>"
                         f"People Without Private Transport: {data['people_no_private_transport']}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=people_no_private_transport,
            size=[(children / 50) for children in school_children],
            colorbar=dict(
                thickness=15,
                title='People Without Private Transport',
                xanchor='left',
                titleside='right'
            ),
            line_width=2,
            sizemode='area',
            opacity=0.8
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Osaka Transit Network Graph with Synthetic Congestion and Demographics',
                    titlefont_size=20,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper") ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.write_html(os.path.join(GRAPHS_DIR, "osaka_transit_graph_interactive.html"))
    print("Interactive graph visualization saved as 'osaka_transit_graph_interactive.html'.")
else:
    print("No edges to visualize with Plotly.")
