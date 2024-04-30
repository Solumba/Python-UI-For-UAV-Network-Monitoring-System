import sys
from flask import Flask, render_template, jsonify, request, redirect, url_for
from neo4j import GraphDatabase
from pymongo import MongoClient
import pandas as pd
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource
# import UAVNetworkSimulation class from utilities.py file


app = Flask(__name__)

# Neo4j connection setup
driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "Shady5000$"))


app = Flask(__name__)
client = MongoClient("mongodb://localhost:27017/")
db = client["ddos_simulation"]
collection = db["simulation_metrics"]
network_collection = db["network_metrics"]

# Appending the path of projectB to sys.path
path_to_add = "../UAV Monitoring System/"

if path_to_add not in sys.path:
    sys.path.append(path_to_add)

# Using imported functions
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "Shady5000$")

from utilities import UAVNetworkSimulation

# Simulation Parameters
num_uavs = 10
connection_range = 30
ground_station_pos = (50, 50)
backbone_range = 50
num_packets = 10

total_time = 60  # Total time to run the simulation in seconds
update_interval = 5  # Time interval between updates in seconds
move_range = 5
target_ids = [5, 7]

network_sim = UAVNetworkSimulation(URI, AUTH)
uav_network = network_sim.create_initial_graph(
    num_uavs, connection_range, ground_station_pos, backbone_range
)
attack_node_id = network_sim.add_attack_node(uav_network, target_ids, num_uavs)
run_ddos_attack = network_sim.run_ddos_simulation
run_network_simulation = network_sim.run_simulation


# Using instance methods


def fetch_data():
    # Fetch data for node 0 and node 5
    filter_query = {"node": {"$in": [0, 5]}}
    data = pd.DataFrame(
        list(
            collection.find(
                filter_query,
                {
                    "_id": 0,
                    "node": 1,
                    "time": 1,
                    "latency": 1,
                    "throughput": 1,
                    "battery": 1,
                },
            )
        )
    )
    print(data.head())  # Print the first few rows to inspect the DataFrame structure
    return data


def fetch_all_data():
    # Fetch all data
    data = pd.DataFrame(list(collection.find({})))
    print(data.head())  # Print the first few rows to inspect the DataFrame structure
    return data


def fetch_network_analysis_data():
    document = network_collection.find_one(
        {},
        {
            "_id": 0,
            "Number of UAVs": 1,
            "Number of Connections": 1,
            "Sparsity": 1,
            "Most Connected UAV": 1,
            "Most Central UAV": 1,
            "Most Critical UAV": 1,
            "Connectivity": 1,
        },
    )

    # Check if a document was found
    if document:
        # Convert the dictionary to a DataFrame
        data = pd.DataFrame([document])  # Wrap the dictionary in a list
    else:
        # Handle the case where no data was found
        print("No data found")
        data = pd.DataFrame()  # Create an empty DataFrame if no document is found

    print(data.head())  # Print the first few rows to inspect the DataFrame structure
    return data


def fetch_data_for_nodes(node_one, node_two):
    filter_query = {"node": {"$in": [node_one, node_two]}}
    data = pd.DataFrame(
        list(
            collection.find(
                filter_query,
                {
                    "_id": 0,
                    "node": 1,
                    "time": 1,
                    "latency": 1,
                    "throughput": 1,
                    "battery": 1,
                },
            )
        )
    )
    return data


def plot_latency(data):
    # Filter default data for UAV 0 and UAV 5
    data_node0 = data[data["node"] == 0]
    data_node5 = data[data["node"] == 5]

    # Create Bokeh plot
    plot = figure(
        title="Latency over Time by UAV",
        x_axis_label="Time (s)",
        y_axis_label="Latency (ms)",
        sizing_mode="scale_width",
    )
    source0 = ColumnDataSource(data_node0)
    source5 = ColumnDataSource(data_node5)

    plot.line(
        "time",
        "latency",
        source=source0,
        line_width=2,
        color="blue",
        legend_label="UAV 0",
    )
    plot.line(
        "time",
        "latency",
        source=source5,
        line_width=2,
        color="red",
        legend_label="UAV 5",
    )

    plot.legend.title = "UAVs"
    plot.legend.location = "top_left"

    script, div = components(plot)
    return script, div


def plot_latency_input(data, node_one, node_two):
    # Filter data for UAV 0 and UAV 5
    data_node_one = data[data["node"] == node_one]
    data_node_two = data[data["node"] == node_two]

    # Create Bokeh plot
    plot = figure(
        title="Latency over Time by UAV",
        x_axis_label="Time (s)",
        y_axis_label="Throughput (mbps)",
        sizing_mode="scale_width",
    )
    source_one = ColumnDataSource(data_node_one)
    source_two = ColumnDataSource(data_node_two)

    plot.line(
        "time",
        "latency",
        source=source_one,
        line_width=2,
        color="blue",
        legend_label=f"UAV {node_one}",
    )
    plot.line(
        "time",
        "latency",
        source=source_two,
        line_width=2,
        color="red",
        legend_label=f"UAV {node_two}",
    )

    plot.legend.title = "UAVs"
    plot.legend.location = "top_left"

    script, div = components(plot)
    return script, div


def plot_throughput(data):
    # Filter data for UAV 0 and UAV 5
    data_node0 = data[data["node"] == 0]
    data_node5 = data[data["node"] == 5]

    # Create Bokeh plot
    plot = figure(
        title="Throughput over Time by UAV",
        x_axis_label="Time (s)",
        y_axis_label="Throughput (mbps)",
        sizing_mode="scale_width",
    )
    source0 = ColumnDataSource(data_node0)
    source5 = ColumnDataSource(data_node5)

    plot.line(
        "time",
        "throughput",
        source=source0,
        line_width=2,
        color="green",
        legend_label="UAV 0",
    )
    plot.line(
        "time",
        "throughput",
        source=source5,
        line_width=2,
        color="yellow",
        legend_label="UAV 5",
    )

    plot.legend.title = "UAVs"
    plot.legend.location = "top_left"

    script, div = components(plot)
    return script, div


def plot_throughput_input(data, node_one, node_two):
    data_node_one = data[data["node"] == node_one]
    data_node_two = data[data["node"] == node_two]

    # Create Bokeh plot
    plot = figure(
        title="Throughput over Time by UAV",
        x_axis_label="Time (s)",
        y_axis_label="Throughput (mbps)",
        sizing_mode="scale_width",
    )
    source_one = ColumnDataSource(data_node_one)
    source_two = ColumnDataSource(data_node_two)

    plot.line(
        "time",
        "throughput",
        source=source_one,
        line_width=2,
        color="green",
        legend_label=f"UAV {node_one}",
    )
    plot.line(
        "time",
        "throughput",
        source=source_two,
        line_width=2,
        color="yellow",
        legend_label=f"UAV {node_two}",
    )

    plot.legend.title = "UAVs"
    plot.legend.location = "top_left"

    script, div = components(plot)
    return script, div


def plot_battery(data):
    # Filter data for node 0 and node 5
    data_node0 = data[data["node"] == 0]
    data_node5 = data[data["node"] == 5]

    # Create Bokeh plot
    plot = figure(
        title="Battery over Time by UAV",
        x_axis_label="Time (s)",
        y_axis_label="Battery (%)",
        sizing_mode="scale_width",
    )
    source0 = ColumnDataSource(data_node0)
    source5 = ColumnDataSource(data_node5)

    plot.line(
        "time",
        "battery",
        source=source0,
        line_width=2,
        color="black",
        legend_label="UAV 0",
    )
    plot.line(
        "time",
        "battery",
        source=source5,
        line_width=2,
        color="red",
        legend_label="UAV 5",
    )

    plot.legend.title = "UAVs"
    plot.legend.location = "top_left"

    script, div = components(plot)
    return script, div


def plot_battery_input(data, node_one, node_two):
    data_node_one = data[data["node"] == node_one]
    data_node_two = data[data["node"] == node_two]

    plot = figure(
        title=f"Battery over Time for UAV {node_one} & {node_two}",
        x_axis_label="Time (s)",
        y_axis_label="Battery (%)",
        sizing_mode="scale_width",
    )
    source_one = ColumnDataSource(data_node_one)
    source_two = ColumnDataSource(data_node_two)

    plot.line(
        "time",
        "battery",
        source=source_one,
        line_width=2,
        color="black",
        legend_label=f"UAV {node_one}",
    )
    plot.line(
        "time",
        "battery",
        source=source_two,
        line_width=2,
        color="red",
        legend_label=f"UAV {node_two}",
    )

    plot.legend.title = "UAVs"
    plot.legend.location = "top_left"

    script, div = components(plot)
    return script, div


@app.route("/")
def index():
    data = fetch_data()

    network_metric = fetch_network_analysis_data()
    print("Data retrieved:", network_metric)
    #getting stats data
    stats_data = fetch_all_data()
    # Calculate Descriptive Statistics
    stats = stats_data.describe().to_dict()
    # Specific Statistics for Display
    specific_stats = {
        "mean_latency": round(stats_data["latency"].mean(), 2),
        "mean_throughput": round(stats_data["throughput"].mean(), 2),
        "max_throughput": round(stats_data["throughput"].max(), 2),
        "min_battery": stats_data["battery"].min(),
    }

    latency_script, latency_div = plot_latency(data)
    throughput_script, throughput_div = plot_throughput(data)
    battery_script, battery_div = plot_battery(data)
    if not network_metric.empty:
        return render_template(
            "indexcopy.html",
            latency_script=latency_script,
            latency_div=latency_div,
            throughput_script=throughput_script,
            throughput_div=throughput_div,
            battery_script=battery_script,
            battery_div=battery_div,
            network_data=network_metric.to_dict(orient="records")[0],
            specific_stats=specific_stats,
        )
    else:
        return render_template("indexcopy.html", network_data=None)


@app.route("/graph")
def get_graph():
    with driver.session() as session:
        # Fetch all nodes and their properties, and relationships if any
        result = session.run("""
            MATCH (n)
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n, r, m
        """)
        nodes = {}
        links = []
        for record in result:
            n = record["n"]
            n_id = n.id
            if n_id not in nodes:
                # Include desired properties from the node 'n'
                nodes[n_id] = {
                    "id": n_id,
                    "name": n.get(
                        "name", ""
                    ),  # Default to empty string if 'name' not present
                    "label": list(n.labels)[0]
                    if n.labels
                    else "None",  # Assume single label for simplicity
                    "uavType": n.get(
                        "uavType", "Unknown"
                    ),  # Assuming 'm' can also have 'uavType'
                }
            if record["m"]:
                m = record["m"]
                m_id = m.id
                if m_id not in nodes:
                    # Include desired properties from the node 'm'
                    nodes[m_id] = {
                        "id": m_id,
                        "name": m.get("name", ""),
                        "label": list(m.labels)[0] if m.labels else "None",
                        "uavType": m.get(
                            "uavType", "Unknown"
                        ),  # Assuming 'm' can also have 'uavType'
                    }
                # Handle case when relationship 'r' may be None
                links.append(
                    {
                        "source": n_id,
                        "target": m_id,
                        "type": record["r"].type if record["r"] else "undefined",
                    }
                )

        graph = {"nodes": list(nodes.values()), "links": links}
        return jsonify(graph)


@app.route("/submit", methods=["POST"])
def plot_battery_data():
    try:
        node_one = request.form["first_node"]
        node_two = request.form["second_node"]
    except ValueError:
        return "Please enter valid node numbers", 400

    n1 = int(node_one)
    n2 = int(node_two)

    data = fetch_data_for_nodes(n1, n2)  # Fetch data for these nodes
    latency_script, latency_div = plot_latency_input(data, n1, n2)
    throughput_script, throughput_div = plot_throughput_input(data, n1, n2)
    battery_script, battery_div = plot_battery_input(data, n1, n2)
    return render_template(
        "indexcopy.html",
        latency_script=latency_script,
        latency_div=latency_div,
        throughput_script=throughput_script,
        throughput_div=throughput_div,
        battery_script=battery_script,
        battery_div=battery_div,
    )


@app.route("/simulations", methods=["POST"])
def run_normal_simulation():
    # Handle the button click here
    print("Button was clicked!")
    run_network_simulation(
        uav_network,
        total_time,
        update_interval,
        move_range,
        connection_range,
        backbone_range,
        num_packets,
    )
    # You can redirect or render a template after handling
    return redirect(url_for("index"))


@app.route("/simulations", methods=["POST"])
def handle_simulations():
    # Handle the button click here
    print("Button was clicked!")
    if "button_action" in request.form:
        button_value = request.form["button_action"]
        if button_value == "normalnetwork":
            run_network_simulation(
                uav_network,
                total_time,
                update_interval,
                move_range,
                connection_range,
                backbone_range,
                num_packets,
            )
            return redirect(url_for("indexcopy"))
        elif button_value == "ddossim":
            run_ddos_attack(
                uav_network,
                total_time,
                update_interval,
                move_range,
                connection_range,
                backbone_range,
                num_packets,
                target_ids,
            )
            return redirect(url_for("indexcopy"))
    return redirect(url_for("indexcopy"))


if __name__ == "__main__":
    app.run(debug=True, port=5002)
