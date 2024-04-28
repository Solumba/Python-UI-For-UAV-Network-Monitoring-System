import sys
from flask import Flask, render_template, jsonify, request
from neo4j import GraphDatabase
from pymongo import MongoClient
import pandas as pd
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource

app = Flask(__name__)

# Neo4j connection setup
driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "Shady5000$"))


app = Flask(__name__)
client = MongoClient("mongodb://localhost:27017/")
db = client['ddos_simulation']
collection = db['metrics']

# Appending the path of projectB to sys.path
path_to_add = '../UAV Monitoring System/'

if path_to_add not in sys.path:
    sys.path.append(path_to_add)

#import UAVNetworkSimulation class from utilities.py file
from utilities import UAVNetworkSimulation

#Using imported functions
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "Shady5000$")

network_sim = UAVNetworkSimulation(URI,AUTH)

# Using instance methods

def fetch_data():
    # Fetch data for node 0 and node 5
    filter_query = {'node': {'$in': [0, 5]}}
    data = pd.DataFrame(list(collection.find(filter_query, {'_id': 0, 'node': 1, 'time': 1, 'latency': 1, 'throughput': 1, 'battery': 1})))
    print(data.head())  # Print the first few rows to inspect the DataFrame structure
    return data

def fetch_data_for_nodes(node_one, node_two):
    filter_query = {'node': {'$in': [node_one, node_two]}}
    data = pd.DataFrame(list(collection.find(filter_query, {'_id': 0, 'node': 1, 'time': 1, 'latency': 1, 'throughput': 1, 'battery': 1})))
    return data

def plot_latency (data):
    # Filter default data for UAV 0 and UAV 5
    data_node0 = data[data['node'] == 0]
    data_node5 = data[data['node'] == 5]
    
    # Create Bokeh plot
    plot = figure(title="Latency over Time by UAV", x_axis_label='Time (s)', y_axis_label='Latency (ms)', sizing_mode="scale_width")
    source0 = ColumnDataSource(data_node0)
    source5 = ColumnDataSource(data_node5)
    
    plot.line('time', 'latency', source=source0, line_width=2, color='blue', legend_label='UAV 0')
    plot.line('time', 'latency', source=source5, line_width=2, color='red', legend_label='UAV 5')
    
    plot.legend.title = 'UAVs'
    plot.legend.location = 'top_left'
    
    script, div = components(plot)
    return script, div

def plot_latency_input (data, node_one, node_two):
    # Filter data for UAV 0 and UAV 5
    data_node_one = data[data['node'] == node_one]
    data_node_two = data[data['node'] == node_two]
    
    # Create Bokeh plot
    plot = figure(title="Throughput over Time by UAV", x_axis_label='Time (s)', y_axis_label='Throughput (mbps)', sizing_mode="scale_width")
    source_one = ColumnDataSource(data_node_one)
    source_two = ColumnDataSource(data_node_two)
    
    plot.line('time', 'latency', source=source_one, line_width=2, color='blue', legend_label=f'UAV {node_one}')
    plot.line('time', 'latency', source=source_two, line_width=2, color='red', legend_label=f'UAV {node_two}')
    
    plot.legend.title = 'UAVs'
    plot.legend.location = 'top_left'
    
    script, div = components(plot)
    return script, div

def plot_throughput (data):
    # Filter data for UAV 0 and UAV 5
    data_node0 = data[data['node'] == 0]
    data_node5 = data[data['node'] == 5]
    
    # Create Bokeh plot
    plot = figure(title="Throughput over Time by UAV", x_axis_label='Time (s)', y_axis_label='Throughput (mbps)', sizing_mode="scale_width")
    source0 = ColumnDataSource(data_node0)
    source5 = ColumnDataSource(data_node5)
    
    plot.line('time', 'throughput', source=source0, line_width=2, color='green', legend_label='UAV 0')
    plot.line('time', 'throughput', source=source5, line_width=2, color='yellow', legend_label='UAV 5')
    
    plot.legend.title = 'UAVs'
    plot.legend.location = 'top_left'
    
    script, div = components(plot)
    return script, div

def plot_throughput_input (data, node_one, node_two):
    data_node_one = data[data['node'] == node_one]
    data_node_two = data[data['node'] == node_two]
    
    # Create Bokeh plot
    plot = figure(title="Throughput over Time by UAV", x_axis_label='Time (s)', y_axis_label='Throughput (mbps)', sizing_mode="scale_width")
    source_one = ColumnDataSource(data_node_one)
    source_two = ColumnDataSource(data_node_two)
    
    plot.line('time', 'throughput', source=source_one, line_width=2, color='green', legend_label=f'UAV {node_one}')
    plot.line('time', 'throughput', source=source_two, line_width=2, color='yellow', legend_label=f'UAV {node_two}')
    
    plot.legend.title = 'UAVs'
    plot.legend.location = 'top_left'
    
    script, div = components(plot)
    return script, div

def plot_battery (data):
    # Filter data for node 0 and node 5
    data_node0 = data[data['node'] == 0]
    data_node5 = data[data['node'] == 5]
    
    # Create Bokeh plot
    plot = figure(title="Battery over Time by UAV", x_axis_label='Time (s)', y_axis_label='Battery (%)', sizing_mode="scale_width")
    source0 = ColumnDataSource(data_node0)
    source5 = ColumnDataSource(data_node5)
    
    plot.line('time', 'battery', source=source0, line_width=2, color='black', legend_label='UAV 0')
    plot.line('time', 'battery', source=source5, line_width=2, color='red', legend_label='UAV 5')
    
    plot.legend.title = 'UAVs'
    plot.legend.location = 'top_left'
    
    script, div = components(plot)
    return script, div

def plot_battery_input(data, node_one, node_two):
    data_node_one = data[data['node'] == node_one]
    data_node_two = data[data['node'] == node_two]
    
    plot = figure(title=f"Battery over Time for UAV {node_one} & {node_two}",
                  x_axis_label='Time (s)', y_axis_label='Battery (%)',
                  sizing_mode="scale_width")
    source_one = ColumnDataSource(data_node_one)
    source_two = ColumnDataSource(data_node_two)
    
    plot.line('time', 'battery', source=source_one, line_width=2, color='black', legend_label=f'UAV {node_one}')
    plot.line('time', 'battery', source=source_two, line_width=2, color='red', legend_label=f'UAV {node_two}')
    
    plot.legend.title = 'UAVs'
    plot.legend.location = 'top_left'
    
    script, div = components(plot)
    return script, div




@app.route("/")
def index():
    data = fetch_data()
    latency_script, latency_div = plot_latency (data)
    throughput_script, throughput_div = plot_throughput (data)
    battery_script, battery_div = plot_battery (data)
    return render_template('indexcopy.html',  latency_script=latency_script, latency_div=latency_div, throughput_script=throughput_script, throughput_div=throughput_div, battery_script=battery_script, battery_div=battery_div)

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
            n = record['n']
            n_id = n.id
            if n_id not in nodes:
                # Include desired properties from the node 'n'
                nodes[n_id] = {
                    'id': n_id,
                    'name': n.get('name', ''),  # Default to empty string if 'name' not present
                    'label': list(n.labels)[0] if n.labels else 'None', # Assume single label for simplicity
                    'uavType': n.get('uavType', 'Unknown')  # Assuming 'm' can also have 'uavType'
                }
            if record['m']:
                m = record['m']
                m_id = m.id
                if m_id not in nodes:
                    # Include desired properties from the node 'm'
                    nodes[m_id] = {
                        'id': m_id,
                        'name': m.get('name', ''),
                        'label': list(m.labels)[0] if m.labels else 'None',
                        'uavType': m.get('uavType', 'Unknown')  # Assuming 'm' can also have 'uavType'
                    }
                # Handle case when relationship 'r' may be None
                links.append({
                    "source": n_id,
                    "target": m_id,
                    "type": record['r'].type if record['r'] else 'undefined'
                })

        graph = {"nodes": list(nodes.values()), "links": links}
        return jsonify(graph)


@app.route('/submit', methods=['POST'])
def plot_battery_data():
    try:
        node_one = request.form['first_node']
        node_two = request.form['second_node']
    except ValueError:
        return "Please enter valid node numbers", 400
    
    n1 = int(node_one)
    n2 = int(node_two)

    data = fetch_data_for_nodes(n1, n2)  # Fetch data for these nodes
    latency_script, latency_div = plot_latency_input (data, n1, n2)
    throughput_script, throughput_div = plot_throughput_input (data, n1, n2)
    battery_script, battery_div = plot_battery_input(data, n1, n2)
    return render_template('indexcopy.html', latency_script=latency_script, latency_div=latency_div, throughput_script=throughput_script, throughput_div=throughput_div,  battery_script=battery_script, battery_div=battery_div)



if __name__ == "__main__":
    app.run(debug=True, port=5002)
