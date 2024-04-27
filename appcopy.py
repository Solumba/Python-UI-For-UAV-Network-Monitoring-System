from flask import Flask, render_template, jsonify
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
client = MongoClient("mongodb://localhost:27017/")  # adjust the connection string as needed
db = client['ddos_simulation']  # your database name
collection = db['metrics']  # your collection name

def fetch_data():
    # Fetch data for node 0 and node 5
    filter_query = {'node': {'$in': [0, 5]}}
    data = pd.DataFrame(list(collection.find(filter_query, {'_id': 0, 'node': 1, 'time': 1, 'latency': 1, 'throughput': 1, 'battery': 1})))
    print(data.head())  # Print the first few rows to inspect the DataFrame structure
    return data

def plot_latency (data):
    # Filter data for node 0 and node 5
    data_node0 = data[data['node'] == 0]
    data_node5 = data[data['node'] == 5]
    
    # Create Bokeh plot
    plot = figure(title="Latency over Time by Node", x_axis_label='Time (s)', y_axis_label='Latency (ms)', sizing_mode="scale_width")
    source0 = ColumnDataSource(data_node0)
    source5 = ColumnDataSource(data_node5)
    
    plot.line('time', 'latency', source=source0, line_width=2, color='blue', legend_label='Node 0')
    plot.line('time', 'latency', source=source5, line_width=2, color='red', legend_label='Node 5')
    
    plot.legend.title = 'Node'
    plot.legend.location = 'top_left'
    
    script, div = components(plot)
    return script, div


def plot_throughput (data):
    # Filter data for node 0 and node 5
    data_node0 = data[data['node'] == 0]
    data_node5 = data[data['node'] == 5]
    
    # Create Bokeh plot
    plot = figure(title="Throughput over Time by Node", x_axis_label='Time (s)', y_axis_label='Throughput (mbps)', sizing_mode="scale_width")
    source0 = ColumnDataSource(data_node0)
    source5 = ColumnDataSource(data_node5)
    
    plot.line('time', 'throughput', source=source0, line_width=2, color='green', legend_label='Node 0')
    plot.line('time', 'throughput', source=source5, line_width=2, color='yellow', legend_label='Node 5')
    
    plot.legend.title = 'Node'
    plot.legend.location = 'top_left'
    
    script, div = components(plot)
    return script, div

def plot_battery (data):
    # Filter data for node 0 and node 5
    data_node0 = data[data['node'] == 0]
    data_node5 = data[data['node'] == 5]
    
    # Create Bokeh plot
    plot = figure(title="Battery over Time by Node", x_axis_label='Time (s)', y_axis_label='Battery (%)', sizing_mode="scale_width")
    source0 = ColumnDataSource(data_node0)
    source5 = ColumnDataSource(data_node5)
    
    plot.line('time', 'battery', source=source0, line_width=2, color='black', legend_label='Node 0')
    plot.line('time', 'battery', source=source5, line_width=2, color='red', legend_label='Node 5')
    
    plot.legend.title = 'Node'
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
                    'label': list(n.labels)[0] if n.labels else 'None'  # Assume single label for simplicity
                }
            if record['m']:
                m = record['m']
                m_id = m.id
                if m_id not in nodes:
                    # Include desired properties from the node 'm'
                    nodes[m_id] = {
                        'id': m_id,
                        'name': m.get('name', ''),
                        'label': list(m.labels)[0] if m.labels else 'None'
                    }
                # Handle case when relationship 'r' may be None
                links.append({
                    "source": n_id,
                    "target": m_id,
                    "type": record['r'].type if record['r'] else 'undefined'
                })

        graph = {"nodes": list(nodes.values()), "links": links}
        return jsonify(graph)



if __name__ == "__main__":
    app.run(debug=True, port=5002)
