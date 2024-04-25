from flask import Flask, jsonify, render_template
from neo4j import GraphDatabase

app = Flask(__name__)
driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "Shady5000$"))

def get_graph_data():
    with driver.session() as session:
        result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")
        data = []
        for record in result:
            node1 = {"id": record["n"].id, "labels": list(record["n"].labels), "properties": dict(record["n"])}
            rel = {"id": record["r"].id, "type": record["r"].type, "properties": dict(record["r"])}
            node2 = {"id": record["m"].id, "labels": list(record["m"].labels), "properties": dict(record["m"])}
            data.append({"node1": node1, "relationship": rel, "node2": node2})
        return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph')
def graph():
    data = get_graph_data()
    # Transform data into a suitable format for visualization
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
