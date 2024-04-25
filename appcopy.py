from flask import Flask, render_template, jsonify
from neo4j import GraphDatabase

app = Flask(__name__)

# Neo4j connection setup

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "Shady5000$"))

@app.route("/")
def index():
    return render_template('indexcopy.html')

@app.route("/graph")
def get_graph():
    with driver.session() as session:
        # Fetch all nodes and relationships (if any)
        result = session.run("MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m")
        nodes = {}
        links = []
        for record in result:
            n_id = record['n'].id
            if n_id not in nodes:
                nodes[n_id] = {'id': n_id}
            if record['m']:
                m_id = record['m'].id
                if m_id not in nodes:
                    nodes[m_id] = {'id': m_id}
                links.append({"source": n_id, "target": m_id, "type": record['r'].type if record['r'] else 'undefined'})

        graph = {"nodes": list(nodes.values()), "links": links}
        return jsonify(graph)


if __name__ == "__main__":
    app.run(debug=True, port=5002)
