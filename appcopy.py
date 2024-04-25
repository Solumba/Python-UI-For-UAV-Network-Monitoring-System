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
