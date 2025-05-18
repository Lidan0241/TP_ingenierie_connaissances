import uuid
import re
import networkx as nx
import ast
import networkx as nx
import community  # alias de python-louvain

# --- Classe Node ---
class Node:
    def __init__(self, name, node_class=None, properties=None):
        self.name = name
        self.node_class = node_class
        self.properties = properties or {}
        self.neighbors = set()
        self.id = self.generate_id()
        self.color = None #Couleur affectée par l'algo de coloriage

    def generate_id(self):
        base = f"{self.node_class or 'Any'}::{self.name}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

    def add_neighbor(self, neighbor_id):
        self.neighbors.add(neighbor_id)

# --- Classe Edge ---
class Edge:
    def __init__(self, source_id, target_id, edge_class=None, properties=None):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_class = edge_class
        self.properties = properties or {}
        self.id = self.generate_id()

    def generate_id(self):
        base = f"{self.edge_class or 'Any'}::{self.source_id}->{self.target_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

# --- Classe OneRingDB ---
class OneRingDB:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    # CRUD Node
    def create_node(self, name, node_class=None, properties=None):
        node = Node(name, node_class, properties)
        self.nodes[node.id] = node
        return node

    def get_node_by_id(self, node_id):
        return self.nodes.get(node_id)

    def find_nodes_by_name(self, name_prefix):
        return [node for node in self.nodes.values() if node.name.startswith(name_prefix)]

    def update_node(self, node_id, new_props):
        node = self.get_node_by_id(node_id)
        if node:
            node.properties.update(new_props)
            return True
        return False

    def delete_node(self, node_id):
        if node_id not in self.nodes:
            return False
        for eid in list(self.edges):
            edge = self.edges[eid]
            if edge.source_id == node_id or edge.target_id == node_id:
                self.delete_edge(eid)
        for n in self.nodes.values():
            n.neighbors.discard(node_id)
        del self.nodes[node_id]
        return True

    # CRUD Edge
    def create_edge(self, source_id, target_id, edge_class=None, properties=None):
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source or target node does not exist.")
        edge = Edge(source_id, target_id, edge_class, properties)
        self.edges[edge.id] = edge
        self.nodes[source_id].add_neighbor(target_id)
        self.nodes[target_id].add_neighbor(source_id)
        return edge

    def get_edge_by_id(self, edge_id):
        return self.edges.get(edge_id)

    def update_edge(self, edge_id, new_props):
        edge = self.get_edge_by_id(edge_id)
        if edge:
            edge.properties.update(new_props)
            return True
        return False

    def delete_edge(self, edge_id):
        edge = self.get_edge_by_id(edge_id)
        if not edge:
            return False
        self.nodes[edge.source_id].neighbors.discard(edge.target_id)
        self.nodes[edge.target_id].neighbors.discard(edge.source_id)
        del self.edges[edge_id]
        return True
    


    def to_networkx(self, directed=False):
        G = nx.DiGraph() if directed else nx.Graph()
        for node in self.nodes.values():
            G.add_node(node.id, label=node.name, class_=node.node_class)
        for edge in self.edges.values():
            G.add_edge(edge.source_id, edge.target_id, **edge.properties)
        return G


# --- ORQueryParser ---
class ORQueryParser:
    def parse(self, query):
        query = query.strip()

        if query.startswith("CREATE (:"):
            return self._parse_create_node(query)
        if query.startswith("CREATE ["):
            return self._parse_create_edge(query)
        if query.startswith("READ (:"):
            return self._parse_read_node(query)
        if query.startswith("UPDATE (:"):
            return self._parse_update_node(query)
        if query.startswith("DELETE (:"):
            return self._parse_delete_node(query)
        if query.startswith("LINK"):
            return self._parse_link_query(query)
        if query.startswith("COLOR"):
            return self._parse_color_query(query)
        if query.startswith("CLUSTER"):
            return self._parse_cluster_query(query)
        if query.startswith("TAG_HUBS"):
            return self._parse_tag_hubs(query)
        if query.startswith("CONNECTED_COMPONENTS"):
            return self._parse_components(query)




        raise ValueError(f"Unsupported ORQL query: {query}")

    def _extract_properties(self, props_str):
        props_str = props_str.strip()
        if not props_str or props_str == "{}":
            return {}
        try:
            # transforme key: value en "key": value
            json_like = re.sub(r'(\w+)\s*:', r'"\1":', props_str)
            return ast.literal_eval(json_like)
        except Exception as e:
            raise ValueError(f"Cannot parse properties: {props_str} → {e}")



    def _parse_create_node(self, query):
        # Utilise une regex qui capture {....} de manière robuste
        pattern = r'CREATE\s+\(:([A-Za-z_][\w]*)\s*(\{.*\})?\s*\)$'
        match = re.match(pattern, query.strip())
        if not match:
            raise ValueError("Invalid CREATE node syntax")
        node_class, props_str = match.groups()
        props = self._extract_properties(props_str or "{}")
        return {
            "action": "CREATE_NODE",
            "class": node_class,
            "properties": props
        }


    def _parse_create_edge(self, query):
        match = re.match(r'CREATE\s+\[:([\w]+)\s*(\{.*\})?\]\s+FROM\s+"([^"]+)"\s+TO\s+"([^"]+)"', query)
        print("QUERY REÇUE :", query)
        if not match:
            raise ValueError("Invalid CREATE edge syntax")
        edge_class, props, source, target = match.groups()
        return {
            "action": "CREATE_EDGE",
            "class": edge_class,
            "properties": self._extract_properties(props or ""),
            "source_name": source,
            "target_name": target
        }

    def _parse_read_node(self, query):
        match_with_where = re.match(r'READ\s+\(:([\w]+)\)\s+WHERE\s+(\w+)\s*=\s*"([^"]+)"', query)
        if match_with_where:
            node_class, key, value = match_with_where.groups()
            return {
                "action": "READ_NODE",
                "class": node_class,
                "filter": {key: value}
            }

        match_class_only = re.match(r'READ\s+\(:([\w]+)\)', query)
        if match_class_only:
            node_class = match_class_only.group(1)
            return {
                "action": "READ_NODE",
                "class": node_class,
                "filter": {}
            }

        raise ValueError("Invalid READ syntax")

        node_class, key, value = match.groups()
        return {
            "action": "READ_NODE",
            "class": node_class,
            "filter": {key: value}
        }

    def _parse_update_node(self, query):
        match = re.match(r'UPDATE\s+\(:([\w]+)\)\s+SET\s+(.*?)\s+WHERE\s+(\w+)\s*=\s*"([^"]+)"', query)
        if not match:
            raise ValueError("Invalid UPDATE syntax")
        node_class, set_clause, where_key, where_val = match.groups()
        updates = {}
        for part in set_clause.split(","):
            k, v = part.strip().split("=")
            updates[k.strip()] = ast.literal_eval(v.strip())
        return {
            "action": "UPDATE_NODE",
            "class": node_class,
            "updates": updates,
            "filter": {where_key: where_val}
        }

    def _parse_delete_node(self, query):
        match = re.match(r'DELETE\s+\(:([\w]+)\)\s+WHERE\s+(\w+)\s*=\s*"([^"]+)"', query)
        if not match:
            raise ValueError("Invalid DELETE syntax")
        node_class, key, val = match.groups()
        return {
            "action": "DELETE_NODE",
            "class": node_class,
            "filter": {key: val}
        }
    
    def _parse_link_query(self, query):
        match = re.match(r'LINK\s+FROM\s+"([^"]+)"\s+TO\s+"([^"]+)"', query)
        if not match:
            raise ValueError("Invalid LINK syntax")
        return {
            "action": "FIND_PATH",
            "source": match.group(1),
            "target": match.group(2)
        }
        
    def _parse_color_query(self, query):
        if query.strip() == "COLOR":
            return {"action": "COLOR_GRAPH"}
        raise ValueError("Invalid COLOR syntax")
    
    def _parse_cluster_query(self, query):
        if query.strip() == "CLUSTER":
            return {"action": "CLUSTER"}

    def _parse_tag_hubs(self, query):
        if query.strip() == "TAG_HUBS":
            return {"action": "TAG_HUBS"}
        
    def _parse_components(self, query):
        if query.strip() == "CONNECTED_COMPONENTS":
            return {"action": "CONNECTED_COMPONENTS"}
        raise ValueError("Invalid CONNECTED_COMPONENTS syntax")




# --- ORQueryExecutor ---
class ORQueryExecutor:
    def __init__(self, db):
        self.db = db
        self.parser = ORQueryParser()

    def execute(self, query):
        parsed = self.parser.parse(query)
        action = parsed["action"]

        if action == "CREATE_NODE":
            return self.db.create_node(
                name=parsed["properties"].get("name"),
                node_class=parsed["class"],
                properties=parsed["properties"]
            )

        elif action == "CREATE_EDGE":
            srcs = self.db.find_nodes_by_name(parsed["source_name"])
            tgts = self.db.find_nodes_by_name(parsed["target_name"])
            if not srcs or not tgts:
                raise ValueError("Source or target node not found")
            return self.db.create_edge(
                source_id=srcs[0].id,
                target_id=tgts[0].id,
                edge_class=parsed["class"],
                properties=parsed["properties"]
            )

        elif action == "READ_NODE":
            return [
                {
                    "name": node.name,
                    "class": node.node_class,
                    "properties": node.properties
                }
                for node in self.db.nodes.values()
                if node.node_class == parsed["class"]
                and all(node.properties.get(k) == v for k, v in parsed["filter"].items())
            ]


        elif action == "UPDATE_NODE":
            updated = []
            for node in self.db.nodes.values():
                if node.node_class == parsed["class"] and \
                   all(node.properties.get(k) == v for k, v in parsed["filter"].items()):
                    node.properties.update(parsed["updates"])
                    updated.append(node)
            return updated

        elif action == "DELETE_NODE":
            deleted = []
            for node in list(self.db.nodes.values()):
                if node.node_class == parsed["class"] and \
                   all(node.properties.get(k) == v for k, v in parsed["filter"].items()):
                    self.db.delete_node(node.id)
                    deleted.append(node)
            return deleted
        
        elif action == "FIND_PATH":
            G = self.db.to_networkx()
            src_nodes = self.db.find_nodes_by_name(parsed["source"])
            tgt_nodes = self.db.find_nodes_by_name(parsed["target"])
            if not src_nodes or not tgt_nodes:
                raise ValueError("Node(s) not found")
            try:
                path = nx.shortest_path(G, src_nodes[0].id, tgt_nodes[0].id)
                return [self.db.get_node_by_id(nid).name for nid in path]
            except nx.NetworkXNoPath:
                return "Aucun chemin trouvé!"
            
        elif action == "COLOR_GRAPH":
            G = self.db.to_networkx()
            colors = nx.coloring.greedy_color(G, strategy="largest_first")

            # Assigner les couleurs aux nœuds (entiers)
            for node_id, color in colors.items():
                self.db.nodes[node_id].color = color
                self.db.nodes[node_id].properties["color"] = color  # pour affichage
            return f" Graphe colorié avec {len(set(colors.values()))} couleurs"
        
        elif action == "CLUSTER":
            G = self.db.to_networkx()
            partition = community.best_partition(G)

            for node_id, community_id in partition.items():
                self.db.nodes[node_id].properties["cluster"] = community_id
                self.db.nodes[node_id].node_class = "Clustered"

            return f"✅ {len(set(partition.values()))} communautés détectées."

        elif action == "TAG_HUBS":
            G = self.db.to_networkx()
            centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            top_k = int(0.05 * len(sorted_nodes)) or 1  # top 5%

            for i in range(top_k):
                node_id = sorted_nodes[i][0]
                node = self.db.nodes[node_id]
                node.properties["role"] = "Hub"
                node.node_class = "Hubs"

            return f"{top_k} hubs taggués."


        elif action == "CONNECTED_COMPONENTS":
            G = self.db.to_networkx()
            comps = list(nx.connected_components(G))
            result = []
            for i, comp in enumerate(comps):
                names = [self.db.nodes[nid].name for nid in comp]
                result.append({"component": i, "size": len(comp), "nodes": names})
            return result


        else:
            raise ValueError(f"Unsupported action: {action}")
