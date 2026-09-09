"""
Generates node/edge/type XML for a 4x4 grid SUMO network.
Layout convention (stated explicitly so it can be changed):
  - Horizontal roads (East-West) = MAIN roads  -> 4 lanes per direction
  - Vertical roads (North-South) = SIDE roads  -> 2 lanes per direction
  - All 16 grid junctions are signalized (traffic_light)
  - Spacing between adjacent junctions = 400 m
"""

N = 3  # 3x3 grid (generalization test: different topology than the 4x4 training network)
SPACING = 400

def node_id(i, j):
    return f"J{i}_{j}"

nodes = []
for i in range(N):
    for j in range(N):
        x, y = i * SPACING, j * SPACING
        nodes.append((node_id(i, j), x, y))

# ---------- nodes.nod.xml ----------
with open("network.nod.xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<nodes>\n')
    for nid, x, y in nodes:
        f.write(f'    <node id="{nid}" x="{x}" y="{y}" type="traffic_light"/>\n')
    f.write('</nodes>\n')

# ---------- types.typ.xml ----------
with open("network.typ.xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<types>\n')
    f.write('    <type id="mainroad" priority="3" numLanes="4" speed="16.67"/>\n')  # ~60 km/h
    f.write('    <type id="sideroad" priority="1" numLanes="2" speed="11.11"/>\n')  # ~40 km/h
    f.write('</types>\n')

# ---------- edges.edg.xml ----------
edges = []
# Horizontal = main roads (both directions), connecting (i,j)-(i+1,j)
for j in range(N):
    for i in range(N - 1):
        a, b = node_id(i, j), node_id(i + 1, j)
        edges.append((f"E_{a}_{b}", a, b, "mainroad"))
        edges.append((f"E_{b}_{a}", b, a, "mainroad"))

# Vertical = side roads (both directions), connecting (i,j)-(i,j+1)
for i in range(N):
    for j in range(N - 1):
        a, b = node_id(i, j), node_id(i, j + 1)
        edges.append((f"E_{a}_{b}", a, b, "sideroad"))
        edges.append((f"E_{b}_{a}", b, a, "sideroad"))

with open("network.edg.xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<edges>\n')
    for eid, frm, to, etype in edges:
        f.write(f'    <edge id="{eid}" from="{frm}" to="{to}" type="{etype}"/>\n')
    f.write('</edges>\n')

# ---------- fringe stub edges (entry/exit points at the grid boundary) ----------
# Each of the 12 boundary junctions gets one outward stub (both directions) so
# vehicles have somewhere to originate/terminate outside the signalized grid.
fringe_nodes = []
fringe_edges = []
fringe_ids = []  # node ids of the outer stub endpoints, used by route generation

def add_fringe(base_id, x, y, dx, dy, road_type, tag):
    fid = f"F_{base_id}_{tag}"
    fx, fy = x + dx, y + dy
    fringe_nodes.append((fid, fx, fy))
    a_id = f"E_{fid}_{base_id}"
    b_id = f"E_{base_id}_{fid}"
    fringe_edges.append((a_id, fid, base_id, road_type))
    fringe_edges.append((b_id, base_id, fid, road_type))
    fringe_ids.append(fid)

for i in range(N):
    for j in range(N):
        nid = node_id(i, j)
        x, y = i * SPACING, j * SPACING
        if i == 0:
            add_fringe(nid, x, y, -SPACING, 0, "mainroad", "W")
        if i == N - 1:
            add_fringe(nid, x, y, SPACING, 0, "mainroad", "E")
        if j == 0:
            add_fringe(nid, x, y, 0, -SPACING, "sideroad", "S")
        if j == N - 1:
            add_fringe(nid, x, y, 0, SPACING, "sideroad", "N")

with open("network.nod.xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<nodes>\n')
    for nid, x, y in nodes:
        f.write(f'    <node id="{nid}" x="{x}" y="{y}" type="traffic_light"/>\n')
    for nid, x, y in fringe_nodes:
        f.write(f'    <node id="{nid}" x="{x}" y="{y}" type="priority"/>\n')
    f.write('</nodes>\n')

with open("network.edg.xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<edges>\n')
    for eid, frm, to, etype in edges:
        f.write(f'    <edge id="{eid}" from="{frm}" to="{to}" type="{etype}"/>\n')
    for eid, frm, to, etype in fringe_edges:
        f.write(f'    <edge id="{eid}" from="{frm}" to="{to}" type="{etype}"/>\n')
    f.write('</edges>\n')

with open("fringe_nodes.txt", "w") as f:
    f.write("\n".join(fringe_ids))

print(f"Generated {len(nodes)} core nodes + {len(fringe_nodes)} fringe nodes")
print(f"Generated {len(edges)} core edges + {len(fringe_edges)} fringe edges")
print("Main (4-lane) horizontal edges:", sum(1 for e in edges if e[3]=='mainroad'))
print("Side (2-lane) vertical edges:", sum(1 for e in edges if e[3]=='sideroad'))
