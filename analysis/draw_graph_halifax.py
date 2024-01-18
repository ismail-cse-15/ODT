import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




G = nx.read_shp('/Volumes/Nur/Nur/TNDP New/Network_ODMatrices_Papers/Network/emme_links.shp')
link_length_dic = nx.get_edge_attributes(G, 'LENGTH')
nodez = nx.read_shp('/Volumes/Nur/Nur/TNDP New/Network_ODMatrices_Papers/Network/emme_nodes.shp')
node_isZone_dic = nx.get_node_attributes(nodez, 'ISZONE')
node_zone_dic = nx.get_node_attributes(nodez, 'EMME_Zone')

pos = nx.get_node_attributes(nodez, 'ID') #preserve ID of GIS
#pos = {k: v for k, v in enumerate(G.nodes())}
rev_pos = {v: k for k, v in pos.items()}



l = [x for x in G.edges()]  # To speed things up in case of large objects
edg = []
included_pos = set()
for sl in l:
    #sl_rev = (sl[1], sl[0])
    if True:#sl_rev in link_length_dic:
        edg.append((pos[sl[0]], pos[sl[1]]))
        included_pos.add(pos[sl[0]])
        included_pos.add(pos[sl[1]])
    # inode = i_dic[sl]
    # jnode = j_dic[sl]
    # edg.append((inode, jnode))
    # included_pos.add(inode)
    # included_pos.add(jnode)
print('total edges: %d, birectional: %d' % (len(l), len(edg)))

for key in list(rev_pos):
    if key not in included_pos:
        rev_pos.pop(key)


X = nx.Graph()  # Empty graph
X.add_nodes_from(included_pos)  # Add nodes preserving coordinates


#nx.draw_networkx_nodes(X, pos, node_size=2, node_color='r')
labels = {}
pos2 = {}
X.add_edges_from(edg)
nx.draw_networkx_edges(X, rev_pos, width=0.2)
#plt.xlim(450000, 470000) #This changes and is problem specific
#plt.ylim(430000, 450000) #This changes and is problem specific

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
#plt.legend()
plt.title('Halifax Peninsula. nodes:' + str(len(pos)) + ', links:' + str(len(edg)) )
plt.show()
#plt.savefig('../output/greater_halifax_graph_directional.pdf',  bbox_inches='tight', dpi=300)
#plt.clf()