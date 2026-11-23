# Historical figure data

These two CSV files are a transcription of the graph and the blue path
in Figure 2 of the manuscript. The final repeated vertex is recorded
explicitly, as shown by Figure 3.

Vertex IDs are `10*x+y` for coordinates `(x,y)` in the 10 by 10 grid.
`figure2_edges.csv` contains the 144 unordered edges; each undirected
edge is stored once, with the smaller ID first, in lexicographic order.
`figure2_trajectory.csv` contains the 20 vertices of the historical
trajectory and their objective values.

These files are reference data, **not** a regenerated training result.
The graph generator with seed 0 reproduces the edge set exactly.
The historical training RNG, tie-breaking and reward-progression data
are not present in the original repository commit `93bf33a`.
