"""Social graph (M3 only) — a homophily-weighted kNN graph over persona feature
vectors, with a Friedkin–Johnsen update: each persona keeps its initial stance as
an anchor and blends in neighbours' stances by a per-persona susceptibility. The
anchor is what stops the network collapsing into fake consensus."""
