import tskit

# Load the tree sequence
ts = tskit.load('datasets/public.trees')

print(f'Total individuals: {ts.num_individuals}')
print(f'Total sites: {ts.num_sites}')
print(f'Total mutations: {ts.num_mutations}')
print()

# Check founder individuals
founders = [ind for ind in ts.individuals() if ind.time == 4.0]
print(f'Founders: {[ind.id for ind in founders]}')
print(f'Founder nodes: {[list(ind.nodes) for ind in founders]}')
print()

# Get genotype matrix
all_nodes = []
for ind in ts.individuals():
	if len(ind.nodes) == 2:
		all_nodes.extend([int(ind.nodes[0]), int(ind.nodes[1])])

G = ts.genotype_matrix(samples=all_nodes)
print(f'Genotype matrix shape: {G.shape}')
print()

# Check first few sites for founders
print('Checking genotypes at first 10 sites for founders:')
node_to_col = {node_id: col for col, node_id in enumerate(all_nodes)}

for site_idx in range(min(10, ts.num_sites)):
	site = ts.sites()[site_idx]
	print(f'\nSite {site_idx} (position {site.position}):')
	print(f'  Mutations on this site: {[(m.node, m.derived_state) for m in site.mutations]}')

	for ind in founders:
		try:
			cols = [node_to_col[n] for n in ind.nodes]
			alleles = [G[site_idx, cols[0]], G[site_idx, cols[1]]]
			dosage = sum(alleles)
			print(f'  Individual {ind.id}: nodes {list(ind.nodes)}, alleles {alleles}, dosage {dosage}')
		except KeyError:
			print(f'  Individual {ind.id}: nodes not in samples')
