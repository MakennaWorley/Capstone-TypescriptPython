import { useMemo, useState } from 'react';

type GeneticValue = number | null;

type FamilyNode = {
	id: number;
	time: number;
	observed: GeneticValue[];
};

type FamilyEdge = {
	source: number;
	target: number;
};

type FamilyTreeData = {
	dataset: string;
	focus_id: number;
	nodes: FamilyNode[];
	edges: FamilyEdge[];
};

type Props = {
	data: FamilyTreeData;
};

export default function FamilyTreeVisualization({ data }: Props) {
	const [hoveredNode, setHoveredNode] = useState<number | null>(null);

	// Constants for layout
	const NODE_RADIUS = 20;
	const WIDTH = 800;
	const HEIGHT = 500;
	const PADDING = 50;

	// X = Spread based on ID order or index
	// Y = Based on 'time' (generation)
	const layout = useMemo(() => {
		const nodesById = new Map<number, { x: number; y: number; node: FamilyNode }>();

		// 1. Group nodes by their time (generation)
		const layers: { [time: number]: FamilyNode[] } = {};
		data.nodes.forEach((node) => {
			if (!layers[node.time]) layers[node.time] = [];
			layers[node.time].push(node);
		});

		const sortedTimes = Object.keys(layers)
			.map(Number)
			.sort((a, b) => a - b);
		const minTime = sortedTimes[0];
		const maxTime = sortedTimes[sortedTimes.length - 1];
		const timeRange = maxTime - minTime || 1;

		// 2. Calculate coordinates for each layer
		sortedTimes.forEach((time) => {
			const nodesInLayer = layers[time];
			// Sort by ID within layer to keep the tree consistent
			nodesInLayer.sort((a, b) => a.id - b.id);

			const t = (time - minTime) / timeRange;
			const y = PADDING + (1 - t) * (HEIGHT - 2 * PADDING);

			nodesInLayer.forEach((node, idx) => {
				// Spread nodes horizontally relative to the count IN THIS LAYER
				const x = nodesInLayer.length === 1 ? WIDTH / 2 : PADDING + (idx / (nodesInLayer.length - 1)) * (WIDTH - 2 * PADDING);

				nodesById.set(node.id, { x, y, node });
			});
		});

		return nodesById;
	}, [data]);

	const hovered = hoveredNode !== null ? layout.get(hoveredNode) : undefined;

	return (
		<div style={{ marginTop: '2rem', padding: '1rem', border: '1px solid #ddd', borderRadius: '12px' }}>
			<h3>
				Family Tree: {data.dataset} (Focus: {data.focus_id})
			</h3>

			<div style={{ position: 'relative' }}>
				<div style={{ overflowX: 'auto' }}>
					<svg width={WIDTH} height={HEIGHT} style={{ border: '1px solid #eee', background: '#fff' }}>
						{/* Draw Edges */}
						{data.edges.map((edge, i) => {
							const start = layout.get(edge.source);
							const end = layout.get(edge.target);
							if (!start || !end) return null;
							return (
								<line
									key={`edge-${i}`}
									x1={start.x}
									y1={start.y}
									x2={end.x}
									y2={end.y}
									stroke="#999"
									strokeWidth="2"
									pointerEvents="none"
								/>
							);
						})}

						{/* Draw Nodes */}
						{Array.from(layout.values()).map(({ x, y, node }) => (
							<g key={node.id} style={{ cursor: 'pointer' }}>
								<circle
									cx={x}
									cy={y}
									r={NODE_RADIUS + 10}
									fill="transparent"
									pointerEvents="all"
									onPointerEnter={() => {
										setHoveredNode(node.id);
									}}
									onPointerLeave={() => {
										setHoveredNode(null);
									}}
								/>
								<circle
									cx={x}
									cy={y}
									r={node.id === data.focus_id ? NODE_RADIUS + 4 : NODE_RADIUS}
									fill={node.id === data.focus_id ? '#3b82f6' : '#fff'}
									stroke={node.id === data.focus_id ? '#1d4ed8' : '#333'}
									strokeWidth={node.id === data.focus_id ? 3 : 1}
									pointerEvents="none"
								/>
								<text
									x={x}
									y={y + 5}
									textAnchor="middle"
									fontSize="10px"
									fontWeight="bold"
									fill={node.id === data.focus_id ? '#fff' : '#000'}
									pointerEvents="none"
								>
									{node.id}
								</text>
							</g>
						))}
					</svg>
				</div>

				{/* Genotype Preview Tooltip */}
				{hovered && (
					<div
						style={{
							position: 'absolute',
							top: 10,
							left: 10,
							zIndex: 50,
							background: 'white',
							padding: '10px',
							border: '1px solid #ccc',
							borderRadius: '4px',
							maxWidth: '200px',
							fontSize: '0.8rem',
							boxShadow: '0 2px 5px rgba(0,0,0,0.1)',
							pointerEvents: 'none',
							color: '#111'
						}}
					>
						<strong>Ind ID: {hovered.node.id}</strong>
						<div style={{ wordBreak: 'break-all', marginTop: '5px' }}>
							<strong>Genotypes:</strong>
							<br />
							{hovered.node.observed
								.slice(0, 20)
								.map((v) => (v === null ? '?' : v))
								.join(', ')}
							...
						</div>
					</div>
				)}
			</div>

			<p style={{ fontSize: '0.8rem', color: '#666', marginTop: '10px' }}>
				* Vertical axis represents <b>Time</b>. Blue node is the focus individual. Hover to see genotype vectors.
			</p>
		</div>
	);
}
