import { useTheme } from '@mui/material/styles';
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
	nerdMode: boolean;
	onNerdModeChange: (value: boolean) => void;
};

export default function FamilyTreeVisualization({ data, nerdMode, onNerdModeChange }: Props) {
	const [hoveredNode, setHoveredNode] = useState<number | null>(null);
	const theme = useTheme();

	// Constants for layout
	const NODE_RADIUS = 20;
	const WIDTH = 800;
	const HEIGHT = 500;
	const PADDING = 50;

	// X = Spread based on ID order or index
	// Y = Based on 'time' (generation)
	const layout = useMemo(() => {
		const focusId = data.focus_id;

		const allNodesById = new Map<number, FamilyNode>();
		data.nodes.forEach((n) => allNodesById.set(n.id, n));

		// Determine parent/child direction using time (higher time = older = parent)
		const parentMap = new Map<number, number[]>(); // childId -> [parentIds]
		const childMap = new Map<number, number[]>(); // parentId -> [childIds]

		data.edges.forEach((edge) => {
			const sNode = allNodesById.get(edge.source);
			const tNode = allNodesById.get(edge.target);
			if (!sNode || !tNode) return;
			let parentId: number, childId: number;
			if (sNode.time >= tNode.time) {
				parentId = edge.source;
				childId = edge.target;
			} else {
				parentId = edge.target;
				childId = edge.source;
			}
			if (!parentMap.has(childId)) parentMap.set(childId, []);
			parentMap.get(childId)!.push(parentId);
			if (!childMap.has(parentId)) childMap.set(parentId, []);
			childMap.get(parentId)!.push(childId);
		});

		const focusParents = parentMap.get(focusId) ?? [];
		const focusChildren = childMap.get(focusId) ?? [];

		// Only show: parents of focus, focus itself, children of focus
		const visibleIds = new Set([focusId, ...focusParents, ...focusChildren]);
		const visibleNodes = data.nodes.filter((n) => visibleIds.has(n.id));

		// Group visible nodes by generation (time)
		const layers: { [time: number]: FamilyNode[] } = {};
		visibleNodes.forEach((node) => {
			if (!layers[node.time]) layers[node.time] = [];
			layers[node.time].push(node);
		});

		const sortedTimes = Object.keys(layers)
			.map(Number)
			.sort((a, b) => a - b);
		const minTime = sortedTimes[0] ?? 0;
		const maxTime = sortedTimes[sortedTimes.length - 1] ?? 0;
		const timeRange = maxTime - minTime || 1;

		const nodesById = new Map<number, { x: number; y: number; node: FamilyNode }>();

		sortedTimes.forEach((time) => {
			const nodesInLayer = layers[time];
			nodesInLayer.sort((a, b) => a.id - b.id);
			const t = (time - minTime) / timeRange;
			const y = PADDING + (1 - t) * (HEIGHT - 2 * PADDING);
			nodesInLayer.forEach((node, idx) => {
				const x = nodesInLayer.length === 1 ? WIDTH / 2 : PADDING + (idx / (nodesInLayer.length - 1)) * (WIDTH - 2 * PADDING);
				nodesById.set(node.id, { x, y, node });
			});
		});

		const familyGroups: { parents: number[]; children: number[] }[] = [];

		// Parents -> focus group
		if (focusParents.length > 0) {
			familyGroups.push({
				parents: [...focusParents].sort((a, b) => a - b),
				children: [focusId]
			});
		}

		// Focus -> children group
		if (focusChildren.length > 0) {
			familyGroups.push({
				parents: [focusId],
				children: [...focusChildren].sort((a, b) => a - b)
			});
		}

		return { nodesById, familyGroups };
	}, [data]);

	const hovered = hoveredNode !== null ? layout.nodesById.get(hoveredNode) : undefined;

	return (
		<div className="family-tree">
			<div className="family-tree-header">
				<h3 className="family-tree-title">
					Family Tree: {data.dataset} focusing on individual {data.focus_id}
				</h3>
			</div>

			<p className="context-text">
				{nerdMode ? (
					<>
						This diagram visualizes the immediate family pedigree for the selected individual. Each circle is a node representing a person
						in the simulated family tree. The <strong>vertical axis</strong> encodes generation time — ancestors with higher temporal
						coordinates appear higher on the plot; descendants appear lower. Only direct parent-child edges are rendered; the genealogical
						relationships are inferred from the temporal ordering of nodes.
					</>
				) : (
					<>
						This diagram shows the family tree for the selected individual. Each circle represents a person. The{' '}
						<strong>vertical position</strong> shows the generational gaps — family members higher up are older ancestors, while those
						lower down are their descendants. You can see the direct family relationships: parents above, children below.
					</>
				)}
			</p>
			<p className="context-text">
				{nerdMode ? (
					<>
						<span className="text-known">Green</span> nodes represent individuals with successfully called genotypes present in the
						dataset. <span className="text-unknown">Red</span> nodes represent individuals with missing genotype data; their genetic
						states must be inferred by the model using information propagated from sequenced relatives. The <strong>blue</strong> node
						highlights the currently selected focus individual. Hover over any node to inspect its genotype vector.
					</>
				) : (
					<>
						<span className="text-known">Green</span> circles represent people whose genetic information is known.{' '}
						<span className="text-unknown">Red</span> circles represent people whose genetic information is unknown — the model tries to
						figure it out based on their family members' information. The <strong>blue</strong> circle is the person you selected. Hover
						over any circle to see their information.
					</>
				)}
			</p>

			<div className="tree-svg-wrapper">
				<div className="tree-svg-scroll">
					<svg
						width={WIDTH}
						height={HEIGHT}
						className="tree-svg"
						role="img"
						aria-label={`Family tree diagram for ${data.dataset}, focused on individual ${data.focus_id}`}
					>
						<title>
							Family tree diagram for {data.dataset}, focused on individual {data.focus_id}
						</title>
						{/* Family tree connectors */}
						{layout.familyGroups.map((group, gi) => {
							const stroke = theme.palette.text.secondary;
							const sw = 2;

							const parentPos = group.parents
								.map((id) => layout.nodesById.get(id))
								.filter((p): p is NonNullable<typeof p> => p != null);
							const childPos = group.children
								.map((id) => layout.nodesById.get(id))
								.filter((c): c is NonNullable<typeof c> => c != null);

							if (parentPos.length === 0 || childPos.length === 0) return null;

							const childrenY = childPos[0].y;
							const lines: React.ReactElement[] = [];

							if (parentPos.length >= 2) {
								const sorted = [...parentPos].sort((a, b) => a.x - b.x);
								const p1 = sorted[0];
								const p2 = sorted[sorted.length - 1];
								const midX = (p1.x + p2.x) / 2;
								const parentY = p1.y;
								const junctionY = (parentY + childrenY) / 2;

								// Couple horizontal line
								lines.push(
									<line
										key={`${gi}-c`}
										x1={p1.x}
										y1={parentY}
										x2={p2.x}
										y2={parentY}
										stroke={stroke}
										strokeWidth={sw}
										pointerEvents="none"
									/>
								);
								// Drop from midpoint to junction
								lines.push(
									<line
										key={`${gi}-d`}
										x1={midX}
										y1={parentY}
										x2={midX}
										y2={junctionY}
										stroke={stroke}
										strokeWidth={sw}
										pointerEvents="none"
									/>
								);

								if (childPos.length === 1) {
									const cx = childPos[0].x;
									// Horizontal from midpoint to child x, then vertical down
									lines.push(
										<line
											key={`${gi}-sh`}
											x1={midX}
											y1={junctionY}
											x2={cx}
											y2={junctionY}
											stroke={stroke}
											strokeWidth={sw}
											pointerEvents="none"
										/>
									);
									lines.push(
										<line
											key={`${gi}-sv`}
											x1={cx}
											y1={junctionY}
											x2={cx}
											y2={childrenY}
											stroke={stroke}
											strokeWidth={sw}
											pointerEvents="none"
										/>
									);
								} else {
									const sortedC = [...childPos].sort((a, b) => a.x - b.x);
									const barLeft = Math.min(sortedC[0].x, midX);
									const barRight = Math.max(sortedC[sortedC.length - 1].x, midX);
									// Sibling bar
									lines.push(
										<line
											key={`${gi}-b`}
											x1={barLeft}
											y1={junctionY}
											x2={barRight}
											y2={junctionY}
											stroke={stroke}
											strokeWidth={sw}
											pointerEvents="none"
										/>
									);
									// Vertical to each child
									sortedC.forEach((c, ci) => {
										lines.push(
											<line
												key={`${gi}-v${ci}`}
												x1={c.x}
												y1={junctionY}
												x2={c.x}
												y2={childrenY}
												stroke={stroke}
												strokeWidth={sw}
												pointerEvents="none"
											/>
										);
									});
								}
							} else {
								// Single parent
								const p = parentPos[0];
								const junctionY = (p.y + childrenY) / 2;

								if (childPos.length === 1) {
									const cx = childPos[0].x;
									// Vertical from parent down to junction, horizontal to child x, vertical to child
									lines.push(
										<line
											key={`${gi}-pd`}
											x1={p.x}
											y1={p.y}
											x2={p.x}
											y2={junctionY}
											stroke={stroke}
											strokeWidth={sw}
											pointerEvents="none"
										/>
									);
									lines.push(
										<line
											key={`${gi}-sh`}
											x1={p.x}
											y1={junctionY}
											x2={cx}
											y2={junctionY}
											stroke={stroke}
											strokeWidth={sw}
											pointerEvents="none"
										/>
									);
									lines.push(
										<line
											key={`${gi}-sv`}
											x1={cx}
											y1={junctionY}
											x2={cx}
											y2={childrenY}
											stroke={stroke}
											strokeWidth={sw}
											pointerEvents="none"
										/>
									);
								} else {
									const sortedC = [...childPos].sort((a, b) => a.x - b.x);
									const barLeft = Math.min(sortedC[0].x, p.x);
									const barRight = Math.max(sortedC[sortedC.length - 1].x, p.x);
									lines.push(
										<line
											key={`${gi}-d`}
											x1={p.x}
											y1={p.y}
											x2={p.x}
											y2={junctionY}
											stroke={stroke}
											strokeWidth={sw}
											pointerEvents="none"
										/>
									);
									lines.push(
										<line
											key={`${gi}-b`}
											x1={barLeft}
											y1={junctionY}
											x2={barRight}
											y2={junctionY}
											stroke={stroke}
											strokeWidth={sw}
											pointerEvents="none"
										/>
									);
									sortedC.forEach((c, ci) => {
										lines.push(
											<line
												key={`${gi}-v${ci}`}
												x1={c.x}
												y1={junctionY}
												x2={c.x}
												y2={childrenY}
												stroke={stroke}
												strokeWidth={sw}
												pointerEvents="none"
											/>
										);
									});
								}
							}

							return <g key={gi}>{lines}</g>;
						})}

						{/* Draw Nodes */}
						{Array.from(layout.nodesById.values()).map(({ x, y, node }) => {
							const hasGenetics = node.observed.some((v) => v !== null);
							const isFocus = node.id === data.focus_id;
							const focusMissingGenetics = isFocus && !hasGenetics;
							const nodeFill = isFocus ? '#2563eb' : hasGenetics ? '#bbf7d0' : '#fecaca';
							const nodeStroke = isFocus ? (hasGenetics ? '#1d4ed8' : '#dc2626') : hasGenetics ? '#16a34a' : '#dc2626';
							return (
								<g
									key={node.id}
									className="tree-node"
									role="img"
									aria-label={`Individual ${node.id}${isFocus ? ' (focus)' : ''}, ${hasGenetics ? 'known' : 'unknown'} genotype`}
								>
									<circle
										cx={x}
										cy={y}
										r={NODE_RADIUS + 10}
										fill="transparent"
										pointerEvents="all"
										tabIndex={0}
										onPointerEnter={() => {
											setHoveredNode(node.id);
										}}
										onPointerLeave={() => {
											setHoveredNode(null);
										}}
										onFocus={() => {
											setHoveredNode(node.id);
										}}
										onBlur={() => {
											setHoveredNode(null);
										}}
									/>
									<circle
										cx={x}
										cy={y}
										r={isFocus ? NODE_RADIUS + 4 : NODE_RADIUS}
										fill={nodeFill}
										stroke={nodeStroke}
										strokeWidth={isFocus ? 3 : 1}
										strokeDasharray={focusMissingGenetics ? '6 3' : undefined}
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
							);
						})}
					</svg>
					{hovered &&
						(() => {
							const TW = 180;
							const TH = 120;
							const gap = NODE_RADIUS + 14;
							const rawX = hovered.x + gap + TW > WIDTH ? hovered.x - gap - TW : hovered.x + gap;
							const rawY = Math.min(Math.max(hovered.y - TH / 2, 4), HEIGHT - TH - 4);
							return (
								<div className="tree-tooltip" style={{ left: rawX, top: rawY }}>
									<div className="tree-tooltip-header">Individual {hovered.node.id}</div>
									<div className="tree-tooltip-label">Genotypes (first 20):</div>
									<div className="tree-tooltip-values">
										{hovered.node.observed
											.slice(0, 20)
											.map((v) => (v === null ? '?' : v))
											.join(', ')}
										&hellip;
									</div>
								</div>
							);
						})()}
				</div>
			</div>

			<p className="tree-legend">
				<span className="legend-item">
					<span className="legend-circle legend-circle-focus" />
					<span className="text-bold">Focused Individual</span>
				</span>
				<span className="legend-item">
					<span className="legend-circle legend-circle-known" />
					<span className="text-known">Known genotype</span>
				</span>
				<span className="legend-item">
					<span className="legend-circle legend-circle-unknown" />
					<span className="text-unknown">Unknown genotype</span>
				</span>
				<span className="empty-state">
					Vertical axis = Time. Click on the nodes to see their {nerdMode ? 'genotype vectors' : 'genotypes'}.
				</span>
				<span className="empty-state">
					{nerdMode
						? 'Shows direct connections only (parent-child edges inferred from temporal ordering). Siblings, marital ties, and collateral relatives are omitted.'
						: 'Shows direct connections only — parents and children of the selected node. Siblings and partners are not shown.'}
				</span>
			</p>
		</div>
	);
}
