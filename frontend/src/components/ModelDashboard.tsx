type Model = {
	model_name: string;
	model_type: string;
};

type ModelDashboardProps = {
	model: Model;
	nerdMode: boolean;
	onNerdModeChange: (value: boolean) => void;
};

// Helper function to render text with variable names highlighted in nerd mode
function renderTextWithVarHighlight(text: string, isNerdMode: boolean): (string | JSX.Element)[] {
	if (!isNerdMode) return [text];

	// Split on variable patterns (snake_case) and wrap them in <code> with nerd-text class
	const parts = text.split(/(\b[a-z_][a-z0-9_]*\b)/g);
	return parts.map((part, idx) => {
		// Check if it's a variable name (contains underscore or is a technical term)
		if (/^[a-z_][a-z0-9_]*$/.test(part) && (part.includes('_') || ['softmax', 'dropout', 'normalization', 'CUDA', 'BFGS'].includes(part))) {
			return (
				<code key={idx} className="nerd-text nerd-code">
					{part}
				</code>
			);
		}
		return part;
	});
}

const MODEL_TYPE_INFO: Record<
	string,
	{
		label: string;
		simple_description: string;
		simple_strengths: string[];
		simple_use_case: string;
		description: string;
		strengths: string[];
		use_case: string;
	}
> = {
	dnn_dosage: {
		label: 'Deep Neural Network',
		simple_description:
			"Like teaching a kid to recognize patterns. The more examples you show it, the better it gets. If you show it pictures of thousands of families and say 'this gene pattern goes with this type of genetics,' it eventually learns to spot those patterns on its own. The downside: it needs a lot of examples to work well, and it takes longer to train. But when you have tons of data, it's the most powerful option.",
		simple_strengths: [
			"Figures out what matters on its own — you don't have to tell it",
			'Gets better the more examples it sees',
			'Can handle thousands of genes at once',
			"Doesn't memorize the examples as much",
			'Runs faster on a graphics card'
		],
		simple_use_case:
			"Pick this when you have a huge amount of data — thousands of people and thousands of genes. The more data you give it, the more accurately it can predict. If you're patient and want the best accuracy possible, use this one. Don't use it if your dataset is small.",
		description:
			'A fully connected feedforward neural network (PyTorch) trained to predict per-site allele dosage (0, 1, 2) for masked individuals. The architecture stacks three hidden layers (256 → 128 → 64 neurons), each followed by batch normalization and ReLU activation. Residual (skip) connections are added between compatible layer widths to mitigate vanishing gradients. Dropout (p=0.3) is applied during training for regularization. The output layer uses softmax over three classes. Input features are constructed via k-hop relative aggregation over the pedigree graph: for each masked individual at each genomic site, the feature vector is [mean_dosage_of_k_hop_relatives, fraction_of_relatives_observed, count_of_relatives]. Training uses cross-entropy loss with class weights inversely proportional to class frequency to handle dosage imbalance. Optimized with Adam; early stopping monitors validation loss with patience of 10 epochs. Supports CUDA and Apple Metal (MPS) acceleration.',
		strengths: [
			'Captures non-linear, high-order interactions between loci that linear models cannot represent',
			'Residual connections stabilise gradient flow in deep stacks and accelerate convergence',
			'Batch normalization reduces internal covariate shift, enabling higher learning rates',
			'Class-weighted cross-entropy handles dosage class imbalance without resampling',
			'Early stopping prevents overfitting when validation loss plateaus',
			'Scales efficiently to high-dimensional inputs (100k+ genomic sites) on GPU',
			'PyTorch backend supports CUDA (NVIDIA) and MPS (Apple Silicon) acceleration'
		],
		use_case:
			'Best suited for medium to large datasets (1,000+ individuals, 1,000+ sites) where non-linear feature interactions are expected and sufficient training data exists to prevent overfitting. Outperforms logistic regression when pedigree-derived features have complex dosage dependencies. Less suitable than Bayesian inference for small datasets or when posterior uncertainty estimates are required. Slower to train than logistic regression but typically achieves the highest raw accuracy on large cohorts. Recommended when GPU hardware is available.'
	},
	multi_log_regression: {
		label: 'Multinomial Logistic Regression',
		simple_description:
			"The simplest one. It looks at what you know about someone's family and adds up different clues. Some clues matter more than others. The one with the highest score wins. Because it's so simple, you can see exactly why it made its choice. It trains super fast and doesn't usually get confused. This is the best place to start.",
		simple_strengths: [
			'You can see the exact reason for every guess',
			'Trains in just seconds',
			'Works fine with tiny datasets',
			"If fancier models can't beat this, they're not worth using",
			'Super easy to explain to anyone'
		],
		simple_use_case:
			"Always try this first. If you want to quickly figure out whether genetics are predictable, run this. It's perfect when you need to explain things to someone without a technical background. On small datasets, it often works as well as complicated models. Use it to compare all other models against.",
		description:
			"A multinomial logistic regression classifier (scikit-learn) that models P(dosage | features) via softmax over three output classes (0, 1, 2). Input features are the same k-hop pedigree aggregation vectors used by other models: [mean_dosage_of_relatives, fraction_observed, count_relatives] per genomic site per masked individual. The model learns one weight vector per class, solved via the L-BFGS multinomial solver with L2 regularisation (C=1.0). Class weights are set to 'balanced' to compensate for dosage distribution skew. Because the decision boundary is a hyperplane in feature space, each coefficient has a direct interpretation: a positive weight on mean_dosage_of_relatives for dosage class 2 means that higher average relative dosage increases the log-odds of predicting class 2. This is the only model in the benchmark that provides this level of interpretability without post-hoc analysis.",
		strengths: [
			'Coefficients are directly interpretable: each weight quantifies the contribution of one feature to one dosage class',
			'L-BFGS solver converges in seconds on any dataset in this benchmark',
			'L2 regularisation controls overfitting without requiring hyperparameter search',
			'Balanced class weights handle dosage imbalance without resampling or custom loss functions',
			'No GPU required; runs on any hardware with minimal memory overhead',
			'Serves as the primary baseline for assessing whether complexity in other models is justified'
		],
		use_case:
			'Use as the first model on any new dataset to establish a performance floor. If logistic regression achieves high accuracy, the genetic signal is linearly separable and more complex models may not add meaningful value. If accuracy is poor, non-linear structure is present and a DNN or GNN is warranted. Always compare every other model against this baseline before drawing conclusions about model superiority. Also the recommended model when auditability of predictions is a hard requirement.'
	},
	hmm_dosage: {
		label: 'Hidden Markov Model',
		simple_description:
			"Genes that sit next to each other on the same chromosome tend to go together when they're passed down. This model notices that pattern and uses it. Instead of looking at each gene alone, it reads them in order — like reading a sentence. It gets an advantage that other models don't have: it understands genes are connected in order.",
		simple_strengths: [
			'Knows that nearby genes go together',
			'Reads genes in order instead of treating them separately',
			'Gives you confidence levels, not just a guess',
			'Works when the order matters',
			'Trains quickly without needing a graphics card'
		],
		simple_use_case:
			'Use this when your genes are in order by chromosome location and nearby genes matter. If you have a medium-sized dataset and want confidence levels for each guess, this is a good choice. Works great for real genetic data where position is important.',
		description:
			'A Gaussian Hidden Markov Model (hmmlearn) trained to infer per-site allele dosage by treating each individual as an observed sequence across genomic sites. The model assumes K=3 hidden states (corresponding loosely to dosage classes 0, 1, 2) with Gaussian emission distributions and a learned transition matrix. State parameters are estimated via Baum-Welch (EM). Emission means and covariances are initialised from empirical statistics of labelled examples (semi-supervised init) to accelerate convergence and avoid degenerate solutions. At inference time, the Viterbi algorithm decodes the most probable hidden state sequence for each individual, and posterior state probabilities are used to produce soft dosage class probabilities. The sequential structure of the model captures linkage disequilibrium — correlation between nearby sites — that i.i.d. models ignore entirely.',
		strengths: [
			'Explicitly models chromosomal-position dependencies via the learned transition matrix',
			'Semi-supervised initialisation from label statistics prevents degenerate local optima in EM',
			'Viterbi decoding produces a globally consistent state sequence rather than per-site greedy decisions',
			'Posterior state probabilities provide well-calibrated soft class estimates',
			'Interpretable: hidden state means and transition probabilities can be inspected directly',
			'Efficient EM convergence with no GPU required',
			'Naturally handles variable-length sequences without padding'
		],
		use_case:
			'Most effective when genomic sites in the feature matrix are ordered by chromosomal position and linkage disequilibrium between adjacent sites is non-negligible. Performance degrades if sites are shuffled or drawn from unrelated chromosomal regions. Recommended when phasing information is available or where sequential correlation structure is a known biological feature. Not recommended when the feature set mixes sites from multiple chromosomes without positional ordering.'
	},
	gnn_dosage: {
		label: 'Graph Neural Network',
		simple_description:
			"This one thinks about your whole family tree, not just you alone. It builds a map of who's related to whom and lets genetic info flow through it — from parents to kids, between siblings, across generations. It gets that if your grandparent has a gene, you probably do too. It's the most family-aware model.",
		simple_strengths: [
			'Uses the whole family tree as its main source of info',
			'Passes genetic signals through many generations',
			'Your guess uses info from parents, kids, and distant relatives',
			'Learns how genes get passed down through families',
			'Works with families of any size'
		],
		simple_use_case:
			"Use this when you have lots of family information. If your data has parents, grandparents, and other relatives of the people you're trying to predict, this will outperform others because it uses those connections. Works best with medium to large datasets with deep family trees.",
		description:
			'A graph convolutional network (PyTorch Geometric) that constructs a feature-correlation graph from the input data and applies message-passing convolutions to aggregate genetic signals across connected individuals. Each individual is a node; edges are drawn between individuals whose feature vectors exceed a Pearson correlation threshold (default 0.5), encoding genetic similarity independently of pedigree topology. Node features are the same k-hop pedigree aggregation vectors used by other models. The architecture stacks three GraphConv layers (256 → 128 → 64 hidden dimensions) with ReLU activations, followed by global mean pooling to produce a graph-level embedding passed through a linear classifier. Trained with cross-entropy loss and Adam. Supports mini-batch training via PyTorch Geometric DataLoader for memory-efficient handling of large pedigrees. GPU-accelerated.',
		strengths: [
			'Constructs edges from feature correlation, capturing genetic similarity that the pedigree graph alone may not encode',
			'Message-passing aggregates information from all graph neighbours, not just direct parents/children',
			'Global mean pooling produces size-invariant graph embeddings that generalise across pedigree shapes',
			'Mini-batch DataLoader enables training on pedigrees too large to fit in memory at once',
			'GPU-accelerated via PyTorch Geometric sparse kernels',
			'Learns relational inheritance patterns end-to-end without hand-crafted pedigree features',
			'Outperforms non-relational models on datasets with dense, multi-generational family structure'
		],
		use_case:
			'Most powerful on datasets with deep pedigrees (10+ generations) and high connectivity, where many individuals share recent common ancestors. The correlation-based edge construction means the model can also exploit genetic similarity between individuals not directly related in the recorded pedigree. Requires more memory and training time than logistic regression or the HMM; GPU acceleration strongly recommended for medium and large datasets. Performance advantage over the DNN diminishes when pedigree connectivity is sparse.'
	},
	bayes_softmax3: {
		label: 'Bayesian Inference',
		simple_description:
			"Most models give one answer: 'You have type 1.' This one gives the full picture: 'You probably have type 1 (70%), might have type 0 (25%), unlikely to have type 2 (5%).' It doesn't just guess — it shows the odds. That makes it honest about what it doesn't know. And it works surprisingly well even with very little data.",
		simple_strengths: [
			'Shows odds instead of just one guess',
			"Honest about what it doesn't know — admits uncertainty",
			'Works well even with tiny datasets',
			'Can use what you already know about genes',
			'Most reliable when a wrong answer is serious'
		],
		simple_use_case:
			"Pick this when you need to trust the model. You always know how confident it is, and it admits when it's unsure. Best for small datasets or when a wrong answer could matter. Takes longer to train, but that extra caution is worth it when the stakes are high.",
		description:
			'A fully Bayesian multinomial logistic regression model implemented in PyMC and sampled via the No-U-Turn Sampler (NUTS) with JAX as the computational backend. Rather than optimising a single point estimate of model weights, NUTS draws samples from the full joint posterior P(weights | data) using Hamiltonian Monte Carlo dynamics. Priors are placed over all weight matrices: Normal(0, 1) for feature weights and Normal(0, 0.5) for per-generation group-level intercepts (hierarchical structure). The likelihood is a categorical distribution parameterised by softmax over three dosage classes. By default the model runs 4 parallel chains with 1,000 tuning steps and 500 draw steps each, producing 2,000 posterior samples. At inference time, predictions are made by averaging the softmax outputs across all posterior samples, yielding calibrated class probabilities. Convergence is assessed via R-hat (target < 1.01) and effective sample size (ESS) diagnostics computed by ArviZ.',
		strengths: [
			'Produces a full posterior distribution over model weights — every prediction comes with a credible interval',
			'NUTS sampler is geometrically ergodic — guaranteed to converge to the true posterior given sufficient samples',
			'Hierarchical priors over generation-level intercepts regularise predictions for generations with few observations',
			'R-hat and ESS diagnostics provide objective convergence evidence unavailable in frequentist models',
			'Posterior predictive averaging produces better-calibrated class probabilities than a single MAP estimate',
			'Robust to small sample sizes due to prior regularisation',
			'JAX backend enables GPU/TPU-accelerated HMC with just-in-time compilation'
		],
		use_case:
			'The strongest choice when statistically rigorous uncertainty quantification is required: credible intervals on dosage probabilities, posterior predictive checks, or sensitivity analysis over prior choices. Hierarchical structure makes it particularly well-suited to datasets spanning multiple generations with unequal sample sizes per generation. Slower than all other models by a significant margin — expect minutes to hours on medium/large datasets without GPU acceleration. Not recommended when training speed is a hard constraint or when the dataset is large enough that the DNN or GNN provides equivalent accuracy without the sampling overhead.'
	}
};

const MODEL_SIZE_INFO: Record<
	string,
	{
		simple: string;
		nerd: string;
	}
> = {
	tiny: {
		simple: 'This model was trained on a small simulated family — 250 individuals spread across 5 generations, with 100 genetic sites measured per person. Think of it as a proof-of-concept: small enough to train in seconds, useful for quickly checking whether the model is working as expected. Because the dataset is so compact, the model has seen fewer examples and may not generalize as well to larger, more complex families.',
		nerd: '250 diploid individuals · 100 genomic sites · 5 generations · 50 samples per generation. Minimal-scale dataset intended for pipeline validation and rapid iteration. Limited feature diversity; expect higher variance in model performance. Useful for debugging and sanity checks but not representative of real-world cohort complexity.'
	},
	small: {
		simple: 'This model was trained on a moderate-sized simulated family — 1,000 individuals across 10 generations, with 1,000 genetic sites per person. This is a solid starting point: the model has seen enough variation to learn meaningful patterns, but training still completes quickly. It strikes a good balance between speed and reliability for most experiments.',
		nerd: '1,000 diploid individuals · 1,000 genomic sites · 10 generations · 50 samples per generation. Compact but representative dataset. Sufficient feature diversity for most model architectures to learn stable decision boundaries. Recommended for exploratory comparisons between model types before scaling up.'
	},
	medium: {
		simple: 'This model was trained on a large simulated family — 5,000 individuals across 25 generations, with 10,000 genetic sites per person. At this scale, the model has been exposed to a wide range of family structures and inheritance patterns. Training takes longer, but the result is a model that can handle more complex scenarios and is generally more accurate.',
		nerd: '5,000 diploid individuals · 10,000 genomic sites · 25 generations · 100 samples per generation. Full-scale dataset that exercises model capacity meaningfully. Captures richer linkage structure and deeper pedigree relationships. Appropriate for benchmarking all five architectures under realistic conditions.'
	},
	large: {
		simple: 'This model was trained on a very large simulated family — 25,000 individuals across 50 generations, with 100,000 genetic sites per person. This is the most demanding scale in the benchmark, designed to reflect the complexity of real-world population studies. Models trained here have seen an enormous range of genetic variation and are the most thoroughly tested — though they also take the longest to run.',
		nerd: '25,000 diploid individuals · 100,000 genomic sites · 50 generations · 500 samples per generation. Large-scale dataset designed to stress-test model capacity and expose performance degradation under high-dimensional inputs. Reflects realistic population cohort complexity. GPU acceleration strongly recommended for DNN, GNN, and Bayesian models at this scale.'
	}
};

const FALLBACK_INFO = {
	label: 'Custom Model',
	simpleDescription: 'This is a custom model. Refer to the model documentation for details on how it works.',
	simpleStrengths: ['User-defined architecture'],
	simpleUseCase: 'Application-specific. Consult the model configuration for details.',
	description: 'This model type does not have a built-in description. Refer to the model documentation for details.',
	strengths: ['User-defined architecture'],
	useCase: 'Application-specific. Consult the model configuration for details.'
};

export default function ModelDashboard({ model, nerdMode, onNerdModeChange }: ModelDashboardProps) {
	const info = MODEL_TYPE_INFO[model.model_type] ?? FALLBACK_INFO;
	const sizeKey = model.model_name?.toLowerCase() ?? '';
	const sizeBlurb = MODEL_SIZE_INFO[sizeKey];

	return (
		<div className="dashboard-wrapper">
			{/* Identity */}
			<div className="grid-2col-mb">
				<div className="info-card">
					<p className="info-card-label">Model Size</p>
					<p className="info-card-value">
						{model.model_name ? model.model_name.charAt(0).toUpperCase() + model.model_name.slice(1) : model.model_name}
					</p>
				</div>
				<div className="info-card">
					<p className="info-card-label">Model Type</p>
					<p className="info-card-value">{info.label}</p>
				</div>
			</div>

			{/* Dataset size blurb */}
			{sizeBlurb && (
				<div className="section-mb">
					<h2 className="section-heading">Training Dataset</h2>
					<p className="context-text">{sizeBlurb.simple}</p>
					{nerdMode &&
						(() => {
							const parts = sizeBlurb.nerd.split('.');
							const stats = parts[0].split('·').map((s) => s.trim());
							const description = parts.slice(1).join('.').trim();
							return (
								<>
									<div className="nerd-text">
										<ul className="nerd-stats-list">
											{stats.map((stat) => (
												<li key={stat}>{stat}</li>
											))}
										</ul>
									</div>
									{description && <p className="nerd-description-para context-text">{description}</p>}
								</>
							);
						})()}
				</div>
			)}

			{/* Training pipeline */}
			<div className="section-mb">
				<h2 className="section-heading">{nerdMode ? 'Training Pipeline' : 'How the Model Was Trained'}</h2>
				{nerdMode ? (
					<>
						<p className="context-text">
							Training follows a 3-phase pipeline applied to three dataset splits derived from the simulation output.
						</p>
						<p className="context-text">
							<strong>Phase 1 — Initial Training:</strong> The model is fit on the training split using the full labelled feature
							matrix. For each masked individual at each genomic site, the input feature vector is{' '}
							<code>[mean_dosage_of_k_hop_relatives, fraction_of_relatives_observed, count_of_relatives]</code>, constructed via k-hop
							relative aggregation over the pedigree graph. The target is allele dosage (0, 1, or 2).
						</p>
						<p className="context-text">
							<strong>Phase 2 — Cross-Validation &amp; Retraining:</strong> The trained model is evaluated on the validation split using
							5-fold cross-validation (KFold, shuffle=True, random_state=123). Per-fold metrics are averaged to produce a robust
							estimate of generalisation performance. The model is then retrained from scratch on the combined train + validation data
							to maximise the information available before final testing.
						</p>
						<p className="context-text">
							<strong>Phase 3 — Final Evaluation:</strong> The retrained model is applied once to a held-out test split that was never
							seen during training or cross-validation. Metrics reported here (precision, recall, F1, ROC/PR AUC, confusion matrix) are
							all computed from this final phase.
						</p>
						<p className="context-text">
							Results are cached per (model_name, model_type, test_dataset) tuple and served from the log on subsequent requests to
							avoid redundant recomputation.
						</p>
					</>
				) : (
					<p className="context-text">
						We hid some people in the dataset and told the model to predict their genetics based on their family. The model learned from
						the people we showed it, then made guesses about the hidden people. We checked those guesses against the real answers.
					</p>
				)}
			</div>

			{/* Description */}
			<div className="section-mb">
				<h2 className="section-heading">About this Model</h2>
				{(() => {
					const text = nerdMode ? info.description : info.simple_description;
					// Split on periods followed by space, then render as separate paragraphs
					const sentences = text.split(/(?<=[.!?])\s+/);
					return sentences.length > 2 ? (
						sentences.map((sentence, idx) => (
							<p key={idx} className="context-text">
								{renderTextWithVarHighlight(sentence, nerdMode)}
							</p>
						))
					) : (
						<p className="context-text">{renderTextWithVarHighlight(text, nerdMode)}</p>
					);
				})()}
			</div>

			{/* Strengths */}
			<div className="section-mb">
				<h2 className="section-heading">Strengths</h2>
				<ul className="strengths-list context-text">
					{(nerdMode ? info.strengths : info.simple_strengths).map((s) => (
						<li key={s}>{s}</li>
					))}
				</ul>
			</div>

			{/* Use case */}
			<div className="use-case-box">
				<h2 className="use-case-label">{nerdMode ? 'Recommended Use Case' : 'Best For'}</h2>
				<p className="context-text">{nerdMode ? info.use_case : info.simple_use_case}</p>
			</div>
		</div>
	);
}
