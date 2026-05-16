# HCMP v1: A More Conservative Approach to Molecular Graph Pretraining

> **Hierarchical Conservative Molecular Pretraining**  
> A molecular graph pretraining framework that tries to avoid contaminating embeddings with overly biased pretraining objectives.

## Preface

In molecular machine learning, pretraining has almost become a default option.

The hope is that a model can first learn some form of “chemical representation” from a large number of unlabeled molecules, and then transfer that representation to downstream tasks: solubility, toxicity, pKa, logP, activity prediction, ADMET, or broader virtual screening problems.

But the question is: **what exactly should pretraining teach the model?**

Many molecular pretraining methods naturally move in one direction: introducing more tasks, more labels, more cheminformatics empirical formulas, more computed chemical properties, and more motif or property prediction objectives. Intuitively, this seems to allow the model to learn more “chemical knowledge.”

However, I have become increasingly skeptical of one assumption: **a higher-information embedding is not necessarily a better embedding.**

A good molecular embedding should not be overly shaped during pretraining by a particular empirical formula, a specific task preference, or a certain type of label noise. Instead, it should be a structural representation that can be effectively connected to a GNN and achieve strong performance after finetuning on concrete downstream tasks.

In other words, the goal of pretraining is not to turn the embedding itself into a complex property predictor. The goal is to make it a clean, robust, and transferable structural foundation.

This is the central motivation behind HCMP v1:

> Pretraining objectives should not heavily depend on cheminformatics empirical formulas. They should be as graph-based, conservative, and low-noise as possible, while avoiding contamination of the embedding by the pretraining task itself.

---

## 1. Four Types of Pretraining Tasks I Want to Avoid

Before designing HCMP, I first clarified one point: some tasks may look “information-rich,” but they are not necessarily suitable as pretraining objectives.

I mainly want to avoid four types of tasks.

### 1.1 Tasks that rely too heavily on empirical formulas

The first type is tasks that rely too heavily on cheminformatics empirical formulas.

For example, directly asking a model to fit certain empirical properties, scoring functions, or highly hand-designed chemical priors can be risky. The embedding may become biased toward fitting these formulas, rather than learning a more general molecular graph representation.

This does not mean empirical formulas are useless. They are certainly useful. But if the pretraining objective itself strongly depends on these formulas, the representation learned by the model may carry strong formula-specific bias.

For downstream tasks, this kind of bias is not always beneficial.

### 1.2 Tasks with unclear semantics and questionable transfer value

The second type is tasks whose semantic targets are unclear and whose transfer value is questionable, such as direct functional group identity prediction or motif identity prediction.

The problem is that **predicting the name of a local motif is not necessarily a meaningful pretraining objective.**

For example, asking a model to learn whether a benzene ring is attached to a nitro group or an amino group is not particularly meaningful by itself. If the input molecular graph already contains atom and bond types, local structures such as nitro and amino groups are almost directly readable. Learning to name them does not necessarily mean the model has learned a more transferable molecular representation.

Such tasks can easily become local pattern recognition:

- seeing an N/O connectivity pattern and predicting nitro;
- seeing an N-H or N-C connectivity pattern and predicting amino;
- seeing C=O-N and predicting amide.

However, the ability to recognize the names of local fragments may not align with the molecule-level representation required by downstream benchmarks.

HCMP does not aim to make the model memorize a motif vocabulary. Instead, I want the model to learn from graph structure:

- how local chemical structures are organized;
- which bonds act as structural boundaries;
- which local structures should be preserved as coherent units;
- how functional-group-like structures naturally emerge from molecular graphs.

Therefore, the second task in HCMP is not motif classification, but conservative cut-bond segmentation.

It does not ask the model to answer “what functional group is this?” Instead, it asks the model to learn “where stable local chemical structure boundaries exist in this molecular graph.”

### 1.3 Tasks that are too simple

The third type is overly simple tasks.

Examples include atom count, heavy atom count, molecular weight, exact molecular weight, and raw ring count.

These tasks are too easy for a model to solve through shortcuts. The model may simply learn that “larger molecules have larger values,” rather than learning genuinely useful chemical structural patterns.

Such tasks may look stable and easy to optimize, but their contribution to representation learning may be very limited. They may even push the model toward the wrong representational direction.

### 1.4 Tasks that are too heavy and difficult to scale

The fourth type is computationally heavy tasks, especially those that rely heavily on computational chemistry.

DFT labels, quantum chemical properties, conformer searches, and complex electronic structure calculations certainly contain rich chemical information. But the problem is that they are difficult to scale to a sufficiently large molecular corpus.

If a pretraining task itself requires extremely high computational cost, it is not suitable as the default foundation for large-scale molecular pretraining.

The goal of HCMP is not to design a small, highly refined, expensive pretraining task. The goal is to design a conservative training framework that can gradually scale to the ChEMBL level.

---

## 2. The Core Position of HCMP: Conservative Does Not Mean Low-Information, but Low-Contamination

The full name of HCMP is **Hierarchical Conservative Molecular Pretraining**.

Here, “conservative” does not mean weak or information-poor. It means:

> using only relatively reliable, structure-derived, low-noise, interpretable weak supervision signals that are unlikely to train the embedding into a specific formula-fitting machine.

I do not think the goal of embedding design is to maximize information content during pretraining.

More precisely, the goal of embedding design should be:

> after being connected to a GNN and finetuned for concrete downstream tasks, the embedding should enable the strongest and most robust performance.

This means pretraining objectives need restraint.

They should help the model learn:

- local chemical rules;
- local chemical structures;
- functional-group-like boundaries;
- structural relationships between molecular graph backbones;
- sufficiently coarse-grained and robust global chemical directions.

But they should not tell the model too early that a molecule should be understood like a certain empirical formula, or that the embedding should be organized like a particular property predictor.

---

## 3. Overall Design: Hierarchical Training from Atoms to Functional Groups to Molecules

The core training objective of HCMP v1 consists of four parts:

```text
L_total = lambda_1 L_BERT
        + lambda_2 L_cut_seg
        + lambda_3 L_prop_rank
        + lambda_4 L_scaf_triplet
```

The four tasks are activated sequentially, forming a hierarchical training process from atoms to local structures and then to the molecular level.

| Stage | Level | Loss | Design purpose | Mainly graph-based? |
| --- | --- | --- | --- | --- |
| Stage 1 | atom/bond | naive atom/bond BERT | Learn local chemical rules | Yes |
| Stage 2 | bond/subgraph | cut-bond segmentation | Learn local chemical structures and functional-group boundaries | Yes |
| Stage 3A | molecule | descriptor threshold pairwise ranking | Learn global chemical directions under weak supervision | No, uses RDKit descriptors |
| Stage 3B | scaffold/molecule | scaffold triplet ranking | Learn scaffold distance geometry | Yes |

It is easy to see that Tasks 1, 2, and 4 are entirely graph-based.

This is the main line of HCMP: as much as possible, pretraining should be built on the molecular graph itself rather than on empirical formulas.

However, fully graph-based tasks may still be insufficiently informative. If a model only learns local atom/bond reconstruction, structural segmentation, and scaffold distance, it may still lack sensitivity to chemical directions related to electronic properties, solubility, polarity, hydrophobicity, charge distribution, and similar global trends.

Therefore, HCMP introduces the third task: selecting a conservative set of RDKit descriptors and using them for threshold pairwise ranking.

This is not intended to turn the embedding into an RDKit descriptor fitter. Instead, it provides the model with some global chemical directions under weak supervision.

---

## 4. Task 1: Naive Atom/Bond BERT — Learning Local Rules from Simple One-Hot Inputs

The first task in HCMP is naive BERT.

The word “naive” is intentional: the model uses the simplest one-hot atom/bond inputs and predicts the types of masked atoms and bonds.

This task is not meant to be fancy.

Its purpose is very clear:

> to let the program first learn local chemical rules.

For example:

- what types of atom neighborhoods are more likely to correspond to certain atom types;
- what local structures correspond to single, double, or aromatic bonds;
- how local patterns such as formal charge, aromaticity, hybridization, and ring membership constrain one another;
- how the type of an atom or bond can be inferred from its surrounding graph environment.

This task is local, graph-based, and low-noise.

It does not force the model to learn a particular downstream property, nor does it inject complex empirical formulas. It only allows the model to master basic chemical syntax from a large number of molecular graphs.

This is also why I insist on using the simplest one-hot inputs.

If the input embedding already contains too many hand-designed strong features, the model may consume artificial features rather than learn from graph structure. HCMP aims to let the model form useful representations through GNN message passing itself.

---

## 5. Task 2: Conservative Cut-Bond Segmentation — Letting the Model Learn Local Chemical Structures

The second task is a molecule segmentation strategy designed to align with human chemical intuition.

But it is not full functional group prediction.

I intentionally avoid direct functional group classification or motif prediction, because such tasks often only ask the model to recognize the names of local patterns, rather than learn molecular representations with genuine transfer value.

HCMP v1 uses bond-level cut prediction:

```text
y_b_cut = 1 if bond b is a boundary between chemically salient segments,
          0 otherwise.
```

The model only needs to determine whether a bond is a stable and interpretable chemical boundary.

This is more conservative than directly predicting motif identity, and it better matches my goal for pretraining.

It does not ask the model to memorize that “this fragment is called nitro” or “that fragment is called amino.” Instead, it asks the model to learn:

- which structures should remain internally intact;
- which bonds connect different chemical units;
- which positions behave like functional group boundaries;
- how rings, unsaturated motifs, heteroatom clusters, and terminal heteroatoms form local chemical structures.

In other words, the goal of this task can even be understood as:

> letting the model learn functional groups from scratch through graph structure.

### 5.1 Priority-based segmentation rule

The current segmentation rule identifies salient motifs by priority:

1. **Ring-system segments**  
   Ring systems are protected first, avoiding cuts inside aromatic rings, fused rings, bridged rings, spiro rings, and other stable ring structures.

2. **Non-ring multiple-bond-seeded, RDKit-conjugation-expanded motifs**  
   Starting from non-ring multiple bonds, the rule expands along RDKit-perceived conjugated bonds to capture carbonyls, alkenes, alkynes, nitriles, enones, amide-like motifs, and ester-like motifs.

3. **Heteroatom connected clusters**  
   This captures nitro-like, peroxide-like, azide-like, sulfur-oxygen, phosphorus-oxygen, and other heteroatom-connected clusters.

4. **Terminal heteroatom segments**  
   This captures remaining terminal heteroatom substituents, such as hydroxyl, thiol, amino, and halogen groups.

Finally, if a bond connects two different assigned segments, or connects an assigned segment with the background skeleton, it is labeled as a cut bond.

If a bond lies inside the same segment, or connects background to background, it is labeled as non-cut.

The key feature of this rule is conservatism:

- do not cut inside rings;
- do not cut inside conjugated motifs;
- do not force cuts through every saturated heteroatom linker;
- do not force every atom into a traditional functional group name;
- provide supervision only when the boundary is stable and graph-structurally clear.

This prevents motif identity prediction from degenerating into a local naming task, while still providing rich local chemical structural signals.

---

## 6. Task 3: RDKit Descriptor Threshold Pairwise Ranking — Learning Global Chemical Directions under Weak Supervision

Among the four tasks, Tasks 1 and 2 are local or mesoscale graph tasks, and Task 4 is a scaffold graph-distance task. They are all conservative and highly graph-based.

But relying entirely on graph tasks may introduce another problem: insufficient global chemical property signals.

For example, it may be difficult for a model to learn only through BERT and segmentation:

- which molecule is more hydrophobic;
- which molecule is more polar;
- which molecule has a more dispersed charge distribution;
- which molecule is more aromatic or conjugated;
- which molecule is more flexible;
- which structural trends relate to solubility or electronic properties.

Therefore, HCMP introduces a third task: selecting 14 RDKit descriptors and using them for pairwise ranking.

There are two key points here.

First, descriptor selection should be conservative.

I try to avoid:

- atom count;
- molecular weight;
- heavy atom count;
- raw count shortcuts;
- stronger and more biased labels such as pKa, logS, QED, and SA score;
- computational chemistry labels.

Second, the supervision format should not be direct regression.

If the model directly regresses RDKit descriptors, the embedding may become a fitter of empirical formulas. This is exactly what I want to avoid.

Therefore, HCMP uses threshold pairwise ranking.

For descriptor q_k, two molecules G_i and G_j are randomly sampled, and their difference is computed:

```text
Delta q_k = q_k(G_i) - q_k(G_j)
```

If the difference is not large enough, the molecular pair is discarded:

```text
|Delta q_k| < tau_k
```

Only when the difference falls into the top 30% is it used as a supervision signal:

```text
|Delta q_k| >= tau_k
```

The model only learns direction: which molecule has a larger descriptor value.

The principle behind this is:

> do not trust the exact numerical value of an empirical formula; only trust the direction when the difference is large enough.

Or more directly:

> I introduce RDKit descriptors not to make the embedding fit RDKit, but to supplement global chemical directions through sufficiently robust pairwise signals.

This is the core meaning of the threshold pairwise design.

It tries to find a compromise between two extremes:

- using no descriptors at all may provide insufficient information;
- directly regressing descriptors may contaminate the embedding;
- using only top-30% pairwise directions preserves part of the global chemical signal while reducing the risk of formula fitting.

---

## 7. Task 4: Scaffold Triplet Ranking — Learning Scaffold Geometry in a Tunable Distance Space

The fourth task comes from a tunable distance I previously designed in a split discussion project.

In that project, I focused on one issue: molecular dataset splitting is not neutral. Different split methods encode different assumptions about generalization, and scaffold split, Butina split, and random split all have clear biases.

Bemis-Murcko scaffold split has several typical problems:

- it overemphasizes rings;
- for acyclic molecules, it often produces coarse categories such as `[NO_SCAFFOLD]`;
- scaffold group distributions are often head-heavy with a long tail;
- scaffold split can easily create strong dataset splitting artifacts.

Therefore, I once designed a tunable molecular distance that combines scaffold distance and functional-group distance, with the goal of constructing more controllable splits.

In HCMP v1, I only use the scaffold distance part as a pretraining task.

The reason is that functional-group-level semantics have already been assigned to Task 2 segmentation, so there is no need to repeat them in molecule-level triplet learning.

### 7.1 Expanded scaffold

The scaffold used in HCMP is not the traditional Bemis-Murcko scaffold.

It uses expanded scaffold extraction, retaining:

- all carbon atoms;
- all ring atoms;
- all ring bonds;
- all multiple bonds and their endpoint atoms;
- deterministic shortest paths connecting retained components.

The resulting scaffold is closer to an expanded backbone: it preserves ring systems, carbon skeletons, multiple-bond systems, and key connection paths.

### 7.2 Iterative MCS scaffold distance

Similarity between two expanded scaffolds is computed using iterative bond-based MCS.

The rough procedure is:

1. run exact atom-type and exact bond-order MCS;
2. find a deterministic match;
3. mask the matched parts;
4. repeat for multiple rounds;
5. normalize by the total number of matched bonds to obtain scaffold similarity.

The distance is defined as:

```text
D_scaffold = 1 - S_scaffold
```

This distance is not a physicochemical property distance, nor is it a biological activity distance.

It is only an interpretable, representation-independent, scaffold-level structural distance.

### 7.3 Triplet supervision

For an anchor G_a, a positive G_p, and a negative G_n, a triplet is kept only when scaffold distance clearly distinguishes the positive from the negative:

```text
D_scaffold(G_a, G_p) + tau_D < D_scaffold(G_a, G_n)
```

The model learns to make the embedding space satisfy:

```text
d_theta(G_a, G_p) < d_theta(G_a, G_n)
```

This follows the same idea as Task 3: supervision is provided only when the structural distance difference is sufficiently clear, so as to avoid ambiguous signals.

---

## 8. Curriculum: Activating the Four Tasks Sequentially

The four tasks in HCMP are not trained from the beginning all at once. Instead, they are activated sequentially.

| Phase | Active losses | Purpose |
| --- | --- | --- |
| Phase 1 | BERT only | Learn local atom/bond rules first |
| Phase 2 | BERT + cut segmentation | Then learn local chemical structures and boundaries |
| Phase 3 | BERT + cut segmentation + descriptor ranking | Add global chemical directions |
| Phase 4 | full loss with scaffold triplet | Add molecular scaffold geometry |

This forms a hierarchical training process from atoms to functional groups and then to molecules.

I think this order is important.

If all molecule-level losses are activated from the very beginning, the model may not yet have stable local representations and may be pushed too early by descriptor ranking or scaffold triplet objectives.

The HCMP curriculum tries to let the model first acquire local chemical syntax, then learn local structural boundaries, and only after that learn global directions and scaffold geometry.

---

## 9. An Important Engineering Principle: All Losses Use the Same Molecule Batch

The training framework of HCMP v1 also follows an important engineering principle:

> all loss components in one training step are computed from the same molecule batch.

That is, a batch enters the model and is forwarded only once. The following losses are then computed simultaneously:

- BERT loss;
- cut segmentation loss;
- descriptor pairwise logistic loss;
- scaffold triplet loss.

This avoids the confusion caused by multiple independent dataloaders sampling different batches, and it makes the multitask signals in each step more consistent.

In the current implementation:

- descriptor pairs are dynamically sampled within the batch;
- scaffold triplets are also sampled from candidates within the batch;
- scaffold distances are queried through a backend/cache;
- for each anchor, only a limited number of partners or candidate pairs are sampled, rather than exhaustively enumerating all batch-internal pairs.

This design is important for large-scale cloud training, because a complete distance matrix or exhaustive triplet mining inside a batch can easily explode in cost.

---

## 10. Current Code Status: From Design to Runnable Experiments

HCMP v1 is no longer just a design document. It is an engineering-oriented research codebase. The main modules include:

- ChEMBL cleaning;
- descriptor computation;
- descriptor threshold estimation;
- priority-based segmentation;
- segmentation visualization;
- expanded scaffold extraction;
- scaffold distance cache;
- graph cache precomputation;
- HCMP model;
- edge-aware graph transformer encoder;
- multi-objective trainer;
- convergence curriculum;
- checkpoint saving;
- downstream finetuning matrix;
- Graph-BERT baseline;
- ablation config generation.

---

## 11. Evaluation: Benchmark Scores

The evaluation goal of HCMP is actually straightforward: **whether it can bring better benchmark performance.**

I do not only want to prove that HCMP is “more interpretable” or “more chemically reasonable.” Those are only design motivations. Ultimately, it must return to the most basic question in molecular machine learning:

> can conservative pretraining make a GNN perform better on downstream benchmarks?

My core hypothesis is:

> a more conservative, more graph-based, and less embedding-contaminating pretraining strategy may be more suitable for downstream finetuning than rich-information pretraining, and therefore may lead to better benchmark performance.

### 11.1 Ten pretraining / initialization settings

The experimental design compares ten settings in total.

The first group is HCMP-style conservative pretraining.

HCMP has a basic BERT task, plus three additional signals:

- cut-bond segmentation;
- descriptor threshold pairwise ranking;
- scaffold triplet ranking.

These three additional signals can be combined arbitrarily. Therefore, there are 2^3 = 8 conservative pretraining settings:

| ID | Pretraining setting |
| -- | --- |
| H0 | BERT only |
| H1 | BERT + cut segmentation |
| H2 | BERT + descriptor ranking |
| H3 | BERT + scaffold triplet |
| H4 | BERT + cut segmentation + descriptor ranking |
| H5 | BERT + cut segmentation + scaffold triplet |
| H6 | BERT + descriptor ranking + scaffold triplet |
| H7 | BERT + cut segmentation + descriptor ranking + scaffold triplet |

The second group consists of two controls:

| ID | Baseline |
| -- | --- |
| B0 | no pretraining / scratch |
| B1 | traditional rich-information BERT pretraining |

Thus, there are ten pretraining / initialization settings in total.

### 11.2 Unified GNN finetuning

All pretraining settings are ultimately connected to the same downstream GNN model and finetuned on the same benchmarks.

This is important.

What I really care about is not how beautiful the loss curve of a pretraining task looks, nor how well the embedding fits the pretraining objective. The real question is:

> after this embedding is connected to a GNN, can it achieve better finetuning performance on concrete benchmarks?

In other words, the pretraining objective is only a means. Downstream benchmark performance is the final test.

### 11.3 Random split and scaffold split

Downstream evaluation uses two split strategies:

- random split;
- scaffold split.

Random split is used to test standard benchmark performance.

Scaffold split is used to test generalization under stronger structural shifts.

### 11.4 The questions I actually want to answer

This experiment ultimately aims to answer several direct questions:

1. Is BERT-only better than scratch?
2. Among the four conservative signals, which ones truly improve benchmark scores?
3. Do the four signals complement one another?
4. Does full HCMP outperform all partial combinations?
5. Does conservative pretraining outperform traditional rich-information BERT pretraining?
6. Does conservative pretraining provide better OOD performance?

If full HCMP or some partial HCMP settings win on benchmarks, then this conservative design is not only “philosophically cleaner,” but also genuinely useful for downstream performance.

If rich-information BERT pretraining wins, that is also valuable: it would suggest that stronger information injection may indeed be more effective for benchmarks, and that the conservatism of HCMP needs to be reevaluated.

Therefore, the evaluation of HCMP is not designed to prove a preset position. It is designed to directly compare:

> conservative graph-based weak supervision vs. rich-information pretraining vs. no pretraining.

---

## 12. The Research Viewpoint I Currently Want to Emphasize Most

HCMP is not only a model structure. It is also a position on molecular pretraining:

> pretraining objectives should not only pursue information content. They should pursue reliability, interpretability, scalability, and low contamination of the embedding.

This differs from many “stack more tasks” styles of pretraining.

The design of HCMP is more like decomposing molecular knowledge into several levels:

- local atom/bond syntax;
- local chemical structure;
- functional-group-like boundaries;
- robust global physicochemical directions;
- scaffold-level graph geometry.

Then, for each level, HCMP designs relatively low-noise weak supervision.

It is not the most aggressive pretraining method, but it may be a more robust starting point.

Therefore, the final goal of HCMP is not merely to propose a pretraining philosophy that sounds more reasonable. It is to answer a concrete question:

> If we use graph-based, conservative, structure-derived weak supervision as much as possible, and cautiously supplement it with a small amount of thresholded descriptor direction, can the model achieve better finetuning performance on real molecular benchmarks?

If the answer is yes, then conservative pretraining is not only a “cleaner” design, but also a practically effective benchmark strategy.

---

## 13. Next Steps

Next, I will focus on the following tasks:

1. **Complete ChEMBL-scale pretraining data preparation**  
   This includes cleaning, descriptor values, descriptor thresholds, graph cache, and scaffold distance cache.

2. **Complete the pretraining matrix**  
   This includes the Graph-BERT baseline, HCMP ablations, and full HCMP.

3. **Run small- to medium-scale sanity checks**  
   First verify loss curves, curriculum transitions, cache hit/miss behavior, checkpoints, and downstream loading, instead of starting with full-scale training immediately.

4. **Complete downstream benchmarks**  
   Compare the 8 HCMP conservative pretraining settings, traditional rich-information BERT pretraining, and no pretraining / scratch under random split and scaffold split.

5. **Write a method-paper-style summary**  
   The focus should not be to package this as “yet another pretraining model,” but to clearly explain why graph-based conservative weak supervision may be necessary, as well as the boundaries and limitations of each loss.

---

## Conclusion

HCMP v1 is still an ongoing research project.

It may not win on every benchmark at the beginning. Some of its designs may eventually prove to contribute little. But I think it raises a question worth testing seriously:

> Can molecular pretraining rely less on strong labels, empirical formulas, and noisy property priors, and instead rely more on graph-based, hierarchical, interpretable weak supervision, while still achieving better benchmark performance?

If the answer is yes, HCMP may be not only a model, but also a more effective molecular benchmark pretraining strategy.

If the answer is no, that would be equally valuable: it would tell us which supposedly “low-noise chemical signals” are not sufficient to support effective transfer, and which downstream properties indeed require stronger and more task-specific pretraining objectives.

Either way, the most important spirit of this project is not to “design a complex model,” but to:

> carefully distinguish whether the model is learning transferable molecular graph regularities, or merely empirical formulas, dataset biases, and task shortcuts.

This is also what I think makes molecular machine learning truly worth studying.
