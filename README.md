# Analogy and similarity in gene embeddings


## Installation

Clone this environment:

```bash
git clone https://github.com/ZadorLaboratory/gene_analogies.git
cd gene_analogies
```

Install micromamba for package management:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Then create a new environment:

```bash
source create_env.sh analogies
```

## Usage

This repo contains an interactive Python program for the interrogation of gene embeddings.

Overall, there are two modalities for querying embeddings: by analogy, and by similarity. The analogy mode will find the closest vector in the embedding space to the vector resulting from the subtraction of two other vectors. The similarity mode will find the closest vector in the embedding space to a given vector.

### Similarity

Querying which genes have the most similar vectors to a given gene is fast and easy. To run this in an interactive mode, use the following command:

```bash
micromamba activate analogies
python pairwise_similarities.py --path "/home/benjami/gene_analogies/data/" --interactive --use-cosine --embeddings geneformer
```

This will load the gene embeddings and prompt you to enter a gene (using its EnsemblID). The script will then output the top N closest genes to the input gene. The results are also saved in a pretty human-readable markdown file with clickable links to the Protein Atlas.

This script can also be used to find the N closest genes overall. To do this, simply omit the `--interactive` flag.

#### Interpretation
These similarities might be called "Coexpression predictive embeddings" for Geneformer. In this case, genes take on similar embeddings when they are equally predictive of other genes' expression levels. For example, take the Globin family.

>#### Top 5 genes most similar to [ENSG00000206172](https://www.proteinatlas.org/ENSG00000206172) (hemoglobin subunit alpha 1):
>| Gene | Similarity | Description |
>|------|------------|-------------|
>| [ENSG00000188536](https://www.proteinatlas.org/ENSG00000188536) | 0.426356 | hemoglobin subunit alpha 2 |
>| [ENSG00000158578](https://www.proteinatlas.org/ENSG00000158578) | 0.406115 | 5'-aminolevulinate synthase 2 |
>| [ENSG00000143546](https://www.proteinatlas.org/ENSG00000143546) | 0.402645 | S100 calcium binding protein A8 |
>| [ENSG00000213934](https://www.proteinatlas.org/ENSG00000213934) | 0.397976 | hemoglobin subunit gamma 1 |
>| [ENSG00000206177](https://www.proteinatlas.org/ENSG00000206177) | 0.395949 | hemoglobin subunit mu |

Olfactory receptors provide another interesting example. 

>#### Top 10 genes most similar to ENSG00000172146 (olfactory receptor family 1 subfamily A member 1):
>| Gene | Similarity | Description |
>|------|------------|-------------|
>| ENSG00000203663 | 0.908487 | olfactory receptor family 2 subfamily L member 2 |
>| ENSG00000184788 | 0.904892 | spermidine/spermine N1-acetyl transferase like 1 |
>| ENSG00000213799 | 0.904683 | zinc finger protein 845 |
>| ENSG00000104848 | 0.902753 | potassium voltage-gated channel subfamily A member 7 |
>| ENSG00000185385 | 0.899623 | olfactory receptor family 7 subfamily A member 17 |

These examples represent the rather rare situation when a family of genes is expressed in a very narrow set of tissues. See also Crystallins (lens proteins), pancreatic enzymes like trypsinogen, and heat shock proteins like HSP70 (ENSG00000204389).

In most other genes, the embeddings are less obviously interpretable by function. This is (likely) because most genes are expressed in many tissues, yet often we are searching for the similar genes in a particular functional context rather than all contexts taken together.

For example, take the gene [Syntaxin-1A](https://www.proteinatlas.org/ENSG00000106089), which is key for synaptic vesicle docking prior to release. One might expect the most similar genes would have similar functions in synaptic vesicle release. However, I cannot make much sense of the most similar genes:
>#### Top 5 genes most similar to [ENSG00000106089](https://www.proteinatlas.org/ENSG00000106089) (syntaxin 1A):
>| Gene | Similarity | Description |
>|------|------------|-------------|
>| [ENSG00000186166](https://www.proteinatlas.org/ENSG00000186166) | 0.933148 | centrosomal AT-AC splicing factor |
>| [ENSG00000168661](https://www.proteinatlas.org/ENSG00000168661) | 0.932601 | zinc finger protein 30 |
>| [ENSG00000126821](https://www.proteinatlas.org/ENSG00000126821) | 0.917730 | sphingosine-1-phosphate phosphatase 1 |
>| [ENSG00000184545](https://www.proteinatlas.org/ENSG00000184545) | 0.914299 | dual specificity phosphatase 8 |
>| [ENSG00000197044](https://www.proteinatlas.org/ENSG00000197044) | 0.910157 | zinc finger protein 441 |

### Analogies

The `query_similar_pairs` script will generate analogies by searching for the closest vector in the embedding space to the vector resulting from the subtraction of two other vectors. The script will output the top N closest vectors to the resulting vector.

To run this in an interactive mode, use the following command:

```bash
micromamba activate analogies
python query_similar_pairs.py --index-file "/home/benjami/gene_analogies/data/geneformer/brain_diff_cosine_20k_index.faiss" --truncate 20000 --difference --path "/home/benjami/gene_analogies/data/" --use-cosine --cpu-only --embeddings geneformer
```

Note that this will take about 400GB of RAM to run, which is almost the entirety of the RAM available on the server. Please run `htop` to ensure that you have enough memory available before running this command. It will take about 5 minutes to load before you can start querying.

The script will then prompt you to enter two genes (using their EnsemblID), and will output the top N closest genes to the resulting vector.

The results are best viewed by opening the output file (called `output.md`) in a markdown viewer. In VSCode, you can use the Markdown Preview Enhanced extension. This will produce outputs that look like this:

> Query pair: ENSG00000107796 - ENSG00000171135
> Top 10 most similar pairs:

| Gene 1 | Gene 2 | ≈ | Gene 3 | Gene 4 | Similarity |
|--------|--------|---|--------|--------|------------|
| [ENSG00000149591](https://www.proteinatlas.org/ENSG00000149591) | [ENSG00000101473](https://www.proteinatlas.org/ENSG00000101473) | ≈ | [ENSG00000107796](https://www.proteinatlas.org/ENSG00000107796) | [ENSG00000171135](https://www.proteinatlas.org/ENSG00000171135) | 0.5397 |
| transgelin [Source:HGNC Symbol;Acc:HGNC:11553] | acyl-CoA thioesterase 8 [Source:HGNC Symbol;Acc:HGNC:15919] | | actin alpha 2, smooth muscle [Source:HGNC Symbol;Acc:HGNC:130] | jagunal homolog 1 [Source:HGNC Symbol;Acc:HGNC:26926] | |

## Data

The index file loaded by this script is a few hundred GB in size, and is not included in this repository. Currently it must be specified by absolute path in the `--index-file` argument. 

## Word2Vec analogies

To perform that same operation with Word2Vec embeddings, use the following command:

```bash
...
```