# Vector analogies in gene embeddings


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

This repo contains an interactive Python program for the automated search of analogies in gene embeddings.

This script will generate analogies by searching for the closest vector in the embedding space to the vector resulting from the subtraction of two other vectors. The script will output the top N closest vectors to the resulting vector.

To run this in an interactive mode, use the following command:

```bash
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