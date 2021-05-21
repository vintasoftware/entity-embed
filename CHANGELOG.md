# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## 0.0.5 (2021-05-20)

### Fixed

- Fix clamping on `MaskedAttention` and `MultitokenAvgEmbed` to small value less than 1. That's the proper behavior to re-scale attentions that sum to less than 1 and ignore ones that sum to 0. This was only causing a minor decrease in F1 score, though.

### Added

- Add --use_gpu option to CLI (before it would always use a GPU if available)
- Colab notebooks (see README)
- Conda compatibility (see README)

### Changed

- Simplify `fix_pool_weights` code. Same behavior.

## 0.0.4 (2021-04-20)

### Fixed

- Fixed `field_mask` on `FieldsEmbedNet` by clamping values to 1. Before, this mask was multiplying field embeddings by the field length in tokens. Now, the correct behavior is implemented: multiply by 0 the empty fields, and by 1 the non-empty fields. This was only causing a minor decrease in F1 score, though.

## 0.0.3 (2021-04-20)

### Added

- Example on how to do pairwise matching of candidate pairs at `notebooks/End-to-End-Matching-Example.ipynb`.
- Enable return of `field_embedding_dict` from `BlockerNet` for assisting pairwise matching. Use `return_field_embeddings` parameter.
- Enable return of attention scores for interpretation from `MultitokenAttentionEmbed`. Use `_forward` method.

### Changed

- Use of `LayerNorm` in `EntityAvgPoolNet` instead of `F.normalize`, it's less "esoteric".
- Zeroing of empty field embeddings in `FieldsEmbedNet` instead of `BlockerNet`.

## 0.0.2 (2021-04-06)

### Added

- Documentation.
- `example-data/` in repo.

### Changed

- Simpler API for validation and test.
- Better naming of various API objects and methods.
- Consider -1 in min_epochs since epochs start from 0.
- Upgrade pytorch-metric-learning to 0.9.98.

## 0.0.1 (2021-03-30)

- First release on PyPI.
