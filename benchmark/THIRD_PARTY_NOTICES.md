# Third-Party Notices

This notice distinguishes the benchmark source license, dataset license,
runtime dependencies, and pretrained label model. Each component remains under
its own license.

## COCOTree benchmark source

Copyright (c) COCOTree authors.

The source code in this benchmark package is released under the MIT License.
See the repository `LICENSE` file for its full terms.

The dataset-construction project and project history are available at:

- <https://github.com/melonkick3090/COCOTree>

## COCOTree dataset

The COCOTree dataset is released under the Creative Commons Attribution 4.0
International license (CC BY 4.0):

- <https://creativecommons.org/licenses/by/4.0/>

The dataset license is separate from the evaluator's MIT source license.
Redistributing or publishing results based on the dataset requires the
attribution specified by the dataset release. Consult the release documentation
for the preferred citation and attribution text.

## Label-similarity model

The paper LQ backend uses:

- `sentence-transformers/all-MiniLM-L6-v2`
- Model page: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- Declared model license: Apache License 2.0

The model is downloaded or resolved from the user's cache at runtime; its
weights are not relicensed by this repository. Users are responsible for
complying with the model license and the terms applicable to their distribution
or deployment.

## Principal Python dependencies

The benchmark may use the following third-party packages. Versions and
transitive dependencies are defined by the installed environment.

| Package | Upstream project | Common upstream license |
|---|---|---|
| NumPy | <https://numpy.org/> | BSD 3-Clause |
| SciPy | <https://scipy.org/> | BSD 3-Clause |
| pycocotools | <https://github.com/cocodataset/cocoapi> | Simplified BSD |
| PyTorch | <https://pytorch.org/> | BSD-style |
| Sentence Transformers | <https://www.sbert.net/> | Apache 2.0 |
| Transformers | <https://github.com/huggingface/transformers> | Apache 2.0 |

This table is a convenience summary, not a replacement for the license files
distributed by each package. The precise dependency graph and license terms
are those of the versions installed for a given run.

## No transfer of rights

Names, logos, model weights, dataset content, and third-party packages are not
covered by the benchmark source-code license unless their own license
explicitly says so. No endorsement by the upstream projects is implied.
