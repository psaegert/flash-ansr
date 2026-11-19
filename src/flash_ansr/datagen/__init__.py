"""Split, throughput-oriented generation of fitted candidate pools (training data for front-directed fine-tunes).

Three stages that run on different hardware and meet only through files:

1. :mod:`flash_ansr.datagen.sample` (GPU nodes): draw training-prior problems, encode them, and decode ``K`` raw
   candidate sequences per problem, packing many problems into one static decode (per-row encoder memory). No
   CPU post-processing on the GPU node.
2. :mod:`flash_ansr.datagen.fit` (CPU nodes): post-process the raw draws exactly as the deployed sampler does
   (extract, simplify, deduplicate), then refit, ladder and rank them with the unchanged deployed pipeline
   (:meth:`FlashANSR._fit_refine` / :meth:`FlashANSR._build_result`), score every candidate on fresh validation
   points, keep the candidates that beat the mean on fresh points, and trace every raw draw to its family.
3. :mod:`flash_ansr.datagen.targets` (CPU): the front of each pool (members = distinct raw expressions) and the
   training targets: every front member at 1/|front| as the model emitted it, and the matched self-distillation
   control. Training reads the target shards.

Problems come from the model's own training prior (its ``catalog_train.yaml`` / ``dataset_train.yaml``), with 512
clean validation rows per problem for fresh-point scoring. File formats: :mod:`flash_ansr.datagen.shards`.
"""
