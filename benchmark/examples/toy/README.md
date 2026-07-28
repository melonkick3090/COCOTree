# Toy fixture

This one-image fixture is intentionally small and uses readable uncompressed
COCO RLE. It checks that prediction-local IDs need not match reference IDs.

`perfect` must score 1.0 for HPQ, OTQ, TQ, BQ, meanNQ, MQ, and LQ under
`configs/toy_exact_v1.json`. `rewired` preserves masks and labels but changes
the wheel parent, so MQ and LQ remain 1.0 while tree-sensitive scores fall.

