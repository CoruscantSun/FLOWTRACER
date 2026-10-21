# FLOWTRACER
FLOWTRACER training code and comparison against SOTA.

1. To run FLOWTRACER and other SOTA on the prepared benchmarks, open 2. SOTA Comparison.ipynb in your IDE and follow the instructions inside.
2. To see how FLOWTRACER is trained/ train FLOWTRACER yourself, open 1. FLOWTRACER Training.ipynb and follow the instructions inside.
3. Due to the large size of image datasets, FLOWTRACER weights and the benchmark images will be hosted on Hugging Face. The sources used in the experiment will be linked under Extras/Feature Extraction.ipynb. The code used to extract DINOv2 features are also included.
4. Ablation.ipynb and CLIP matching.ipynb are included in Extras for reference. Experiments requiring the CNN trajectory cache (5gb+) are omitted. Included are: 8D metrics vs CNN, FLOWTRACER vs DINOv2
