# FLOWTRACER
FLOWTRACER training code and comparison against SOTA.

1. Download FLOWTRACER into your google drive. Make sure this location exists: My Drive/FLOWTRACER.
2. To run FLOWTRACER and other SOTA on the prepared benchmarks, open 2. SOTA Comparison.ipynb in your IDE and follow the instructions inside.
3. To see how FLOWTRACER is trained/ train FLOWTRACER yourself, open 1. FLOWTRACER Training.ipynb and follow the instructions inside.
4. Due to the large sizes, FLOWTRACER weights, Training Features and the benchmark images will be hosted on Hugging Face. The sources used in the experiment will be linked under Extras/Feature Extraction.ipynb. The code used to extract DINOv2 features are also included.
5. Ablation.ipynb and CLIP matching.ipynb are included in Extras for reference. Experiments requiring the CNN trajectory cache (5gb+) are omitted. Included are: 8D metrics vs CNN, FLOWTRACER vs DINOv2
