# FLOWTRACER
This project is part of the dissertation titled "Synthetic Image Detector Measuring Rectified Flow Trajectories" for Masters in Computer Science, University of Technology Malaysia. This repository contains FLOWTRACER training code and comparison against SOTA.

1. Download this repository as zip. Create a folder called "FLOWTRACER" somewhere on your device, and extract the contents from the zip into it.
2. Create a folder called "FLOWTRACER" in your Google Drive. Upload all files from FLOWTRACER into it. It should look like this:
<img width="528" height="502" alt="Directory Image" src="https://github.com/user-attachments/assets/e13e83d6-676f-4218-8a59-0308a9f53a12" />

3. Use an IDE that could connect to Google Colab (VSCode). Select 'Open Folder' and choose the FLOWTRACER directory on your device.
4. To run FLOWTRACER and other SOTA on the prepared benchmarks, open 2. SOTA Comparison.ipynb in your IDE and follow the instructions inside. Keep in mind that compute units may run out, and it will take some time before it recharges. A red error message usually occurs when no more compute units are available.
5. To see how FLOWTRACER is trained/ train FLOWTRACER yourself, open 1. FLOWTRACER Training.ipynb and follow the instructions inside.
6. Due to the large sizes, FLOWTRACER weights, Training Features and the benchmark images will be hosted on Hugging Face. The sources used in the experiment will be linked under Extras/Feature Extraction.ipynb. The code used to extract DINOv2 features are also included.
7. Ablation.ipynb and CLIP matching.ipynb are included in Extras for reference. Experiments requiring the CNN trajectory cache (5gb+) are omitted. Included are: 8D metrics vs CNN, FLOWTRACER vs DINOv2
