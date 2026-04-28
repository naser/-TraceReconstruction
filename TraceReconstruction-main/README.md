#  Execution Trace Reconstruction Using Diffusion-Based Generative Models
This is the official repository for the Execution Trace Reconstruction Using Diffusion-Based Generative Models, accepted in ICSE 2025. In this work, we introduce a novel application of diffusion-based generative models for reconstructing traces, conduct a thorough evaluation of four models using several distinct system call sequence datasets, and compare against other approaches. 
## Datasets
The trace data used to make each dataset are publicly available:

Dataset | Publication | Data
--- | --- | --- 
*Phoronix Test Suites (PTS)*  - compress-gzip, ffmpeg, scimark2, stream, ramspeed, phpbench, pybench, iozone, and unpack-linux | [Automatic benchmark profiling through advanced workflow-based trace analysis](https://inria.hal.science/hal-02047273/document) by Alexis Martin, Vania Marangozova-Martin | [link](https://zenodo.org/records/437207)
*Apache* | [On Improving Deep Learning Trace Analysis with System Call Arguments](https://arxiv.org/pdf/2103.06915) by Quentin Fournier, Daniel Aloise, Seyed Vahid Azhari, and François Tetreault | [link](https://zenodo.org/records/4091287)
*PLAID* | [Methods for Host-based Intrusion Detection with Deep Learning](https://dl.acm.org/doi/full/10.1145/3461462) by John H. Ring, IV, Colin M. Van Oort, Samson Durst, Vanessa White, Joseph P. Near, and Christian Skalka | [link](https://gitlab.com/jhring/uvm_ids)
*ELK* | [Enhancing empirical software performance engineering research with kernel-level events: A comprehensive system tracing approach](https://www.sciencedirect.com/science/article/pii/S0164121224001626) by Morteza Noferesti and Naser Ezzati-Jivan | [link](https://github.com/mnoferestibrocku/dataset-repo)

The sequence datasets created using these sources are all found in the `Datasets` folder. 

## Training and Evaluation
The code used to train and evaluate the $`SSSD^{S4}`$, $`SSSD^{SA}`$, $`CSDI^{S4}`$, and DiffWave models can be found at [this GitHub repo](https://github.com/AI4HealthUOL/SSSD). The results obtained using these models using different sequences lengths (50, 100, 150, 200) and blackout sizes (1, 5, 10, 20, 30, 40) are found in the `Results` folder.
