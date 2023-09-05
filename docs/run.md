# Instruction for Code usage 

## Setup
- Get Code
```shell
 git clone https://github.com/Naiftt/FedSIS.git
```
- Build Environment
```shell
cd FedSIS
conda env create -f environment.yaml
conda activate fedsis
```


## Dataset Pre-Processing
Please refer to [datasets.md](datasets.md) for instruction on downloading and pre-processing each client data.



## Pre-trained weights
Download the below model and add it in the [FedSIS](../) directory
- Pretrained CelebA Model on 90% of the data: [Download](https://drive.google.com/file/d/1kqDrT9Wrh_xVlG4-NPt9NNeEU9r_Xfz5/view?usp=drive_link)
- Pretrained CelebA Model trained on 100% of the Data: [Download](https://drive.google.com/file/d/1HI2kTZ67DH7k75fqDoLpCv9kh7-spmfX/view?usp=drive_link)
- Splitted CelebA Dataset: [Download](https://drive.google.com/file/d/1tw-dgu8_UNz-d58wHIn_6YQLV9VcYO02/view?usp=drive_link)



## Training 
We provide [scripts](../train_scripts) to execute different experiments.
```shell
# Navigate to the folder
cd train_scripts
```

### Standard Benchmarks
#### (1) Training with MCIO datasets
- Benchmark 1 contains MSU-MFSD (M), CASIA-MFSD (C), Idiap Replay Attack (I), and OULU-NPU (O) as the  4 clients.
- CelebA-Spoof is added as an auxiliary client.  
- We follow a leave-one-domain-out protocol. Hence we will have 4 combinations for this benchmark.
- When training, we will have 4 clients, where CelebA-Spoof will be the fourth client. Modify the _'combination'_ argument in the script with the current combination of interest. 
- To mitigate the impact of statistical variations and the non-deterministic nature of predictions in FedSIS, we repeat each experiement 3 times. Modify the argument _'seed'_ in the range [0-2] for each run.


```bash
chmod +x run_mcio.sh
# To see other arg options, please open the script file
./run_mcio.sh
```

#### (2) Training with WCS datasets
- Benchmark 2 contains WMCA (W), CASIA-CeFA (C), and CASIA-SURF (S) as the  3 clients.
- CelebA-Spoof is added as an auxiliary client.  
- We follow a leave-one-domain-out protocol. Hence we will have 3 combinations for this benchmark.
- When training, we will have 3 clients, where CelebA-Spoof will be the third client. Modify the _'combination'_ argument in the script with the current combination of interest. 
- To mitigate the impact of statistical variations and the non-deterministic nature of predictions in FedSIS, we repeat each experiement 3 times. Modify the argument _'seed'_ in the range [0-2] for each run.


```bash
chmod +x run_wcs.sh
# To see other arg options, please open the script file
./run_wcs.sh
```

### Effect of One Attack Per Client
- This setup is designed to challenge FedSIS further by ensuring that each client has only one attack type in addition to the existing non-iid data distribution. 
- We perform this on the MCIO datasets.
- We run this experiment under-two sub-settings with mild variations. (Change the _'datasetname'_  flag in the following script.)
    - mcio_exp1: When we split the attack samples of a client (say M) into 2 sub-client with equal halves of the data, the bonafide samples are the same(repeated) in both the sub-clients.
    - mcio_exp2: When we split the attack samples of a client (say M) into 2 sub-client with equal halves of the data, the bonafide samples are also split into two equal halves.

```bash
chmod +x run_mcio_exp.sh
# To see other arg options, please open the script file
./run_mcio_exp.sh
```

## Compute Metrics
Post training, use the following command to compute the final metrics.

```bash
cd ..
python eval.py --exp_path path/to/exp/folder/
```