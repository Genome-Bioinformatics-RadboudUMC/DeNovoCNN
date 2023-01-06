# DeNovoCNN

A deep learning approach to call de novo mutations (DNMs) on whole-exome (WES) and whole-genome sequencing (WGS) data. DeNovoCNN uses trio BAM/CRAM + VCF (or tab-separated list of variants) files to generate image-like genomic sequence representations and detect DNMs with high accuracy. <br>
<br>
DeNovoCNN is a combination of three models for the calling of substitution, deletion and insertion DNMs. Each of the model is a 9-layers CNN with [squeeze-and-excitation](https://arxiv.org/pdf/1709.01507.pdf) blocks. DeNovoCNN is trained on ~50k manually curated DNM and IV (inherited and non-DNM variants) sequencing data, generated using [Illumina](https://www.illumina.com/) sequencer and [Sureselect Human
All Exon V5](https://www.agilent.com/cs/library/datasheets/public/AllExondatasheet-5990-9857EN.pdf)/[Sureselect Human
All Exon V4](https://www.agilent.com/cs/library/flyers/Public/5990-9857en_lo.pdf) capture kits.  <br>
<br>
DeNovoCNN returns a tab-separated file of format:
> Chromosome | Start position | End position | Reference | Variant | DNM posterior probability | Mean coverage

We used **DNM posterior probability >= 0.5** to create a filtered tab-separated file with the list of variants that are likely to be *de novo*.

## Versions

1.0 corresponds to a version that is used in the [publication](https://doi.org/10.1093/nar/gkac511).

2.0 is currently under development which has several important changes: improved accuracy, pytorch implementation, ONNX optimization for faster inference.

## How does it work?

DeNovoCNN reads BAM files and iterates through potential DNM locations using the input VCF files to generate snapshots of genomic regions. It stacks trio BAM files to generate and RGB image representation which are passed into a CNN with squeeze-and-excitation blocks to classify each image as either DNM or IV (inherited variant, non-DNM).<br>
<br>

## Manual installation
We advise to use our docker container (see Usage section). In case it's not possible, the easiest way of installing is creating an [Anaconda](https://www.anaconda.com/) environment.

```bash
#Create environment
cd .../DeNovoCNN
conda env create -f environment.yml
conda activate tensorflow_env
```

## Usage

### Docker

DeNovoCNN is available as a docker container.

The example of DeNovoCNN usage for prediction (to use pretrained models, corresponding arguments shoud remain unchanged):
```bash
docker run \
  -v "YOUR_INPUT_DIRECTORY":"/input" \
  -v "YOUR_OUTPUT_DIRECTORY:/output" \
  gelana/denovocnn:1.0 \
  /app/apply_denovocnn.sh\
    --workdir=/output \
    --child-vcf=/input/<CHILD_VCF> \
    --father-vcf=/input/<FATHER_VCF> \
    --mother-vcf=/input/<MOTHER_VCF> \
    --child-bam=/input/<CHILD_BAM> \
    --father-bam=/input/<FATHER_BAM> \
    --mother-bam=/input/<MOTHER_BAM> \
    --snp-model=/app/models/snp \
    --in-model=/app/models/ins \
    --del-model=/app/models/del \
    --genome=/input/<REFERENCE_GENOME> \
    --output=predictions.csv
```
Parameters description and usage are described earlier in the previous section.

### Singularity

```bash
singularity build denovocnn.sif docker://gelana/denovocnn:1.0
```

```bash
singularity run -B YOUR_INPUT_DIRECTORY:/input,YOUR_OUTPUT_DIRECTORY:/output \
    denovocnn.sif \
    /app/apply_denovocnn.sh \
    --workdir=/output \
    --child-vcf=/input/<CHILD_VCF> \
    --father-vcf=/input/<FATHER_VCF> \
    --mother-vcf=/input/<MOTHER_VCF> \
    --child-bam=/input/<CHILD_BAM> \
    --father-bam=/input/<FATHER_BAM> \
    --mother-bam=/input/<MOTHER_BAM> \
    --snp-model=/app/models/snp \
    --in-model=/app/models/ins \
    --del-model=/app/models/del \
    --genome=/input/<REFERENCE_GENOME> \
    --output=predictions.csv
```


### Manual prediction
To use the pretrained models, you can provide the paths to the models from 'models' folder.

If you're running DeNovoCNN on WGS data, it is recommended to split the VCF files or variants of interest into 10 or more parts and run each of them separately and if possible in parallel. The separation could be done using the following commands:
```bash
   bcftools isec -C $BGZIPPED_CHILD_VCF $BGZIPPED_FATHER_VCF $BGZIPPED_MOTHER_VCF > all_variants.txt
   split -d -l 10000 --additional-suffix=.txt all_variants.txt part_variants

```
The resulting list of variants could be passed as `-v` parameter. <br>
<br>
To run DeNovoCNN on all possible locations:
```bash
cd .../DeNovoCNN

./apply_denovocnn.sh \
-w=<WORKING_DIRECTORY> \
-cv=<CHILD_VCF> \
-fv=<FATHER_VCF> \
-mv=<MOTHER_VCF> \
-cb=<CHILD_BAM> \
-fb=<FATHER_BAM> \
-mb=<MOTHER_BAM> \
-sm=<SNP_MODEL> \
-im=<INSERTION_MODEL> \
-dm=<DELETION_MODEL> \
-g=<REFERENCE_GENOME> \
-o=predictions.csv
```

To run DeNovoCNN on a specified list (VARIANT_LIST_TSV) of locations:

```bash
./apply_denovocnn.sh \
-w=<WORKING_DIRECTORY> \
-v=<VARIANT_LIST_TSV>
-cb=<CHILD_BAM> \
-fb=<FATHER_BAM> \
-mb=<MOTHER_BAM> \
-sm=<SNP_MODEL> \
-im=<INSERTION_MODEL> \
-dm=<DELETION_MODEL> \
-g=<REFERENCE_GENOME> \
-o=predictions.csv
```
VARIANT_LIST_TSV is a tab-separated file of format:
> Chromosome | Start position | Reference | Variant | Additional info

It could be generated by filtering the locations of interest of the result of this command:

```bash
   bcftools isec -C $BGZIPPED_CHILD_VCF $BGZIPPED_FATHER_VCF $BGZIPPED_MOTHER_VCF > all_variants_list.txt
```

## Citation
If you use any of the materials in the repository, we would appreciate it if you cited our [manuscript](https://doi.org/10.1093/nar/gkac511).

## Version 2.0
In order to setup version 2.0 locally it is recommended to create a virtual environment using python3.10.<br>
Use requirements-gpu.txt if GPU is available. You may need to make some adjustments depending on the available torch and torchvision versions. torch and torchvision versions can be found at [torch](https://download.pytorch.org/whl/torch/) and [torchvision](https://download.pytorch.org/whl/torchvision/), respectively.<br>
If you prefer the CPU version, use requirements-cpu.txt.

```python
sudo apt-get install python3.10-venv
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
# If you only want to use CPU implementation use this: pip install -r requirements-cpu.txt
sudo apt install bcftools
sudo apt install tabix
```

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

Copyright (c) 2023 Karolis Sablauskas <br>
Copyright (c) 2023 Gelana Khazeeva
