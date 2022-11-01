#!/usr/bin/env bash

# Help command
if [[ ("$1" == "-h") || ("$1" == "--help") ]]; then
  echo "Usage: ./`basename $0` [-wd] [-v] [-cv] [-fv] [-mv] [-cb] [-fb] [-mb] [-sm] [-im] [-dm] [-r] [-g] [-df] [-o]"
  echo "    -w,--workdir : Path to working directory."
  echo "    -v,--variant-list : Path to the list of variants."
  echo "    -cv,--child-vcf : Path to child vcf file, should be specified if --variant_list is not passed."
  echo "    -fv,--father-vcf : Path to father vcf file, should be specified if --variant_list is not passed."
  echo "    -mv,--mother-vcf : Path to mother vcf file, should be specified if --variant_list is not passed."
  echo "    -cb,--child-bam : Path to child bam/cram file."
  echo "    -fb,--father-bam : Path to father bam/cram file."
  echo "    -mb,--mother-bam : Path to mother bam/cram file."
  echo "    -sm,--snp-model : Path to substitution model."
  echo "    -im,--in-model : Path to insertion model."
  echo "    -dm,--del-model : Path to deletion model."
  echo "    -r,--region : Chromosome to analyse (1, 2, ... 22, X)."
  echo "    -g,--genome : Reference genome path."
  echo "    -df,--output_denovocnn_format: true or false,  default: false. If set to true, the output file will contain normalized variants and end coordinate."
  echo "    -o,--output : Output file name (will be saved to workdir)."
  exit 0
fi

# Parsing of the input arguments
for i in "$@"
do
case $i in
    -w=*|--workdir=*)
    WORKDIR="${i#*=}"
    shift
    ;;
    -v=*|--variant-list=*)
    VARIANTS_LIST="${i#*=}"
    shift
    ;;
    -cv=*|--child-vcf=*)
    CHILD_VCF="${i#*=}"
    shift
    ;;
    -fv=*|--father-vcf=*)
    FATHER_VCF="${i#*=}"
    shift
    ;;
    -mv=*|--mother-vcf=*)
    MOTHER_VCF="${i#*=}"
    shift
    ;;
    -cb=*|--child-bam=*)
    CHILD_BAM="${i#*=}"
    shift
    ;;
    -fb=*|--father-bam=*)
    FATHER_BAM="${i#*=}"
    shift
    ;;
    -mb=*|--mother-bam=*)
    MOTHER_BAM="${i#*=}"
    shift
    ;;
    -sm=*|--snp-model=*)
    SNP_MODEL="${i#*=}"
    shift
    ;;
    -im=*|--in-model=*)
    IN_MODEL="${i#*=}"
    shift
    ;;
    -dm=*|--del-model=*)
    DEL_MODEL="${i#*=}"
    shift
    ;;
    -r=*|--region=*)
    REGION="${i#*=}"
    shift
    ;;
    -g=*|--genome=*)
    GENOME="${i#*=}"
    shift
    ;;
    -df=*|--output_denovocnn_format=*)
    OUTPUT_DENOVOCNN_FORMAT="${i#*=}"
    shift
    ;;
    -o=*|--output=*)
    OUTPUT="${i#*=}"
    shift
    ;;
    *)
          # unknown option
    ;;
esac
done


# Check the correctness of the arguments
if [[ (${VARIANTS_LIST} = "") ]]; then
  # VCFs
  if [[ (${CHILD_VCF} = "") ]]; then
      echo "Error: Path to child vcf file --child-vcf must be provided!"
      exit
  fi
  if [[ (${FATHER_VCF} = "") ]]; then
      echo "Error: Path to father vcf file --father-vcf must be provided!"
      exit
  fi
  if [[ (${MOTHER_VCF} = "")]]; then
      echo "Error: Path to mother vcf file --mother-vcf must be provided!"
      exit
  fi
fi

# BAMs
if [[ ${CHILD_BAM} = "" ]]; then
    echo "Error: Path to child bam file --child-bam must be provided!"
    exit
fi
if [[ ${FATHER_BAM} = "" ]]; then
    echo "Error: Path to father bam file --father-bam must be provided!"
    exit
fi
if [[ ${MOTHER_BAM} = "" ]]; then
    echo "Error: Path to mother bam file --mother-bam must be provided!"
    exit
fi

# MODELS
if [[ ${SNP_MODEL} = "" ]]; then
    echo "Error: Path to SNP model --snp-model must be provided!"
    exit
fi
if [[ ${IN_MODEL} = "" ]]; then
    echo "Error: Path to insertion model --in-model must be provided!"
    exit
fi
if [[ ${DEL_MODEL} = "" ]]; then
    echo "Error: Path to deletion model --del-model must be provided!"
    exit
fi

# Reference genome
if [[ ${GENOME} = "" ]]; then
    echo "Error: GENOME --genome must be provided!"
    exit
fi

# DeNovoCNN format
if [[ ${OUTPUT_DENOVOCNN_FORMAT} = "" ]]; then
    OUTPUT_DENOVOCNN_FORMAT="false"
fi

if [[ ${OUTPUT_DENOVOCNN_FORMAT} != "false" ]] && [[ ${OUTPUT_DENOVOCNN_FORMAT} != "true" ]]; then
    echo "Error: OUTPUT DENOVOCNN FORMAT--output_denovocnn_format must be true or false!"
    exit
fi

# Working directory
if [[ ${WORKDIR} = "" ]]; then
    echo "Error: WORKING DIRECTORY --workdir must be provided!"
    exit
fi

echo "Start preprocessing step..."

mkdir $WORKDIR

# create variant list file by excluding the obviously inherited variants

if [[ (${VARIANTS_LIST} = "") ]]; then

  echo "Creating variant list file by excluding the obviously inherited variants..."

  ## copy or create child gziped vcf
  if [ $CHILD_VCF =  "*gz" ]; then
      cp $CHILD_VCF $WORKDIR/child.vcf.gz
  else
      bcftools sort $CHILD_VCF > $WORKDIR/child.vcf
      bgzip $WORKDIR/child.vcf
  fi
  BGZIPPED_CHILD_VCF=$WORKDIR/child.vcf.gz
  tabix  -p vcf $BGZIPPED_CHILD_VCF

  ## copy or create father gziped vcf
  if [ $FATHER_VCF =  "*gz" ]; then
      cp $FATHER_VCF $WORKDIR/father.vcf.gz
  else
      bcftools sort $FATHER_VCF > $WORKDIR/father.vcf
      bgzip $WORKDIR/father.vcf
  fi
  BGZIPPED_FATHER_VCF=$WORKDIR/father.vcf.gz
  tabix  -p vcf $BGZIPPED_FATHER_VCF

  ## copy or create mother gziped vcf
  if [ $MOTHER_VCF =  "*gz" ]; then
      cp $MOTHER_VCF $WORKDIR/mother.vcf.gz
  else
      bcftools sort $MOTHER_VCF > $WORKDIR/mother.vcf
      bgzip $WORKDIR/mother.vcf
  fi
  BGZIPPED_MOTHER_VCF=$WORKDIR/mother.vcf.gz
  tabix  -p vcf $BGZIPPED_MOTHER_VCF

  ## create variant list file by excluding the obviously inherited variants
  if [[ ${REGION} = "" ]]; then
      bcftools isec -C $BGZIPPED_CHILD_VCF $BGZIPPED_FATHER_VCF $BGZIPPED_MOTHER_VCF > $WORKDIR/variants_list.txt
      VARIANTS_LIST=$WORKDIR/variants_list.txt
  else
      bcftools isec -r $REGION -C $BGZIPPED_CHILD_VCF $BGZIPPED_FATHER_VCF $BGZIPPED_MOTHER_VCF > $WORKDIR/variants_list_chr${REGION}.txt
      bcftools isec -r chr${REGION} -C $BGZIPPED_CHILD_VCF $BGZIPPED_FATHER_VCF $BGZIPPED_MOTHER_VCF >> $WORKDIR/variants_list_chr${REGION}.txt
      VARIANTS_LIST=$WORKDIR/variants_list_chr${REGION}.txt
  fi

  echo "Variants list created in:"
  echo $VARIANTS_LIST
fi

echo "Preprocessing step finished."
echo "Running DenovoCNN..."

script_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Run Python command
KERAS_BACKEND=tensorflow python $script_path/main.py \
--mode=predict \
--ref_genome=$GENOME \
--child_bam=$CHILD_BAM \
--father_bam=$FATHER_BAM \
--mother_bam=$MOTHER_BAM \
--snp_model=$SNP_MODEL \
--ins_model=$IN_MODEL \
--del_model=$DEL_MODEL \
--variants_list=$VARIANTS_LIST \
--output_denovocnn_format=$OUTPUT_DENOVOCNN_FORMAT \
--output_path=$WORKDIR/$OUTPUT

echo "DenovoCNN finished."
echo "Output in:"
echo $WORKDIR/$OUTPUT

if [[ (${VARIANTS_LIST} = "") ]]; then
  # Cleanup
  rm $WORKDIR/child.vcf.gz*
  rm $WORKDIR/father.vcf.gz*
  rm $WORKDIR/mother.vcf.gz*
fi
