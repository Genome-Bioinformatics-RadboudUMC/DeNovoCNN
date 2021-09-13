#!/usr/bin/env bash

# Help command
if [[ ("$1" == "-h") || ("$1" == "--help") ]]; then
  echo "Usage: ./`basename $0` [-wd] [-cv] [-fv] [-mv] [-cb] [-fb] [-mb] [-g] [-o]"
  echo "    -w,--workdir : Path to working directory."
  echo "    -vl,--variants-list : Path to the file containing list of the variants to check."
  echo "    -cv,--child-vcf : Path to child vcf file."
  echo "    -fv,--father-vcf : Path to father vcf file."
  echo "    -mv,--mother-vcf : Path to mother vcf file."
  echo "    -cb,--child-bam : Path to child bam file."
  echo "    -fb,--father-bam : Path to father bam file."
  echo "    -mb,--mother-bam : Path to mother bam file."
  echo "    -sm,--snp-model : Path to SNP model."
  echo "    -im,--in-model : Path to insertion model."
  echo "    -dm,--del-model : Path to deletion model."
  echo "    -g,--genome : Path to the reference genome fasta file."
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
    -vl=*|--variants-list=*)
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
    -g=*|--genome=*)
    GENOME="${i#*=}"
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
# VCFs
if [[ (${CHILD_VCF} = "") && (${VARIANTS_LIST} = "") ]]; then
    echo "Error: Path to child vcf file --child-vcf must be provided!"
    exit
fi
if [[ (${FATHER_VCF} = "") && (${VARIANTS_LIST} = "") ]]; then
    echo "Error: Path to father vcf file --father-vcf must be provided!"
    exit
fi
if [[ (${MOTHER_VCF} = "") && (${VARIANTS_LIST} = "") ]]; then
    echo "Error: Path to mother vcf file --mother-vcf must be provided!"
    exit
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

# Working directory
if [[ ${WORKDIR} = "" ]]; then
    echo "Error: WORKING DIRECTORY --workdir must be provided!"
    exit
fi

echo "Start preprocessing step..."

mkdir $WORKDIR

# Create variants list if no VARIANTS_LIST is specified
if [[ ${VARIANTS_LIST} = "" ]]; then
    # create variant list file by excluding the obviously inherited variants
    echo "Creating variant list file by excluding the obviously inherited variants..."
 
    ## copy or create child gziped vcf 
    if [ $CHILD_VCF =  *"gz" ]; then
        cp $CHILD_VCF $WORKDIR/child.vcf.gz
    else
        bcftools sort $CHILD_VCF > $WORKDIR/child.vcf
        bgzip $WORKDIR/child.vcf
    fi
    BGZIPPED_CHILD_VCF=$WORKDIR/child.vcf.gz
    tabix  -p vcf $BGZIPPED_CHILD_VCF
    
    ## copy or create father gziped vcf
    if [ $FATHER_VCF =  *"gz" ]; then
        cp $FATHER_VCF $WORKDIR/father.vcf.gz
    else
        bcftools sort $FATHER_VCF > $WORKDIR/father.vcf
        bgzip $WORKDIR/father.vcf
    fi
    BGZIPPED_FATHER_VCF=$WORKDIR/father.vcf.gz
    tabix  -p vcf $BGZIPPED_FATHER_VCF
    
    ## copy or create mother gziped vcf
    if [ $MOTHER_VCF =  *"gz" ]; then
        cp $MOTHER_VCF $WORKDIR/mother.vcf.gz
    else
        bcftools sort $MOTHER_VCF > $WORKDIR/mother.vcf
        bgzip $WORKDIR/mother.vcf
    fi
    BGZIPPED_MOTHER_VCF=$WORKDIR/mother.vcf.gz
    tabix  -p vcf $BGZIPPED_MOTHER_VCF
    
    ## create variant list file by excluding the obviously inherited variants
    bcftools isec -C $BGZIPPED_CHILD_VCF $BGZIPPED_FATHER_VCF $BGZIPPED_MOTHER_VCF > $WORKDIR/variants_list.txt
    VARIANTS_LIST=$WORKDIR/variants_list.txt
    
    echo "Variants list created in:"
    echo $VARIANTS_LIST
fi

echo "Preprocessing step finished."


echo "Running DenovoCNN..."

# Run Python command
KERAS_BACKEND=tensorflow python ./main.py \
--mode=predict \
--genome=$GENOME \
--child-bam=$CHILD_BAM \
--father-bam=$FATHER_BAM \
--mother-bam=$MOTHER_BAM \
--snp-model=$SNP_MODEL \
--in-model=$IN_MODEL \
--del-model=$DEL_MODEL \
--variants-list=$VARIANTS_LIST \
--create-filtered-file=true \
--output=$WORKDIR/$OUTPUT

echo "DenovoCNN finished."
echo "Output in:"
echo $WORKDIR/$OUTPUT


# Cleanup
rm $WORKDIR/child.vcf.gz*
rm $WORKDIR/father.vcf.gz*
rm $WORKDIR/mother.vcf.gz*
