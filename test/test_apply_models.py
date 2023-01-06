import os
import unittest

from denovonet.dataset import apply_models_on_trio_onnx

PATH_TO_FIXTURES = os.path.join("test", "fixtures")
PATH_TO_BAM = os.path.join(PATH_TO_FIXTURES, "NA12878.chr22.tiny.bam")
PATH_TO_CRAM = os.path.join(PATH_TO_FIXTURES, "NA12878.chr22.tiny.cram")
PATH_TO_VCF = os.path.join(PATH_TO_FIXTURES, "NA12878.chr22.tiny.giab.vcf")


class TestONNXModels(unittest.TestCase):
    pass
    # def test_apply_models_on_trio(self):
    #     variants_list = ""
    #     output_path = ""
    #     child_bam = ""
    #     father_bam = ""
    #     mother_bam = ""
    #     snp_model = ""
    #     del_model = ""
    #     ins_model = ""
    #     ref_genome = ""
    #     output_denovocnn_format = ""
    #     convert_to_inner_format = ""
    #     n_jobs = 1
    #     apply_models_on_trio_onnx(
    #         variants_list,
    #         output_path,
    #         child_bam,
    #         father_bam,
    #         mother_bam,
    #         snp_model,
    #         del_model,
    #         ins_model,
    #         ref_genome,
    #         output_denovocnn_format,
    #         convert_to_inner_format,
    #         n_jobs,
    #     )


if __name__ == "__main__":
    unittest.main()
