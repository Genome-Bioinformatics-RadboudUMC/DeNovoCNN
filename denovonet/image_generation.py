import pandas as pd
import numpy as np
import pysam


# extract_bases returns all bases from a specified region of the provided bam file, as well as indels. This data is split up into phased reads of both haplotypes and unphased reads.
def extract_bases(bam_path, ref_path, chr, pos, radius):
    reference_genome = pysam.FastaFile(ref_path)
    hp_bam = pysam.AlignmentFile(bam_path, "rb", reference_filename=reference_genome.filename.decode('utf-8'))
    hp1_seq, hp2_seq, unp_seq = [], [], []
    hp1_seq_indel, hp2_seq_indel, unp_seq_indel = [], [], []
    pos = pos - 1

    for pileupcolumn in hp_bam.pileup(chr, pos - radius, pos + radius + 1, truncate=True):
        if pileupcolumn.pos in range(pos - radius, pos + radius + 1):
            unp_col, hp1_col, hp2_col = [], [], []
            unp_indel, hp1_indel, hp2_indel = [], [], []

            for pileupread in pileupcolumn.pileups:
                if not pileupread.is_refskip:
                    if not pileupread.is_del:
                        nb = pileupread.alignment.query_sequence[pileupread.query_position]

                        if (pileupread.alignment.has_tag('HP')):
                            if pileupread.alignment.get_tag('HP') == 1:
                                hp1_col.append(nb)
                                hp1_indel.append(pileupread.indel)
                            else:
                                hp2_col.append(nb)
                                hp2_indel.append(pileupread.indel)
                        else:
                            unp_col.append(nb)
                            unp_indel.append(pileupread.indel)

            hp1_seq.append(hp1_col)
            hp2_seq.append(hp2_col)
            unp_seq.append(unp_col)

            hp1_seq_indel.append(hp1_indel)
            hp2_seq_indel.append(hp2_indel)
            unp_seq_indel.append(unp_indel)
    hp_bam.close()

    return hp1_seq, hp2_seq, unp_seq, hp1_seq_indel, hp2_seq_indel, unp_seq_indel


# add_qual extracts the quality of each position and the read depth, adding those to the image.
def add_qual(image, bam_path, ref_path, chr, pos, radius, parent):
    # Get median quality for each position and haplotype
    reference_genome = pysam.FastaFile(ref_path)

    hp_bam = pysam.AlignmentFile(bam_path, "rb", reference_filename=reference_genome.filename.decode('utf-8'))
    hp1_q, hp2_q, unp_q = [], [], []
    hp1_n, hp2_n, unp_n = [], [], []
    pos = pos - 1

    for pileupcolumn in hp_bam.pileup(chr, pos - radius, pos + radius + 1, truncate=True):
        if pileupcolumn.pos in range(pos - radius, pos + radius + 1):
            unp_col, hp1_col, hp2_col = [], [], []

            for pileupread in pileupcolumn.pileups:
                if not pileupread.is_refskip:
                    if not pileupread.is_del:
                        read_q = pileupread.alignment.query_qualities[pileupread.query_position]

                        if (pileupread.alignment.has_tag('HP')):
                            if pileupread.alignment.get_tag('HP') == 1:
                                hp1_col.append(read_q)
                            else:
                                hp2_col.append(read_q)
                        else:
                            unp_col.append(read_q)

            hp1_q.append(np.median(hp1_col))
            hp2_q.append(np.median(hp2_col))
            unp_q.append(np.median(unp_col))

            hp1_n.append(len(hp1_col))
            hp2_n.append(len(hp2_col))
            unp_n.append(len(unp_col))
    hp_bam.close()

    # Split qualities
    if parent:
        qual1 = [np.mean([x, y, z]) if not (np.isnan(x) or np.isnan(y) or np.isnan(z)) else max(0, x, y, z) for x, y, z
                 in zip(hp1_q, hp2_q, unp_q)]
        qual2 = qual1

        n1 = hp1_n
        n2 = hp2_n
    else:
        qual1 = [np.mean([x, x, y]) if not (np.isnan(x) or np.isnan(y)) else max(0, x, y) for x, y in
                 zip(hp1_q, unp_q)]  # Weighted to favour phased reads
        qual2 = [np.mean([x, x, y]) if not (np.isnan(x) or np.isnan(y)) else max(0, x, y) for x, y in
                 zip(hp2_q, unp_q)]  # 0 in front because max nan shenanigans

        n1 = [int(x + 0.5 * y) for x, y in zip(hp1_n, unp_n)]
        n2 = [int(x + 0.5 * y) for x, y in zip(hp2_n, unp_n)]

    # Add qual to image
    for idx, _ in enumerate(qual1):
        if not np.isnan(qual1[idx]):
            image[idx][0] = int(qual1[idx])
        if not np.isnan(qual2[idx]):
            image[idx][8] = int(qual2[idx])
        image[idx][1] = n1[idx]
        image[idx][9] = n2[idx]

    return image


# Adds deletions to the correct locations.
def add_deletions(seq, indel_seq):
    for idx in range(0, len(indel_seq)):
        for indel in indel_seq[idx]:
            if indel < 0:  # Deletions are encoded as -x, where x is the length of the deletion
                for j in range(1, abs(indel) + 1):
                    if idx + j < len(seq):  # Add deletion to the correct locations
                        seq[idx + j].append('DEL')
    return seq


# Creates one encoded row of the image
def create_encoding_half(frac_list, parent):
    # Locations in the encoding are structured as follows: row 0 = quality encoding, 1 = read depth, 2:5 = bases, 6 = ins_n, 7 = ins_length.
    brightness = 255
    img = pd.DataFrame(np.zeros((8, len(frac_list))))
    for idx, column in enumerate(frac_list):
        for base in column:
            if base[0] == 'A':
                img[idx][2] = int(base[1] * brightness)
            if base[0] == 'C':
                img[idx][3] = int(base[1] * brightness)
            if base[0] == 'G':
                img[idx][4] = int(base[1] * brightness)
            if base[0] == 'T':
                img[idx][5] = int(base[1] * brightness)
    return img


# Adds the unphased reads to the proband image, at half the strength of phased reads. This way phased data is prioritized, but unphased data is not lost.
def add_unphased(hp, unp):
    combined = []
    for pos, col in enumerate(hp):
        a_freq, c_freq, g_freq, t_freq = 0, 0, 0, 0
        for base in col:
            if base[0] == 'A':
                a_freq += base[1]
            if base[0] == 'C':
                c_freq += base[1]
            if base[0] == 'G':
                g_freq += base[1]
            if base[0] == 'T':
                t_freq += base[1]
        for base in unp[pos]:
            if base[0] == 'A':
                a_freq = np.mean([base[1], a_freq, a_freq])
            if base[0] == 'C':
                c_freq = np.mean([base[1], c_freq, c_freq])
            if base[0] == 'G':
                g_freq = np.mean([base[1], g_freq, g_freq])
            if base[0] == 'T':
                t_freq = np.mean([base[1], t_freq, t_freq])
        combined.append([('A', a_freq), ('C', c_freq), ('G', g_freq), ('T', t_freq)])
    return combined


def proband_imgen(hp1, hp2, unp):
    # Get frequency of each base for each haplotype
    hp1 = [[base.upper() for base in pos] for pos in hp1]
    frac_list1 = [[(base, pos.count(base) / len(pos)) for base in set(pos)] for pos in hp1]

    hp2 = [[base.upper() for base in pos] for pos in hp2]
    frac_list2 = [[(base, pos.count(base) / len(pos)) for base in set(pos)] for pos in hp2]

    unp = [[base.upper() for base in pos] for pos in unp]
    frac_list_unp = [[(base, pos.count(base) / len(pos)) for base in set(pos)] for pos in unp]

    # Add unphased reads at 0.5 strength
    frac_list1 = add_unphased(frac_list1, frac_list_unp)
    frac_list2 = add_unphased(frac_list2, frac_list_unp)

    # Generate image half
    image1 = create_encoding_half(frac_list1, False)
    image2 = create_encoding_half(frac_list2, False)

    # Complete image
    image = pd.concat([image1, image2], ignore_index=True)

    return image


def parent_imgen(hp1, hp2, unp):
    # Combine all haplotypes
    seq = []
    width = max(len(hp1), len(hp2), len(unp))
    for idx in range(0, width):
        pos = []
        if idx < len(hp1):
            pos += hp1[idx]
        if idx < len(hp2):
            pos += hp2[idx]
        if idx < len(unp):
            pos += unp[idx]
        seq.append(pos)

    # Get frequency of each base
    seq = [[base.upper() for base in pos] for pos in seq]
    frac_list = [[(base, pos.count(base) / len(pos)) for base in set(pos)] for pos in seq]

    # Generate image half
    image = create_encoding_half(frac_list, True)

    # Complete image
    image = pd.concat([image, image], ignore_index=True)

    return image


# add_ins adds insertion data to the image
def add_ins(image, hp1, hp2, unp, parent):
    # Seperate haplotypes
    seq1 = []
    seq2 = []
    if parent:
        width = max(len(hp1), len(hp2), len(unp))
        for idx in range(0, width):
            pos = []
            if idx < len(hp1):
                pos += hp1[idx]
            if idx < len(hp2):
                pos += hp2[idx]
            if idx < len(unp):
                pos += unp[idx]
            seq1.append(pos)
    else:
        unp = [[x / 2 for x in pos] for pos in unp]
        width1 = max(len(hp1), len(unp))
        for idx in range(0, width1):
            pos = []
            if idx < len(hp1):
                pos += hp1[idx]
            if idx < len(unp):
                pos += unp[idx]
            seq1.append(pos)
        width2 = max(len(hp2), len(unp))
        for idx in range(0, width2):
            pos = []
            if idx < len(hp2):
                pos += hp2[idx]
            if idx < len(unp):
                pos += unp[idx]
            seq2.append(pos)

    # Extract length and number for each allele
    len1 = []  # Stores the fraction of reads containing an insertion for each location
    max1 = []  # Stores max length insertion for each location
    for pos in seq1:
        if len(pos) > 0:
            max1.append(max(pos))
        else:
            max1.append(0)
        ins_n = 0
        for read in pos:
            if read > 0:
                ins_n += 1
        if len(pos) == 0:
            len1.append(0)
        else:
            len1.append(ins_n / len(pos))

    if parent:
        len2 = len1
        max2 = max1
    else:
        len2 = []
        max2 = []
        for pos in seq2:
            if len(pos) > 0:
                max2.append(max(pos))
            else:
                max2.append(0)
            ins_n = 0
            for read in pos:
                if read > 0:
                    ins_n += 1
            if len(pos) == 0:
                len2.append(0)
            else:
                len2.append(ins_n / len(pos))

    # Add values to image
    for idx, _ in enumerate(len1):
        image[idx][6] = int(min(255, len1[idx] * 255))  # relative freq * 255
        image[idx][7] = int(max(min(255, max1[idx] * 10), 0))  # Largest ins at location * 10

        image[idx][14] = int(min(255, len2[idx] * 255))
        image[idx][15] = int(max(min(255, max2[idx] * 10), 0))

    return image


# The function that combines extracts all bam data and encodes the image for a single member of the trio
def encode_image(bam_path, ref_path, chr, pos, radius, parent=False):
    hp1_seq, hp2_seq, unp_seq, hp1_seq_indel, hp2_seq_indel, unp_seq_indel = extract_bases(bam_path, ref_path, chr, pos,
                                                                                           radius)

    # Add deletions to correct places
    hp1_final = add_deletions(hp1_seq, hp1_seq_indel)
    hp2_final = add_deletions(hp2_seq, hp2_seq_indel)
    unp_final = add_deletions(unp_seq, unp_seq_indel)

    # Encode bases
    if parent:
        image = parent_imgen(hp1_final, hp2_final, unp_final)
        # Add insertions + qual
        image = add_ins(image, hp1_seq_indel, hp2_seq_indel, unp_seq_indel, True)
        image = add_qual(image, bam_path, ref_path, chr, pos, radius, True)
    else:
        image = proband_imgen(hp1_final, hp2_final, unp_final)
        # Add insertions + qual
        image = add_ins(image, hp1_seq_indel, hp2_seq_indel, unp_seq_indel, False)
        image = add_qual(image, bam_path, ref_path, chr, pos, radius, False)
    return image


""" gen_trio_img takes the bam/cram paths of the trio members and the location, then returns an encoded image

Here cbam, fbam and mbam should be a string with the path to the respective bam/cram files.
Chromosome should be a string formatted like: 'chr22'.
Pos denotes the location of the variant as an integer.
Radius specifies the number of bases added on both sides of the target position, the final image has width: 2 * radius + 1 
"""


def gen_trio_img(cbam, fbam, mbam, ref_path, chromosome, pos, radius=50):
    cimg = encode_image(cbam, ref_path, chromosome, pos, radius, False)
    fimg = encode_image(fbam, ref_path, chromosome, pos, radius, True)
    mimg = encode_image(mbam, ref_path, chromosome, pos, radius, True)

    trio_img = np.zeros((16, radius * 2 + 1, 3))

    trio_img[:, :, 0] = cimg
    trio_img[:, :, 1] = fimg
    trio_img[:, :, 2] = mimg

    return trio_img
