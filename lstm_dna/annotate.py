import numpy as np
from typing import List, Tuple, Optional
from ngslite import Chromosome, FeatureArray, GenericFeature, translate, rev_comp
from lstm_dna.predictor import Predictor


class DNARegion:

    min_orf_coverage = 0.9
    start_codons = ('ATG', 'GTG', 'TTG')

    def __init__(
            self,
            chromosome_position: int,
            seq: str,
            min_protein_len: int):

        self.chromosome_position = chromosome_position
        self.seq = seq
        self.min_protein_len = min_protein_len

    def __init_orf_regions(self):

        ret = []
        seq = self.seq

        for frame in range(3):
            translation = translate(seq[frame:])
            proteins = translation.split('*')
            start = 1 + frame  # 1-based inclusive
            for prot in proteins:
                cds_len = len(prot) * 3 + 3  # including '*'
                if len(prot) > self.min_protein_len:
                    end = start + cds_len - 1
                    ret.append((start, end))
                start += cds_len

        return ret

    def __iteratively_retain(
            self,
            orf_regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Iteratively go through each ORF until the whole DNA region is covered
        """

        ret = []

        arr = np.zeros((len(self.seq), ))

        for start, end in orf_regions:
            ret.append((start, end))
            arr[start-1:end] = 1
            orf_coverage = np.sum(arr) / len(arr)
            if orf_coverage > self.min_orf_coverage:
                break

        return ret

    def __find_start_codon(
            self,
            start: int,
            end: int) -> int:

        p = start  # 1-based inclusive
        while p <= end - 2:
            codon = self.seq[p-1:p+2]
            if codon.upper() in self.start_codons:
                return p
            p += 3

        return start

    def __trim(
            self,
            orf_regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        For each ORF, trim to the start codon
        """

        ret = []

        for start, end in orf_regions:
            if start == 1:
                new_start = start  # Trust the model prediction
            else:
                new_start = self.__find_start_codon(start, end)
            ret.append((new_start, end))

        return ret

    def __shift(
            self,
            orf_regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Shift to the original genome location
        """

        shift = self.chromosome_position - 1
        return [(start + shift, end + shift) for start, end in orf_regions]

    def get_orf_regions(self) -> List[Tuple[int, int]]:

        orf_regions = self.__init_orf_regions()

        orf_regions = sorted(orf_regions, key=lambda x: x[1] - x[0], reverse=True)  # longest to shortest

        orf_regions = self.__iteratively_retain(orf_regions=orf_regions)

        orf_regions = self.__trim(orf_regions=orf_regions)

        orf_regions = self.__shift(orf_regions=orf_regions)

        return orf_regions


def prediction_to_regions(
        prediction: List[bool]) -> List[Tuple[int, int]]:

    regions = []

    is_plus = False
    start, end = None, None
    for i in range(len(prediction)):

        see_start = not is_plus and prediction[i]
        see_end = is_plus and not prediction[i]

        if see_start:
            is_plus = True
            start = i + 1

        elif see_end:
            is_plus = False
            end = i
            assert start and end
            assert end >= start
            regions.append((start, end))

    if is_plus:
        regions.append((start, len(prediction)))

    return regions


def is_proper_protein(protein: str) -> bool:
    p = protein
    a = p.endswith('*')
    b = '*' not in p[:-1]
    return a and b


def regions_to_features(
        regions: List[Tuple[int, int]],
        seqname: str,
        genome_size: int) -> FeatureArray:

    features = []
    for start, end in regions:
        f = GenericFeature(
            seqname=seqname,
            type_='CDS',
            start=start,
            end=end,
            strand='+')
        features.append(f)

    return FeatureArray(
        seqname=seqname,
        genome_size=genome_size,
        features=features,
        circular=True)


class CDSAnnotator:

    dna: str
    seqname: str
    circular: bool

    def __init__(
            self,
            predictor: Predictor,
            min_protein_len: int):

        self.predictor = predictor
        self.min_protein_len = min_protein_len

    def correct_regions(
            self,
            regions: List[Tuple[int, int]],
            dna: str) -> List[Tuple[int, int]]:

        ret = []

        for region in regions:
            start, end = region
            cds = dna[start - 1:end]

            protein = translate(cds)

            if len(protein) < self.min_protein_len:
                continue

            if is_proper_protein(protein=protein):
                ret.append(region)

            else:
                dna_region = DNARegion(
                    chromosome_position=start,
                    seq=cds,
                    min_protein_len=self.min_protein_len)
                ret += dna_region.get_orf_regions()

        return ret

    def get_features(self, dna: str) -> FeatureArray:

        prediction = self.predictor.predict(
            dna=dna)

        regions = prediction_to_regions(
            prediction=prediction)

        regions = self.correct_regions(
            regions=regions,
            dna=dna)

        features = regions_to_features(
            regions=regions,
            seqname=self.seqname,
            genome_size=len(dna)
        )

        return features

    def get_forward_features(self) -> FeatureArray:

        return self.get_features(dna=self.dna)

    def get_reverse_features(self) -> FeatureArray:

        features = self.get_features(rev_comp(self.dna))
        features.reverse()

        return features

    def annotate(
            self,
            dna: str,
            seqname: str,
            circular: bool) -> Chromosome:

        self.dna = dna
        self.seqname = seqname
        self.circular = circular

        features = self.get_forward_features() + self.get_reverse_features()

        return Chromosome(
            seqname=seqname,
            sequence=dna,
            features=features,
            circular=circular)
