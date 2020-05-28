from typing import List, Tuple, Optional
from ngslite import Chromosome, FeatureArray, GenericFeature, translate, rev_comp
from lstm_dna.predictor import Predictor


class DNARegion:

    def __init__(
            self,
            chromosome_position: int,
            seq: str,
            min_protein_len: int):
        """
        Args:
            chromosome_position: 1-based chromosome position
            seq: DNA sequence
            min_protein_len: min protein length
        """
        self.chromosome_position = chromosome_position
        self.seq = seq
        self.min_protein_len = min_protein_len

    def get_left_protein(self) -> Optional[str]:
        prot = translate(self.seq).split('*')[0] + '*'

        if len(prot) >= self.min_protein_len:
            return prot
        else:
            return None

    def get_right_protein(self):

        # Left strip extra (1 or 2) bases
        extra = len(self.seq) % 3
        prot = translate(self.seq[extra:])

        if not prot.endswith('*'):
            return None

        prot = prot[:-1]  # rstrip '*'
        prot = prot.split('*')[-1]  # get the last orf
        pos = prot.find('M')
        prot = prot[pos:] + '*'

        if len(prot) >= self.min_protein_len:
            return prot
        else:
            return None

    def get_left_region(
            self,
            left_protein: str) -> Tuple[int, int]:

        cds_len = len(left_protein) * 3

        start = self.chromosome_position
        end = self.chromosome_position + cds_len - 1  # -1 for 1-based inclusive

        return start, end

    def get_right_region(
            self,
            right_protein: str) -> Tuple[int, int]:

        cds_len = len(right_protein) * 3
        last_chromosome_pos = self.chromosome_position + len(self.seq) - 1  # 1-based

        end = last_chromosome_pos
        start = end - cds_len + 1  # +1 because inclusive index

        return start, end

    def get_orf_regions(self) -> List[Tuple[int, int]]:

        ret = []

        left_protein = self.get_left_protein()
        right_protein = self.get_right_protein()

        if left_protein is not None:
            lr = self.get_left_region(left_protein=left_protein)
            ret.append(lr)

        if right_protein is not None:
            rr = self.get_right_region(right_protein=right_protein)
            ret.append(rr)

        return ret


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
